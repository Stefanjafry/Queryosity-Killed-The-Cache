"""
Seed control for the neighbor-vs-random wrinkle.

At seed 42, neighbor refinement beat the random-weights floor on the larger
workloads (tpcds 262144/524288, job 262144).  Every BO run so far used seed
42, so that gap could be one lucky perturbation rather than real refinement.
This runner re-runs random and neighbor across several seeds for ONE config
and reports whether the neighbor-minus-random gap survives reseeding.

Matrices, the good-start set, and the greedy/multistart references are
seed-independent, so they are built once and reused across seeds; only the
random sampling and neighbor perturbation consume the seed.

Pre-registered verdict (per arm): the refinement SURVIVES if neighbor beats
random by > 0.1 pp at every seed (consistent, same sign); it WASHES OUT if
the mean gap is < 0.1 pp or the sign flips across seeds.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.bayesopt.bo_tuner import run_search
from src.bayesopt.data import load_workload_pages
from src.bayesopt.objective import ExactSimObjective
from src.bayesopt.step_scorer import (
    ScorerWeights,
    good_start_set,
    greedy_schedule,
    make_step_scorer,
    multistart_greedy_schedules,
    row_normalize,
)
from src.simulator.cache_simulator import (
    compute_directional_matrix,
    compute_overlap_matrix,
)
from src.utilities.constants import PROJECT_ROOT, WORKLOAD_DIRS

_SURVIVE_PP = 0.1


def _verdict(deltas_pp: list[float]) -> str:
    if all(d > _SURVIVE_PP for d in deltas_pp):
        return "SURVIVES"
    mean = sum(deltas_pp) / len(deltas_pp)
    if mean < _SURVIVE_PP or len({d > 0 for d in deltas_pp}) > 1:
        return "WASHES OUT"
    return "INCONCLUSIVE"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=list(WORKLOAD_DIRS), required=True)
    p.add_argument("--cache-pages", type=int, required=True)
    p.add_argument("--page-access-dir", type=Path, required=True)
    p.add_argument("--exclude", default="")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--num-starts", type=int, default=4)
    p.add_argument("--eval-budget", type=int, default=60)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--cache-mode", choices=["pressure", "fit"], default="fit")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "experiment_logs" / "bo_seedsweep")
    args = p.parse_args(argv)
    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print(f"Loading page sets from {args.page_access_dir}…")
    wp = load_workload_pages(args.page_access_dir, exclude=exclude)
    pc = [len(ps) for ps in wp.page_sets]
    print(f"  {wp.n} queries, C = {args.cache_pages:,} pages")

    t0 = time.perf_counter()
    M = compute_overlap_matrix(wp.page_sets)
    D = compute_directional_matrix(wp.page_sets, args.cache_pages)
    M_norm = row_normalize(M)
    D_norm = row_normalize(D)
    print(f"  matrices in {time.perf_counter() - t0:.1f}s (built once, reused)")

    obj = ExactSimObjective(wp.page_sets, args.cache_pages, d_matrix=D)

    def fhit(sched: list[int]) -> float:
        m, _ = obj.evaluate_schedule(sched)
        return m.hit_ratio

    def make_objective(immediate: list[list[float]]):
        starts = good_start_set(immediate, pc, args.num_starts)

        def objective(w: ScorerWeights) -> tuple[float, int]:
            score = make_step_scorer(immediate, w, pc, args.cache_pages)
            scheds = multistart_greedy_schedules(score, wp.n, pc, starts)
            return max(fhit(s) for s in scheds), len(scheds)

        return objective

    obj_m = make_objective(M_norm)
    obj_d = make_objective(D_norm)
    greedy_d = fhit(greedy_schedule(
        make_step_scorer(D_norm, ScorerWeights(), pc, args.cache_pages), wp.n, pc))

    # per seed: best_fhit[backend][arm]
    per_seed: list[dict[str, dict[str, float]]] = []
    for seed in seeds:
        row: dict[str, dict[str, float]] = {"random": {}, "neighbor": {}}
        for backend in ("random", "neighbor"):
            for arm, objective in (("M", obj_m), ("D", obj_d)):
                res = run_search(objective, backend, args.eval_budget,
                                 patience=args.patience, seed=seed,
                                 cache_mode=args.cache_mode)
                row[backend][arm] = res.best_fhit
        per_seed.append(row)
        print(f"  seed {seed}: "
              f"randD={row['random']['D']:.4f} neighD={row['neighbor']['D']:.4f} "
              f"(Δ {(row['neighbor']['D'] - row['random']['D']) * 100:+.3f}pp)  "
              f"randM={row['random']['M']:.4f} neighM={row['neighbor']['M']:.4f} "
              f"(Δ {(row['neighbor']['M'] - row['random']['M']) * 100:+.3f}pp)")

    d_deltas = [(r["neighbor"]["D"] - r["random"]["D"]) * 100 for r in per_seed]
    m_deltas = [(r["neighbor"]["M"] - r["random"]["M"]) * 100 for r in per_seed]
    d_verdict = _verdict(d_deltas)
    m_verdict = _verdict(m_deltas)

    summary = {
        "workload": args.workload, "cache_pages": args.cache_pages,
        "n_queries": wp.n, "seeds": seeds, "eval_budget": args.eval_budget,
        "greedy_D": greedy_d,
        "per_seed": [
            {"seed": s, "random": r["random"], "neighbor": r["neighbor"],
             "neigh_minus_rand_pp": {
                 "M": (r["neighbor"]["M"] - r["random"]["M"]) * 100,
                 "D": (r["neighbor"]["D"] - r["random"]["D"]) * 100}}
            for s, r in zip(seeds, per_seed)],
        "neigh_minus_rand_mean_pp": {
            "M": sum(m_deltas) / len(m_deltas),
            "D": sum(d_deltas) / len(d_deltas)},
        "verdict": {"M": m_verdict, "D": d_verdict},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"seedsweep_{args.workload}_{args.cache_pages}.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n=== seed sweep: {args.workload} {args.cache_pages} "
          f"(neighbor vs random, budget={args.eval_budget}) ===")
    print(f"  neighbor-random gap per seed:  D {[round(d, 3) for d in d_deltas]} "
          f" M {[round(m, 3) for m in m_deltas]}")
    print(f"  mean gap:  D {sum(d_deltas) / len(d_deltas):+.3f}pp  "
          f"M {sum(m_deltas) / len(m_deltas):+.3f}pp")
    print(f"  VERDICT (>{_SURVIVE_PP}pp every seed):  D {d_verdict}   M {m_verdict}")
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
