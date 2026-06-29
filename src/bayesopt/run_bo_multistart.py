"""
BO on the multistart-greedy consumer — paired M vs D.

Each BO trial proposes scorer weights; multistart-greedy builds one
schedule per good-start (K starts), the exact simulator scores all K, the
trial value is the best F_hit, and the trial costs K exact evals.  BO-M
and BO-D share the search space and differ only in the immediate matrix.

Reports each backend (random floor, neighbor, optional tpe) at equal eval
budget for both matrices, against the references greedy-D and
multistart-greedy-D (warm-up greedy-equivalent).  The decisive reads:
does a learning backend beat the random floor, does BO-D beat BO-M, and
does either clear the references — all judged by the exact simulator.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from src.bayesopt.bo_tuner import (
    BOResult,
    convergence_of,
    history_records,
    run_search,
    weights_from_vec,
)
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


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=list(WORKLOAD_DIRS), required=True)
    p.add_argument("--cache-pages", type=int, required=True)
    p.add_argument("--page-access-dir", type=Path, required=True)
    p.add_argument("--exclude", default="")
    p.add_argument("--num-starts", type=int, default=4)
    p.add_argument("--eval-budget", type=int, default=200,
                   help="exact-sim evaluations per backend per matrix")
    p.add_argument("--backends", default="random,neighbor",
                   help="comma-separated: random,neighbor,tpe")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-mode", choices=["pressure", "fit"], default="fit")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "experiment_logs" / "bo_multistart")
    args = p.parse_args(argv)
    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    print(f"Loading page sets from {args.page_access_dir}…")
    wp = load_workload_pages(args.page_access_dir, exclude=exclude)
    pc = [len(ps) for ps in wp.page_sets]
    print(f"  {wp.n} queries, C = {args.cache_pages:,} pages")

    t0 = time.perf_counter()
    M = compute_overlap_matrix(wp.page_sets)
    D = compute_directional_matrix(wp.page_sets, args.cache_pages)
    M_norm = row_normalize(M)
    D_norm = row_normalize(D)
    print(f"  matrices in {time.perf_counter() - t0:.1f}s")

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

        return objective, starts

    # references
    greedy_d = fhit(greedy_schedule(
        make_step_scorer(D_norm, ScorerWeights(), pc, args.cache_pages), wp.n, pc))
    greedy_m = fhit(greedy_schedule(
        make_step_scorer(M_norm, ScorerWeights(), pc, args.cache_pages), wp.n, pc))
    obj_m, starts_m = make_objective(M_norm)
    obj_d, starts_d = make_objective(D_norm)
    ms_greedy_d = obj_d(ScorerWeights())[0]   # multistart, aux weights 0
    ms_greedy_m = obj_m(ScorerWeights())[0]

    results: dict[str, dict[str, BOResult]] = {}
    for backend in backends:
        results[backend] = {}
        for arm, objective in (("M", obj_m), ("D", obj_d)):
            t = time.perf_counter()
            try:
                res = run_search(objective, backend, args.eval_budget,
                                 patience=args.patience, seed=args.seed,
                                 cache_mode=args.cache_mode)
            except Exception as exc:  # keep the rest of the run alive
                print(f"  [warn] {backend} BO-{arm} failed: {exc!r} — skipping")
                continue
            dt = time.perf_counter() - t
            results[backend][arm] = res
            print(f"  {backend:9s} BO-{arm}: best F_hit={res.best_fhit:.4f} "
                  f"({res.n_trials} trials / {res.n_evals} evals, "
                  f"stop={res.early_stop_reason}, {dt:.1f}s)")
        if not results[backend]:
            del results[backend]

    def serialize(r: BOResult) -> dict[str, object]:
        c = convergence_of(r)
        return {
            "best_fhit": r.best_fhit, "best_vec": r.best_vec,
            "n_trials": r.n_trials, "n_evals": r.n_evals,
            "early_stop": r.early_stop_reason,
            "convergence": {
                "best_trial_index": c.best_trial_index,
                "best_from_warmup": c.best_from_warmup,
                "n_warmup": c.n_warmup, "dominant_term": c.dominant_term,
                "best_so_far": c.best_so_far},
            "history": history_records(r)}

    summary = {
        "workload": args.workload, "cache_pages": args.cache_pages,
        "n_queries": wp.n, "num_starts": args.num_starts,
        "eval_budget": args.eval_budget, "seed": args.seed,
        "references": {
            "greedy_M": greedy_m, "greedy_D": greedy_d,
            "multistart_greedy_M": ms_greedy_m,
            "multistart_greedy_D": ms_greedy_d,
            "good_starts_M": starts_m, "good_starts_D": starts_d},
        "backends": {b: {arm: serialize(r) for arm, r in arms.items()}
                     for b, arms in results.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "bo_multistart_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    # flat per-trial CSV across every backend/arm for easy plotting
    csv_path = args.out_dir / "bo_multistart_trials.csv"
    fields = ["backend", "arm", "trial", "is_warmup", "evals", "fhit",
              "best_so_far", "w_regret", "w_cache", "w_connector", "connector_rho"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for b, arms in results.items():
            for arm, r in arms.items():
                for rec in history_records(r):
                    row: dict[str, object] = {"backend": b, "arm": arm}
                    row.update({k: rec.get(k, "") for k in fields if k not in row})
                    writer.writerow(row)

    print(f"\n=== BO on multistart-greedy (K={args.num_starts}, "
          f"budget={args.eval_budget} evals) ===")
    print(f"  refs: greedy_M={greedy_m:.4f} greedy_D={greedy_d:.4f}  "
          f"ms_greedy_M={ms_greedy_m:.4f} ms_greedy_D={ms_greedy_d:.4f}")
    rnd = results.get("random")
    for backend in results:
        if "M" not in results[backend] or "D" not in results[backend]:
            continue
        rm = results[backend]["M"].best_fhit
        rd = results[backend]["D"].best_fhit
        dm = (rd - rm) * 100.0
        vs_g = (rd - greedy_d) * 100.0
        line = (f"  {backend:9s} BO-M={rm:.4f} BO-D={rd:.4f}  "
                f"D-M {dm:+.3f}pp  D vs greedy_D {vs_g:+.3f}pp")
        if (rnd is not None and backend != "random"
                and "M" in rnd and "D" in rnd):
            line += (f"  vs random: M {(rm - rnd['M'].best_fhit) * 100:+.3f} "
                     f"D {(rd - rnd['D'].best_fhit) * 100:+.3f}pp")
        print(line)
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
