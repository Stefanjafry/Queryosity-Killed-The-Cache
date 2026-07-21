"""
Export schedules for wall-clock runs (run_sweep schedules-file format).

Regenerates the chosen method orderings for one (workload, cache) — exactly
the schedules the full grid produced, since everything is deterministic under
the same seed — and writes them as query-id strings in the format
``run_sweep.py`` expects:

    {"cache_label": "<cache>", "schedules": {"<method>": "q1,q3,q2,...", ...}}

Methods are the four the paper compares — GA_M (Queryosity baseline),
GA_D, sweep_D, sweep_beam_D — plus a seeded random floor. Real runs are
~20 min each, so export only the schedules a given experiment needs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.bayesopt.data import load_workload_pages
from src.bayesopt.objective import ExactSimObjective
from src.bayesopt.run_baselines import compute_baseline_schedule
from src.bayesopt.step_scorer import (
    ScorerWeights,
    beam_search_schedule,
    good_start_set,
    make_step_scorer,
    multistart_greedy_schedules,
    row_normalize,
)
from src.simulator.cache_simulator import compute_directional_matrix
from src.utilities.constants import WORKLOAD_DIRS

ALL_METHODS = ["random", "GA_M", "GA_D", "sweep_D", "sweep_beam_D"]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=list(WORKLOAD_DIRS), required=True)
    p.add_argument("--cache-pages", type=int, required=True)
    p.add_argument("--page-access-dir", type=Path, required=True)
    p.add_argument("--exclude", default="")
    p.add_argument("--methods", default="GA_M,GA_D,sweep_D,sweep_beam_D",
                   help=f"comma-separated subset of: {','.join(ALL_METHODS)}")
    p.add_argument("--num-starts", type=int, default=4)
    p.add_argument("--regret-grid", default="",
                   help="w_regret sweep grid; blank = default fine grid.")
    p.add_argument("--sweep-beam-widths", default="2,3,4")
    p.add_argument("--ga-pop", type=int, default=100)
    p.add_argument("--ga-gen", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}
    want = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in want:
        if m not in ALL_METHODS:
            raise SystemExit(f"unknown method {m!r}; choose from {ALL_METHODS}")

    print(f"Loading {args.page_access_dir}…")
    wp = load_workload_pages(args.page_access_dir, exclude=exclude)
    n = wp.n
    pc = [len(ps) for ps in wp.page_sets]
    D = compute_directional_matrix(wp.page_sets, args.cache_pages)
    D_norm = row_normalize(D)
    obj = ExactSimObjective(wp.page_sets, args.cache_pages, d_matrix=D)

    def fhit(sched: list[int]) -> float:
        m, _ = obj.evaluate_schedule(sched)
        return m.hit_ratio

    def best_sched(scheds: list[list[int]]) -> list[int]:
        return max(scheds, key=fhit)

    def schedule_for(method: str) -> list[int]:
        if method == "random":
            rng = random.Random(args.seed)
            perm = list(range(n))
            rng.shuffle(perm)
            return perm
        if method in ("GA_M", "GA_D"):
            key = "ga_m" if method.endswith("M") else "ga_d"
            return compute_baseline_schedule(key, wp.page_sets, wp.query_ids, D,
                                             args.cache_pages, args.seed,
                                             args.ga_pop, args.ga_gen)
        if method == "sweep_D":
            grid = ([float(x) for x in args.regret_grid.split(",") if x.strip()]
                    if args.regret_grid.strip()
                    else [round(0.1 * i, 1) for i in range(11)] + [1.5, 2.0])
            starts = good_start_set(D_norm, pc, args.num_starts)
            best_s: list[int] = []
            best_f = -1.0
            for wr in grid:
                sc = make_step_scorer(D_norm, ScorerWeights(w_regret=wr), pc,
                                      args.cache_pages)
                cand = best_sched(multistart_greedy_schedules(sc, n, pc, starts))
                f = fhit(cand)
                if f > best_f:
                    best_f, best_s = f, cand
            return best_s
        if method == "sweep_beam_D":
            grid = ([float(x) for x in args.regret_grid.split(",") if x.strip()]
                    if args.regret_grid.strip()
                    else [round(0.1 * i, 1) for i in range(11)] + [1.5, 2.0])
            widths = [int(b) for b in args.sweep_beam_widths.split(",") if b.strip()]
            best_s2: list[int] = []
            best_f2 = -1.0
            for wr in grid:
                sc = make_step_scorer(D_norm, ScorerWeights(w_regret=wr), pc,
                                      args.cache_pages)
                for bw in widths:
                    cand = best_sched(beam_search_schedule(sc, n, pc, bw))
                    f = fhit(cand)
                    if f > best_f2:
                        best_f2, best_s2 = f, cand
            return best_s2
        raise SystemExit(f"unhandled method {method}")

    schedules: dict[str, str] = {}
    for method in want:
        sched = schedule_for(method)
        qids = ",".join(wp.query_ids[i] for i in sched)
        schedules[method] = qids
        print(f"  {method:18s} F_hit={fhit(sched):.4f}  {qids[:60]}…")

    payload = {"cache_label": str(args.cache_pages), "schedules": schedules}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {len(schedules)} schedules -> {args.out}")


if __name__ == "__main__":
    main()
