"""
Final paper grid — every method at every (workload, cache) in one pinned run.

Methods (all judged by the exact clock-sweep simulator):
  random              shuffled baseline floor
  greedy_M / greedy_D single-step scorer, symmetric / directional
  multistart_M/_D     multistart-greedy, K starts (tuned default K=4)
  beam_M / beam_D     beam search, width B (default 4)
  windowed_greedy_M   windowed triple-intersection as a greedy step (Queryosity
                      fitness, single consumer) — the windowed-symmetric column
  GA_M                GA on windowed-symmetric fitness = Queryosity baseline
  GA_D                GA on single-step directional fitness
  BO_M / BO_D         BO-tuned scorer (neighbor backend) on multistart-K

Two clean axes the grid exposes:
  matrix:    *_D vs *_M under the same consumer (directional vs symmetric)
  windowing: GA_M / windowed_greedy_M vs single-step-M (windowed vs edge-local)

Eval cost is recorded per cell (exact sims consumed): 1 for greedy/random/
windowed, K for multistart, B for beam, pop*gen for GA, BO n_evals for BO.

Writes one combined grid JSON and a long-format CSV; everything from one run
with a single pinned seed for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from src.bayesopt.bo_tuner import run_search
from src.bayesopt.data import load_workload_pages
from src.bayesopt.objective import ExactSimObjective
from src.bayesopt.run_baselines import compute_baseline_schedule
from src.bayesopt.step_scorer import (
    ScorerWeights,
    beam_search_schedule,
    good_start_set,
    greedy_schedule,
    make_step_scorer,
    multistart_greedy_schedules,
    row_normalize,
)
from src.bayesopt.windowed_scorer import greedy_windowed_schedule
from src.simulator.cache_simulator import (
    compute_directional_matrix,
    compute_overlap_matrix,
)
from src.utilities.constants import PROJECT_ROOT, WORKLOAD_DIRS

METHOD_ORDER = [
    "random", "greedy_M", "greedy_D", "multistart_M", "multistart_D",
    "beam_M", "beam_D", "windowed_greedy_M", "GA_M", "GA_D", "BO_M", "BO_D",
]


def run_config(wp, M, M_norm, D, D_norm, pc, cache_pages, args
               ) -> dict[str, tuple[float, int]]:
    """Run every method for one (workload, cache); return method -> (F_hit, evals)."""
    n = wp.n
    obj = ExactSimObjective(wp.page_sets, cache_pages, d_matrix=D)

    def fhit(sched: list[int]) -> float:
        m, _ = obj.evaluate_schedule(sched)
        return m.hit_ratio

    def best(scheds: list[list[int]]) -> float:
        return max(fhit(s) for s in scheds)

    out: dict[str, tuple[float, int]] = {}

    rng = random.Random(args.seed)
    perm = list(range(n))
    rng.shuffle(perm)
    out["random"] = (fhit(perm), 1)

    for arm, imm in (("M", M_norm), ("D", D_norm)):
        sc = make_step_scorer(imm, ScorerWeights(), pc, cache_pages)
        out[f"greedy_{arm}"] = (fhit(greedy_schedule(sc, n, pc)), 1)
        starts = good_start_set(imm, pc, args.num_starts)
        ms = multistart_greedy_schedules(sc, n, pc, starts)
        out[f"multistart_{arm}"] = (best(ms), len(ms))
        bm = beam_search_schedule(sc, n, pc, args.beam_width)
        out[f"beam_{arm}"] = (best(bm), len(bm))

    out["windowed_greedy_M"] = (fhit(greedy_windowed_schedule(M, pc, cache_pages)), 1)

    ga_cost = args.ga_pop * args.ga_gen
    out["GA_M"] = (fhit(compute_baseline_schedule(
        "ga_m", wp.page_sets, wp.query_ids, D, cache_pages,
        args.seed, args.ga_pop, args.ga_gen)), ga_cost)
    out["GA_D"] = (fhit(compute_baseline_schedule(
        "ga_d", wp.page_sets, wp.query_ids, D, cache_pages,
        args.seed, args.ga_pop, args.ga_gen)), ga_cost)

    if args.with_bo:
        for arm, imm in (("M", M_norm), ("D", D_norm)):
            starts = good_start_set(imm, pc, args.num_starts)

            def objective(w: ScorerWeights, _imm=imm, _starts=starts
                          ) -> tuple[float, int]:
                sc = make_step_scorer(_imm, w, pc, cache_pages)
                ms = multistart_greedy_schedules(sc, n, pc, _starts)
                return best(ms), len(ms)

            res = run_search(objective, args.bo_backend, args.bo_budget,
                             patience=args.patience, seed=args.seed,
                             cache_mode="fit")
            out[f"BO_{arm}"] = (res.best_fhit, res.n_evals)

    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--page-access-root", type=Path, default=Path("page_access"),
                   help="dir containing <workload>_6gb subdirs")
    p.add_argument("--suffix", default="_6gb")
    p.add_argument("--workloads", default="tpch,tpcds,job")
    p.add_argument("--caches", default="102400,262144,524288")
    p.add_argument("--exclude", default="")
    p.add_argument("--num-starts", type=int, default=4)
    p.add_argument("--beam-width", type=int, default=4)
    p.add_argument("--ga-pop", type=int, default=100)
    p.add_argument("--ga-gen", type=int, default=200)
    p.add_argument("--with-bo", action="store_true", default=True)
    p.add_argument("--no-bo", dest="with_bo", action="store_false")
    p.add_argument("--bo-backend", default="neighbor")
    p.add_argument("--bo-budget", type=int, default=60)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "experiment_logs" / "paper_grid_full")
    args = p.parse_args(argv)
    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    caches = [int(c) for c in args.caches.split(",") if c.strip()]

    methods = list(METHOD_ORDER) if args.with_bo else \
        [m for m in METHOD_ORDER if not m.startswith("BO_")]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid: list[dict[str, object]] = []
    t_start = time.perf_counter()

    for w in workloads:
        if w not in WORKLOAD_DIRS:
            print(f"  [skip] unknown workload {w}")
            continue
        pdir = args.page_access_root / f"{w}{args.suffix}"
        print(f"\n### {w} ({pdir})")
        wp = load_workload_pages(pdir, exclude=exclude)
        pc = [len(ps) for ps in wp.page_sets]
        M = compute_overlap_matrix(wp.page_sets)
        M_norm = row_normalize(M)
        for c in caches:
            t = time.perf_counter()
            D = compute_directional_matrix(wp.page_sets, c)
            D_norm = row_normalize(D)
            res = run_config(wp, M, M_norm, D, D_norm, pc, c, args)
            dt = time.perf_counter() - t
            for method, (f, ev) in res.items():
                grid.append({"workload": w, "cache_pages": c, "method": method,
                             "fhit": f, "evals": ev})
            best_method = max(res, key=lambda m: res[m][0])
            print(f"  C={c:,}: best={best_method} ({res[best_method][0]:.4f})  "
                  f"greedy_D={res['greedy_D'][0]:.4f} GA_M={res['GA_M'][0]:.4f}  "
                  f"[{dt:.0f}s]")

    # write long-format CSV + JSON
    csv_path = args.out_dir / "paper_grid_full.csv"
    with csv_path.open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["workload", "cache_pages",
                                             "method", "fhit", "evals"])
        wtr.writeheader()
        for row in grid:
            wtr.writerow(row)
    (args.out_dir / "paper_grid_full.json").write_text(json.dumps(
        {"seed": args.seed, "ga": [args.ga_pop, args.ga_gen],
         "bo": {"backend": args.bo_backend, "budget": args.bo_budget,
                "enabled": args.with_bo},
         "num_starts": args.num_starts, "beam_width": args.beam_width,
         "methods": methods, "grid": grid}, indent=2))

    # readable pivot: rows = (workload, cache), cols = methods (F_hit)
    print(f"\n=== paper grid (F_hit) — {time.perf_counter() - t_start:.0f}s total ===")
    header = "  " + f"{'workload':8s}{'cache':>9s}  " + \
        "".join(f"{m:>10s}" for m in methods)
    print(header)
    by_cell = {(r["workload"], r["cache_pages"], r["method"]): r["fhit"]
               for r in grid}
    for w in workloads:
        for c in caches:
            cells = "".join(
                f"{by_cell.get((w, c, m), float('nan')):>10.4f}" for m in methods)
            print(f"  {w:8s}{c:>9,}  {cells}")
    print(f"  wrote: {csv_path}")


if __name__ == "__main__":
    main()
