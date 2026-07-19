"""
Scheduling-time benchmark.

Measures the *optimizer* wall-clock \u2014 how long each method takes to produce
a schedule \u2014 with no database in the loop. This is the scheduler-overhead
metric (cf. Queryosity Figure 15), and it is confound-free: no Postgres, no
OS cache, no timeouts. It isolates the cost of the search itself.

Methods timed per (workload, cache):
  - GA_M / GA_D      : the genetic algorithm (population x generations search)
  - sweep_D          : the 13-point regret sweep on the multistart consumer
  - sweep_beam_D     : the regret sweep x beam widths {2,3,4}

Each method is timed over several repetitions; mean and std are reported.
The GA cost is dominated by population*generations fitness evaluations; the
sweep cost by 13 (or 13*3) schedule builds \u2014 so the sweep should be far
cheaper, and this quantifies by how much.

Usage:
    python -m src.bayesopt.bench_schedule_time \\
        --workload tpch --cache-pages 262144 \\
        --page-access-dir page_access/tpch_6gb --reps 5
"""

from __future__ import annotations

import argparse
import time
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
from src.simulator.cache_simulator import (
    compute_directional_matrix,
    compute_overlap_matrix,
)
from src.utilities.constants import WORKLOAD_DIRS

DEFAULT_REGRET_GRID = [round(0.1 * i, 1) for i in range(11)] + [1.5, 2.0]
DEFAULT_BEAM_WIDTHS = (2, 3, 4)


def _time(fn, reps: int) -> tuple[float, float, list[float]]:
    """Run fn reps times, return (mean_s, std_s, samples)."""
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / len(samples)
    return mean, var ** 0.5, samples


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=list(WORKLOAD_DIRS), required=True)
    p.add_argument("--cache-pages", type=int, required=True)
    p.add_argument("--page-access-dir", type=Path, required=True)
    p.add_argument("--exclude", default="")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ga-pop", type=int, default=100)
    p.add_argument("--ga-gen", type=int, default=200)
    p.add_argument("--num-starts", type=int, default=4)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}

    wp = load_workload_pages(args.page_access_dir, exclude=exclude)
    pc = [len(ps) for ps in wp.page_sets]
    page_sets_int = [frozenset(ps) for ps in wp.page_sets]

    # matrices (built once; matrix-build time is shared by all methods and is
    # reported separately so the per-method numbers are just the search).
    t0 = time.perf_counter()
    M = compute_overlap_matrix(wp.page_sets)
    D = compute_directional_matrix(wp.page_sets, args.cache_pages)
    matrix_s = time.perf_counter() - t0
    M_norm = row_normalize(M)
    D_norm = row_normalize(D)

    grid = DEFAULT_REGRET_GRID
    n = wp.n

    # Exact simulator is used only for the SHARED final validation step (one
    # evaluation), exactly as the reference method does: "the final best
    # schedule is re-evaluated using the exact clock-sweep simulation."  It is
    # NOT charged per-candidate here, because that would unfairly load the
    # sweep with dozens of exact-sim calls while the GA searches with its own
    # cheap approximate fitness. This benchmark isolates the scheduling/search
    # cost (cf. Queryosity Figure 15), which is the fair per-method comparison.
    obj = ExactSimObjective(wp.page_sets, args.cache_pages, d_matrix=D)

    # --- method closures: SCHEDULE CONSTRUCTION / SEARCH only ---
    def ga_m():
        compute_baseline_schedule("ga_m", page_sets_int, wp.query_ids, D,
                                  args.cache_pages, args.seed,
                                  args.ga_pop, args.ga_gen)

    def ga_d():
        compute_baseline_schedule("ga_d", page_sets_int, wp.query_ids, D,
                                  args.cache_pages, args.seed,
                                  args.ga_pop, args.ga_gen)

    def sweep_multistart():
        # build all 13 candidate schedules; selection among them is the
        # shared exact-sim step, timed separately below.
        starts = good_start_set(D_norm, pc, args.num_starts)
        for wr in grid:
            sc = make_step_scorer(D_norm, ScorerWeights(w_regret=wr), pc,
                                  args.cache_pages)
            multistart_greedy_schedules(sc, n, pc, starts)

    def sweep_beam():
        for wr in grid:
            sc = make_step_scorer(D_norm, ScorerWeights(w_regret=wr), pc,
                                  args.cache_pages)
            for bw in DEFAULT_BEAM_WIDTHS:
                beam_search_schedule(sc, n, pc, bw)

    methods = [
        ("GA_M", ga_m),
        ("GA_D", ga_d),
        ("sweep_D", sweep_multistart),
        ("sweep_beam_D", sweep_beam),
    ]

    print(f"Scheduling-time benchmark  "
          f"(workload={args.workload}, cache={args.cache_pages:,}, "
          f"n={n} queries, reps={args.reps})")
    print(f"  matrix build (shared, once): {matrix_s * 1000:.1f} ms")
    print(f"  {'method':14s} {'search mean':>12s} {'std':>10s}   {'vs GA_M':>12s}")
    print("  " + "-" * 54)

    def fmt(x: float) -> str:
        return f"{x * 1000:.1f} ms" if x < 1 else f"{x:.2f} s"

    results = {}
    ga_m_mean = None
    for name, fn in methods:
        mean, std, _ = _time(fn, args.reps)
        results[name] = mean
        if name == "GA_M":
            ga_m_mean = mean
        speedup = ""
        if ga_m_mean and name.startswith("sweep") and mean > 0:
            speedup = f"{ga_m_mean / mean:.0f}x faster"
        print(f"  {name:14s} {fmt(mean):>12s} {fmt(std):>10s}   {speedup:>12s}")

    # shared final validation: ONE exact-sim evaluation (what every method,
    # including the reference GA, pays once on its final schedule).
    starts = good_start_set(D_norm, pc, args.num_starts)
    sc0 = make_step_scorer(D_norm, ScorerWeights(w_regret=0.5), pc,
                           args.cache_pages)
    final = multistart_greedy_schedules(sc0, n, pc, starts)[0]
    t0 = time.perf_counter()
    obj.evaluate_schedule(final)
    one_exact = time.perf_counter() - t0

    print("  " + "-" * 54)
    print(f"  shared final validation (1 exact-sim eval): {fmt(one_exact)}")
    print(f"    \u2014 every method (incl. the reference GA) pays this once on its")
    print(f"      final schedule; it is not part of the search cost above.")

    if ga_m_mean:
        best_sweep = min(results["sweep_D"], results["sweep_beam_D"])
        print(f"\n  Search cost: GA_M takes {ga_m_mean / best_sweep:.0f}x longer "
              f"than the fastest sweep to produce a schedule.")


if __name__ == "__main__":
    main()
