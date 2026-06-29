"""
Schedule-level directional curve — greedy on M vs greedy on D.

The residual audit established at the *matrix* level that D = alpha*M + E
with E orthogonal to M and concentrated on overflowing queries.  This
runner establishes the *schedule* level counterpart with the correct
consumer: single-step greedy nearest-neighbour on the symmetric overlap
matrix M versus on the directional matrix D, judged by the exact
clock-sweep simulator.

Both greedies share a seed (largest-pagecount start), so the only
difference is the matrix they walk.  Their F_hit gap is the
schedule-level directional gain; plotting it against the matrix
residual ||E||/||D|| across configs is the gain-vs-residual curve and
the empirical form of the demo-paper claim — directional scoring beats
symmetric overlap in proportion to cache-overflow pressure.

This deliberately uses single-step edge-D, not the windowed kernel:
the windowed inclusion-exclusion collapses in the overflow regime
where E lives (see run_residual_scorer), so single-step is both the
correct and the physically appropriate consumer there.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.bayesopt.data import load_workload_pages
from src.bayesopt.objective import ExactSimObjective
from src.bayesopt.residual_scorer import residual_frac, row_alpha_residual
from src.scheduler.greedy_directional import greedy_directional_schedule
from src.simulator.cache_simulator import (
    compute_directional_matrix,
    compute_overlap_matrix,
)
from src.utilities.constants import PROJECT_ROOT, WORKLOAD_DIRS

_TIE_PP = 0.1


def _diff_positions(a: list[int], b: list[int]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=list(WORKLOAD_DIRS), required=True)
    p.add_argument("--cache-pages", type=int, required=True)
    p.add_argument("--page-access-dir", type=Path, required=True)
    p.add_argument("--exclude", default="")
    p.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "experiment_logs" / "directional_curve",
    )
    args = p.parse_args(argv)
    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}

    print(f"Loading page sets from {args.page_access_dir}…")
    wp = load_workload_pages(args.page_access_dir, exclude=exclude)
    page_counts = [len(ps) for ps in wp.page_sets]
    print(f"  {wp.n} queries, {wp.total_unique_pages:,} unique pages, "
          f"C = {args.cache_pages:,} pages")

    t0 = time.perf_counter()
    M = compute_overlap_matrix(wp.page_sets)
    D = compute_directional_matrix(wp.page_sets, args.cache_pages)
    _, E = row_alpha_residual(M, D)
    e_frac = residual_frac(M, D, E)
    print(f"  M, D, decomposition in {time.perf_counter() - t0:.1f}s  "
          f"(||E||/||D|| = {e_frac:.4f})")

    objective = ExactSimObjective(wp.page_sets, args.cache_pages, d_matrix=D)

    sched_m = greedy_directional_schedule(M, page_counts)
    sched_d = greedy_directional_schedule(D, page_counts)
    m_metrics, _ = objective.evaluate_schedule(sched_m)
    d_metrics, _ = objective.evaluate_schedule(sched_d)

    gain_pp = (d_metrics.hit_ratio - m_metrics.hit_ratio) * 100.0
    changed = _diff_positions(sched_d, sched_m)
    verdict = ("tie" if abs(gain_pp) < _TIE_PP
               else ("D_wins" if gain_pp > 0 else "M_wins"))

    summary = {
        "workload": args.workload,
        "cache_pages": args.cache_pages,
        "n_queries": wp.n,
        "e_frac_of_d": e_frac,
        "greedy_M_fhit": m_metrics.hit_ratio,
        "greedy_D_fhit": d_metrics.hit_ratio,
        "directional_gain_pp": gain_pp,
        "positions_changed": changed,
        "verdict": verdict,
        "tie_threshold_pp": _TIE_PP,
        "greedy_M_schedule": sched_m,
        "greedy_D_schedule": sched_d,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "directional_curve_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    print("\n=== greedy(M) vs greedy(D), single-step, exact F_hit ===")
    print(f"  ||E||/||D||        : {e_frac:.4f}")
    print(f"  greedy(M) F_hit    : {m_metrics.hit_ratio:.6f}")
    print(f"  greedy(D) F_hit    : {d_metrics.hit_ratio:.6f}")
    print(f"  directional gain   : {gain_pp:+.3f} pp  [{verdict}]"
          f"  ({changed}/{wp.n} pos changed)")
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
