"""
Phase 2 runner — alphaM_window vs alphaM_plus_E under the exact simulator.

For one (workload, cache) pair this builds ``M``, ``D``, the
decomposition ``alpha``/``E``, then runs the greedy windowed consumer
under two scorers:

* ``alphaM_window``   (lambda_e = 0): windowed inclusion-exclusion on
  the survival-scaled symmetric overlap, no directional residual.
* ``alphaM_plus_E``   (lambda_e > 0): same, plus the E window-sum.

Both schedules are judged by the exact clock-sweep simulator
(``ExactSimObjective``).  The headline is the F_hit delta of
``alphaM_plus_E`` over ``alphaM_window`` and the matrix residual
fraction ``||E||/||D||`` — one point on the gain-vs-residual curve.

This is the necessary-and-sufficient Phase-2 gate: a large matrix
residual does not imply a schedule gain.  If E does not move F_hit even
where the residual is large (TPC-H, small cache), the directional
approach is a clean negative.

The exact simulator is the judge.  ``greedy_directional`` (single-step
edge D) is reported only as a reference baseline, not an arm.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.bayesopt.data import load_workload_pages
from src.bayesopt.objective import ExactSimObjective
from src.bayesopt.residual_scorer import (
    greedy_windowed_schedule,
    make_scorer,
    residual_frac,
    row_alpha_residual,
)
from src.scheduler.greedy_directional import greedy_directional_schedule
from src.simulator.cache_simulator import (
    compute_directional_matrix,
    compute_overlap_matrix,
)
from src.utilities.constants import PROJECT_ROOT, WORKLOAD_DIRS

# pp deltas below this are treated as tie/noise (matches the audit thresholds).
_TIE_PP = 0.1


def _diff_positions(a: list[int], b: list[int]) -> int:
    """Count positions where two schedules differ."""
    return sum(1 for x, y in zip(a, b) if x != y)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=list(WORKLOAD_DIRS), required=True)
    p.add_argument("--cache-pages", type=int, required=True)
    p.add_argument("--page-access-dir", type=Path, required=True)
    p.add_argument("--exclude", default="")
    p.add_argument(
        "--lambda-e", default="0.5,1.0,2.0",
        help="Comma-separated E weights for alphaM_plus_E (0.0 baseline "
             "is always run as alphaM_window).",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "experiment_logs" / "residual_scorer",
    )
    args = p.parse_args(argv)

    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}
    lambdas = [float(x) for x in args.lambda_e.split(",") if x.strip()]

    print(f"Loading page sets from {args.page_access_dir}…")
    wp = load_workload_pages(args.page_access_dir, exclude=exclude)
    page_counts = [len(ps) for ps in wp.page_sets]
    print(f"  {wp.n} queries, {wp.total_unique_pages:,} unique pages, "
          f"C = {args.cache_pages:,} pages")

    t0 = time.perf_counter()
    M = compute_overlap_matrix(wp.page_sets)
    D = compute_directional_matrix(wp.page_sets, args.cache_pages)
    alpha, E = row_alpha_residual(M, D)
    e_frac = residual_frac(M, D, E)
    print(f"  M, D, decomposition in {time.perf_counter() - t0:.1f}s  "
          f"(||E||/||D|| = {e_frac:.4f})")

    objective = ExactSimObjective(wp.page_sets, args.cache_pages, d_matrix=D)

    # alphaM_window baseline (lambda_e = 0).
    base_scorer = make_scorer(alpha, M, E, page_counts, args.cache_pages, 0.0)
    base_sched = greedy_windowed_schedule(base_scorer, wp.n, page_counts)
    base_metrics, _ = objective.evaluate_schedule(base_sched)
    base_fhit = base_metrics.hit_ratio

    # Reference only: single-step edge-D greedy.
    ref_sched = greedy_directional_schedule(D, page_counts)
    ref_metrics, _ = objective.evaluate_schedule(ref_sched)

    arms: list[dict[str, object]] = []
    arm_rows: list[tuple[float, float, int, float]] = []  # le, fhit, changed, gain
    for le in lambdas:
        scorer = make_scorer(alpha, M, E, page_counts, args.cache_pages, le)
        sched = greedy_windowed_schedule(scorer, wp.n, page_counts)
        metrics, _ = objective.evaluate_schedule(sched)
        gain_pp = (metrics.hit_ratio - base_fhit) * 100.0
        changed = _diff_positions(sched, base_sched)
        arm_rows.append((le, metrics.hit_ratio, changed, gain_pp))
        arms.append({
            "lambda_e": le,
            "hit_ratio": metrics.hit_ratio,
            "gain_pp_over_alphaM_window": gain_pp,
            "positions_changed_vs_baseline": changed,
            "schedule": sched,
            "edge_sum_d": metrics.edge_sum_d,
        })

    best_gain_pp = max((g for _, _, _, g in arm_rows), default=0.0)

    summary = {
        "workload": args.workload,
        "cache_pages": args.cache_pages,
        "n_queries": wp.n,
        "e_frac_of_d": e_frac,
        "alphaM_window": {
            "hit_ratio": base_fhit,
            "schedule": base_sched,
            "edge_sum_d": base_metrics.edge_sum_d,
        },
        "greedy_directional_reference": {
            "hit_ratio": ref_metrics.hit_ratio,
            "delta_pp_vs_alphaM_window": (ref_metrics.hit_ratio - base_fhit) * 100.0,
        },
        "alphaM_plus_E": arms,
        "best_gain_pp": best_gain_pp,
        "tie_threshold_pp": _TIE_PP,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "residual_scorer_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    print("\n=== alphaM_window vs alphaM_plus_E (exact F_hit) ===")
    print(f"  ||E||/||D||                : {e_frac:.4f}")
    print(f"  alphaM_window  F_hit       : {base_fhit:.6f}")
    print(f"  greedy_directional (ref)   : {ref_metrics.hit_ratio:.6f}"
          f"  ({(ref_metrics.hit_ratio - base_fhit) * 100:+.3f} pp)")
    for le, fhit, changed, gain_pp in arm_rows:
        verdict = "tie" if abs(gain_pp) < _TIE_PP else \
                  ("WIN" if gain_pp > 0 else "loss")
        print(f"  alphaM_plus_E le={le:<4} F_hit={fhit:.6f}"
              f"  {gain_pp:+.3f} pp  [{verdict}]"
              f"  ({changed}/{wp.n} pos changed)")
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
