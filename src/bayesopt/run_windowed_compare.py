"""
Windowing-axis comparison: windowed-M vs single-step-M vs single-step-D.

Same consumer (greedy), three immediate signals, exact-sim judge.  Two
clean axes fall out:
  * windowing:  windowed-M  vs  single-step-M  (same matrix, multi- vs
    single-predecessor)
  * matrix:     single-step-M vs single-step-D (same step structure,
    symmetric vs directional)

Hypothesis (run at all three caches to see the crossover): windowed-M ~=
single-step-M at small cache (budget covers one predecessor) and may exceed
it at large cache (several survivors contribute), while single-step-D leads
at small cache and collapses to M at large cache.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.bayesopt.data import load_workload_pages
from src.bayesopt.objective import ExactSimObjective
from src.bayesopt.step_scorer import (
    ScorerWeights,
    greedy_schedule,
    make_step_scorer,
    row_normalize,
)
from src.bayesopt.windowed_scorer import greedy_windowed_schedule
from src.simulator.cache_simulator import (
    compute_directional_matrix,
    compute_overlap_matrix,
)
from src.utilities.constants import PROJECT_ROOT, WORKLOAD_DIRS

_TIE_PP = 0.1


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=list(WORKLOAD_DIRS), required=True)
    p.add_argument("--cache-pages", type=int, required=True)
    p.add_argument("--page-access-dir", type=Path, required=True)
    p.add_argument("--exclude", default="")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "experiment_logs" / "windowed_compare")
    args = p.parse_args(argv)
    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}

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

    ss_m = fhit(greedy_schedule(
        make_step_scorer(M_norm, ScorerWeights(), pc, args.cache_pages), wp.n, pc))
    ss_d = fhit(greedy_schedule(
        make_step_scorer(D_norm, ScorerWeights(), pc, args.cache_pages), wp.n, pc))
    win_m = fhit(greedy_windowed_schedule(M, pc, args.cache_pages))

    windowing_pp = (win_m - ss_m) * 100.0   # does windowing help, same matrix?
    matrix_pp = (ss_d - ss_m) * 100.0        # does direction help, same step?
    win_vs_d_pp = (win_m - ss_d) * 100.0

    summary = {
        "workload": args.workload, "cache_pages": args.cache_pages,
        "n_queries": wp.n,
        "single_step_M": ss_m, "single_step_D": ss_d, "windowed_M": win_m,
        "windowing_axis_pp": windowing_pp,
        "matrix_axis_pp": matrix_pp,
        "windowed_M_vs_single_step_D_pp": win_vs_d_pp,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"windowed_{args.workload}_{args.cache_pages}.json"
    out.write_text(json.dumps(summary, indent=2))

    def tag(pp: float) -> str:
        return "tie" if abs(pp) < _TIE_PP else ("+" if pp > 0 else "-")

    print(f"\n=== windowing axis: {args.workload} {args.cache_pages} (greedy) ===")
    print(f"  single_step_M = {ss_m:.4f}")
    print(f"  single_step_D = {ss_d:.4f}")
    print(f"  windowed_M    = {win_m:.4f}")
    print(f"  windowing (windowed_M - single_step_M): {windowing_pp:+.3f}pp "
          f"[{tag(windowing_pp)}]")
    print(f"  matrix    (single_step_D - single_step_M): {matrix_pp:+.3f}pp "
          f"[{tag(matrix_pp)}]")
    print(f"  windowed_M - single_step_D: {win_vs_d_pp:+.3f}pp [{tag(win_vs_d_pp)}]")
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
