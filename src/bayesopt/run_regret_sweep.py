"""
1-D regret sweep — the interpretable replacement for BO.

The BO characterization showed that of the four scorer weights, only
``w_regret`` earns nonzero weight on the workloads where the directional
matrix wins (``w_cache`` and ``w_connector`` collapse to zero in 8/9
configs).  A four-parameter Bayesian optimizer is therefore overkill: a
single-parameter sweep of ``w_regret`` recovers essentially all of BO's
benefit, is exhaustive (no search noise), and needs no BoTorch/Optuna
dependency.

This runner mirrors ``run_bo_multistart`` exactly — same M/D matrices,
same ``good_start_set``, same exact clock-sweep scoring — but replaces the
BO backends with an exhaustive sweep of ``w_regret`` over a fixed grid.  It
evaluates two consumers so the paper reports more than greedy alone:

- ``multistart``  : multistart-greedy, K starts (the BO consumer).
- ``beam``        : beam search; the best width in ``--beam-widths`` is
                    kept per (arm, w_regret).  Width 1 is excluded by
                    default because beam-1 is plain greedy.

For each (arm, consumer) it reports the best F_hit over the grid and the
``w_regret`` that achieved it, alongside greedy/multistart references, and
writes a summary JSON shaped like the BO runner's so it drops straight
into the existing comparison tooling.

Usage
-----
    python -m src.bayesopt.run_regret_sweep \\
        --workload tpch --cache-pages 262144 \\
        --page-access-dir page_access/tpch_6gb --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.bayesopt.data import load_workload_pages
from src.bayesopt.objective import ExactSimObjective
from src.bayesopt.step_scorer import (
    ScorerWeights,
    beam_search_schedule,
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

# Default fine grid for w_regret (0..1 in 0.1 steps, then 1.5, 2.0).
DEFAULT_REGRET_GRID = [round(0.1 * i, 1) for i in range(11)] + [1.5, 2.0]
DEFAULT_BEAM_WIDTHS = (2, 3, 4)  # width 1 == greedy, excluded


@dataclass
class SweepResult:
    """Best point found by the sweep for one (arm, consumer)."""

    best_fhit: float
    best_w_regret: float
    best_extra: dict[str, float]  # e.g. {"beam_width": 4} for beam
    n_evals: int
    grid_fhits: list[tuple[float, float]] = field(default_factory=list)
    # list of (w_regret, best_fhit_at_that_regret)


def _sweep_multistart(
    immediate: list[list[float]],
    pagecounts: list[int],
    cache_pages: int,
    n: int,
    fhit,
    grid: list[float],
) -> SweepResult:
    """Exhaustive w_regret sweep on the multistart-greedy consumer."""
    starts = good_start_set(immediate, pagecounts, k=4)
    best = SweepResult(-1.0, 0.0, {}, 0)
    for wr in grid:
        w = ScorerWeights(w_regret=wr)
        score = make_step_scorer(immediate, w, pagecounts, cache_pages)
        scheds = multistart_greedy_schedules(score, n, pagecounts, starts)
        f = max(fhit(s) for s in scheds)
        best.n_evals += len(scheds)
        best.grid_fhits.append((wr, f))
        if f > best.best_fhit:
            best.best_fhit, best.best_w_regret, best.best_extra = f, wr, {}
    return best


def _sweep_beam(
    immediate: list[list[float]],
    pagecounts: list[int],
    cache_pages: int,
    n: int,
    fhit,
    grid: list[float],
    beam_widths: tuple[int, ...],
) -> SweepResult:
    """
    w_regret sweep on the beam consumer.

    For every (w_regret, width) pair the best schedule the beam returns is
    exact-scored; the single best cell over the whole grid is kept.
    """
    best = SweepResult(-1.0, 0.0, {}, 0)
    for wr in grid:
        w = ScorerWeights(w_regret=wr)
        score = make_step_scorer(immediate, w, pagecounts, cache_pages)
        best_at_wr = -1.0
        for bw in beam_widths:
            beams = beam_search_schedule(score, n, pagecounts, bw)
            f = max(fhit(s) for s in beams)
            best.n_evals += len(beams)
            if f > best_at_wr:
                best_at_wr = f
            if f > best.best_fhit:
                best.best_fhit = f
                best.best_w_regret = wr
                best.best_extra = {"beam_width": float(bw)}
        best.grid_fhits.append((wr, best_at_wr))
    return best


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=list(WORKLOAD_DIRS), required=True)
    p.add_argument("--cache-pages", type=int, required=True)
    p.add_argument("--page-access-dir", type=Path, required=True)
    p.add_argument("--exclude", default="")
    p.add_argument("--seed", type=int, default=42)  # kept for logging parity
    p.add_argument(
        "--regret-grid", default="",
        help="Comma-separated w_regret values; blank = default fine grid.",
    )
    p.add_argument(
        "--beam-widths", default=",".join(str(b) for b in DEFAULT_BEAM_WIDTHS),
        help="Comma-separated beam widths (width 1 == greedy, avoid).",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "experiment_logs" / "regret_sweep",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}
    grid = (
        [float(x) for x in args.regret_grid.split(",") if x.strip()]
        if args.regret_grid.strip()
        else DEFAULT_REGRET_GRID
    )
    beam_widths = tuple(
        int(b) for b in args.beam_widths.split(",") if b.strip()
    )

    print(f"Loading page sets from {args.page_access_dir}…")
    wp = load_workload_pages(args.page_access_dir, exclude=exclude)
    pc = [len(ps) for ps in wp.page_sets]
    print(f"  {wp.n} queries, C = {args.cache_pages:,} pages")
    print(f"  regret grid: {grid}")
    print(f"  beam widths: {beam_widths}")

    M = compute_overlap_matrix(wp.page_sets)
    D = compute_directional_matrix(wp.page_sets, args.cache_pages)
    M_norm = row_normalize(M)
    D_norm = row_normalize(D)

    obj = ExactSimObjective(wp.page_sets, args.cache_pages, d_matrix=D)

    def fhit(schedule: list[int]) -> float:
        m, _ = obj.evaluate_schedule(schedule)
        return m.hit_ratio

    # references (aux weights all zero == plain greedy on each matrix)
    greedy_d = fhit(greedy_schedule(
        make_step_scorer(D_norm, ScorerWeights(), pc, args.cache_pages),
        wp.n, pc))
    greedy_m = fhit(greedy_schedule(
        make_step_scorer(M_norm, ScorerWeights(), pc, args.cache_pages),
        wp.n, pc))

    arms = {"M": M_norm, "D": D_norm}
    consumers = ("multistart", "beam")

    results: dict[str, dict[str, SweepResult]] = {}
    for consumer in consumers:
        results[consumer] = {}
        for arm, immediate in arms.items():
            t = time.perf_counter()
            if consumer == "multistart":
                res = _sweep_multistart(
                    immediate, pc, args.cache_pages, wp.n, fhit, grid)
            else:
                res = _sweep_beam(
                    immediate, pc, args.cache_pages, wp.n, fhit, grid,
                    beam_widths)
            dt = time.perf_counter() - t
            results[consumer][arm] = res
            extra = (
                f" width={int(res.best_extra['beam_width'])}"
                if "beam_width" in res.best_extra else ""
            )
            print(
                f"  {consumer:10s} {arm}: F_hit={res.best_fhit:.4f} "
                f"@ w_regret={res.best_w_regret:g}{extra} "
                f"({res.n_evals} evals, {dt:.1f}s)"
            )

    # references line
    print(f"  refs: greedy_M={greedy_m:.4f}  greedy_D={greedy_d:.4f}")
    for consumer in consumers:
        rd = results[consumer]["D"].best_fhit
        rm = results[consumer]["M"].best_fhit
        print(
            f"  {consumer:10s}  sweep-D={rd:.4f} sweep-M={rm:.4f}  "
            f"D-M {(rd - rm) * 100:+.3f}pp  "
            f"D vs greedy_D {(rd - greedy_d) * 100:+.3f}pp"
        )

    # write summary JSON (shaped like the BO runner: consumer -> arm -> result)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "workload": args.workload,
        "cache_pages": args.cache_pages,
        "page_access_dir": str(args.page_access_dir),
        "mode": "regret_sweep",
        "n_queries": wp.n,
        "excluded": sorted(exclude),
        "seed": args.seed,
        "regret_grid": grid,
        "beam_widths": list(beam_widths),
        "references": {"greedy_M": greedy_m, "greedy_D": greedy_d},
        "consumers": {
            consumer: {
                arm: {
                    "best_fhit": r.best_fhit,
                    "best_w_regret": r.best_w_regret,
                    "best_extra": r.best_extra,
                    "n_evals": r.n_evals,
                    "grid_fhits": r.grid_fhits,
                }
                for arm, r in results[consumer].items()
            }
            for consumer in consumers
        },
    }
    out = args.out_dir / "regret_sweep_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
