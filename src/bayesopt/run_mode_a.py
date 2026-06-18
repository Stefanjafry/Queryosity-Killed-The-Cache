"""
Run one Mode A scorer search from the command line.

Search is simulator-only (no database): page sets are loaded from the
profiling CSVs, the structured scorer is tuned by warm-up + random +
neighbour refinement, and every candidate schedule is judged by the
exact clock-sweep simulator.  All artifacts land under
``<output-dir>/<run-id>/``.

Example
-------
    python -m src.bayesopt.run_mode_a \\
        --workload tpch --cache-pages 102400 \\
        --page-access-dir page_access/tpch_6gb \\
        --model-family d --consumer multistart_greedy \\
        --max-trials 120 --early-stop-patience 15

Baselines (Greedy-D, GA+M, GA+D, …) are regenerated separately by
``src.bayesopt.run_baselines`` on the same profiles, caches, and
exact-sim metric; this runner does not reimplement them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.bayesopt.data import load_workload_pages
from src.bayesopt.mode_a.consumers import CONSUMERS
from src.bayesopt.mode_a.runner import run_mode_a
from src.bayesopt.mode_a.scorers import SCORER_FAMILIES
from src.utilities.constants import PROJECT_ROOT, WORKLOAD_DIRS

FAMILY_ALIASES = {"m": "m", "d": "d", "hybrid": "hybrid"}


def _int_csv(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(",") if x.strip())


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=list(WORKLOAD_DIRS), required=True)
    p.add_argument("--cache-pages", type=int, required=True)
    p.add_argument("--page-access-dir", type=Path, required=True)
    p.add_argument("--model-family", choices=list(SCORER_FAMILIES),
                   required=True)
    p.add_argument("--consumer", choices=list(CONSUMERS), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-trials", type=int, default=200)
    p.add_argument("--max-schedule-evals", type=int, default=100000)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--num-starts", type=int, default=5)
    p.add_argument("--beam-widths", type=_int_csv, default=(2, 4, 8))
    p.add_argument("--topk-values", type=_int_csv, default=(3, 5, 10))
    p.add_argument("--search-mode",
                   choices=["warmup_random_neighbor", "warmup_random",
                            "random_only", "botorch_gp"],
                   default="warmup_random_neighbor")
    p.add_argument("--exclude", default="")
    p.add_argument("--log-all-candidates", action="store_true")
    p.add_argument("--output-dir", type=Path,
                   default=PROJECT_ROOT / "experiment_logs" / "mode_a")
    p.add_argument("--run-id", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and execute one Mode A run."""
    args = build_parser().parse_args(argv)
    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}

    print(f"Loading page sets from {args.page_access_dir}…")
    wp = load_workload_pages(args.page_access_dir, exclude=exclude)
    print(f"  {wp.n} queries, {wp.total_unique_pages:,} unique pages, "
          f"C = {args.cache_pages:,} pages")

    print(f"Mode A: family={args.model_family} consumer={args.consumer} "
          f"seed={args.seed} max_trials={args.max_trials}…")
    res = run_mode_a(
        page_sets=wp.page_sets,
        query_ids=wp.query_ids,
        workload=args.workload,
        cache_pages=args.cache_pages,
        family=args.model_family,
        consumer=args.consumer,
        seed=args.seed,
        max_trials=args.max_trials,
        max_schedule_evals=args.max_schedule_evals,
        early_stop_patience=args.early_stop_patience,
        num_starts=args.num_starts,
        beam_widths=args.beam_widths,
        topk_values=args.topk_values,
        search_mode=args.search_mode,
        output_dir=args.output_dir,
        run_id=args.run_id,
        log_all_candidates=args.log_all_candidates,
    )

    print(f"\n{'─' * 56}")
    print(f"  Best F_hit       : {res.best_hit_ratio:.4f} "
          f"({res.best_hit_ratio * 100:.2f}%)")
    print(f"  Best cost (1−F)  : {res.best_cost:.6f}")
    print(f"  Best source      : {res.best_source}")
    print(f"  Best config      : {res.best_config}")
    print(f"  Param trials     : {res.num_parameter_trials}")
    print(f"  Exact evals      : {res.num_exact_evals}")
    print(f"  Early stopped    : {res.early_stopped}")
    print(f"  Artifacts        : {res.output_dir}")
    print(f"{'─' * 56}\n")


if __name__ == "__main__":
    main()
