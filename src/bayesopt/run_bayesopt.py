"""
Run a Mode B priority-vector search against the exact cache simulator.

Usage
-----
    python -m src.bayesopt.run_bayesopt \\
        --workload tpch --cache-pages 102400 \\
        --page-access-dir page_access/tpch_6gb \\
        --backend smac --budget 100 --seed 42

Search is **simulator-only** (no database connection): page sets are
loaded from the profiling CSVs, every candidate schedule is scored by
the exact clock-sweep simulation, and the incumbent schedule is written
out as query ids ready for later wall-clock validation.  The
equal-budget random baseline is the same command with
``--backend random``.

Outputs (under --out-dir):
    <run>.trials.jsonl    one record per evaluation (full trial log)
    <run>.summary.json    incumbent, convergence curve, run metadata
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from src.bayesopt.data import load_workload_pages
from src.bayesopt.objective import ExactSimObjective
from src.bayesopt.search import (
    BACKENDS,
    MODE,
    PrioritySpace,
    default_n_init,
    run_search,
)
from src.simulator.cache_simulator import compute_directional_matrix
from src.utilities.constants import PROJECT_ROOT, WORKLOAD_DIRS

GP_ARM_MAX_N = 40
"""Soft ceiling above which the vanilla-GP arm is flagged as misapplied."""


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=list(WORKLOAD_DIRS),
        required=True,
        help="Workload label recorded in logs and filenames.",
    )
    parser.add_argument(
        "--cache-pages",
        type=int,
        required=True,
        help="Clock-sweep cache capacity in 8 KB pages.",
    )
    parser.add_argument(
        "--page-access-dir",
        type=Path,
        required=True,
        help="Per-query page-access CSV directory.  Required explicitly "
        "(no default) so runs cannot silently pick up truncated "
        "profiles; use the _6gb directories.",
    )
    parser.add_argument(
        "--backend",
        choices=list(BACKENDS),
        required=True,
        help="Optimizer backend.  'random' is the mandatory "
        "equal-budget baseline.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="Total evaluation budget (initial design included).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-init",
        type=int,
        default=None,
        help="Initial-design size for smac/botorch_gp.  Default: "
        "max(5, min(25, budget // 4)), shared across backends so "
        "arms are comparable.",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated query ids to drop before searching "
        "(e.g. q13 when a schedule is destined for TPC-H wall-clock).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "experiment_logs" / "bo",
        help="Directory for trial logs and summaries "
        "(default: experiment_logs/bo).",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional tag appended to output filenames.",
    )
    parser.add_argument(
        "--no-edge-sum",
        action="store_true",
        help="Skip building the directional matrix for the per-trial "
        "edge-sum diagnostic.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and execute one Mode B search run."""
    args = build_parser().parse_args(argv)

    if os.environ.get("PYTHONHASHSEED") != "0":
        print(
            "WARNING: PYTHONHASHSEED is not 0; project convention is "
            "'export PYTHONHASHSEED=0' for reproducible runs.",
            file=sys.stderr,
        )

    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}
    print(f"Loading page sets from {args.page_access_dir}…")
    workload_pages = load_workload_pages(args.page_access_dir, exclude=exclude)
    n = workload_pages.n
    print(
        f"  {n} queries, {workload_pages.total_unique_pages:,} unique pages, "
        f"cache C = {args.cache_pages:,} pages"
    )
    if exclude:
        print(f"  Excluded: {sorted(exclude)}")
    if args.backend == "botorch_gp" and n > GP_ARM_MAX_N:
        print(
            f"WARNING: botorch_gp is the small-n vanilla GP arm; n={n} is "
            f"outside its intended range (TPC-H ≈ 22).  The "
            f"high-dimensional arm is a separate, pending decision.",
            file=sys.stderr,
        )

    d_matrix = None
    if not args.no_edge_sum:
        t0 = time.perf_counter()
        d_matrix = compute_directional_matrix(
            workload_pages.page_sets, args.cache_pages
        )
        print(f"  Directional matrix (diagnostic) in {time.perf_counter() - t0:.1f}s")

    objective = ExactSimObjective(
        workload_pages.page_sets, args.cache_pages, d_matrix=d_matrix
    )
    space = PrioritySpace(n)
    n_init = default_n_init(args.budget) if args.n_init is None else args.n_init

    tag = f"_{args.label}" if args.label else ""
    stem = (
        f"{args.workload}_c{args.cache_pages}_{args.backend}"
        f"_b{args.budget}_s{args.seed}{tag}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = args.out_dir / f"{stem}.trials.jsonl"
    summary_path = args.out_dir / f"{stem}.summary.json"

    static_fields: dict[str, object] = {
        "workload": args.workload,
        "cache_pages": args.cache_pages,
        "page_access_dir": str(args.page_access_dir),
        "backend": args.backend,
        "mode": MODE,
        "seed": args.seed,
        "budget": args.budget,
        "n_init": n_init,
        "n_queries": n,
        "excluded": sorted(exclude),
    }

    print(
        f"\nRunning backend={args.backend} budget={args.budget} "
        f"n_init={n_init} seed={args.seed}…"
    )
    with open(trials_path, "w") as stream:
        run = run_search(
            args.backend,
            space,
            objective,
            workload_pages.query_ids,
            budget=args.budget,
            seed=args.seed,
            n_init=n_init,
            static_fields=static_fields,
            trials_stream=stream,
            workdir=args.out_dir / "smac3_output",
        )

    summary: dict[str, object] = dict(static_fields)
    summary.update(asdict(run))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'─' * 56}")
    print(f"  Trials run        : {run.n_trials} / {run.budget}")
    print(f"  Best at trial     : {run.best_trial}")
    print(f"  Best F_hit        : {run.best_hit_ratio:.4f} "
          f"({run.best_hit_ratio * 100:.2f}%)")
    print(f"  Best cost (1−F)   : {run.best_cost:.6f}")
    print(f"  Best page reads   : {run.best_page_reads:,}")
    print(f"  Wall time         : {run.wall_seconds:.1f}s")
    print(f"  Best order        : {' → '.join(run.best_schedule_qids)}")
    print(f"  Trials log        : {trials_path}")
    print(f"  Summary           : {summary_path}")
    print(f"{'─' * 56}\n")


if __name__ == "__main__":
    main()
