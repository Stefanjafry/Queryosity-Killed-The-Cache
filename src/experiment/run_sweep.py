"""
Wall-clock experiment orchestrator.

Runs a set of named schedules against a live PostgreSQL instance, each
repeated N times with a cold shared_buffers (and optionally a cold OS
page cache), and records per-query wall-clock time plus shared-buffer
hit/read counts to a tidy long-format CSV.

The CSV is one row per (schedule, rep, query) plus one TOTAL row per
(schedule, rep), with enough columns to compute mean/std across reps and
to plot cumulative shared-buffer reads and workload completion time.

Example
-------
    python -m src.experiment.run_sweep \
        --workload tpch \
        --schedules-file experiment_logs/schedules_tpch_100k.json \
        --reps 3 \
        --flush-cmd "sudo systemctl restart postgresql-16" \
        --out experiment_logs/results_tpch_100k_warm.csv

Add --drop-os-cache (and ensure the drop command has passwordless sudo)
for a true cold-disk pass.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

from src.executor.executor import ExecutionResult, execute_schedule
from src.executor.run_executor import flush_buffer_cache
from src.postgres.connection import close_connection, create_connection
from src.utilities.configurations import (
    PG_HOST,
    PG_PASSWORD,
    PG_PORT,
    PG_SCHEMA,
    PG_STATEMENT_TIMEOUT_MS,
    PG_USER,
)
from src.utilities.constants import DB_DEFAULTS
from src.utilities.workload import load_queries


CSV_FIELDS = [
    "cache_label",
    "os_cache",
    "schedule",
    "rep",
    "position",
    "query_id",
    "elapsed_ms",
    "shared_hit_blocks",
    "shared_read_blocks",
]


def drop_os_cache(drop_cmd: str) -> None:
    """
    Drop the Linux OS page cache via *drop_cmd*.

    Typically ``sudo sysctl -w vm.drop_caches=3`` (requires passwordless
    sudo for that exact command).  Combined with the shared_buffers
    flush, this gives a true cold-disk read for the next schedule.

    Parameters
    ----------
    drop_cmd : str
        Shell command to drop the OS page cache.

    Raises
    ------
    RuntimeError
        If the command exits non-zero.
    """
    print(f"  Dropping OS page cache: {drop_cmd}")
    result = subprocess.run(drop_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"OS cache drop failed (rc={result.returncode}): {err}")


def run_one(
    queries,
    order: list[str],
    *,
    flush_cmd: str,
    host: str,
    port: int,
    user: str,
    password: str,
    db_name: str,
    schema: str,
    timeout_ms: int,
    os_drop_cmd: str | None,
) -> ExecutionResult:
    """
    Flush caches and execute a single schedule once.

    Parameters
    ----------
    queries : dict
        Mapping query_id -> SQL.
    order : list[str]
        Execution order of query IDs.
    flush_cmd : str
        Command to restart Postgres (flush shared_buffers).
    host, port, user, password, db_name, schema : str / int
        Connection parameters.
    timeout_ms : int
        Per-statement timeout.
    os_drop_cmd : str | None
        If given, drop the OS page cache with this command before the
        shared_buffers flush.

    Returns
    -------
    ExecutionResult
        Per-query and aggregate statistics for this run.
    """
    if os_drop_cmd:
        drop_os_cache(os_drop_cmd)
    flush_buffer_cache(flush_cmd=flush_cmd, host=host, port=port)

    conn = create_connection(
        db_name=db_name,
        user=user,
        password=password,
        host=host,
        port=port,
        schema=schema,
        statement_timeout_ms=timeout_ms,
    )
    try:
        return execute_schedule(queries, order, conn)
    finally:
        close_connection(conn)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", default="tpch", choices=list(DB_DEFAULTS))
    parser.add_argument(
        "--schedules-file",
        required=True,
        help="JSON file: {cache_label, schedules: {label: 'q1,q2,...'}}",
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--flush-cmd",
        default="sudo systemctl restart postgresql-16",
        help="Command to flush shared_buffers (restart Postgres).",
    )
    parser.add_argument(
        "--drop-os-cache",
        action="store_true",
        help="Also drop the OS page cache before each rep (true cold disk).",
    )
    parser.add_argument(
        "--drop-cache-cmd",
        default="sudo sysctl -w vm.drop_caches=3",
        help="Command used when --drop-os-cache is set.",
    )
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--host", default=PG_HOST)
    parser.add_argument("--port", type=int, default=PG_PORT)
    parser.add_argument("--user", default=PG_USER)
    parser.add_argument("--password", default=PG_PASSWORD)
    parser.add_argument("--schema", default=PG_SCHEMA)
    parser.add_argument("--timeout-ms", type=int, default=PG_STATEMENT_TIMEOUT_MS)
    args = parser.parse_args(argv)

    cfg = json.loads(Path(args.schedules_file).read_text())
    cache_label = cfg.get("cache_label", "NA")
    schedules: dict[str, str] = cfg["schedules"]
    os_cache_tag = "cold" if args.drop_os_cache else "warm"
    os_drop_cmd = args.drop_cache_cmd if args.drop_os_cache else None

    db_name = DB_DEFAULTS[args.workload]
    queries = load_queries(args.workload)

    # Validate every schedule references known queries before we start the
    # (potentially hours-long) run.
    for label, order_str in schedules.items():
        order = [q.strip() for q in order_str.split(",")]
        missing = [q for q in order if q not in queries]
        if missing:
            parser.error(f"Schedule '{label}' has unknown queries: {missing}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Accumulate per-(schedule, rep) totals for the end-of-run summary.
    totals: dict[str, list[tuple[float, int, int]]] = {k: [] for k in schedules}

    print(
        f"\nWall-clock sweep: workload={args.workload}  cache={cache_label}  "
        f"os_cache={os_cache_tag}  reps={args.reps}  "
        f"schedules={list(schedules)}\n"
    )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for label, order_str in schedules.items():
            order = [q.strip() for q in order_str.split(",")]
            for rep in range(1, args.reps + 1):
                print(f"=== {label}  rep {rep}/{args.reps}  ({os_cache_tag}) ===")
                result = run_one(
                    queries,
                    order,
                    flush_cmd=args.flush_cmd,
                    host=args.host,
                    port=args.port,
                    user=args.user,
                    password=args.password,
                    db_name=db_name,
                    schema=args.schema,
                    timeout_ms=args.timeout_ms,
                    os_drop_cmd=os_drop_cmd,
                )

                for pos, qr in enumerate(result.query_results, 1):
                    writer.writerow(
                        {
                            "cache_label": cache_label,
                            "os_cache": os_cache_tag,
                            "schedule": label,
                            "rep": rep,
                            "position": pos,
                            "query_id": qr.query_id,
                            "elapsed_ms": f"{qr.elapsed_ms:.3f}",
                            "shared_hit_blocks": qr.shared_hit_blocks,
                            "shared_read_blocks": qr.shared_read_blocks,
                        }
                    )
                # TOTAL row (position 0 marks the aggregate).
                writer.writerow(
                    {
                        "cache_label": cache_label,
                        "os_cache": os_cache_tag,
                        "schedule": label,
                        "rep": rep,
                        "position": 0,
                        "query_id": "TOTAL",
                        "elapsed_ms": f"{result.total_elapsed_ms:.3f}",
                        "shared_hit_blocks": result.total_shared_hit_blocks,
                        "shared_read_blocks": result.total_shared_read_blocks,
                    }
                )
                f.flush()  # persist after every rep so a crash loses at most one rep

                totals[label].append(
                    (
                        result.total_elapsed_ms,
                        result.total_shared_hit_blocks,
                        result.total_shared_read_blocks,
                    )
                )
                print(
                    f"    total_time={result.total_elapsed_ms:,.1f} ms  "
                    f"hits={result.total_shared_hit_blocks:,}  "
                    f"reads={result.total_shared_read_blocks:,}  "
                    f"hit%={result.hit_ratio * 100:.2f}\n"
                )

    # ---- Summary table ----
    print(f"\nWrote {out_path}")
    print(f"\n{'=' * 72}")
    print(f"SUMMARY  (workload={args.workload}, cache={cache_label}, {os_cache_tag})")
    print(f"{'=' * 72}")
    print(
        f"{'schedule':<10} {'time_ms mean':>14} {'time_ms std':>12} "
        f"{'reads mean':>12} {'hit% mean':>10}"
    )
    print("-" * 72)
    for label in schedules:
        runs = totals[label]
        times = [t for t, _, _ in runs]
        reads = [r for _, _, r in runs]
        hits = [h for _, h, _ in runs]
        hit_pcts = [
            100.0 * h / (h + r) if (h + r) else 0.0
            for (_, h, r) in runs
        ]
        t_mean = statistics.mean(times)
        t_std = statistics.stdev(times) if len(times) > 1 else 0.0
        r_mean = statistics.mean(reads)
        hp_mean = statistics.mean(hit_pcts)
        print(
            f"{label:<10} {t_mean:>14,.1f} {t_std:>12,.1f} "
            f"{r_mean:>12,.0f} {hp_mean:>9.2f}%"
        )
    print("-" * 72)
    print(
        "Note: 'reads' = shared_read_blocks (shared_buffers misses). With a "
        "warm OS\ncache these are largely served from RAM, not disk. Compare "
        "against the\n--drop-os-cache pass for the true cold-disk picture."
    )


if __name__ == "__main__":
    sys.exit(main())
