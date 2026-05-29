"""
Profile all queries in a workload by capturing page-level buffer cache access.

Each query is run from a cold cache (buffer cache flushed via Docker
restart or ``--flush-cmd``), then ``pg_buffercache`` is inspected to
record exactly which pages were loaded.  Results are saved as CSV files
in ``page_access/<workload>/`` by default; pass ``--output-dir`` to send
them elsewhere — e.g. one directory per simulated cache size when
calibrating the directional matrix at multiple buffer capacities:

    SET shared_buffers='800MB';  -- restart Postgres
    python -m src.profiler.run_profiler \\
        --workload tpch \\
        --output-dir page_access/tpch_sb800mb \\
        --flush-cmd "sudo systemctl restart postgresql-16"

Then point ``src.scheduler.run_scheduler`` at the matching directory via
``--page-access-dir page_access/tpch_sb800mb`` so the simulator sees
residue sets captured at the cache size it is about to simulate, rather
than residues filtered through a different (larger) cache.

Usage
-----
    python -m src.profiler.run_profiler --workload tpcds
    python -m src.profiler.run_profiler --workload tpch
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

from src.executor.run_executor import flush_buffer_cache
from src.postgres.connection import close_connection, create_connection
from src.profiler.page_profiler import profile_query, save_page_access
from src.utilities.configurations import (
    PG_CONTAINER_NAME,
    PG_HOST,
    PG_PASSWORD,
    PG_PORT,
    PG_SCHEMA,
    PG_STATEMENT_TIMEOUT_MS,
    PG_USER,
)
from src.utilities.constants import DB_DEFAULTS, PROJECT_ROOT, WORKLOAD_DIRS
from src.utilities.workload import load_queries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Profile queries via pg_buffercache",
    )
    parser.add_argument(
        "--workload",
        choices=list(WORKLOAD_DIRS),
        default="tpch",
        help="Query workload to profile (default: tpch)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write per-query CSVs into.  Defaults to "
             "page_access/<workload>/.  Use a per-cache-size directory "
             "(e.g. page_access/tpch_sb800mb) when re-profiling at a "
             "non-default shared_buffers so the simulator can match the "
             "regime it is going to simulate.",
    )
    parser.add_argument(
        "--container",
        default=PG_CONTAINER_NAME,
        help=f"Docker container name (default: {PG_CONTAINER_NAME}). "
             "Ignored when --flush-cmd is given.",
    )
    parser.add_argument(
        "--flush-cmd",
        default=None,
        help="Shell command to flush the buffer cache, e.g. "
             "'sudo systemctl restart postgresql-16' for native installs. "
             "Falls back to 'docker restart <--container>' if omitted.",
    )
    parser.add_argument("--host", default=PG_HOST)
    parser.add_argument("--port", type=int, default=PG_PORT)
    parser.add_argument("--user", default=PG_USER)
    parser.add_argument("--password", default=PG_PASSWORD)
    parser.add_argument("--schema", default=PG_SCHEMA)
    parser.add_argument("--timeout-ms", type=int, default=PG_STATEMENT_TIMEOUT_MS)
    args = parser.parse_args(argv)

    db_name = DB_DEFAULTS[args.workload]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "page_access" / args.workload
    )

    print(f"Loading {args.workload} queries…")
    queries = load_queries(args.workload)
    print(f"  Found {len(queries)} queries")

    print(f"  Output directory: {output_dir}")

    # Ensure pg_buffercache extension is available
    print("  Ensuring pg_buffercache extension…")
    init_conn = create_connection(
        db_name=db_name,
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
        schema=args.schema,
        statement_timeout_ms=args.timeout_ms,
    )
    try:
        with init_conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_buffercache;")
    finally:
        close_connection(init_conn)

    t_total = time.perf_counter()

    for i, (query_id, query_sql) in enumerate(queries.items(), 1):
        print(f"\n[{i}/{len(queries)}] Profiling {query_id}…")

        flush_args = (
            {"flush_cmd": args.flush_cmd}
            if args.flush_cmd
            else {"container_name": args.container}
        )
        flush_buffer_cache(**flush_args, host=args.host, port=args.port)

        conn = create_connection(
            db_name=db_name,
            user=args.user,
            password=args.password,
            host=args.host,
            port=args.port,
            schema=args.schema,
            statement_timeout_ms=args.timeout_ms,
        )

        try:
            pages = profile_query(query_id, query_sql, conn)
            path = save_page_access(query_id, pages, output_dir)
            print(f"    Saved to {path}")
        except Exception as exc:
            print(f"    FAILED: {exc}")
        finally:
            close_connection(conn)

    elapsed = time.perf_counter() - t_total
    print(f"\nProfiling complete in {elapsed:.1f}s")
    print(f"Page access data saved to {output_dir}")


if __name__ == "__main__":
    main()
