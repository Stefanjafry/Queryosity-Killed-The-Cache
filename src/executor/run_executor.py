"""
Execute queries against PostgreSQL in a specified order and report actual
buffer cache statistics.

The buffer cache is always flushed (via Docker container restart) before
each schedule run to ensure cold-start, reproducible measurements.

Usage
-----
    # Execute in default (natural) order:
    python -m src.executor.run_executor --workload tpch

    # Execute in a custom order:
    python -m src.executor.run_executor --workload tpch --order q3,q5,q1,q10

    # Compare baseline order vs. a GA-optimized order:
    python -m src.executor.run_executor --workload tpch --order q5,q3,q1 --compare-baseline
"""

from __future__ import annotations

import argparse
import logging
import random
import subprocess
import time

logging.basicConfig(level=logging.WARNING)

from psycopg import Connection

from src.executor.executor import execute_schedule, print_execution_result
from src.postgres.connection import close_connection, create_connection
from src.utilities.configurations import (
    BASELINE_SEED,
    PG_CONTAINER_NAME,
    PG_HOST,
    PG_PASSWORD,
    PG_PORT,
    PG_SCHEMA,
    PG_STATEMENT_TIMEOUT_MS,
    PG_USER,
)
from src.utilities.constants import DB_DEFAULTS, WORKLOAD_DIRS
from src.utilities.workload import load_queries
from src.visualization.serializers import dump_executor_data


def flush_buffer_cache(
    container_name: str | None = None,
    *,
    flush_cmd: str | None = None,
    host: str = "127.0.0.1",
    port: int = 5432,
) -> None:
    """
    Flush PostgreSQL shared buffers by restarting the server.

    Exactly one of *container_name* or *flush_cmd* must be supplied:

    * ``container_name`` -- legacy Docker setup; runs
      ``docker restart <name>``.
    * ``flush_cmd`` -- arbitrary shell command for non-Docker setups,
      e.g. ``"sudo systemctl restart postgresql-16"`` on a native
      install.

    Either approach clears PostgreSQL's shared_buffers (the postgres
    process restarts).  Neither approach clears the host OS page cache;
    this is the same behaviour the original Docker-based path had, so
    wall-clock numbers remain comparable between setups.

    After running the restart command, blocks until the server again
    accepts a TCP connection on *host*:*port*.

    Parameters
    ----------
    container_name : str | None
        Docker container name (legacy path).  Mutually exclusive with
        ``flush_cmd``.
    flush_cmd : str | None
        Shell command for non-Docker setups.  Mutually exclusive with
        ``container_name``.
    host : str
        Host to probe while waiting for the server to come back.
    port : int
        Port to probe while waiting for the server to come back.

    Raises
    ------
    ValueError
        If neither or both of ``container_name`` and ``flush_cmd`` are
        given.
    RuntimeError
        If the restart command fails or the server does not become
        ready within the wait timeout.
    """
    if (container_name is None) == (flush_cmd is None):
        raise ValueError(
            "flush_buffer_cache: pass exactly one of "
            "container_name or flush_cmd"
        )

    if container_name is not None:
        print(f"  Flushing buffer cache (docker restart {container_name})…")
        result = subprocess.run(
            ["docker", "restart", container_name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to restart container '{container_name}': "
                f"{result.stderr.strip()}"
            )
    else:
        assert flush_cmd is not None  # narrowing for type checker
        print(f"  Flushing buffer cache ({flush_cmd})…")
        result = subprocess.run(
            flush_cmd,
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"Flush command failed (rc={result.returncode}): {err}"
            )

    _wait_for_pg(host=host, port=port)
    print("  Buffer cache flushed — PostgreSQL is ready.")


def _wait_for_pg(
    host: str = "127.0.0.1",
    port: int = 5432,
    timeout_s: int = 30,
) -> None:
    """
    Block until PostgreSQL accepts a TCP connection on *host*:*port*.

    Probes via psycopg rather than ``pg_isready`` so we do not depend
    on the libpq client binary being on PATH (it is not, by default,
    on PGDG RHEL installs).

    Parameters
    ----------
    host : str
        Host to probe.
    port : int
        Port to probe.
    timeout_s : int
        Maximum seconds to wait before raising.

    Raises
    ------
    RuntimeError
        If PostgreSQL does not become ready within the timeout.
    """
    import psycopg  # local import to avoid a hard dep at module load

    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            conn = psycopg.connect(
                host=host,
                port=port,
                dbname="postgres",
                user="postgres",
                password="postgres",
                connect_timeout=2,
            )
            conn.close()
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5)
    raise RuntimeError(
        f"PostgreSQL at {host}:{port} did not become ready within "
        f"{timeout_s}s (last error: {last_exc})"
    )


def _connect(args, db_name: str) -> Connection:
    """
    Create a PostgreSQL connection from parsed CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments containing connection parameters.
    db_name : str
        Name of the database to connect to.

    Returns
    -------
    Connection
        Active PostgreSQL connection.
    """
    return create_connection(
        db_name=db_name,
        user=args.user,
        password=args.password,
        host=args.host,
        port=args.port,
        schema=args.schema,
        statement_timeout_ms=args.timeout_ms,
    )


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments, execute queries, and print cache statistics.

    The buffer cache is flushed before each schedule run.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments.  Uses sys.argv when None.
    """
    parser = argparse.ArgumentParser(
        description="Execute queries and report actual buffer cache statistics",
    )
    parser.add_argument(
        "--workload",
        choices=list(WORKLOAD_DIRS),
        default="tpch",
        help="Query workload (default: tpch)",
    )
    parser.add_argument(
        "--order",
        type=str,
        default=None,
        help="Comma-separated query IDs in desired execution order "
        "(e.g. q3,q1,q5). If omitted, uses natural file order.",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also execute in default order for comparison",
    )
    parser.add_argument(
        "--container",
        default=PG_CONTAINER_NAME,
        help=f"Docker container name for cache flushing (default: {PG_CONTAINER_NAME}). "
             "Ignored when --flush-cmd is given.",
    )
    parser.add_argument(
        "--flush-cmd",
        default=None,
        help="Shell command to run to flush the buffer cache (e.g. "
             "'sudo systemctl restart postgresql-16' for native installs). "
             "If omitted, falls back to 'docker restart <--container>'.",
    )
    parser.add_argument("--host", default=PG_HOST)
    parser.add_argument("--port", type=int, default=PG_PORT)
    parser.add_argument("--user", default=PG_USER)
    parser.add_argument("--password", default=PG_PASSWORD)
    parser.add_argument("--schema", default=PG_SCHEMA)
    parser.add_argument("--timeout-ms", type=int, default=PG_STATEMENT_TIMEOUT_MS)
    args = parser.parse_args(argv)

    db_name = DB_DEFAULTS[args.workload]

    print(f"Loading {args.workload} queries…")
    queries = load_queries(args.workload)
    query_ids = list(queries.keys())
    print(f"  Found {len(queries)} queries")

    if args.order:
        schedule = [q.strip() for q in args.order.split(",")]
        missing = [q for q in schedule if q not in queries]
        if missing:
            parser.error(f"Unknown query IDs: {', '.join(missing)}")
    else:
        schedule = query_ids

    baseline = None

    if args.compare_baseline:
        rng = random.Random(BASELINE_SEED)
        random_order = list(query_ids)
        rng.shuffle(random_order)
        if args.flush_cmd:
            flush_buffer_cache(flush_cmd=args.flush_cmd, host=args.host, port=args.port)
        else:
            flush_buffer_cache(container_name=args.container, host=args.host, port=args.port)
        print(f"\nConnecting to database '{db_name}' at {args.host}:{args.port}…")
        conn = _connect(args, db_name)
        try:
            print("\nExecuting baseline (random order)…")
            print(f"  Order: {' → '.join(random_order)}")
            baseline = execute_schedule(queries, random_order, conn)
            print_execution_result(baseline, "Baseline (random order)")
        finally:
            close_connection(conn)

    if args.flush_cmd:
        flush_buffer_cache(flush_cmd=args.flush_cmd, host=args.host, port=args.port)
    else:
        flush_buffer_cache(container_name=args.container, host=args.host, port=args.port)
    print(f"\nConnecting to database '{db_name}' at {args.host}:{args.port}…")
    conn = _connect(args, db_name)
    try:
        label = "Custom order" if args.order else "Default order"
        print(f"\nExecuting {label.lower()}…")
        result = execute_schedule(queries, schedule, conn)
        print_execution_result(result, label)
    finally:
        close_connection(conn)

    if baseline is not None:
        improvement = result.hit_ratio - baseline.hit_ratio
        print(f"\n{'─' * 48}")
        print(
            f"  Hit ratio improvement : "
            f"{improvement:+.4f}  ({improvement * 100:+.2f}pp)"
        )
        time_diff = result.total_elapsed_ms - baseline.total_elapsed_ms
        print(f"  Time difference       : {time_diff:+.1f} ms")
        dump_executor_data(baseline, result, workload=args.workload)

        print(f"{'─' * 48}\n")


if __name__ == "__main__":
    main()
