"""
Profile queries by capturing actual page accesses via pg_buffercache.

For each query, the buffer cache is flushed (cold start), the query is
executed, and pg_buffercache is inspected to record which (table, block)
pages were loaded.  Results are saved as CSV files in the page_access
directory.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

from psycopg import Connection

logger = logging.getLogger(__name__)

from psycopg.sql import SQL

BUFFERCACHE_QUERY = """
SELECT c.relname, b.relblocknumber
FROM pg_buffercache b
JOIN pg_class c ON b.relfilenode = c.relfilenode
WHERE b.reldatabase = (SELECT oid FROM pg_database WHERE datname = current_database())
  AND b.relblocknumber IS NOT NULL
  AND c.relkind = 'r'
ORDER BY c.relname, b.relblocknumber;
"""


def profile_query(
    query_id: str,
    query: SQL,
    connection: Connection,
) -> list[tuple[str, int]]:
    """
    Execute a query and capture which pages it loaded into the buffer cache.

    The caller is responsible for flushing the buffer cache before calling
    this function to ensure a cold start.

    Parameters
    ----------
    query_id : str
        Identifier for the query.
    query : psycopg.sql.SQL
        SQL statement to execute, as a psycopg ``Composable`` so that
        the EXPLAIN wrapper is built via ``SQL.format`` and any nested
        composition is preserved.  Passing a raw ``str`` here would
        bypass psycopg's escaping and serialise composed identifiers as
        their template literals (a silent footgun).
    connection : Connection
        Active PostgreSQL connection.

    Returns
    -------
    list[tuple[str, int]]
        List of (table_name, block_number) pairs found in the buffer cache
        after executing the query.
    """
    t0 = time.perf_counter()
    explain_stmt = SQL("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {q}").format(
        q=query,
    )
    with connection.cursor() as cur:
        cur.execute(explain_stmt)
        cur.fetchone()
    elapsed = time.perf_counter() - t0
    print(f"    Query executed in {elapsed:.1f}s")

    with connection.cursor() as cur:
        cur.execute(BUFFERCACHE_QUERY)
        rows = cur.fetchall()

    pages = [(row[0], row[1]) for row in rows]
    print(f"    Captured {len(pages):,} pages in buffer cache")
    return pages


def save_page_access(
    query_id: str,
    pages: list[tuple[str, int]],
    output_dir: Path,
) -> Path:
    """
    Save page access data to a CSV file.

    Parameters
    ----------
    query_id : str
        Identifier for the query.
    pages : list[tuple[str, int]]
        List of (table_name, block_number) pairs.
    output_dir : Path
        Directory in which to save the file.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{query_id}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["table", "block"])
        writer.writerows(pages)
    return path


def load_page_access(path: Path) -> set[tuple[str, int]]:
    """
    Load page access data from a CSV file.

    Parameters
    ----------
    path : Path
        Path to a CSV file with columns (table, block).

    Returns
    -------
    set[tuple[str, int]]
        Set of (table_name, block_number) pairs.
    """
    pages: set[tuple[str, int]] = set()
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            pages.add((row[0], int(row[1])))
    return pages


def load_all_page_access(directory: Path) -> dict[str, set[tuple[str, int]]]:
    """
    Load page access data for all queries in a directory.

    Parameters
    ----------
    directory : Path
        Directory containing per-query CSV files.

    Returns
    -------
    dict[str, set[tuple[str, int]]]
        Mapping from query_id to set of (table, block) pairs.
    """
    result: dict[str, set[tuple[str, int]]] = {}
    for path in sorted(directory.glob("*.csv")):
        query_id = path.stem
        result[query_id] = load_page_access(path)
    return result
