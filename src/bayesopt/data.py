"""
Load page-access profiles for BO runs without a database connection.

The BO search phase is simulator-only: it needs each query's
integer-encoded page set and nothing else.  Query identifiers come from
the CSV file stems in the page-access directory and are ordered with
the same natural-sort rule as ``src.utilities.workload.load_queries``
so that index *i* here refers to the same query as index *i* everywhere
else in the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.profiler.page_profiler import load_all_page_access
from src.simulator.cache_simulator import encode_page_sets


def _natural_sort_key(stem: str) -> list[int | str]:
    """
    Sort key splitting digit runs so 'query2' precedes 'query10'.

    Mirrors the key used by ``load_queries``; the two must stay in
    lockstep or query indices will diverge between the BO runner and
    the rest of the pipeline.
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", stem)
    ]


@dataclass(frozen=True)
class WorkloadPages:
    """
    Integer-encoded page sets for one workload at one profiling pass.

    Attributes
    ----------
    query_ids : list[str]
        Query identifiers in natural-sort order; ``query_ids[i]``
        labels ``page_sets[i]``.
    page_sets : list[frozenset[int]]
        Per-query page sets encoded via ``encode_page_sets``.
    total_unique_pages : int
        Number of distinct pages across the workload.
    """

    query_ids: list[str]
    page_sets: list[frozenset[int]]
    total_unique_pages: int

    @property
    def n(self) -> int:
        """Number of queries."""
        return len(self.query_ids)


def load_workload_pages(
    page_access_dir: Path,
    exclude: set[str] | None = None,
) -> WorkloadPages:
    """
    Load and encode per-query page sets from a page-access directory.

    Parameters
    ----------
    page_access_dir : Path
        Directory of per-query CSVs (one per query, named by query id).
    exclude : set[str] | None
        Query ids to drop before encoding (e.g. wall-clock exclusions).

    Returns
    -------
    WorkloadPages
        Encoded page sets aligned with natural-sorted query ids.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or contains no CSV files.
    ValueError
        If *exclude* names a query id not present in the directory, or
        if any loaded page set is empty.
    """
    if not page_access_dir.is_dir():
        raise FileNotFoundError(f"page-access dir not found: {page_access_dir}")

    all_pages = load_all_page_access(page_access_dir)
    if not all_pages:
        raise FileNotFoundError(f"no page-access CSVs in: {page_access_dir}")

    exclude = exclude or set()
    unknown = exclude - set(all_pages)
    if unknown:
        raise ValueError(
            f"--exclude names unknown query ids: {sorted(unknown)} "
            f"(available: {sorted(all_pages)[:5]}…)"
        )

    query_ids = sorted(
        (qid for qid in all_pages if qid not in exclude),
        key=_natural_sort_key,
    )
    raw_sets = [all_pages[qid] for qid in query_ids]
    empty = [qid for qid, ps in zip(query_ids, raw_sets) if not ps]
    if empty:
        raise ValueError(
            f"empty page sets for {empty}; refusing to search over "
            f"queries the simulator cannot score"
        )

    page_sets, page_to_id = encode_page_sets(raw_sets)
    return WorkloadPages(
        query_ids=query_ids,
        page_sets=page_sets,
        total_unique_pages=len(page_to_id),
    )


__all__ = ["WorkloadPages", "load_workload_pages"]
