"""
Exact clock-sweep simulator objective for Mode B.

The BO **search objective** is the exact page-level clock-sweep
simulation (``simulate_schedule_page_level``) — Level 2 of the
objective ladder.  The directional edge-sum is recorded per trial as a
*diagnostic only* and never decides anything.  Cost is defined once,
project-wide for Mode B, as::

    cost = 1.0 − F_hit  (the simulated miss ratio)

Because the total page-request count is invariant under permutation,
minimizing ``1 − F_hit`` is exactly equivalent to minimizing total
simulated page reads; both are logged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.simulator.cache_simulator import (
    PageClockSweepCache,
    simulate_schedule_page_level,
)


@dataclass(frozen=True)
class PerQueryTrace:
    """
    Per-query simulation record at one schedule position.

    Attributes
    ----------
    position : int
        0-based execution position in the schedule.
    query_index : int
        Query index executed at this position.
    requests : int
        Pages requested by this query.
    hits : int
        Pages served from cache.
    misses : int
        ``requests − hits``.
    """

    position: int
    query_index: int
    requests: int
    hits: int
    misses: int


def simulate_schedule_page_level_traced(
    page_sets: list[frozenset[int]],
    schedule: Sequence[int],
    cache_capacity_pages: int,
) -> tuple[int, int, list[PerQueryTrace]]:
    """
    Simulate a schedule and return per-query metrics from the same pass.

    Replays the schedule through one clock-sweep cache, capturing each
    query's request and hit counts as it goes — a single O(n) pass, not
    prefix re-simulation.  The aggregate totals match
    :func:`simulate_schedule_page_level` exactly.

    Parameters
    ----------
    page_sets : list[frozenset[int]]
        Integer-encoded per-query page sets.
    schedule : Sequence[int]
        Permutation of ``range(len(page_sets))``.
    cache_capacity_pages : int
        Cache capacity in pages.

    Returns
    -------
    tuple[int, int, list[PerQueryTrace]]
        ``(total_requests, total_hits, per_query_traces)``.
    """
    cache = PageClockSweepCache(cache_capacity_pages)
    total_requests = 0
    total_hits = 0
    traces: list[PerQueryTrace] = []
    for position, idx in enumerate(schedule):
        pages = page_sets[idx]
        req = len(pages)
        hits = cache.batch_access(pages)
        total_requests += req
        total_hits += hits
        traces.append(
            PerQueryTrace(
                position=position,
                query_index=idx,
                requests=req,
                hits=hits,
                misses=req - hits,
            )
        )
    return total_requests, total_hits, traces


def is_valid_permutation(schedule: Sequence[int], n: int) -> bool:
    """
    Check that *schedule* is a permutation of ``range(n)``.

    Parameters
    ----------
    schedule : Sequence[int]
        Candidate schedule.
    n : int
        Expected number of queries.

    Returns
    -------
    bool
        True iff every query index appears exactly once.
    """
    return len(schedule) == n and sorted(schedule) == list(range(n))


@dataclass(frozen=True)
class EvalMetrics:
    """
    Exact-simulator metrics for one schedule.

    Attributes
    ----------
    cost : float
        ``1 − hit_ratio``; the quantity all Mode B optimizers minimize.
    hit_ratio : float
        Simulated F_hit.
    total_hits : int
        Simulated page hits.
    total_requests : int
        Simulated page requests (schedule-invariant).
    page_reads : int
        Simulated disk reads, ``total_requests − total_hits``.
    edge_sum_d : int | None
        Diagnostic single-step directional edge sum
        ``Σ_k D[π(k−1)][π(k)]``; None when no D matrix was supplied.
    """

    cost: float
    hit_ratio: float
    total_hits: int
    total_requests: int
    page_reads: int
    edge_sum_d: int | None


class ExactSimObjective:
    """
    Memoized exact-simulator objective over a fixed workload and cache.

    Distinct priority vectors frequently decode to the same permutation
    (the decode-by-sort landscape is piecewise constant), so results
    are memoized on the decoded schedule.  Memoization saves simulator
    time only — it never changes values and trial budgets are counted
    by the recorder regardless of memo hits.

    Parameters
    ----------
    page_sets : list[frozenset[int]]
        Integer-encoded per-query page sets.
    cache_capacity_pages : int
        Clock-sweep cache capacity in 8 KB pages.
    d_matrix : list[list[int]] | None
        Directional matrix for the diagnostic edge sum.  Optional.
    """

    def __init__(
        self,
        page_sets: list[frozenset[int]],
        cache_capacity_pages: int,
        d_matrix: list[list[int]] | None = None,
    ) -> None:
        if cache_capacity_pages <= 0:
            raise ValueError("cache_capacity_pages must be positive")
        if d_matrix is not None and len(d_matrix) != len(page_sets):
            raise ValueError("d_matrix size does not match page_sets")
        self.page_sets = page_sets
        self.cache_capacity_pages = cache_capacity_pages
        self.d_matrix = d_matrix
        self._memo: dict[tuple[int, ...], EvalMetrics] = {}

    @property
    def n(self) -> int:
        """Number of queries."""
        return len(self.page_sets)

    def evaluate_schedule(
        self, schedule: Sequence[int]
    ) -> tuple[EvalMetrics, bool]:
        """
        Score one schedule with the exact clock-sweep simulator.

        Parameters
        ----------
        schedule : Sequence[int]
            Permutation of ``range(n)``.

        Returns
        -------
        tuple[EvalMetrics, bool]
            The metrics and a flag that is True when the result came
            from the memo cache (no simulation was run).
        """
        key = tuple(schedule)
        cached = self._memo.get(key)
        if cached is not None:
            return cached, True

        sim = simulate_schedule_page_level(
            self.page_sets, list(schedule), self.cache_capacity_pages
        )
        edge_sum: int | None = None
        if self.d_matrix is not None:
            edge_sum = sum(
                self.d_matrix[a][b] for a, b in zip(schedule, schedule[1:])
            )
        metrics = EvalMetrics(
            cost=1.0 - sim.hit_ratio,
            hit_ratio=sim.hit_ratio,
            total_hits=sim.total_hits,
            total_requests=sim.total_requests,
            page_reads=sim.total_requests - sim.total_hits,
            edge_sum_d=edge_sum,
        )
        self._memo[key] = metrics
        return metrics, False


__all__ = [
    "EvalMetrics",
    "ExactSimObjective",
    "PerQueryTrace",
    "is_valid_permutation",
    "simulate_schedule_page_level_traced",
]
