"""
Windowed-M step term — Queryosity's multi-predecessor fitness as a step value.

The paper's GA+M fitness scores a whole schedule by walking, for each query,
its predecessors within a cache-page budget and summing pairwise overlap
minus a triple-intersection discount.  Here that exact computation is applied
to a *single candidate* j given the already-placed prefix, so windowing can
drive a greedy/beam consumer step by step and be compared, same-consumer,
against single-step-M and single-step-D.

Why this isolates the windowing axis: at small cache the page budget covers
only the most recent predecessor, so the windowed value collapses to the
single edge overlap[last][j] — windowed-M ~= single-step-M.  At large cache
the budget spans several predecessors, so multiple survivors contribute,
which is where windowing *could* beat single-step.  The crossover (if any)
is the result.

This is the SYMMETRIC arm only.  Windowed-D collapsed to windowed-M in prior
work (the windowing factors out D's row scaling), so it is deliberately not
built here.
"""

from __future__ import annotations

from collections.abc import Sequence


def windowed_immediate_value(
    j: int,
    placed: Sequence[int],
    overlap_matrix: Sequence[Sequence[int]],
    page_counts: Sequence[int],
    cache_pages: int,
) -> float:
    """
    Windowed multi-predecessor hit estimate for placing j after *placed*.

    Walks the placed prefix newest-first within a page budget of
    *cache_pages*, summing overlap[prev][j] minus the triple-intersection
    discount against already-counted predecessors, capped at j's page count.
    Faithful to ``approximate_schedule_fitness`` applied to one candidate.
    """
    budget = cache_pages
    hits = 0.0
    counted: list[int] = []
    for prev in reversed(placed):
        budget -= page_counts[prev]
        if budget < 0:
            break
        incremental = float(overlap_matrix[prev][j])
        discount = 0.0
        for cp in counted:
            if page_counts[cp] > 0:
                discount += (overlap_matrix[prev][cp]
                             * overlap_matrix[cp][j] / page_counts[cp])
        hits += max(0.0, incremental - discount)
        counted.append(prev)
    return min(hits, float(page_counts[j]))


def greedy_windowed_schedule(
    overlap_matrix: Sequence[Sequence[int]],
    page_counts: Sequence[int],
    cache_pages: int,
    start: int | None = None,
) -> list[int]:
    """Greedy on the windowed-M step value; ties break to the lower index."""
    n = len(page_counts)
    if n == 0:
        return []
    if start is None:
        start = max(range(n), key=lambda i: page_counts[i])
    visited = [False] * n
    schedule = [start]
    visited[start] = True
    for _ in range(n - 1):
        best_j = -1
        best = float("-inf")
        for j in range(n):
            if visited[j]:
                continue
            v = windowed_immediate_value(j, schedule, overlap_matrix,
                                         page_counts, cache_pages)
            if v > best:
                best = v
                best_j = j
        schedule.append(best_j)
        visited[best_j] = True
    return schedule


__all__ = ["windowed_immediate_value", "greedy_windowed_schedule"]
