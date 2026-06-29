"""
Phase 2 — repaired windowed residual scorer (greedy consumer).

Decomposes ``D = alpha_i * M + E`` (the decomposition audited in
``run_residual_audit``) and scores a candidate query ``q`` against the
recently-scheduled window::

    score(q | window) = alphaM_window_hits(q, window)
                      + lambda_e * sum_{p in window} E[p][q]

The ``alphaM`` component reuses Queryosity's windowed inclusion-exclusion
(``approximate_schedule_fitness``, Algorithm 2) with each predecessor's
contribution scaled by its survival coefficient ``alpha_p``; the
triple-intersection correction is applied in that survived space, where
it still has page-set meaning.  ``E`` is a *signed scalar* residual with
no page set, so it is aggregated as a plain window-sum and never
inclusion-excluded.

``lambda_e == 0.0`` reproduces the pure ``alphaM_window`` scorer
*exactly* (shared code path), so the only difference between the two
arms of the Phase-2 comparison is the ``E`` term — the gain is
attributable to ``E`` and nothing else.

The greedy consumer builds a schedule by repeatedly appending the
unvisited query with the highest score given the current prefix.  The
exact clock-sweep simulator (not this estimate) judges the result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def row_alpha_residual(
    m_matrix: Sequence[Sequence[float]],
    d_matrix: Sequence[Sequence[float]],
) -> tuple[list[float], list[list[float]]]:
    """
    Per-row least-squares survival coefficient and directional residual.

    ``alpha_i = <D_i, M_i> / <M_i, M_i>`` over off-diagonal entries and
    ``E[i][j] = D[i][j] - alpha_i * M[i][j]`` (diagonal left at 0).
    """
    n = len(m_matrix)
    alpha = [0.0] * n
    residual = [[0.0] * n for _ in range(n)]
    for i in range(n):
        num = 0.0
        den = 0.0
        for j in range(n):
            if i == j:
                continue
            num += d_matrix[i][j] * m_matrix[i][j]
            den += m_matrix[i][j] * m_matrix[i][j]
        a = num / den if den > 0.0 else 0.0
        alpha[i] = a
        for j in range(n):
            if i == j:
                continue
            residual[i][j] = d_matrix[i][j] - a * m_matrix[i][j]
    return alpha, residual


def alpha_m_window_hits(
    q: int,
    placed: Sequence[int],
    alpha: Sequence[float],
    m_matrix: Sequence[Sequence[float]],
    page_counts: Sequence[int],
    cache_capacity: int,
) -> float:
    """
    Windowed inclusion-exclusion hit estimate for *q* given the *placed*
    prefix, with each predecessor's contribution scaled by ``alpha_p``.

    Mirrors ``approximate_schedule_fitness``'s inner loop: scan
    predecessors most-recent-first until the cache budget is spent;
    accumulate ``alpha_p * M[p][q]`` discounted by the alpha-scaled
    triple-intersection with already-counted predecessors; cap at
    ``page_counts[q]``.
    """
    budget = cache_capacity
    hits = 0.0
    counted: list[int] = []
    for prev in reversed(placed):
        budget -= page_counts[prev]
        if budget < 0:
            break
        raw = alpha[prev] * m_matrix[prev][q]
        discount = 0.0
        for cp in counted:
            if page_counts[cp] > 0:
                discount += (
                    (alpha[prev] * m_matrix[prev][cp])
                    * (alpha[cp] * m_matrix[cp][q])
                    / page_counts[cp]
                )
        hits += max(0.0, raw - discount)
        counted.append(prev)
    pcq = float(page_counts[q])
    return hits if hits <= pcq else pcq


def e_window_sum(
    q: int,
    placed: Sequence[int],
    residual: Sequence[Sequence[float]],
    page_counts: Sequence[int],
    cache_capacity: int,
) -> float:
    """
    Plain window-sum of ``E[p][q]`` over predecessors *p* that fall
    within the same cache budget window used by ``alpha_m_window_hits``.

    No inclusion-exclusion: ``E`` is a scalar residual without a page
    set, so triple-intersection would be meaningless.
    """
    budget = cache_capacity
    total = 0.0
    for prev in reversed(placed):
        budget -= page_counts[prev]
        if budget < 0:
            break
        total += residual[prev][q]
    return total


def make_scorer(
    alpha: Sequence[float],
    m_matrix: Sequence[Sequence[float]],
    residual: Sequence[Sequence[float]],
    page_counts: Sequence[int],
    cache_capacity: int,
    lambda_e: float,
) -> Callable[[int, list[int]], float]:
    """
    Build ``score(q, placed)`` = alphaM window hits + ``lambda_e`` * E
    window-sum.  ``lambda_e == 0.0`` is the pure ``alphaM_window`` scorer.
    """
    def score(q: int, placed: list[int]) -> float:
        base = alpha_m_window_hits(
            q, placed, alpha, m_matrix, page_counts, cache_capacity
        )
        if lambda_e == 0.0:
            return base
        return base + lambda_e * e_window_sum(
            q, placed, residual, page_counts, cache_capacity
        )

    return score


def greedy_windowed_schedule(
    scorer: Callable[[int, list[int]], float],
    n: int,
    page_counts: Sequence[int],
    start: int | None = None,
) -> list[int]:
    """
    Greedy schedule: repeatedly append the unvisited query with the
    highest ``scorer(q, prefix)``.  Ties break on the lower index.

    *start* defaults to the largest-pagecount query, matching
    ``greedy_directional_schedule`` so the two greedies share a seed.
    """
    if n == 0:
        return []
    if start is None:
        start = max(range(n), key=lambda i: page_counts[i])
    if not 0 <= start < n:
        raise ValueError(f"start={start} out of range for n={n}")

    visited = [False] * n
    schedule = [start]
    visited[start] = True
    for _ in range(n - 1):
        best_j = -1
        best_score = float("-inf")
        for j in range(n):
            if visited[j]:
                continue
            s = scorer(j, schedule)
            if s > best_score:
                best_score = s
                best_j = j
        schedule.append(best_j)
        visited[best_j] = True
    return schedule


def residual_frac(
    m_matrix: Sequence[Sequence[float]],
    d_matrix: Sequence[Sequence[float]],
    residual: Sequence[Sequence[float]],
) -> float:
    """``||E||_F / ||D||_F`` over off-diagonal entries (the audit metric)."""
    n = len(m_matrix)
    d2 = 0.0
    e2 = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d2 += d_matrix[i][j] * d_matrix[i][j]
            e2 += residual[i][j] * residual[i][j]
    return (e2 ** 0.5) / ((d2 ** 0.5) + 1e-9)


__all__ = [
    "row_alpha_residual",
    "alpha_m_window_hits",
    "e_window_sum",
    "make_scorer",
    "greedy_windowed_schedule",
    "residual_frac",
]
