"""
Greedy nearest-neighbour scheduler on the directional utility matrix.

At each step, the algorithm extends the partial schedule by appending
the unvisited query Qⱼ that maximises ``D[current][Qⱼ]`` — i.e. the
next query whose page set has the largest expected hit count given the
residual left by the most recently scheduled query.

This is intentionally simple.  It exists as a free baseline: if it
beats the random baseline by a non-trivial margin, the directional
matrix is itself useful as a scheduling signal, independent of the GA.
If the GA with directional fitness then beats it further, the GA's
search adds value on top of the heuristic.

Functions
---------
greedy_directional_schedule
    Return a permutation built by repeated nearest-neighbour selection
    on the directional matrix.
"""

from __future__ import annotations


def greedy_directional_schedule(
    directional_matrix: list[list[int]],
    page_counts: list[int] | None = None,
    start: int | None = None,
) -> list[int]:
    """
    Build a schedule by greedy nearest-neighbour on the directional matrix.

    Starting from *start* (or the query with the largest page count if
    ``page_counts`` is supplied, else query 0), the schedule is grown
    by repeatedly choosing the unvisited query maximising
    ``D[current][candidate]``.  Ties break on the lower query index.

    Parameters
    ----------
    directional_matrix : list[list[int]]
        Asymmetric *n* x *n* matrix from
        ``compute_directional_matrix``.
    page_counts : list[int] or None
        Per-query page counts.  Used only to pick the starting query
        when *start* is None: the query with the most pages tends to
        be a good seed because it produces the largest residual.
    start : int or None
        Index of the query to schedule first.  If None, derived from
        *page_counts* as described above, or 0 if both are None.

    Returns
    -------
    list[int]
        Permutation of ``range(n)``.

    Raises
    ------
    ValueError
        If the matrix is empty or non-square, or if *start* is out of
        range.
    """
    n = len(directional_matrix)
    if n == 0:
        return []
    for row in directional_matrix:
        if len(row) != n:
            raise ValueError("directional_matrix must be square")

    if start is None:
        if page_counts is not None and len(page_counts) == n:
            start = max(range(n), key=lambda i: page_counts[i])
        else:
            start = 0
    if not 0 <= start < n:
        raise ValueError(f"start={start} out of range for n={n}")

    visited = [False] * n
    schedule = [start]
    visited[start] = True
    current = start

    for _ in range(n - 1):
        best_j = -1
        best_score = -1
        row = directional_matrix[current]
        for j in range(n):
            if visited[j]:
                continue
            if row[j] > best_score:
                best_score = row[j]
                best_j = j
        if best_j == -1:
            # No unvisited candidates with non-negative weight; fall
            # back to the lowest-index unvisited query.
            for j in range(n):
                if not visited[j]:
                    best_j = j
                    break
        schedule.append(best_j)
        visited[best_j] = True
        current = best_j

    return schedule


__all__ = ["greedy_directional_schedule"]
