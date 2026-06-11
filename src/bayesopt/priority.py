"""
Random-key (priority-vector) encoding of query schedules.

A continuous vector ``p = (p_0, …, p_{n-1})`` with ``p_i ∈ [0, 1]`` is
decoded into a permutation by sorting query indices by **descending**
priority.  Ties break on the lower query index, which makes decoding a
deterministic function of the vector regardless of platform, hash seed
or sort implementation.

This is the standard random-key trick that lets continuous black-box
optimizers (SMAC, GP-based BO) operate on a permutation space.  Note
the resulting objective landscape is piecewise constant: a priority
perturbation changes the schedule only when it flips a pairwise order.
"""

from __future__ import annotations

from typing import Sequence


def decode_priority_vector(priorities: Sequence[float]) -> list[int]:
    """
    Decode a priority vector into a query execution order.

    Parameters
    ----------
    priorities : Sequence[float]
        One priority per query.  Higher priority executes earlier.

    Returns
    -------
    list[int]
        Permutation of ``range(len(priorities))`` sorted by descending
        priority, ties broken by ascending query index.

    Raises
    ------
    ValueError
        If *priorities* is empty.
    """
    n = len(priorities)
    if n == 0:
        raise ValueError("priority vector must be non-empty")
    return sorted(range(n), key=lambda i: (-priorities[i], i))


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


__all__ = ["decode_priority_vector", "is_valid_permutation"]
