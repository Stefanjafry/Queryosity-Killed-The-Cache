"""Tests for the windowed-M step term."""

from __future__ import annotations

from src.bayesopt.windowed_scorer import (
    greedy_windowed_schedule,
    windowed_immediate_value,
)


def test_single_predecessor_equals_overlap() -> None:
    # one placed predecessor, ample budget -> value == overlap[p][j] (capped)
    M = [[0, 30, 20], [30, 0, 10], [20, 10, 0]]
    pc = [100, 100, 100]
    v = windowed_immediate_value(2, [0], M, pc, cache_pages=10_000)
    assert v == 20.0  # M[0][2], below pc[2]


def test_value_capped_at_target_pagecount() -> None:
    M = [[0, 500], [500, 0]]
    pc = [100, 80]
    v = windowed_immediate_value(1, [0], M, pc, cache_pages=10_000)
    assert v == 80.0  # capped at pc[1]


def test_empty_window_when_cache_smaller_than_predecessor() -> None:
    # budget goes negative on the first predecessor -> no contribution
    M = [[0, 40], [40, 0]]
    pc = [5000, 100]
    v = windowed_immediate_value(1, [0], M, pc, cache_pages=1000)
    assert v == 0.0


def test_triple_intersection_discount_reduces_double_count() -> None:
    # two predecessors both overlapping j and each other -> discounted sum
    M = [[0, 60, 50], [60, 0, 50], [50, 50, 0]]
    pc = [100, 100, 100]
    # placed = [0, 1] (1 most recent); ample budget so both count
    v = windowed_immediate_value(2, [0, 1], M, pc, cache_pages=10_000)
    # prev=1: incremental 50, no counted yet -> +50
    # prev=0: incremental 50, discount = M[0][1]*M[1][2]/pc[1] = 60*50/100 = 30 -> +20
    assert abs(v - 70.0) < 1e-9


def test_greedy_windowed_returns_valid_permutation() -> None:
    M = [[0, 30, 20, 10], [30, 0, 25, 15], [20, 25, 0, 35], [10, 15, 35, 0]]
    pc = [100, 200, 150, 120]
    sched = greedy_windowed_schedule(M, pc, cache_pages=500)
    assert sorted(sched) == [0, 1, 2, 3]
    assert sched[0] == 1  # largest pagecount start
