"""Tests for the candidate-set frontier-knee detector."""

from __future__ import annotations

from src.bayesopt.run_candidate_sets import _frontier_knee


def test_knee_stops_when_gain_flattens() -> None:
    # gains 1->4 (+1.77pp), 4->8 (~0) -> knee at 4
    pts = [(1, 0.2292), (4, 0.2469), (8, 0.2469)]
    assert _frontier_knee(pts) == 4


def test_knee_keeps_going_while_improving() -> None:
    pts = [(1, 0.2292), (2, 0.2412), (4, 0.2429), (8, 0.2481)]
    assert _frontier_knee(pts) == 8


def test_knee_at_first_when_wider_hurts() -> None:
    # immediately worse -> stay at the cheapest
    pts = [(1, 0.2465), (2, 0.2355), (4, 0.2425), (8, 0.2282)]
    assert _frontier_knee(pts) == 1


def test_knee_single_point() -> None:
    assert _frontier_knee([(4, 0.5)]) == 4
