"""Tests for the Phase-2 residual scorer kernel."""

from __future__ import annotations

from src.bayesopt.residual_scorer import (
    alpha_m_window_hits,
    e_window_sum,
    greedy_windowed_schedule,
    make_scorer,
    residual_frac,
    row_alpha_residual,
)


def _sym(n: int) -> list[list[float]]:
    m = [[0.0] * n for _ in range(n)]
    val = 1
    for i in range(n):
        for j in range(i + 1, n):
            m[i][j] = m[j][i] = float(val * 10)
            val += 1
    return m


def test_row_alpha_residual_recovers_pure_scaling() -> None:
    M = _sym(5)
    alpha_true = [1.0, 0.5, 0.25, 0.8, 0.0]
    D = [[alpha_true[i] * M[i][j] if i != j else 0.0 for j in range(5)]
         for i in range(5)]
    alpha, E = row_alpha_residual(M, D)
    for i in range(5):
        if i == 4:  # zero row -> alpha falls back to 0
            continue
        assert abs(alpha[i] - alpha_true[i]) < 1e-9
        assert all(abs(E[i][j]) < 1e-6 for j in range(5))
    assert residual_frac(M, D, E) < 1e-6


def test_lambda_e_zero_is_pure_alpha_m_window() -> None:
    M = _sym(6)
    D = [[0.4 * M[i][j] if i != j else 0.0 for j in range(6)] for i in range(6)]
    alpha, E = row_alpha_residual(M, D)
    pc = [100] * 6
    scorer0 = make_scorer(alpha, M, E, pc, 250, 0.0)
    placed = [0, 1]
    for q in (2, 3, 4, 5):
        assert scorer0(q, placed) == alpha_m_window_hits(q, placed, alpha, M, pc, 250)


def test_e_window_sum_respects_cache_budget() -> None:
    # Three predecessors of 100 pages each; budget 250 admits the two most
    # recent (250 - 100 - 100 = 50 >= 0; third would push budget negative).
    n = 5
    E = [[0.0] * n for _ in range(n)]
    E[0][4] = 7.0  # oldest predecessor, should be excluded
    E[1][4] = 3.0
    E[2][4] = 2.0
    pc = [100] * n
    placed = [0, 1, 2]  # order: 0 oldest, 2 most recent
    s = e_window_sum(4, placed, E, pc, 250)
    assert s == E[2][4] + E[1][4]  # 0 is out of budget


def test_greedy_returns_valid_permutation_and_e_can_change_it() -> None:
    M = _sym(7)
    D = [[0.3 * M[i][j] if i != j else 0.0 for j in range(7)] for i in range(7)]
    alpha, E = row_alpha_residual(M, D)
    pc = [200] + [100] * 6  # query 0 has most pages -> it is the greedy start
    # Inject a strong residual on an edge FROM the start node, so it is in
    # the window when its successor is scored and actually steers greedy.
    E[0][5] = 50000.0
    s0 = greedy_windowed_schedule(make_scorer(alpha, M, E, pc, 400, 0.0), 7, pc)
    s1 = greedy_windowed_schedule(make_scorer(alpha, M, E, pc, 400, 5.0), 7, pc)
    assert sorted(s0) == list(range(7))
    assert sorted(s1) == list(range(7))
    assert s0[0] == s1[0] == 0  # same seed
    assert s1[1] == 5  # E pulls query 5 right after the start
    assert s0 != s1  # E changed the schedule
