"""Tests for the paired single-step scorer kernel."""

from __future__ import annotations

from src.bayesopt.step_scorer import (
    ScorerWeights,
    cache_feature,
    future_value,
    greedy_schedule,
    make_step_scorer,
    regret,
    row_normalize,
)


def _sym(n: int) -> list[list[float]]:
    m = [[0.0] * n for _ in range(n)]
    v = 1
    for i in range(n):
        for j in range(i + 1, n):
            m[i][j] = m[j][i] = float(v * 7 % 50 + 1)
            v += 1
    return m


def test_row_normalize_preserves_argmax() -> None:
    M = _sym(6)
    Mn = row_normalize(M)
    for i in range(6):
        raw_best = max((j for j in range(6) if j != i), key=lambda j: M[i][j])
        nrm_best = max((j for j in range(6) if j != i), key=lambda j: Mn[i][j])
        assert raw_best == nrm_best
        # each off-diagonal row sums to 1
        assert abs(sum(Mn[i][j] for j in range(6) if j != i) - 1.0) < 1e-9


def test_immediate_only_scorer_equals_matrix_entry() -> None:
    M = _sym(5)
    Mn = row_normalize(M)
    pc = [100] * 5
    score = make_step_scorer(Mn, ScorerWeights(), pc, 250)
    for i in range(5):
        for j in range(5):
            if i != j:
                assert score(i, j, [0, 1, 2, 3, 4]) == Mn[i][j]


def test_greedy_on_normalized_matches_greedy_on_raw() -> None:
    # Greedy-D / Greedy-M = D_only / M_only of this scorer.
    M = _sym(7)
    Mn = row_normalize(M)
    pc = [100, 200, 150, 120, 90, 110, 130]
    s_norm = greedy_schedule(make_step_scorer(Mn, ScorerWeights(), pc, 300), 7, pc)

    # Reference greedy directly on the raw matrix, same start rule.
    start = max(range(7), key=lambda i: pc[i])
    visited = [False] * 7
    ref = [start]
    visited[start] = True
    for _ in range(6):
        i = ref[-1]
        bj, bv = -1, float("-inf")
        for j in range(7):
            if not visited[j] and M[i][j] > bv:
                bv, bj = M[i][j], j
        ref.append(bj)
        visited[bj] = True
    assert s_norm == ref
    assert sorted(s_norm) == list(range(7))


def test_future_regret_cache_sane() -> None:
    M = _sym(5)
    Mn = row_normalize(M)
    U = [0, 1, 2, 3, 4]
    # future is a non-negative top-k mean
    assert future_value(0, U, Mn, topk=3) >= 0.0
    # regret is clipped at zero
    assert regret(0, 1, U, Mn) >= 0.0
    # cache_fit saturates at 1 when query fits
    assert cache_feature(0, 1000, [100, 100, 100, 100, 100], "fit") == 1.0
    # cache pressure scales with size
    assert cache_feature(0, 100, [300, 100, 100, 100, 100], "pressure") == 3.0


def test_regret_zero_when_i_is_best_predecessor() -> None:
    # Build immediate where i=2 is the strongest predecessor of j=0.
    n = 4
    imm = [[0.0] * n for _ in range(n)]
    imm[2][0] = 0.9
    imm[1][0] = 0.4
    imm[3][0] = 0.1
    assert regret(2, 0, [0, 1, 3], imm) == 0.0       # i=2 already best
    assert regret(1, 0, [0, 2, 3], imm) > 0.0        # i=1, 2 is better -> regret


def test_beam_width_one_equals_greedy() -> None:
    from src.bayesopt.step_scorer import beam_search_schedule
    M = _sym(7)
    Mn = row_normalize(M)
    pc = [100, 200, 150, 120, 90, 110, 130]
    score = make_step_scorer(Mn, ScorerWeights(), pc, 300)
    g = greedy_schedule(score, 7, pc)
    b = beam_search_schedule(score, 7, pc, beam_width=1)
    assert b == [g]


def test_connector_is_degree_count_not_sum() -> None:
    from src.bayesopt.step_scorer import connector_out
    # j=0 connects strongest to one (0.9); relative threshold rho*max.
    n = 4
    X = [[0.0] * n for _ in range(n)]
    X[0][1] = 0.9
    X[0][2] = 0.1
    X[0][3] = 0.2
    # rho=0.5 -> thr=0.45 -> only 0.9 qualifies -> 1/3
    assert abs(connector_out(0, [0, 1, 2, 3], X, rho=0.5) - 1 / 3) < 1e-9
    # rho=0.1 -> thr=0.09 -> all three qualify -> 1.0
    assert abs(connector_out(0, [0, 1, 2, 3], X, rho=0.1) - 1.0) < 1e-9


def test_connector_out_in_equal_on_raw_symmetric_diverge_on_asymmetric() -> None:
    from src.bayesopt.step_scorer import connector_in, connector_out
    M = _sym(6)  # RAW symmetric matrix (relative threshold -> out == in)
    U = list(range(6))
    for j in range(6):
        assert abs(connector_out(j, U, M, 0.25) - connector_in(j, U, M, 0.25)) < 1e-9
    # asymmetric matrix: out (what j feeds) and in (what feeds j) differ
    A = [[0.0] * 4 for _ in range(4)]
    A[0][1] = 0.9   # 0 feeds 1
    A[0][3] = 0.8   # 0 feeds 3  -> out(0) counts {1,3}
    A[2][0] = 0.9   # 2 feeds 0  -> in(0) counts {2}
    assert connector_out(0, [0, 1, 2, 3], A, 0.25) > connector_in(0, [0, 1, 2, 3], A, 0.25)


def test_net_reuse_gap_reads_both_matrices() -> None:
    from src.bayesopt.step_scorer import net_reuse_gap
    n = 3
    D = [[0.0, 0.3, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    M = [[0.0, 0.5, 0.4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    # sum_k D[0][k] - (M[0][k]-D[0][k]) = (0.3-0.2)+(0.2-0.2) = 0.1
    assert abs(net_reuse_gap(0, [0, 1, 2], D, M) - 0.1) < 1e-9


def test_multistart_runs_from_distinct_starts() -> None:
    from src.bayesopt.step_scorer import (
        good_start_set, multistart_greedy_schedules)
    M = _sym(7)
    Mn = row_normalize(M)
    pc = [100, 250, 150, 120, 90, 110, 300]
    starts = good_start_set(Mn, pc, k=4)
    assert len(starts) == 4
    assert len(set(starts)) == 4               # distinct
    assert pc.index(max(pc)) in starts          # largest pagecount included
    score = make_step_scorer(Mn, ScorerWeights(), pc, 300)
    scheds = multistart_greedy_schedules(score, 7, pc, starts)
    assert len(scheds) == 4
    for s in scheds:
        assert sorted(s) == list(range(7))
    assert {s[0] for s in scheds} == set(starts)  # each starts where told
