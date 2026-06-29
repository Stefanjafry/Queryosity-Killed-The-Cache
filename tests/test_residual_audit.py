"""Tests for the Phase-1 residual decomposition audit math.

Exercises ``audit_matrices`` on hand-built matrices with known ``alpha``
and ``E`` so the three decision regimes are pinned:

* pure collapse  (D = alpha_i * M)        -> R² ~ 1, E ~ 0, do NOT continue
* concentrated residual on overflow row   -> high E, continue
* injected D > M                          -> violation counted
"""

from __future__ import annotations

import numpy as np

from src.bayesopt.run_residual_audit import audit_matrices


def _sym_overlap(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.integers(10, 100, (n, n))
    m = m + m.T
    np.fill_diagonal(m, 0)
    return m


def test_pure_collapse_recovers_alpha_and_declines_phase2() -> None:
    n = 6
    M = _sym_overlap(n, 0)
    alpha = np.array([1.0, 0.8, 0.6, 0.4, 0.9, 0.5])
    D = np.round(M * alpha[:, None]).astype(int)
    np.fill_diagonal(D, 0)

    res = audit_matrices(
        M.tolist(), D.tolist(), [50] * n, [50] * n,
        [f"q{i}" for i in range(n)], cache_pages=9999,
        total_unique_pages=1000,
    )

    recovered = np.array([r.alpha_ls for r in res.rows])
    assert np.allclose(recovered, alpha, atol=0.02)
    assert res.overall_r2 > 0.99
    assert res.e_frac_of_d < 0.02
    assert res.d_gt_m_violations == 0
    assert not res.nonfinite
    assert res.verdict["continue_to_phase2"] is False


def test_concentrated_residual_on_overflow_row_triggers_phase2() -> None:
    n = 6
    M = _sym_overlap(n, 1)
    alpha = np.array([1.0, 0.8, 0.6, 0.4, 0.9, 0.5])
    D = np.round(M * alpha[:, None]).astype(int)
    # Row 2 overflows and its survival is successor-dependent: full overlap
    # available to a couple of successors, almost nothing to the rest.
    for j in range(n):
        if j == 2:
            continue
        D[2][j] = M[2][j] if j in (1, 4) else int(round(0.15 * M[2][j]))
    np.fill_diagonal(D, 0)
    D = np.minimum(D, M)

    pagecounts = [50, 50, 800, 50, 50, 50]  # only row 2 overflows C=100
    res = audit_matrices(
        M.tolist(), D.tolist(), pagecounts, [50, 50, 90, 50, 50, 50],
        [f"q{i}" for i in range(n)], cache_pages=100,
        total_unique_pages=1000,
        continue_e_frac=0.10, overflow_concentration=0.50,
    )

    assert res.rows[2].e_row_frac > 0.2
    assert max(res.rows[i].e_row_frac for i in range(n) if i != 2) < 0.05
    assert res.e_frac_of_d >= 0.10
    assert res.overflow_energy_frac >= 0.50
    assert res.d_gt_m_violations == 0
    assert res.verdict["continue_to_phase2"] is True
    # top positive residual edge should be one of the surviving successors
    i, j, _ = res.top_pos_edges[0]
    assert i == 2 and j in (1, 4)


def test_d_greater_than_m_is_flagged() -> None:
    n = 5
    M = _sym_overlap(n, 2)
    D = np.round(M * 0.5).astype(int)
    np.fill_diagonal(D, 0)
    D[0][1] = M[0][1] + 5  # impossible: D <= M must hold

    res = audit_matrices(
        M.tolist(), D.tolist(), [50] * n, [50] * n,
        [f"q{i}" for i in range(n)], cache_pages=9999,
        total_unique_pages=1000,
    )
    assert res.d_gt_m_violations >= 1
    assert res.verdict["continue_to_phase2"] is False
