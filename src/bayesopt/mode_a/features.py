"""
Feature construction for Mode A (BO-tuned structured scorer).

Given a workload's page sets and a cache capacity, this module builds
the symmetric overlap matrix ``M``, the directional matrix ``D``, and a
table of **normalized** edge- and node-level features that the M / D /
Hybrid scorers consume.  All scorer inputs are normalized to stable
ranges (``[0, 1]`` or ``[-1, 1]``) with epsilon guards so that the
lambda ranges in the search space mean the same thing across workloads
and cache sizes; raw matrix values are retained only for logging.

Mathematical facts the construction relies on (asserted, not assumed):

* ``D[i][j] = |R(Qᵢ; C) ∩ P(Qⱼ)|`` with ``R ⊆ P(Qᵢ)``, hence
  ``D[i][j] ≤ M[i][j]`` for every pair.  A violation means the matrix
  construction is buggy; :func:`build_features` counts violations and
  callers treat any non-zero count as a hard failure.
* Consequently ``gap_raw = M − D ≥ 0`` (a one-sided feature) and
  ``survivor_ratio = D / (M + ε) ∈ [0, 1]``, meaningful only on edges
  where ``M > 0`` (non-overlapping pairs give a trivial 0).

Top-K aggregations exclude the diagonal (``j → j``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.simulator.cache_simulator import (
    compute_directional_matrix,
    compute_overlap_matrix,
)

EPSILON = 1e-9
"""Epsilon guard for all divisions; small enough not to perturb ratios."""

DEFAULT_TOPK_VALUES = (3, 5, 10)
"""Top-K values supported for the out-potential features."""


def _safe_div(num: float, den: float) -> float:
    """Division guarded against a zero (or near-zero) denominator."""
    return num / (den + EPSILON)


def _normalize_unit(values: list[float]) -> list[float]:
    """Scale a non-negative vector to [0, 1] by its max (0-vector → 0s)."""
    m = max(values) if values else 0.0
    if m <= EPSILON:
        return [0.0 for _ in values]
    return [v / m for v in values]


def _normalize_signed(values: list[float]) -> list[float]:
    """Scale a signed vector to [-1, 1] by its max abs (0-vector → 0s)."""
    m = max((abs(v) for v in values), default=0.0)
    if m <= EPSILON:
        return [0.0 for _ in values]
    return [v / m for v in values]


def _topk_row_mean(row: list[int], self_index: int, k: int) -> float:
    """Mean of the K largest off-diagonal entries of *row* (K capped at n-1)."""
    vals = [row[j] for j in range(len(row)) if j != self_index]
    if not vals:
        return 0.0
    k_eff = min(k, len(vals))
    vals.sort(reverse=True)
    return sum(vals[:k_eff]) / k_eff


@dataclass(frozen=True)
class FeatureTable:
    """
    Precomputed Mode A features for one (workload, cache) pair.

    Edge features are ``n × n`` lists; node features are length-``n``
    lists.  Normalized features feed the scorers; ``*_raw`` fields are
    for logging only.  Top-K node features are keyed by the K value so a
    config that searches over ``topK`` can select the matching table.

    Attributes
    ----------
    n : int
        Number of queries.
    cache_pages : int
        Cache capacity used to build ``D``.
    M, D : list[list[int]]
        Raw symmetric and directional matrices (logged, not scored).
    pagecount : list[int]
        Per-query page counts.
    M_norm, D_norm : list[list[float]]
        Row-relative normalized matrices in ``[0, 1]``; within a fixed
        row ``i`` these are monotone in the raw row, so
        ``argmaxⱼ D_norm[i][j] = argmaxⱼ D[i][j]`` (the Greedy-D rule).
    asym_norm : list[list[float]]
        ``D[i][j] − D[j][i]`` normalized to ``[-1, 1]`` by max abs.
    gap_raw : list[list[int]]
        ``M[i][j] − D[i][j]`` (≥ 0); logged raw.
    gap_norm : list[list[float]]
        ``gap_raw`` normalized to ``[0, 1]`` by global max gap.
    survivor_ratio : list[list[float]]
        ``D / (M + ε)`` clipped to ``[0, 1]``; ~0 off the overlap edges.
    size_norm : list[float]
        ``pagecount[j] / max pagecount`` in ``[0, 1]``.
    cache_pressure : list[float]
        ``pagecount[j] / C`` clipped to ``[0, 1]``.
    in_D, out_D : list[float]
        Normalized incoming / outgoing directional strength (column /
        row sums excluding self), each in ``[0, 1]``.
    balance_D : list[float]
        ``out_D_raw − in_D_raw`` normalized to ``[-1, 1]``.
    out_D_topk, out_M_topk : dict[int, list[float]]
        Per-K top-K outgoing means, normalized to ``[0, 1]``.
    dle_violations : int
        Count of pairs violating ``D ≤ M`` (must be 0).
    """

    n: int
    cache_pages: int
    M: list[list[int]]
    D: list[list[int]]
    pagecount: list[int]
    M_norm: list[list[float]]
    D_norm: list[list[float]]
    asym_norm: list[list[float]]
    gap_raw: list[list[int]]
    gap_norm: list[list[float]]
    survivor_ratio: list[list[float]]
    size_norm: list[float]
    cache_pressure: list[float]
    in_D: list[float]
    out_D: list[float]
    balance_D: list[float]
    out_D_topk: dict[int, list[float]]
    out_M_topk: dict[int, list[float]]
    dle_violations: int = 0
    topk_values: tuple[int, ...] = field(default=DEFAULT_TOPK_VALUES)


def build_features(
    page_sets: list[frozenset[int]],
    cache_pages: int,
    topk_values: tuple[int, ...] = DEFAULT_TOPK_VALUES,
) -> FeatureTable:
    """
    Build the full Mode A feature table for one workload and cache size.

    Parameters
    ----------
    page_sets : list[frozenset[int]]
        Integer-encoded per-query page sets.
    cache_pages : int
        Clock-sweep cache capacity in pages.
    topk_values : tuple[int, ...]
        K values to precompute out-potential features for.

    Returns
    -------
    FeatureTable
        All matrices and normalized features; ``dle_violations`` records
        any ``D > M`` pairs (expected 0 on valid input).

    Raises
    ------
    ValueError
        If *page_sets* is empty or *cache_pages* is not positive.
    """
    n = len(page_sets)
    if n == 0:
        raise ValueError("page_sets must be non-empty")
    if cache_pages <= 0:
        raise ValueError("cache_pages must be positive")

    M = compute_overlap_matrix(page_sets)
    D = compute_directional_matrix(page_sets, cache_pages)
    pagecount = [len(ps) for ps in page_sets]

    # Sanity: D[i][j] <= M[i][j] everywhere (theorem; violations == bug).
    dle_violations = sum(
        1
        for i in range(n)
        for j in range(n)
        if i != j and D[i][j] > M[i][j]
    )

    # Row-relative normalized matrices (monotone per row ⇒ preserve argmax).
    D_norm: list[list[float]] = []
    M_norm: list[list[float]] = []
    for i in range(n):
        d_rowmax = max((D[i][j] for j in range(n) if j != i), default=0)
        m_rowmax = max((M[i][j] for j in range(n) if j != i), default=0)
        D_norm.append([_safe_div(D[i][j], d_rowmax) if j != i else 0.0
                       for j in range(n)])
        M_norm.append([_safe_div(M[i][j], m_rowmax) if j != i else 0.0
                       for j in range(n)])

    # Asymmetry D[i][j] - D[j][i], normalized to [-1, 1] by global max abs.
    asym_raw = [[D[i][j] - D[j][i] for j in range(n)] for i in range(n)]
    max_abs_asym = max(
        (abs(asym_raw[i][j]) for i in range(n) for j in range(n)),
        default=0,
    )
    asym_norm = [
        [_safe_div(asym_raw[i][j], max_abs_asym) if max_abs_asym > 0 else 0.0
         for j in range(n)]
        for i in range(n)
    ]

    # Gap M - D (>= 0), raw for logs, normalized for scoring.
    gap_raw = [[M[i][j] - D[i][j] for j in range(n)] for i in range(n)]
    max_gap = max(
        (gap_raw[i][j] for i in range(n) for j in range(n)), default=0
    )
    gap_norm = [
        [_safe_div(gap_raw[i][j], max_gap) if i != j else 0.0
         for j in range(n)]
        for i in range(n)
    ]

    # Survivor ratio D/(M+eps), clipped to [0, 1]; ~0 off overlap edges.
    survivor_ratio = [
        [min(1.0, _safe_div(D[i][j], M[i][j])) if i != j else 0.0
         for j in range(n)]
        for i in range(n)
    ]

    size_norm = _normalize_unit([float(c) for c in pagecount])
    cache_pressure = [min(1.0, c / cache_pages) for c in pagecount]

    # Directional strength: row sums (out) and column sums (in), no self.
    out_D_raw = [sum(D[j][k] for k in range(n) if k != j) for j in range(n)]
    in_D_raw = [sum(D[k][j] for k in range(n) if k != j) for j in range(n)]
    out_D = _normalize_unit([float(v) for v in out_D_raw])
    in_D = _normalize_unit([float(v) for v in in_D_raw])
    balance_D = _normalize_signed(
        [float(out_D_raw[j] - in_D_raw[j]) for j in range(n)]
    )

    out_D_topk: dict[int, list[float]] = {}
    out_M_topk: dict[int, list[float]] = {}
    for k in topk_values:
        out_D_topk[k] = _normalize_unit(
            [_topk_row_mean(D[j], j, k) for j in range(n)]
        )
        out_M_topk[k] = _normalize_unit(
            [_topk_row_mean(M[j], j, k) for j in range(n)]
        )

    return FeatureTable(
        n=n,
        cache_pages=cache_pages,
        M=M,
        D=D,
        pagecount=pagecount,
        M_norm=M_norm,
        D_norm=D_norm,
        asym_norm=asym_norm,
        gap_raw=gap_raw,
        gap_norm=gap_norm,
        survivor_ratio=survivor_ratio,
        size_norm=size_norm,
        cache_pressure=cache_pressure,
        in_D=in_D,
        out_D=out_D,
        balance_D=balance_D,
        out_D_topk=out_D_topk,
        out_M_topk=out_M_topk,
        dle_violations=dle_violations,
        topk_values=tuple(topk_values),
    )


__all__ = [
    "EPSILON",
    "DEFAULT_TOPK_VALUES",
    "FeatureTable",
    "build_features",
]
