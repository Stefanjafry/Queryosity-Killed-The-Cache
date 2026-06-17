"""
Structured edge scorers for Mode A: M-only, D-only, and Hybrid.

Each scorer maps a transition ``i → j`` (current query *i*, candidate
next query *j*) to a deterministic scalar, using **normalized** feature
values so the lambda weights mean the same thing across workloads and
cache sizes.  The main signal (``M_norm`` for the M family, ``D_norm``
for D and Hybrid) is anchored at coefficient 1 and cannot be tuned
away, so the optimizer can reweight the auxiliary terms but never
remove the matrix it is built on.

Containment: with every lambda at 0, ``score_D`` reduces to
``D_norm[i][j]``, whose per-row argmax equals ``argmaxⱼ D[i][j]`` — the
Greedy-D rule.  The pure-D config is therefore Greedy-D at the edge
level (see the multistart containment test for the schedule-level
statement).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bayesopt.mode_a.features import FeatureTable

SCORER_FAMILIES = ("m", "d", "hybrid")
"""Implemented scorer families."""


@dataclass(frozen=True)
class ScorerConfig:
    """
    A scoring policy: family, top-K choice, and the lambda weights.

    Only the weights relevant to *family* are used; the others are
    ignored, so one config type covers all three families.  ``topk`` is
    the K whose out-potential table the scorer reads.

    Attributes
    ----------
    family : str
        One of :data:`SCORER_FAMILIES`.
    topk : int
        Top-K value selecting the out-potential feature table.
    lambda_out, lambda_size, lambda_balance, lambda_asym,
    lambda_ratio, lambda_gap : float
        Auxiliary-term weights (see the scorer formulas).
    """

    family: str
    topk: int = 5
    lambda_out: float = 0.0
    lambda_size: float = 0.0
    lambda_balance: float = 0.0
    lambda_asym: float = 0.0
    lambda_ratio: float = 0.0
    lambda_gap: float = 0.0


def score_edge(
    cfg: ScorerConfig, feats: FeatureTable, i: int, j: int
) -> float:
    """
    Deterministic score for transition ``i → j`` under *cfg*.

    Dispatches on ``cfg.family``.  All terms are normalized features;
    raw matrices are never used in scoring.

    M-only::

        M_norm[i][j] + λ_out·out_M_topK[j] − λ_size·size_norm[j]

    D-only::

        D_norm[i][j] + λ_out·out_D_topK[j] + λ_balance·balance_D[j]
                     + λ_asym·asym[i][j]   − λ_size·size_norm[j]

    Hybrid (M enters only as eviction-loss info, not as a raw blend)::

        D_norm[i][j] + λ_gap·gap_norm[i][j] + λ_ratio·survivor_ratio[i][j]
                     + λ_out·out_D_topK[j]  − λ_size·size_norm[j]
    """
    if cfg.family == "m":
        out = feats.out_M_topk[cfg.topk][j]
        return (
            feats.M_norm[i][j]
            + cfg.lambda_out * out
            - cfg.lambda_size * feats.size_norm[j]
        )
    if cfg.family == "d":
        out = feats.out_D_topk[cfg.topk][j]
        return (
            feats.D_norm[i][j]
            + cfg.lambda_out * out
            + cfg.lambda_balance * feats.balance_D[j]
            + cfg.lambda_asym * feats.asym_norm[i][j]
            - cfg.lambda_size * feats.size_norm[j]
        )
    if cfg.family == "hybrid":
        out = feats.out_D_topk[cfg.topk][j]
        return (
            feats.D_norm[i][j]
            + cfg.lambda_gap * feats.gap_norm[i][j]
            + cfg.lambda_ratio * feats.survivor_ratio[i][j]
            + cfg.lambda_out * out
            - cfg.lambda_size * feats.size_norm[j]
        )
    raise ValueError(f"unknown scorer family {cfg.family!r}")


__all__ = ["SCORER_FAMILIES", "ScorerConfig", "score_edge"]
