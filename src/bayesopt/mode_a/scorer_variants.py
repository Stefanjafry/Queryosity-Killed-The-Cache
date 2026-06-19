"""
Stage-2 scorer-repair variants for Mode A Track 4 (additive).

Forensic-guided scorer structures that test how to balance *immediate*
directional reuse (``D_norm[i][j]``) against *future* reuse, especially
under small cache where future-potential terms misplace large consumer
queries (the q3 failure: a 786K-page query placed after a zero-overlap
predecessor, losing 102K hits).

This module is separate from the locked production ``scorers.py`` and
does not touch it.  It evaluates a controlled variant grid by **direct
deterministic multistart** — no optimizer, no neighbour refinement, no
BoTorch — so the ablation measures scorer *structure*, not search.

Dynamic features (recomputed exactly at each greedy step over the
current unscheduled set ``U``, never the eventual schedule — no
leakage):

* ``out_rem[j, U]``  = mean top-K ``D_norm[j][k]`` for ``k ∈ U \\ {j}``
* ``in_rem[j, U]``   = mean top-K ``D_norm[k][j]`` for ``k ∈ U \\ {j}``
* ``producer_gate[j]`` = out_rem / (out_rem + in_rem + ε)
* ``role_out[j]``    = out_rem · producer_gate          (producer reward)
* ``role_balance[j]``= out_rem − in_rem
* ``cache_fit[j]``   = min(1, C / pagecount[j])         (static, ≤1)

``D_norm[j][k]`` is row-normalized by j's global row max — a fixed
matrix property, so restricting the average to U introduces no leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bayesopt.mode_a.consumers import Candidate, _start_set
from src.bayesopt.mode_a.features import EPSILON, FeatureTable

# alpha_D anchor sweep values (discrete, per spec).
ALPHA_D_VALUES = (1, 2, 3, 5)


@dataclass(frozen=True)
class VariantSpec:
    """
    One scorer-repair variant.

    Attributes
    ----------
    name : str
        Human-readable variant id (column key in the output table).
    alpha_D : int
        Coefficient on the immediate ``D_norm`` anchor.
    topk : int
        K for the dynamic top-K means and any static out-potential.
    use_static_out : bool
        Add static ``out_D_topK[j]`` (precomputed, schedule-independent).
    use_remaining_out : bool
        Add dynamic ``out_rem[j, U]``.
    use_role_out : bool
        Add dynamic ``role_out[j, U]`` (producer-gated future reward).
    use_balance : bool
        Add static ``balance_D[j]``.
    use_size : bool
        Subtract ``size_norm[j]`` (size penalty).
    use_cache_pressure : bool
        Subtract ``cache_pressure[j]`` (alternative size penalty).
    cache_fit_on_future : bool
        Multiply the future-reuse term (role_out or remaining_out) by
        ``cache_fit[j]`` — damps future reward for huge high-pressure
        queries under small cache.
    """

    name: str
    alpha_D: int = 1
    topk: int = 5
    use_static_out: bool = False
    use_remaining_out: bool = False
    use_role_out: bool = False
    use_balance: bool = False
    use_size: bool = False
    use_cache_pressure: bool = False
    cache_fit_on_future: bool = False


def cache_fit(feats: FeatureTable, j: int) -> float:
    """``min(1, C / pagecount[j])`` — ≤1, small for cache-exceeding queries."""
    pc = feats.pagecount[j]
    if pc <= 0:
        return 1.0
    return min(1.0, feats.cache_pages / pc)


def _topk_mean_over_U(
    values: list[float], j: int, remaining: list[int], k: int
) -> float:
    """Mean of the top-K of ``values[u]`` for ``u`` in *remaining* minus j."""
    pool = [values[u] for u in remaining if u != j]
    if not pool:
        return 0.0
    pool.sort(reverse=True)
    k_eff = min(k, len(pool))
    return sum(pool[:k_eff]) / k_eff


def _dyn_features(
    feats: FeatureTable, j: int, remaining: list[int], k: int
) -> tuple[float, float, float, float, float]:
    """
    Exact dynamic features for candidate j over the remaining set U.

    Returns
    -------
    (out_rem, in_rem, producer_gate, role_out, role_balance)
    """
    # out_rem: j's outgoing D_norm to remaining queries (row j).
    out_rem = _topk_mean_over_U(
        [feats.D_norm[j][u] if u != j else 0.0 for u in range(feats.n)],
        j, remaining, k,
    )
    # in_rem: incoming D_norm from remaining queries (column j).
    in_rem = _topk_mean_over_U(
        [feats.D_norm[u][j] if u != j else 0.0 for u in range(feats.n)],
        j, remaining, k,
    )
    producer_gate = out_rem / (out_rem + in_rem + EPSILON)
    role_out = out_rem * producer_gate
    role_balance = out_rem - in_rem
    return out_rem, in_rem, producer_gate, role_out, role_balance


def variant_score(
    spec: VariantSpec,
    feats: FeatureTable,
    i: int,
    j: int,
    remaining: list[int],
) -> tuple[float, dict[str, float]]:
    """
    Score transition ``i -> j`` under *spec*, given remaining set U.

    Returns the scalar score and a dict of the dynamic feature values
    (for edge logging).  All terms use normalized features; the future
    term is the producer-gated ``role_out`` if enabled else dynamic
    ``out_rem``, optionally multiplied by ``cache_fit[j]``.
    """
    k = spec.topk
    out_rem, in_rem, pgate, role_out, role_balance = _dyn_features(
        feats, j, remaining, k
    )
    cf = cache_fit(feats, j)

    score = spec.alpha_D * feats.D_norm[i][j]

    if spec.use_static_out:
        score += feats.out_D_topk[k][j]

    future = 0.0
    if spec.use_role_out:
        future = role_out
    elif spec.use_remaining_out:
        future = out_rem
    if spec.use_role_out or spec.use_remaining_out:
        if spec.cache_fit_on_future:
            future *= cf
        score += future

    if spec.use_balance:
        score += feats.balance_D[j]
    if spec.use_size:
        score -= feats.size_norm[j]
    if spec.use_cache_pressure:
        score -= feats.cache_pressure[j]

    diag = {
        "out_rem": round(out_rem, 6),
        "in_rem": round(in_rem, 6),
        "producer_gate": round(pgate, 6),
        "role_out": round(role_out, 6),
        "role_balance": round(role_balance, 6),
        "cache_fit": round(cf, 6),
        "alpha_D": float(spec.alpha_D),
    }
    return score, diag


def multistart_greedy_variant(
    spec: VariantSpec, feats: FeatureTable, num_starts: int = 5
) -> list[Candidate]:
    """
    Deterministic multistart greedy under a variant scorer.

    Mirrors the production multistart's start set (so pure-D contains
    Greedy-D) but builds each schedule with :func:`variant_score`,
    recomputing dynamic features over the shrinking remaining set.
    """
    # Reuse the production start-set rule via a pure-D proxy config so the
    # Greedy-D seed (largest pagecount) is always included.
    from src.bayesopt.mode_a.scorers import ScorerConfig
    proxy = ScorerConfig(family="d", topk=spec.topk)
    starts = _start_set(proxy, feats, min(num_starts, feats.n))

    out: list[Candidate] = []
    n = feats.n
    for start in starts:
        visited = [False] * n
        visited[start] = True
        schedule = [start]
        current = start
        remaining = [q for q in range(n) if q != start]
        for _ in range(n - 1):
            best_j = -1
            best_score = float("-inf")
            for j in remaining:
                s, _ = variant_score(spec, feats, current, j, remaining)
                if s > best_score:
                    best_score = s
                    best_j = j
            schedule.append(best_j)
            visited[best_j] = True
            remaining.remove(best_j)
            current = best_j
        out.append(Candidate(tuple(schedule), source=f"start={start}"))
    return out


def default_variant_grid() -> list[VariantSpec]:
    """
    The controlled Stage-2 variant grid (no combinatorial explosion).

    Auxiliary weights are fixed at 1 (structure test, not lambda tuning);
    alpha_D variants are expanded over ``ALPHA_D_VALUES``.
    """
    grid: list[VariantSpec] = [
        VariantSpec(name="D_only"),
    ]
    # alpha_D * D only
    for a in ALPHA_D_VALUES:
        grid.append(VariantSpec(name=f"aD{a}_only", alpha_D=a))
    # single-feature additions on the plain anchor
    grid += [
        VariantSpec(name="D_static_out", use_static_out=True),
        VariantSpec(name="D_remaining_out", use_remaining_out=True),
        VariantSpec(name="D_role_out", use_role_out=True),
        VariantSpec(name="D_balance", use_balance=True),
        VariantSpec(name="D_size", use_size=True),
        VariantSpec(name="D_cachepressure", use_cache_pressure=True),
    ]
    # alpha_D-scaled combinations over the anchor sweep
    for a in ALPHA_D_VALUES:
        grid += [
            VariantSpec(name=f"aD{a}_remaining_out",
                        alpha_D=a, use_remaining_out=True),
            VariantSpec(name=f"aD{a}_role_out",
                        alpha_D=a, use_role_out=True),
            VariantSpec(name=f"aD{a}_remaining_out_balance",
                        alpha_D=a, use_remaining_out=True, use_balance=True),
            VariantSpec(name=f"aD{a}_role_out_size",
                        alpha_D=a, use_role_out=True, use_size=True),
            VariantSpec(name=f"aD{a}_role_out_cachefit_size",
                        alpha_D=a, use_role_out=True, use_size=True,
                        cache_fit_on_future=True),
        ]
    return grid


__all__ = [
    "ALPHA_D_VALUES",
    "VariantSpec",
    "variant_score",
    "multistart_greedy_variant",
    "cache_fit",
    "default_variant_grid",
]
