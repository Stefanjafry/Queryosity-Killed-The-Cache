"""
Dynamic Scorer v2 for Mode A (Stage-2, additive, direct-eval only).

Extends the Stage-1 dynamic features with a **missed-opportunity**
penalty that directly targets the q3 failure: placing a consumer query
after a weak predecessor while a much better predecessor is still
unscheduled.  All features are recomputed exactly over the current
remaining set U at each greedy step (no future leakage).

Features (j = candidate, i = current tail, U = unscheduled set, C = cache):

* immediate handoff   D_norm[i][j]
* out_rem[j,U]        mean top-K D_norm[j][k], k ∈ U\\{j}        (future value)
* in_rem[j,U]         mean top-K D_norm[k][j], k ∈ U\\{j}        (consumer need)
* producer_gate[j,U]  out_rem / (out_rem + in_rem + ε)
* role_out[j,U]       out_rem · producer_gate                  (gated future)
* role_balance[j,U]   out_rem − in_rem
* cache_fit[j,C]      min(1, C / pagecount[j])                 (≤1)
* cache_pressure[j]   min(1, pagecount[j] / C)
* missed_opportunity[i,j,U]
      = max(0, best_in_remaining[j,U] − D_norm[i][j])
      where best_in_remaining[j,U] = max D_norm[k][j], k ∈ U\\{j}
      — penalizes stranding a consumer after a weak predecessor when a
      strictly better predecessor is still available (the q3 fix).

This module is separate from ``scorers.py`` (production) and
``scorer_variants.py`` (Stage-1); it does not modify either.  Evaluation
is deterministic multistart greedy, direct exact-sim judging, no BO and
no refinement — Stage 2 measures scorer *structure* only.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bayesopt.mode_a.consumers import Candidate, _start_set
from src.bayesopt.mode_a.features import EPSILON, FeatureTable
from src.bayesopt.mode_a.scorer_variants import _topk_mean_over_U, cache_fit


@dataclass(frozen=True)
class DynamicSpec:
    """
    A Dynamic Scorer v2 configuration.

    ``formula`` selects which terms are active; the lambdas weight them.
    alpha_D scales the immediate-D anchor.  Auxiliary terms not used by
    the chosen formula are ignored.

    Formulas:
      * ``D_only``                 alpha_D·D
      * ``D_balance``              + lambda_balance·role_balance
      * ``remaining_out_balance``  + lambda_future·out_rem + lambda_balance·role_balance
      * ``role_out_size``          + lambda_future·role_out − lambda_size·cache_pressure
      * ``dynamic_missed_opportunity``
                                   + lambda_future·role_out·cache_fit
                                   − lambda_wait·missed_opportunity
      * ``dynamic_full``           all of the above terms combined
    """

    formula: str
    alpha_D: float = 1.0
    lambda_future: float = 1.0
    lambda_balance: float = 1.0
    lambda_size: float = 1.0
    lambda_wait: float = 1.0
    topk: int = 5


DYNAMIC_FORMULAS = (
    "D_only",
    "D_balance",
    "remaining_out_balance",
    "role_out_size",
    "dynamic_missed_opportunity",
    "dynamic_full",
)


def _dyn_v2(
    feats: FeatureTable, i: int, j: int, remaining: list[int], k: int
) -> dict[str, float]:
    """All v2 dynamic features for transition i→j over remaining set U."""
    n = feats.n
    out_rem = _topk_mean_over_U(
        [feats.D_norm[j][u] if u != j else 0.0 for u in range(n)],
        j, remaining, k,
    )
    in_rem = _topk_mean_over_U(
        [feats.D_norm[u][j] if u != j else 0.0 for u in range(n)],
        j, remaining, k,
    )
    producer_gate = out_rem / (out_rem + in_rem + EPSILON)
    role_out = out_rem * producer_gate
    role_balance = out_rem - in_rem
    # best available predecessor for j among the remaining set (excl. j).
    best_in_remaining = max(
        (feats.D_norm[u][j] for u in remaining if u != j), default=0.0
    )
    missed = max(0.0, best_in_remaining - feats.D_norm[i][j])
    return {
        "out_rem": out_rem,
        "in_rem": in_rem,
        "producer_gate": producer_gate,
        "role_out": role_out,
        "role_balance": role_balance,
        "best_in_remaining": best_in_remaining,
        "missed_opportunity": missed,
        "cache_fit": cache_fit(feats, j),
        "cache_pressure": min(1.0, feats.pagecount[j] / feats.cache_pages),
    }


def dynamic_score(
    spec: DynamicSpec, feats: FeatureTable, i: int, j: int,
    remaining: list[int],
) -> tuple[float, dict[str, float]]:
    """
    Score transition i→j under a Dynamic Scorer v2 spec.

    Returns the scalar score and the dict of dynamic feature values (for
    edge logging).  All inputs are normalized features.
    """
    f = _dyn_v2(feats, i, j, remaining, spec.topk)
    base = spec.alpha_D * feats.D_norm[i][j]
    formula = spec.formula

    if formula == "D_only":
        score = base
    elif formula == "D_balance":
        score = base + spec.lambda_balance * f["role_balance"]
    elif formula == "remaining_out_balance":
        score = (base
                 + spec.lambda_future * f["out_rem"]
                 + spec.lambda_balance * f["role_balance"])
    elif formula == "role_out_size":
        score = (base
                 + spec.lambda_future * f["role_out"]
                 - spec.lambda_size * f["cache_pressure"])
    elif formula == "dynamic_missed_opportunity":
        score = (base
                 + spec.lambda_future * f["role_out"] * f["cache_fit"]
                 - spec.lambda_wait * f["missed_opportunity"])
    elif formula == "dynamic_full":
        score = (base
                 + spec.lambda_future * f["role_out"] * f["cache_fit"]
                 + spec.lambda_balance * f["role_balance"]
                 - spec.lambda_size * f["cache_pressure"]
                 - spec.lambda_wait * f["missed_opportunity"])
    else:
        raise ValueError(f"unknown dynamic formula {formula!r}")
    return score, f


def multistart_greedy_dynamic(
    spec: DynamicSpec, feats: FeatureTable, num_starts: int = 5
) -> list[Candidate]:
    """
    Deterministic multistart greedy under a Dynamic Scorer v2 spec.

    Mirrors the production start set (so pure-D contains Greedy-D) and
    recomputes all dynamic features over the shrinking remaining set.
    """
    from src.bayesopt.mode_a.scorers import ScorerConfig
    proxy = ScorerConfig(family="d", topk=spec.topk)
    starts = _start_set(proxy, feats, min(num_starts, feats.n))

    out: list[Candidate] = []
    n = feats.n
    for start in starts:
        schedule = [start]
        current = start
        remaining = [q for q in range(n) if q != start]
        for _ in range(n - 1):
            best_j = -1
            best_score = float("-inf")
            for j in remaining:
                s, _ = dynamic_score(spec, feats, current, j, remaining)
                if s > best_score:
                    best_score = s
                    best_j = j
            schedule.append(best_j)
            remaining.remove(best_j)
            current = best_j
        out.append(Candidate(tuple(schedule), source=f"start={start}"))
    return out


def default_dynamic_grid() -> list[DynamicSpec]:
    """
    A small hand-designed Stage-2 grid (no combinatorial explosion).

    Controlled settings from the Track-4 findings: alpha_D ∈ {1,2,3},
    a couple of lambda settings per formula, topk fixed at 5 except where
    a sweep is cheap.  The point is to test whether the v2 structures —
    especially missed_opportunity — beat the Stage-1 best.
    """
    grid: list[DynamicSpec] = []
    # controls
    grid.append(DynamicSpec(formula="D_only", alpha_D=1))
    for a in (1, 2, 3):
        grid.append(DynamicSpec(formula="D_balance", alpha_D=a,
                                lambda_balance=1.0))
    # remaining_out_balance (Stage-1 small-cache champion shape)
    for a in (2, 3):
        for lf in (0.5, 1.0):
            grid.append(DynamicSpec(formula="remaining_out_balance",
                                    alpha_D=a, lambda_future=lf,
                                    lambda_balance=1.0))
    # role_out_size (Stage-1 large-cache champion shape)
    for a in (2, 3):
        for ls in (0.5, 1.0):
            grid.append(DynamicSpec(formula="role_out_size", alpha_D=a,
                                    lambda_future=1.0, lambda_size=ls))
    # the new one: missed_opportunity
    for a in (1, 2, 3):
        for lw in (0.5, 1.0, 2.0):
            grid.append(DynamicSpec(formula="dynamic_missed_opportunity",
                                    alpha_D=a, lambda_future=1.0,
                                    lambda_wait=lw))
    # dynamic_full: a few balanced settings
    for a in (2, 3):
        for lw in (1.0, 2.0):
            grid.append(DynamicSpec(formula="dynamic_full", alpha_D=a,
                                    lambda_future=1.0, lambda_balance=0.5,
                                    lambda_size=0.5, lambda_wait=lw))
    return grid


def spec_name(spec: DynamicSpec) -> str:
    """Compact identifier for a spec (table key / log id)."""
    parts = [spec.formula, f"aD{spec.alpha_D:g}"]
    if spec.formula not in ("D_only",):
        if "future" in spec.formula or "out" in spec.formula \
                or spec.formula in ("role_out_size",
                                    "dynamic_missed_opportunity",
                                    "dynamic_full"):
            parts.append(f"lf{spec.lambda_future:g}")
    if spec.formula in ("D_balance", "remaining_out_balance", "dynamic_full"):
        parts.append(f"lb{spec.lambda_balance:g}")
    if spec.formula in ("role_out_size", "dynamic_full"):
        parts.append(f"ls{spec.lambda_size:g}")
    if spec.formula in ("dynamic_missed_opportunity", "dynamic_full"):
        parts.append(f"lw{spec.lambda_wait:g}")
    return "_".join(parts)


__all__ = [
    "DynamicSpec",
    "DYNAMIC_FORMULAS",
    "dynamic_score",
    "multistart_greedy_dynamic",
    "default_dynamic_grid",
    "spec_name",
]
