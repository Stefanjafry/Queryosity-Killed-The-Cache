"""
Search space and proposal mechanisms for the Mode A scorer search.

This module owns the structured weight space and three deterministic
ways to propose configs:

* **warm-up** — meaningful hand-set configs per scorer family, always
  including the pure-D / pure-M anchor so the incumbent baseline is in
  the search from trial 1;
* **random** — uniform samples of the lambda weights (and topk /
  beam_width) within range, the honesty floor that tests whether
  refinement beats guessing;
* **neighbor refinement** — small ±steps around an incumbent in weight
  space, with categorical neighbours for topk and beam_width.

Nothing here evaluates schedules; the runner does that.  All randomness
flows through an explicit ``random.Random`` so a seed fully determines
the proposal stream.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from src.bayesopt.mode_a.scorers import ScorerConfig

# Lambda ranges (inclusive).  lambda_gap is signed; the rest are >= 0.
LAMBDA_RANGES: dict[str, tuple[float, float]] = {
    "lambda_out": (0.0, 2.0),
    "lambda_size": (0.0, 2.0),
    "lambda_balance": (0.0, 2.0),
    "lambda_asym": (0.0, 2.0),
    "lambda_ratio": (0.0, 2.0),
    "lambda_gap": (-2.0, 2.0),
}

# Which lambdas each family actually uses (others stay 0 and unsearched).
FAMILY_LAMBDAS: dict[str, tuple[str, ...]] = {
    "m": ("lambda_out", "lambda_size"),
    "d": ("lambda_out", "lambda_size", "lambda_balance", "lambda_asym"),
    "hybrid": ("lambda_gap", "lambda_ratio", "lambda_out", "lambda_size"),
}

NEIGHBOR_STEPS = (-0.5, -0.25, 0.25, 0.5)
"""Additive perturbations tried per continuous lambda during refinement."""


@dataclass(frozen=True)
class Proposal:
    """A scorer config plus its provenance (warmup / random / neighbor)."""

    config: ScorerConfig
    source: str


def _clip(name: str, value: float) -> float:
    lo, hi = LAMBDA_RANGES[name]
    return max(lo, min(hi, value))


def warmup_configs(
    family: str, topk_values: tuple[int, ...]
) -> list[Proposal]:
    """
    Deterministic warm-up configs for *family* at the first topk value.

    The first entry is always the pure anchor (all lambdas 0): pure-D
    for ``d``/``hybrid`` (the Greedy-D scorer) and pure-M for ``m``.

    Parameters
    ----------
    family : str
        Scorer family.
    topk_values : tuple[int, ...]
        Available K values; warm-ups use ``topk_values[0]``.

    Returns
    -------
    list[Proposal]
        Hand-set configs covering the family's main terms.
    """
    k = topk_values[0]
    base = ScorerConfig(family=family, topk=k)
    out: list[Proposal] = [Proposal(base, "warmup:pure")]

    if family == "m":
        out += [
            Proposal(replace(base, lambda_out=1.0), "warmup:M+out"),
            Proposal(replace(base, lambda_size=1.0), "warmup:M+size"),
            Proposal(replace(base, lambda_out=1.0, lambda_size=1.0),
                     "warmup:M+out+size"),
        ]
    elif family == "d":
        out += [
            Proposal(replace(base, lambda_out=1.0), "warmup:D+out"),
            Proposal(replace(base, lambda_balance=1.0), "warmup:D+balance"),
            Proposal(replace(base, lambda_asym=1.0), "warmup:D+asym"),
            Proposal(replace(base, lambda_size=1.0), "warmup:D+size"),
            Proposal(replace(base, lambda_out=1.0, lambda_size=1.0),
                     "warmup:D+out+size"),
        ]
    elif family == "hybrid":
        out += [
            Proposal(replace(base, lambda_ratio=1.0),
                     "warmup:D+survivor_ratio"),
            Proposal(replace(base, lambda_gap=1.0), "warmup:D+gap"),
            Proposal(replace(base, lambda_ratio=1.0, lambda_size=1.0),
                     "warmup:D+ratio+size"),
            Proposal(replace(base, lambda_gap=1.0, lambda_ratio=1.0),
                     "warmup:D+gap+ratio"),
        ]
    else:
        raise ValueError(f"unknown family {family!r}")
    return out


def random_config(
    family: str,
    topk_values: tuple[int, ...],
    rng: random.Random,
) -> Proposal:
    """
    Sample one random config: family lambdas uniform in range, random K.

    Lambdas the family does not use stay at 0, so the random baseline
    explores exactly the family's own subspace.
    """
    used = FAMILY_LAMBDAS[family]
    kwargs: dict[str, float] = {}
    for name in used:
        lo, hi = LAMBDA_RANGES[name]
        kwargs[name] = rng.uniform(lo, hi)
    cfg = ScorerConfig(
        family=family, topk=rng.choice(topk_values), **kwargs
    )
    return Proposal(cfg, "random")


def neighbor_configs(
    incumbent: ScorerConfig,
    topk_values: tuple[int, ...],
) -> list[Proposal]:
    """
    Deterministic neighbours of *incumbent* in weight space.

    For each lambda the family uses, try ±steps (clipped to range); for
    ``topk`` try the adjacent allowed values.  Schedules are never
    perturbed — this is weight-space refinement only.

    Returns
    -------
    list[Proposal]
        Unique neighbour configs (the incumbent itself is excluded).
    """
    used = FAMILY_LAMBDAS[incumbent.family]
    seen: set[tuple] = set()
    out: list[Proposal] = []

    def emit(cfg: ScorerConfig, src: str) -> None:
        key = (
            cfg.family, cfg.topk, cfg.lambda_out, cfg.lambda_size,
            cfg.lambda_balance, cfg.lambda_asym, cfg.lambda_ratio,
            cfg.lambda_gap,
        )
        if key in seen:
            return
        seen.add(key)
        out.append(Proposal(cfg, src))

    emit(incumbent, "_self")  # reserve so we never re-emit the incumbent
    for name in used:
        current = getattr(incumbent, name)
        for step in NEIGHBOR_STEPS:
            nxt = _clip(name, current + step)
            if nxt != current:
                emit(replace(incumbent, **{name: nxt}),
                     f"neighbor:{name}{step:+g}")
    # Categorical neighbours for topk.
    idx = topk_values.index(incumbent.topk) if incumbent.topk in topk_values \
        else 0
    for di in (-1, 1):
        ni = idx + di
        if 0 <= ni < len(topk_values):
            emit(replace(incumbent, topk=topk_values[ni]),
                 f"neighbor:topk={topk_values[ni]}")

    return [p for p in out if p.source != "_self"]


__all__ = [
    "LAMBDA_RANGES",
    "FAMILY_LAMBDAS",
    "NEIGHBOR_STEPS",
    "Proposal",
    "warmup_configs",
    "random_config",
    "neighbor_configs",
]
