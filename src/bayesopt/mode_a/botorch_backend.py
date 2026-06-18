"""
BoTorch GP backend for Mode A scorer-weight search (Track 2).

A real Bayesian-optimization loop over the **continuous lambda weights**
of one scorer family, stratified by the small discrete ``topk`` set.
The objective is the deterministic exact-simulator cost of the schedule
the consumer builds from a weight vector, so the GP models a noise-free
response surface and the recommendation is simply the best observed
point (no posterior-mean denoising needed).

Design choices, per the Track-2 spec:

* **topK by stratum.**  ``topk ∈ {3, 5, 10}`` is handled by fitting one
  GP per topk value over that stratum's observed points and maximizing
  the acquisition within each stratum, then taking the best acquisition
  candidate across strata.  This keeps the GP purely continuous and
  avoids encoding a 3-level categorical into the kernel.
* **Only the family's own lambdas are searched.**  Each family uses a
  subset of the lambda dimensions (see ``FAMILY_LAMBDAS``); the GP
  operates on the unit cube of just those dimensions, mapped to/from
  the real lambda ranges.
* **Warm-up anchors + Sobol initial design** seed the GP; acquisition
  (LogExpectedImprovement, minimizing cost ⇒ maximizing −cost) then
  drives suggestions.

This module imports torch/botorch lazily inside :func:`propose_botorch`
so the rest of Mode A never pays for (or requires) the BO stack.  The
caller supplies an ``evaluate`` callback that maps a ScorerConfig to a
realized cost and records the trial; this backend only chooses configs.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable

from src.bayesopt.mode_a.scorers import ScorerConfig
from src.bayesopt.mode_a.search_space import (
    FAMILY_LAMBDAS,
    LAMBDA_RANGES,
    warmup_configs,
)

# Evaluate callback: given a config and a provenance label, build+score
# its schedules, record the trial, and return the realized cost.
EvaluateFn = Callable[[ScorerConfig, str], float]


def _bounds_for(family: str) -> list[tuple[str, float, float]]:
    """Ordered (name, lo, hi) for the lambda dims this family searches."""
    return [(n, *LAMBDA_RANGES[n]) for n in FAMILY_LAMBDAS[family]]


def _config_from_unit(
    family: str, topk: int, unit: list[float],
    dims: list[tuple[str, float, float]],
) -> ScorerConfig:
    """Map a unit-cube point to a ScorerConfig in real lambda ranges."""
    kwargs: dict[str, float] = {}
    for (name, lo, hi), u in zip(dims, unit):
        kwargs[name] = lo + (hi - lo) * float(min(1.0, max(0.0, u)))
    return ScorerConfig(family=family, topk=topk, **kwargs)


def _config_to_unit(
    cfg: ScorerConfig, dims: list[tuple[str, float, float]]
) -> list[float]:
    """Map a ScorerConfig's searched lambdas back to the unit cube."""
    out = []
    for name, lo, hi in dims:
        v = getattr(cfg, name)
        out.append(0.0 if hi == lo else (v - lo) / (hi - lo))
    return out


def _as_floats(values) -> list[float]:
    """Coerce a sequence (incl. torch .tolist() output) to list[float].

    Isolates the torch-derived ``Unknown`` type so the rest of the
    module sees a concrete ``list[float]`` regardless of whether the
    type checker can resolve torch.
    """
    return [float(v) for v in values]


def propose_botorch(
    *,
    family: str,
    topk_values: tuple[int, ...],
    evaluate: EvaluateFn,
    n_init: int,
    n_bo: int,
    seed: int,
    should_stop: Callable[[], bool],
) -> dict[str, int]:
    """
    Run the warm-up → Sobol init → GP-acquisition loop for one family.

    Parameters
    ----------
    family : str
        Scorer family; only its lambda dims are searched.
    topk_values : tuple[int, ...]
        Discrete topk strata to optimize over.
    evaluate : EvaluateFn
        Callback ``(config, source) -> cost`` that records the trial and
        returns the realized (deterministic) cost.
    n_init : int
        Sobol initial-design points (in addition to warm-up anchors).
    n_bo : int
        Acquisition-driven suggestions to make after the initial design.
    seed : int
        Seed for torch and the Sobol engine.
    should_stop : Callable[[], bool]
        Returns True when an external budget (eval cap) is exhausted;
        checked before each evaluation so the loop never overruns.

    Returns
    -------
    dict[str, int]
        Counts: ``n_warmup``, ``n_init``, ``n_bo`` actually evaluated.

    Notes
    -----
    Imports torch/botorch lazily.  A GP fit/acquisition failure on any
    iteration falls back to a Sobol draw for that step so a long run
    survives occasional ill-conditioning.
    """
    import torch
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim.optimize import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from torch.quasirandom import SobolEngine

    torch.manual_seed(seed)
    dtype = torch.double
    dims = _bounds_for(family)
    d = len(dims)
    counts = {"n_warmup": 0, "n_init": 0, "n_bo": 0}

    # Per-stratum observation store: topk -> (list[unit-point], list[cost]).
    obs_x: dict[int, list[list[float]]] = {k: [] for k in topk_values}
    obs_y: dict[int, list[float]] = {k: [] for k in topk_values}

    def observe(cfg: ScorerConfig, source: str) -> None:
        cost = evaluate(cfg, source)
        obs_x[cfg.topk].append(_config_to_unit(cfg, dims))
        obs_y[cfg.topk].append(cost)

    # --- Warm-up anchors (known-good starting configs) ---
    for prop in warmup_configs(family, topk_values):
        if should_stop():
            return counts
        observe(prop.config, prop.source)
        counts["n_warmup"] += 1

    # --- Sobol initial design, spread across strata ---
    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    for i in range(n_init):
        if should_stop():
            return counts
        unit = _as_floats(sobol.draw(1).to(dtype).squeeze(0).tolist())
        topk = topk_values[i % len(topk_values)]
        cfg = _config_from_unit(family, topk, unit, dims)
        observe(cfg, "botorch_init")
        counts["n_init"] += 1

    bounds = torch.stack(
        [torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)]
    )

    # --- GP-acquisition loop ---
    for _ in range(n_bo):
        if should_stop():
            break
        best_cfg: ScorerConfig | None = None
        best_acq = -math.inf
        fallback_unit: list[float] | None = None

        for topk in topk_values:
            xs, ys = obs_x[topk], obs_y[topk]
            if len(xs) < 3:
                continue  # too few points in this stratum to fit a GP
            try:
                train_x = torch.tensor(xs, dtype=dtype)
                # Maximize −cost (BoTorch maximizes); Standardize outcomes.
                train_y = torch.tensor(
                    [[-c] for c in ys], dtype=dtype
                )
                model = SingleTaskGP(
                    train_x, train_y, outcome_transform=Standardize(m=1)
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
                acqf = LogExpectedImprovement(
                    model, best_f=train_y.max()
                )
                cand, acq_val = optimize_acqf(
                    acqf, bounds=bounds, q=1,
                    num_restarts=8, raw_samples=128,
                )
                if acq_val is None or cand is None:
                    continue
                v = float(acq_val)
                if v > best_acq:
                    best_acq = v
                    unit = _as_floats(cand.squeeze(0).clamp(0.0, 1.0).tolist())
                    best_cfg = _config_from_unit(family, topk, unit, dims)
            except Exception:
                # Ill-conditioned fit: remember a Sobol fallback point.
                fallback_unit = _as_floats(sobol.draw(1).to(dtype).squeeze(0).tolist())

        if best_cfg is None:
            # No stratum could fit a GP (or all failed): Sobol fallback.
            if fallback_unit is None:
                fallback_unit = _as_floats(sobol.draw(1).to(dtype).squeeze(0).tolist())
            topk = topk_values[counts["n_bo"] % len(topk_values)]
            best_cfg = _config_from_unit(family, topk, fallback_unit, dims)
            observe(best_cfg, "botorch_fallback")
        else:
            observe(best_cfg, "botorch_acq")
        counts["n_bo"] += 1

    return counts


__all__ = ["propose_botorch", "EvaluateFn"]
