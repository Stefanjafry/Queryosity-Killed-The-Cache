"""
BoTorch single-task GP backend for Mode B — the small-n arm.

Vanilla GP + LogEI is only sensible at TPC-H scale (n ≈ 22); at
n = 93/113 a plain GP is the wrong tool and a dedicated
high-dimensional arm (TuRBO vs SAASBO — decision pending) is used
instead.  The CLI warns when this backend is pointed at large n.

Loop: Sobol initial design → fit ``SingleTaskGP`` (standardized
outcomes) → maximize LogExpectedImprovement over ``[0, 1]^n`` → evaluate
through the shared recorder → repeat until the budget is consumed.
The GP maximizes ``−cost``; the recorder still logs cost.

A failed GP fit on a given iteration falls back to one Sobol point so
long runs survive occasional ill-conditioning; fallbacks are counted
and reported by the runner.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.bayesopt.search import TrialRecorder

logger = logging.getLogger(__name__)


def run(
    recorder: TrialRecorder,
    *,
    n: int,
    budget: int,
    seed: int,
    n_init: int,
    workdir: Path | None = None,
) -> None:
    """
    Run vanilla GP BO with LogEI over the priority hypercube.

    Parameters
    ----------
    recorder : TrialRecorder
        Shared trial recorder.
    n : int
        Number of queries.
    budget : int
        Total evaluation budget, Sobol initial design included.
    seed : int
        Seed for torch and the (scrambled) Sobol engine.
    n_init : int
        Sobol initial-design size.
    workdir : Path | None
        Unused; present for backend-interface uniformity.
    """
    # Imported lazily so smac/random runs never import torch.
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

    sobol = SobolEngine(dimension=n, scramble=True, seed=seed)
    n_init = min(n_init, budget)
    x_train = sobol.draw(n_init).to(dtype)
    y_vals = [-recorder(x.tolist()) for x in x_train]  # maximize −cost
    y_train = torch.tensor(y_vals, dtype=dtype).unsqueeze(-1)

    bounds = torch.stack(
        [torch.zeros(n, dtype=dtype), torch.ones(n, dtype=dtype)]
    )
    fit_failures = 0

    while recorder.n_trials < budget:
        try:
            model = SingleTaskGP(
                x_train, y_train, outcome_transform=Standardize(m=1)
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            acqf = LogExpectedImprovement(model, best_f=y_train.max())
            candidate, _ = optimize_acqf(
                acqf,
                bounds=bounds,
                q=1,
                num_restarts=8,
                raw_samples=128,
            )
            x_next = candidate.squeeze(0).clamp(0.0, 1.0)
        except Exception as exc:  # GP fit/acq failure: stay alive
            fit_failures += 1
            logger.warning("GP iteration failed (%s); Sobol fallback", exc)
            x_next = sobol.draw(1).to(dtype).squeeze(0)

        y_next = -recorder(x_next.tolist())
        x_train = torch.cat([x_train, x_next.unsqueeze(0)], dim=0)
        y_train = torch.cat(
            [y_train, torch.tensor([[y_next]], dtype=dtype)], dim=0
        )

    if fit_failures:
        logger.warning("botorch_gp finished with %d fit fallbacks", fit_failures)


__all__ = ["run"]
