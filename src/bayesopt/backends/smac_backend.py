"""
SMAC3 random-forest SMBO backend for Mode B.

Uses the HyperparameterOptimization facade — random-forest surrogate
with a Sobol initial design — which is the SMAC configuration suited to
higher-dimensional spaces (n = 22…113 priorities).  The target function
is the shared recorder: the exact clock-sweep cost, never the edge sum.

Determinism: the simulator objective is deterministic, so the scenario
is declared ``deterministic=True`` (one evaluation per configuration)
with fixed seeds throughout — deliberately *not* the noisy-objective
pattern from Rafael's BO work, which does not apply here.
"""

from __future__ import annotations

from pathlib import Path

from src.bayesopt.search import TrialRecorder


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
    Run SMAC3 RF-SMBO over the unit hypercube of priorities.

    Parameters
    ----------
    recorder : TrialRecorder
        Shared trial recorder (the SMAC target function).
    n : int
        Number of queries.
    budget : int
        Total SMAC trial budget, initial design included.
    seed : int
        SMAC scenario seed.
    n_init : int
        Sobol initial-design size.
    workdir : Path | None
        Directory for SMAC's run artifacts (runhistory etc.).
        Defaults to ``smac3_output`` under the current directory.
    """
    # Imported lazily so other backends never pay for (or require) smac.
    from ConfigSpace import ConfigurationSpace
    from smac import HyperparameterOptimizationFacade as HPOFacade
    from smac import Scenario

    names = [f"p{i:04d}" for i in range(n)]
    configspace = ConfigurationSpace({name: (0.0, 1.0) for name in names})

    output = Path(workdir) if workdir is not None else Path("smac3_output")
    scenario = Scenario(
        configspace,
        name=f"modeb_n{n}_b{budget}_s{seed}",
        output_directory=output,
        deterministic=True,
        n_trials=budget,
        seed=seed,
    )

    def target(config, seed: int = 0) -> float:
        return recorder([float(config[name]) for name in names])

    initial_design = HPOFacade.get_initial_design(scenario, n_configs=n_init)
    facade = HPOFacade(
        scenario,
        target,
        initial_design=initial_design,
        overwrite=True,
    )
    facade.optimize()


__all__ = ["run"]
