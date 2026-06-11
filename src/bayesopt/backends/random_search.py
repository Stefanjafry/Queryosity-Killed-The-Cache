"""
Equal-budget random priority search — the Mode B honesty floor.

Draws i.i.d. uniform priority vectors and evaluates each through the
shared recorder.  Every BO arm must beat this at the *same* evaluation
budget before any claim that its surrogate learned anything.
"""

from __future__ import annotations

import random
from pathlib import Path

from src.bayesopt.search import TrialRecorder


def run(
    recorder: TrialRecorder,
    *,
    n: int,
    budget: int,
    seed: int,
    n_init: int = 0,
    workdir: Path | None = None,
) -> None:
    """
    Evaluate *budget* uniform-random priority vectors.

    Parameters
    ----------
    recorder : TrialRecorder
        Shared trial recorder.
    n : int
        Number of queries (vector dimensionality).
    budget : int
        Number of vectors to draw and evaluate.
    seed : int
        Seed for the stdlib Mersenne Twister; runs are fully
        deterministic given (n, budget, seed).
    n_init : int
        Unused; random search has no surrogate phase.
    workdir : Path | None
        Unused.
    """
    rng = random.Random(seed)
    for _ in range(budget):
        recorder([rng.random() for _ in range(n)])


__all__ = ["run"]
