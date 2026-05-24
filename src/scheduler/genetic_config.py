"""
Configuration dataclass for the genetic algorithm scheduler.

This module defines GAConfig, which holds all tunable hyperparameters
for the GA: population size, generation count, crossover and mutation
rates, tournament size, elitism count, cache capacity, and random seed.

Classes
-------
GAConfig
    Hyperparameter configuration for the genetic algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    # Imported for type hints only.  The DQN class transitively requires
    # onnxruntime, which we do not want to force on users of the LRU
    # fitness path.  The string annotation below is resolved lazily.
    from src.simulator.dqn_simulator import DQN


FitnessType = Literal["lru", "dqn"]
"""
Types of supported fitness functions.

- ``"lru"``: simulation of a clock-sweep (LRU-family) cache; the metric
  is cache hit ratio.
- ``"dqn"``: Deep Q-Network surrogate; the metric is the sum of the
  Q-values output by the network.
"""


ApproxMode = Literal["symmetric", "directional"]
"""
Variant of approximate fitness used when ``use_approximate_fitness`` is True.

- ``"symmetric"``: precomputed pairwise overlap matrix ``M[i][j] =
  |P(Qᵢ) ∩ P(Qⱼ)|`` with the windowed triple-intersection discount.
- ``"directional"``: precomputed directional utility matrix
  ``D[i][j] = |R(Qᵢ; C) ∩ P(Qⱼ)|`` with edge-local fitness summation,
  capturing the asymmetry between large→small and small→large
  transitions under eviction pressure.
"""


@dataclass
class GAConfig:
    """
    Configuration for the genetic algorithm scheduler.

    Attributes
    ----------
    population_size : int
        Number of individuals per generation.
    num_generations : int
        Maximum number of generations to evolve.
    crossover_rate : float
        Probability of applying crossover to a pair of parents.
    mutation_rate : float
        Probability of applying swap mutation to an offspring.
    tournament_size : int
        Number of candidates drawn for tournament selection.
    elite_count : int
        Number of top individuals preserved unchanged across generations.
    cache_capacity_pages : int
        Clock-sweep cache capacity in 8 KB pages used for fitness evaluation.
    all_tables : list[str]
        Ordered list of table names referenced by any query in the
        workload.  Required for DQN state encoding.
    max_pages : dict[str, int]
        Maximum page count observed per table across the workload.  Used
        to normalise DQN state vectors.
    fitness_type : FitnessType
        Which fitness function the GA should use during evolution.
    dqn : DQN | None
        DQN surrogate evaluator.  Required when ``fitness_type == "dqn"``.
    seed : int | None
        Random seed for reproducibility.  None means non-deterministic.
    use_approximate_fitness : bool
        When True and page-level data is available (and
        ``fitness_type == "lru"``), use the precomputed overlap matrix
        for fast approximate fitness during evolution.  The final best
        schedule is always validated with the exact clock-sweep simulation.
    approx_mode : ApproxMode
        Variant of approximate fitness.  ``"symmetric"`` uses the
        precomputed page-overlap matrix; ``"directional"`` uses the
        residual-aware directional utility matrix.  Ignored when
        ``use_approximate_fitness`` is False.
    """

    population_size: int = 100
    num_generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.3
    tournament_size: int = 3
    elite_count: int = 2
    cache_capacity_pages: int = 1000
    all_tables: list[str] = field(default_factory=list)
    max_pages: dict[str, int] = field(default_factory=dict)
    fitness_type: FitnessType = "lru"
    dqn: "DQN | None" = None
    seed: int | None = None
    use_approximate_fitness: bool = False
    approx_mode: ApproxMode = "symmetric"
