"""
Genetic algorithm primitives for query schedule evolution.

Provides the core building blocks used by the GA scheduler: fitness
evaluation, tournament selection, order crossover, swap mutation, and
the Individual dataclass hierarchy that encapsulates a schedule with
lazily cached fitness.

Classes
-------
FitnessFunction
    Protocol describing the callable interface every fitness evaluator
    must implement.  Both the clock-sweep simulator and the DQN surrogate
    conform to it, letting the GA swap evaluators via config.
Individual
    A candidate query execution order with lazy cached fitness evaluated
    via table-level clock-sweep simulation.
IndividualWithPageSet
    Extends Individual to evaluate fitness using page-level simulation
    from pg_buffercache data.
IndividualApproximate
    Extends Individual to evaluate fitness using the fast pairwise
    overlap-matrix approximation.  Used during evolution only; the final
    best schedule is always re-scored by the exact clock-sweep simulator.

Functions
---------
make_individual
    Factory that constructs the appropriate Individual subtype based on
    whether page sets are available and which fitness mode is configured.
select_parents
    Selects multiple parents from the population via tournament selection.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Optional, Protocol

from src.scheduler.genetic_config import GAConfig
from src.simulator.access_profile import AccessProfile
from src.simulator.cache_simulator import (
    approximate_schedule_fitness,
    approximate_schedule_fitness_directional,
    simulate_schedule,
    simulate_schedule_page_level,
)


class FitnessFunction(Protocol):
    """
    Protocol defining the interface for query-schedule fitness functions.

    Any callable conforming to this protocol can be used as a fitness
    function in the genetic algorithm, allowing the simulation-based and
    DQN-based evaluators to be used interchangeably.

    Methods
    -------
    __call__(individual, profiles, cache_capacity_pages, page_sets)
        Evaluate the fitness of a query schedule.
    """

    def __call__(
        self,
        individual: list[int],
        profiles: list[AccessProfile],
        cache_capacity_pages: int,
        page_sets: list[frozenset[int]] | None,
    ) -> float:
        """
        Evaluate the fitness of a query schedule.

        Parameters
        ----------
        individual : list[int]
            Permutation of query indices representing the schedule to
            evaluate.
        profiles : list[AccessProfile]
            Access profiles for all queries, indexed by query index.
        cache_capacity_pages : int
            Clock-sweep cache capacity in 8 KB pages.
        page_sets : list[frozenset[int]] or None
            Optional precomputed page sets for each query.  Implementations
            may use this for page-level granularity or ignore it.

        Returns
        -------
        float
            Fitness score of the schedule.  Higher is better.
        """
        ...
def _fitness_cache_simulation(
    individual: list[int],
    profiles: list[AccessProfile],
    cache_capacity_pages: int,
    page_sets: list[frozenset[int]] | None = None,
) -> float:
    """
    Evaluate the cache hit-ratio fitness of an individual.

    When page_sets are provided, uses page-level clock-sweep simulation.
    Otherwise falls back to table-level simulation.

    Parameters
    ----------
    individual : list[int]
        Permutation of query indices representing an execution order.
    profiles : list[AccessProfile]
        Access profiles for each query, indexed by query index.
    cache_capacity_pages : int
        Clock-sweep cache capacity in 8 KB pages.
    page_sets : list[frozenset[int]] | None
        Integer-encoded page sets from pg_buffercache profiling.

    Returns
    -------
    float
        Cache hit ratio in the range [0.0, 1.0].
    """
    if page_sets is not None:
        result = simulate_schedule_page_level(
            page_sets, individual, cache_capacity_pages,
        )
    else:
        result = simulate_schedule(
            profiles, individual, cache_capacity_pages
        )
    return result.hit_ratio


def _tournament_select(
    population: list[Individual],
    k: int,
    rng: random.Random,
) -> Individual:
    """
    Select one individual from the population via k-tournament selection.

    Randomly samples k individuals and returns a copy of the one with the
    highest fitness.

    Parameters
    ----------
    population : list[Individual]
        Current population of individuals.
    k : int
        Number of candidates to draw for the tournament.
    rng : random.Random
        Random number generator instance.

    Returns
    -------
    Individual
        The selected individual (not a copy).
    """
    candidates = rng.sample(range(len(population)), k)
    winner = max(candidates, key=lambda i: population[i].fitness())
    return population[winner]


def _order_crossover(
    parent1: list[int],
    parent2: list[int],
    rng: random.Random,
) -> list[int]:
    """
    Produce an offspring using Order Crossover (OX).

    A random substring is copied from parent1 into the child at the same
    positions.  The remaining positions are filled with elements from
    parent2, preserving their relative order.

    Parameters
    ----------
    parent1 : list[int]
        First parent permutation.
    parent2 : list[int]
        Second parent permutation.
    rng : random.Random
        Random number generator instance.

    Returns
    -------
    list[int]
        Offspring permutation.
    """
    n = len(parent1)
    start, end = sorted(rng.sample(range(n), 2))

    child = [-1] * n
    child[start : end + 1] = parent1[start : end + 1]

    inherited = set(child[start : end + 1])
    fill = [g for g in parent2 if g not in inherited]

    pos = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[pos]
            pos += 1

    return child


def _swap_mutation(individual: list[int], rng: random.Random) -> None:
    """
    Apply swap mutation to an individual in-place.

    Two randomly chosen positions are swapped.

    Parameters
    ----------
    individual : list[int]
        Permutation to mutate.
    rng : random.Random
        Random number generator instance.
    """
    n = len(individual)
    i, j = rng.sample(range(n), 2)
    individual[i], individual[j] = individual[j], individual[i]


@dataclass
class Individual:
    """
    A single individual in the GA population.

    Encapsulates a query execution order (schedule) along with the context
    needed to evaluate its fitness.  Fitness is computed lazily and cached
    to avoid redundant simulation calls.

    Attributes
    ----------
    schedule : list[int]
        Permutation of query indices representing an execution order.
    profiles : list[AccessProfile]
        Access profiles for each query, indexed by query index.
    cache_capacity_pages : int
        Clock-sweep cache capacity in 8 KB pages used for fitness evaluation.
    _rng : random.Random
        Random number generator instance used for mutation and crossover.
    _config : GAConfig
        Algorithm configuration, including mutation and crossover rates.
    _fitness_fn : FitnessFunction
        Callable used to score schedules.  Selected by ``make_individual``
        based on ``config.fitness_type``.
    _fitness : float or None
        Cached fitness value.  None until first call to fitness().
    """

    schedule: list[int]
    profiles: list[AccessProfile]
    cache_capacity_pages: int
    _rng: random.Random
    _config: GAConfig
    _fitness_fn: FitnessFunction
    _fitness: Optional[float] = field(init=False, default=None)

    def fitness(self) -> float:
        """
        Return the fitness of this individual.

        Evaluates and caches the fitness on the first call via
        ``self._fitness_fn``.  Subsequent calls return the cached value.

        Returns
        -------
        float
            Fitness score produced by the configured fitness function.
        """
        if self._fitness is None:
            self._fitness = self._fitness_fn(
                self.schedule,
                self.profiles,
                self.cache_capacity_pages,
                None,
            )
        return self._fitness

    def clone(self) -> Individual:
        """
        Produce a mutated copy of this individual.

        Shallow-copies all attributes and deep-copies the schedule.
        Swap mutation is applied with probability config.mutation_rate.

        Returns
        -------
        Individual
            A new individual with a possibly mutated schedule.  The
            cached fitness is inherited and remains valid unless the
            schedule was mutated.
        """
        # Shallow copy all attributes
        new = copy.copy(self)
        # Deep copy schedule
        new.schedule = list(self.schedule)
        if new._rng.random() < new._config.mutation_rate:
            _swap_mutation(new.schedule, new._rng)
            # Schedule changed — invalidate inherited fitness cache.
            new._fitness = None
        return new

    def __matmul__(
        self,
        other: Individual,
    ) -> Individual:
        """
        Produce an offspring by crossing this individual with another.

        With probability config.crossover_rate, applies order crossover
        between the two parents, then applies swap mutation with
        probability config.mutation_rate. If crossover is not applied,
        returns a clone of self with possible mutation.

        Parameters
        ----------
        other : Individual
            The second parent.

        Returns
        -------
        Individual
            A new offspring individual.
        """
        if self._rng.random() < self._config.crossover_rate:
            new_schedule = _order_crossover(
                self.schedule,
                other.schedule,
                rng=self._rng
            )
            new = self.clone()
            new._fitness = None
            new.schedule = new_schedule
            if new._rng.random() < new._config.mutation_rate:
                _swap_mutation(new.schedule, new._rng)
        else:
            new = self.clone()
        return new


@dataclass
class IndividualWithPageSet(Individual):
    """
    An Individual that evaluates fitness using page-level simulation.

    Extends Individual with a set of per-query page sets captured from
    pg_buffercache, enabling finer-grained cache hit estimation.

    Attributes
    ----------
    page_sets : list[frozenset[int]]
        Integer-encoded page sets used for page-level fitness evaluation.
    """

    page_sets: list[frozenset[int]]

    def fitness(self) -> float:
        """
        Return the page-level fitness of this individual.

        Evaluates and caches the fitness on the first call via
        ``self._fitness_fn``, forwarding ``self.page_sets`` for
        implementations that make use of page-level data.

        Returns
        -------
        float
            Fitness score produced by the configured fitness function.
        """
        if self._fitness is None:
            self._fitness = self._fitness_fn(
                self.schedule,
                self.profiles,
                self.cache_capacity_pages,
                self.page_sets,
            )
        return self._fitness


@dataclass
class IndividualApproximate(Individual):
    """
    An Individual that evaluates fitness using the pairwise overlap approximation.

    Used during GA evolution to reduce per-evaluation cost from
    O(total_pages) to O(n_queries · w).  The final best schedule is
    always re-scored by the exact clock-sweep simulator in run_ga.

    Attributes
    ----------
    overlap_matrix : list[list[int]]
        Symmetric pairwise page-overlap matrix from compute_overlap_matrix.
    page_counts : list[int]
        Number of distinct pages per query, same indexing as overlap_matrix.
    """

    overlap_matrix: list[list[int]]
    page_counts: list[int]

    def fitness(self) -> float:
        """
        Return the approximate cache hit-ratio fitness of this individual.

        Evaluates and caches the fitness on the first call using the
        precomputed overlap matrix.  Subsequent calls return the cached value.

        Returns
        -------
        float
            Approximate cache hit ratio in the range [0.0, 1.0].
        """
        if self._fitness is None:
            self._fitness = approximate_schedule_fitness(
                self.overlap_matrix,
                self.page_counts,
                self.schedule,
                self.cache_capacity_pages,
            )
        return self._fitness


@dataclass
class IndividualDirectional(Individual):
    """
    An Individual that evaluates fitness using the directional utility matrix.

    Used during GA evolution when ``config.approx_mode == "directional"``.
    The directional matrix ``D[i][j] = |R(Qᵢ; C) ∩ P(Qⱼ)|`` captures
    the asymmetry between large→small and small→large transitions that
    the symmetric overlap matrix misses.  The final best schedule is
    always re-scored by the exact clock-sweep simulator in run_ga.

    Attributes
    ----------
    directional_matrix : list[list[int]]
        Asymmetric pairwise directional matrix from
        ``compute_directional_matrix``.  Must be built with the same
        cache capacity used by the simulator.
    page_counts : list[int]
        Number of distinct pages per query, same indexing as directional_matrix.
    """

    directional_matrix: list[list[int]]
    page_counts: list[int]

    def fitness(self) -> float:
        """
        Return the directional approximate fitness of this individual.

        Evaluates and caches the fitness on the first call as the
        edge-local sum ``Σ D[π(k-1)][π(k)]`` divided by total page
        requests.  Subsequent calls return the cached value.

        Returns
        -------
        float
            Approximate cache hit ratio in the range [0.0, 1.0].
        """
        if self._fitness is None:
            self._fitness = approximate_schedule_fitness_directional(
                self.directional_matrix,
                self.page_counts,
                self.schedule,
            )
        return self._fitness


def make_individual(
    schedule: list[int],
    profiles: list[AccessProfile],
    cache_capacity_pages: int,
    rng: random.Random,
    config: GAConfig,
    page_sets: Optional[list[frozenset[int]]],
    overlap_matrix: Optional[list[list[int]]] = None,
    page_counts: Optional[list[int]] = None,
    directional_matrix: Optional[list[list[int]]] = None,
) -> Individual:
    """
    Construct the appropriate Individual subtype for the configured fitness mode.

    The fitness function is chosen from ``config.fitness_type``:
    ``"lru"`` selects the clock-sweep simulator and ``"dqn"`` selects
    the DQN surrogate (``config.dqn.infer``).

    Dispatch rules:
    * When ``config.use_approximate_fitness`` is True, ``fitness_type``
      is ``"lru"``, ``config.approx_mode == "directional"``, and page-level
      data (``page_sets``, ``directional_matrix``, ``page_counts``) is
      provided, returns an ``IndividualDirectional``.
    * Else when ``config.use_approximate_fitness`` is True, ``fitness_type``
      is ``"lru"``, and page-level data (``page_sets``, ``overlap_matrix``,
      ``page_counts``) is provided, returns an ``IndividualApproximate``.
    * Else, if ``page_sets`` is provided, returns an ``IndividualWithPageSet``.
    * Else, returns a plain ``Individual``.

    Parameters
    ----------
    schedule : list[int]
        Permutation of query indices representing an execution order.
    profiles : list[AccessProfile]
        Access profiles for each query, indexed by query index.
    cache_capacity_pages : int
        Clock-sweep cache capacity in 8 KB pages.
    rng : random.Random
        Random number generator instance.
    config : GAConfig
        Algorithm configuration.
    page_sets : list[frozenset[int]] or None
        Integer-encoded page sets for page-level simulation.  If None,
        an Individual using table-level simulation is returned.
    overlap_matrix : list[list[int]] or None
        Precomputed pairwise overlap matrix.  Required when
        ``config.use_approximate_fitness`` is True and
        ``config.approx_mode == "symmetric"``.
    page_counts : list[int] or None
        Per-query page counts.  Required when
        ``config.use_approximate_fitness`` is True.
    directional_matrix : list[list[int]] or None
        Precomputed directional utility matrix.  Required when
        ``config.use_approximate_fitness`` is True and
        ``config.approx_mode == "directional"``.

    Returns
    -------
    Individual
        IndividualDirectional, IndividualApproximate, IndividualWithPageSet,
        or Individual as dictated by the dispatch rules above.

    Raises
    ------
    ValueError
        If ``config.fitness_type`` is not a recognised value.
    AssertionError
        If ``config.fitness_type`` is ``"dqn"`` but ``config.dqn`` is None.
    """
    if config.fitness_type == "lru":
        fitness_fn: FitnessFunction = _fitness_cache_simulation
    elif config.fitness_type == "dqn":
        assert config.dqn is not None, (
            "DQN fitness was chosen but config.dqn is None"
        )
        fitness_fn = config.dqn.infer
    else:
        raise ValueError(f"Unknown fitness type: {config.fitness_type}")

    if (
        config.fitness_type == "lru"
        and config.use_approximate_fitness
        and config.approx_mode == "directional"
        and page_sets is not None
        and directional_matrix is not None
        and page_counts is not None
    ):
        return IndividualDirectional(
            schedule=schedule,
            profiles=profiles,
            cache_capacity_pages=cache_capacity_pages,
            _rng=rng,
            _config=config,
            _fitness_fn=fitness_fn,
            directional_matrix=directional_matrix,
            page_counts=page_counts,
        )

    if (
        config.fitness_type == "lru"
        and config.use_approximate_fitness
        and page_sets is not None
        and overlap_matrix is not None
        and page_counts is not None
    ):
        return IndividualApproximate(
            schedule=schedule,
            profiles=profiles,
            cache_capacity_pages=cache_capacity_pages,
            _rng=rng,
            _config=config,
            _fitness_fn=fitness_fn,
            overlap_matrix=overlap_matrix,
            page_counts=page_counts,
        )
    if page_sets is not None:
        return IndividualWithPageSet(
            schedule=schedule,
            profiles=profiles,
            cache_capacity_pages=cache_capacity_pages,
            _rng=rng,
            _config=config,
            _fitness_fn=fitness_fn,
            page_sets=page_sets,
        )
    return Individual(
        schedule=schedule,
        profiles=profiles,
        cache_capacity_pages=cache_capacity_pages,
        _rng=rng,
        _config=config,
        _fitness_fn=fitness_fn,
    )


def select_parents(
    population: list[Individual],
    config: GAConfig,
    rng: random.Random,
    n_parents: int = 2,
) -> tuple[Individual, ...]:
    """
    Select multiple parents from the population via tournament selection.

    Parameters
    ----------
    population : list[Individual]
        Current population to select from.
    config : GAConfig
        Algorithm configuration, providing tournament_size.
    rng : random.Random
        Random number generator instance.
    n_parents : int
        Number of parents to select.  Defaults to 2.

    Returns
    -------
    tuple[Individual, ...]
        Selected parents, each chosen independently by tournament selection.
    """
    parents = []
    for _ in range(n_parents):
        parent = _tournament_select(population, config.tournament_size, rng)
        parents.append(parent)
    return tuple(parents)


__all__ = [
    "_fitness_cache_simulation",
    "_tournament_select",
    "_order_crossover",
    "_swap_mutation",
    "FitnessFunction",
    "Individual",
    "IndividualWithPageSet",
    "IndividualApproximate",
    "IndividualDirectional",
    "make_individual",
    "select_parents",
]
