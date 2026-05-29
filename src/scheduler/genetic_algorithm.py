"""
Genetic algorithm-based query scheduler.

Uses tournament selection, order crossover (OX), swap mutation, and
elitism to evolve a population of query permutations that maximise
the simulated clock-sweep cache hit ratio.

Optimisations
-------------
* **Fitness memoization** — identical permutations (from elitism or
  convergent crossover) are not re-evaluated.
* **Approximate fitness via overlap matrix** — when page-level data is
  available, a precomputed pairwise overlap matrix reduces per-eval
  cost from O(total_pages) to O(n_queries · w).  The exact clock-sweep
  simulation is still used for the final best schedule.
"""

from __future__ import annotations

import random
from typing import Callable

from src.scheduler.base_scheduler import ScheduleResult, SchedulerBase
from src.scheduler.genetic_config import GAConfig
from src.scheduler.genetic_utils import (
    Individual,
    make_individual,
    select_parents,
)
from src.simulator.access_profile import AccessProfile
from src.simulator.cache_simulator import (
    SimulationResult,
    compute_directional_matrix,
    compute_overlap_matrix,
    simulate_schedule,
    simulate_schedule_page_level,
)


class GAScheduler(SchedulerBase):
    """
    Genetic algorithm-based query scheduler.

    Uses tournament selection, order crossover, swap mutation, and
    elitism to evolve a population of query permutations that maximise
    the simulated clock-sweep cache hit ratio.

    Parameters
    ----------
    config : GAConfig | None
        Algorithm parameters.  Defaults are used when None.
    on_generation : callable or None
        Optional callback invoked as ``on_generation(gen, best_fitness)``
        after each generation completes.
    """

    def __init__(
        self,
        config: GAConfig | None = None,
        on_generation: Callable[[int, float], None] | None = None,
    ) -> None:
        self.config = config or GAConfig()
        self.on_generation = on_generation

    def schedule(
        self,
        profiles: list[AccessProfile],
        page_sets: list[frozenset[int]] | None = None,
    ) -> ScheduleResult:
        """
        Run the genetic algorithm to find a cache-friendly query schedule.

        Parameters
        ----------
        profiles : list[AccessProfile]
            One access profile per query.
        page_sets : list[frozenset[int]] | None
            Integer-encoded page sets from pg_buffercache profiling.
            When provided, fitness is evaluated at page-level granularity.

        Returns
        -------
        ScheduleResult
            Best schedule, its fitness, simulation result, and fitness
            history.
        """
        return run_ga(
            profiles,
            self.config,
            on_generation=self.on_generation,
            page_sets=page_sets,
        )


def run_ga(
    profiles: list[AccessProfile],
    config: GAConfig | None = None,
    on_generation: Callable[[int, float], None] | None = None,
    page_sets: list[frozenset[int]] | None = None,
) -> ScheduleResult:
    """
    Run the genetic algorithm to find a cache-friendly query schedule.

    Evolves a population of permutations using tournament selection,
    order crossover, swap mutation, and elitism.  Fitness is the cache
    hit ratio obtained by simulating each schedule through a clock-sweep
    cache.

    When ``config.use_approximate_fitness`` is True and *page_sets* are
    provided, the precomputed pairwise overlap matrix is used for fast
    surrogate fitness during evolution.  The final best schedule is
    always validated with the exact clock-sweep simulation.

    Parameters
    ----------
    profiles : list[AccessProfile]
        One access profile per query.  The index in this list is the
        query's integer ID used in chromosomes.
    config : GAConfig | None
        Algorithm parameters.  Defaults are used when None.
    on_generation : callable or None
        Optional callback invoked as ``on_generation(gen, best_fitness)``
        after each generation completes.
    page_sets : list[frozenset[int]] | None
        Integer-encoded page sets from pg_buffercache profiling.  When
        provided, fitness is evaluated at page-level granularity.

    Returns
    -------
    ScheduleResult
        Best schedule, its fitness, full simulation result, and the
        per-generation fitness history.
    """
    if config is None:
        config = GAConfig()

    rng = random.Random(config.seed)
    n = len(profiles)
    cache_capacity_pages = config.cache_capacity_pages

    if n == 0:
        return ScheduleResult(
            best_schedule=[],
            best_fitness=0.0,
            best_simulation=SimulationResult(total_requests=0, total_hits=0),
        )

    if n == 1:
        if page_sets is not None:
            sim = simulate_schedule_page_level(page_sets, [0], cache_capacity_pages)
        else:
            sim = simulate_schedule(profiles, [0], cache_capacity_pages)
        return ScheduleResult(
            best_schedule=[0],
            best_fitness=sim.hit_ratio,
            best_simulation=sim,
            fitness_history=[sim.hit_ratio],
        )

    # When the approximate-fitness path is active, precompute the
    # relevant matrix (symmetric overlap or directional utility) and
    # per-query page counts once up front.  These are shared by every
    # Individual in the population.
    overlap_matrix: list[list[int]] | None = None
    directional_matrix: list[list[int]] | None = None
    page_counts: list[int] | None = None
    if config.use_approximate_fitness and page_sets is not None:
        page_counts = [len(ps) for ps in page_sets]
        if config.approx_mode == "directional":
            directional_matrix = compute_directional_matrix(
                page_sets, cache_capacity_pages,
            )
        else:
            overlap_matrix = compute_overlap_matrix(page_sets)

    population: list[Individual] = []
    for _ in range(config.population_size):
        perm = list(range(n))
        rng.shuffle(perm)
        population.append(
            make_individual(
                schedule=perm,
                profiles=profiles,
                cache_capacity_pages=cache_capacity_pages,
                rng=rng,
                config=config,
                page_sets=page_sets,
                overlap_matrix=overlap_matrix,
                page_counts=page_counts,
                directional_matrix=directional_matrix,
            )
        )

    fitness_history: list[float] = []

    # --- Evolution loop ---------------------------------------------------
    for gen in range(config.num_generations):
        ranked: list[Individual] = sorted(
            population,
            key=lambda individual: individual.fitness(),
            reverse=True,
        )
        elites: list[Individual] = ranked[:config.elite_count]

        new_population: list[Individual] = list(elites)

        # Generate children until population reaches the configured size
        while len(new_population) < config.population_size:
            p1, p2 = select_parents(population, config, rng)
            child = p1 @ p2
            new_population.append(child)

        population = new_population

        best_gen_fitness = max(individual.fitness() for individual in population)
        fitness_history.append(best_gen_fitness)

        if on_generation is not None:
            on_generation(gen, best_gen_fitness)

    # --- Final result (always exact simulation) ---------------------------
    best_idx = max(range(len(population)), key=lambda i: population[i].fitness())
    best_schedule = population[best_idx].schedule
    if page_sets is not None:
        best_sim = simulate_schedule_page_level(
            page_sets, best_schedule, cache_capacity_pages,
        )
    else:
        best_sim = simulate_schedule(
            profiles, best_schedule, cache_capacity_pages,
        )

    return ScheduleResult(
        best_schedule=best_schedule,
        best_fitness=best_sim.hit_ratio,
        best_simulation=best_sim,
        fitness_history=fitness_history,
    )
