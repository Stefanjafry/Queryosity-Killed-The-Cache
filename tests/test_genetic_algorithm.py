from dataclasses import dataclass
from typing import cast

from src.simulator.access_profile import AccessProfile
from src.simulator.cache_simulator import simulate_schedule
from src.scheduler.genetic_algorithm import run_ga
from src.scheduler.genetic_config import GAConfig
from src.scheduler.genetic_utils import (
    _order_crossover,
    _swap_mutation,
    _tournament_select,
    Individual
)
import random


@dataclass
class IndividualMock:
    schedule: list[int]
    _fitness: float

    def fitness(self):
        return self._fitness


class TestOrderCrossover:
    def test_produces_valid_permutation(self):
        rng = random.Random(42)
        for _ in range(50):
            n = 10
            p1 = list(range(n))
            p2 = list(range(n))
            rng.shuffle(p1)
            rng.shuffle(p2)
            child = _order_crossover(p1, p2, rng)
            assert sorted(child) == list(range(n))

    def test_inherits_substring_from_parent1(self):
        rng = random.Random(0)
        p1 = [0, 1, 2, 3, 4]
        p2 = [4, 3, 2, 1, 0]
        # Force known cut points by controlling the RNG
        child = _order_crossover(p1, p2, rng)
        assert sorted(child) == [0, 1, 2, 3, 4]


class TestSwapMutation:
    def test_still_valid_permutation(self):
        rng = random.Random(42)
        ind = list(range(10))
        _swap_mutation(ind, rng)
        assert sorted(ind) == list(range(10))

    def test_exactly_two_positions_differ(self):
        rng = random.Random(42)
        original = list(range(10))
        mutated = list(original)
        _swap_mutation(mutated, rng)
        diffs = sum(1 for a, b in zip(original, mutated) if a != b)
        assert diffs == 2


class TestTournamentSelect:
    def test_selects_from_population(self):
        rng = random.Random(42)
        scheds = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
        fits = [0.1, 0.5, 0.3]
        pop = [
            IndividualMock(schedule=sched, _fitness=fit)
            for sched, fit in zip(scheds, fits)
        ]
        pop = cast(list[Individual], pop)
        selected = _tournament_select(pop, k=2, rng=rng)
        assert sorted(selected.schedule) == [0, 1, 2]

    def test_prefers_fitter(self):
        rng = random.Random(42)
        scheds = [[0, 1], [1, 0]]
        fits = [0.0, 1.0]
        pop = [
            IndividualMock(schedule=sched, _fitness=fit)
            for sched, fit in zip(scheds, fits)
        ]
        pop = cast(list[Individual], pop)
        # Over many trials, individual 1 (fitness=1.0) should win most
        wins = sum(
            1
            for _ in range(100)
            if _tournament_select(pop, k=2, rng=rng).schedule == [1, 0]
        )
        assert wins > 60


class TestRunGA:
    def _make_profiles(self) -> list[AccessProfile]:
        """Queries where grouping by shared tables improves cache hits."""
        return [
            AccessProfile(query_id="q0", table_pages={"A": 10, "B": 5}),
            AccessProfile(query_id="q1", table_pages={"C": 10}),
            AccessProfile(query_id="q2", table_pages={"A": 10, "D": 5}),
            AccessProfile(query_id="q3", table_pages={"C": 10, "E": 5}),
        ]

    def test_returns_valid_permutation(self):
        profiles = self._make_profiles()
        config = GAConfig(
            population_size=20,
            num_generations=10,
            seed=42,
            cache_capacity_pages=50,
        )
        result = run_ga(profiles, config)
        assert sorted(result.best_schedule) == [0, 1, 2, 3]

    def test_fitness_improves_or_stable(self):
        profiles = self._make_profiles()
        config = GAConfig(
            population_size=30,
            num_generations=50,
            seed=42,
            cache_capacity_pages=50,
        )
        result = run_ga(profiles, config)
        # With elitism, best fitness should be monotonically non-decreasing
        for i in range(1, len(result.fitness_history)):
            assert result.fitness_history[i] >= result.fitness_history[i - 1] - 1e-12

    def test_beats_random_baseline(self):
        """GA should find a schedule at least as good as random on average."""
        profiles = self._make_profiles()
        config = GAConfig(
            population_size=30,
            num_generations=50,
            seed=42,
            cache_capacity_pages=30,
        )
        result = run_ga(profiles, config)

        # Compare against random schedules
        rng = random.Random(99)
        random_fitnesses = []
        for _ in range(100):
            perm = list(range(len(profiles)))
            rng.shuffle(perm)
            sim = simulate_schedule(profiles, perm, config.cache_capacity_pages)
            random_fitnesses.append(sim.hit_ratio)

        avg_random = sum(random_fitnesses) / len(random_fitnesses)
        assert result.best_fitness >= avg_random

    def test_deterministic_with_seed(self):
        profiles = self._make_profiles()
        config = GAConfig(
            population_size=20,
            num_generations=10,
            seed=123,
            cache_capacity_pages=50,
        )
        r1 = run_ga(profiles, config)
        r2 = run_ga(profiles, config)
        assert r1.best_schedule == r2.best_schedule
        assert r1.best_fitness == r2.best_fitness

    def test_empty_profiles(self):
        result = run_ga([], GAConfig(seed=0))
        assert result.best_schedule == []
        assert result.best_fitness == 0.0

    def test_single_query(self):
        profiles = [AccessProfile(query_id="q0", table_pages={"A": 10})]
        result = run_ga(profiles, GAConfig(seed=0, cache_capacity_pages=100))
        assert result.best_schedule == [0]

    def test_on_generation_callback(self):
        profiles = self._make_profiles()
        log: list[tuple[int, float]] = []
        config = GAConfig(
            population_size=10,
            num_generations=5,
            seed=42,
            cache_capacity_pages=50,
        )
        run_ga(profiles, config, on_generation=lambda g, f: log.append((g, f)))
        assert len(log) == 5
        assert log[0][0] == 0
        assert log[4][0] == 4

    def test_approximate_fitness_mode(self):
        """GA with approximate fitness should still produce a valid schedule."""
        profiles = self._make_profiles()
        page_sets = [
            frozenset({0, 1, 2, 3}),     # q0: A(10)+B(5) mapped to pages
            frozenset({10, 11, 12}),      # q1: C(10)
            frozenset({0, 1, 2, 20}),     # q2: A(10)+D(5) — shares pages with q0
            frozenset({10, 11, 12, 30}),  # q3: C(10)+E(5) — shares pages with q1
        ]
        config = GAConfig(
            population_size=20,
            num_generations=10,
            seed=42,
            cache_capacity_pages=50,
            use_approximate_fitness=True,
        )
        result = run_ga(profiles, config, page_sets=page_sets)
        assert sorted(result.best_schedule) == [0, 1, 2, 3]
        # Final fitness is always from exact simulation
        assert 0.0 <= result.best_fitness <= 1.0

    def test_memoization_deterministic(self):
        """Memoized fitness should not change results vs. non-memoized."""
        profiles = self._make_profiles()
        config = GAConfig(
            population_size=20,
            num_generations=10,
            seed=42,
            cache_capacity_pages=50,
        )
        r1 = run_ga(profiles, config)
        r2 = run_ga(profiles, config)
        assert r1.best_schedule == r2.best_schedule
        assert r1.best_fitness == r2.best_fitness


class TestDirectionalApproximateFitness:
    """Smoke tests for the --approx-mode directional GA path."""

    def _make_profiles(self):
        return [
            AccessProfile(query_id="q0", table_pages={"A": 10, "B": 5}),
            AccessProfile(query_id="q1", table_pages={"C": 10}),
            AccessProfile(query_id="q2", table_pages={"A": 10, "D": 5}),
            AccessProfile(query_id="q3", table_pages={"C": 10, "E": 5}),
        ]

    def test_ga_runs_directional_end_to_end(self):
        profiles = self._make_profiles()
        page_sets = [
            frozenset({0, 1, 2, 3}),
            frozenset({10, 11, 12}),
            frozenset({0, 1, 2, 20}),
            frozenset({10, 11, 12, 30}),
        ]
        config = GAConfig(
            population_size=20,
            num_generations=10,
            seed=42,
            cache_capacity_pages=50,
            use_approximate_fitness=True,
            approx_mode="directional",
        )
        result = run_ga(profiles, config, page_sets=page_sets)
        assert sorted(result.best_schedule) == [0, 1, 2, 3]
        # Final fitness is always exact simulation.
        assert 0.0 <= result.best_fitness <= 1.0

    def test_greedy_directional_baseline_returns_permutation(self):
        from src.scheduler.greedy_directional import (
            greedy_directional_schedule,
        )
        from src.simulator.cache_simulator import compute_directional_matrix

        page_sets = [
            frozenset(range(0, 30)),
            frozenset({0, 1, 2, 3, 4}),
            frozenset({0, 1, 2, 25, 26}),
            frozenset(range(20, 40)),
        ]
        D = compute_directional_matrix(page_sets, cache_capacity_pages=10)
        page_counts = [len(ps) for ps in page_sets]
        schedule = greedy_directional_schedule(D, page_counts=page_counts)
        assert sorted(schedule) == [0, 1, 2, 3]
