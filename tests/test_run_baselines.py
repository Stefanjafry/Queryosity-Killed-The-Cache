"""
Tests for the incumbent-baseline regeneration runner.

Synthetic page sets only; GA budgets are shrunk for speed.  The point
is wiring correctness (valid schedules, determinism, same metric units
as the BO logs), not schedule quality.
"""

from __future__ import annotations

import json
import random

import pytest

from src.bayesopt.objective import is_valid_permutation
from src.bayesopt.run_baselines import (
    BASELINE_METHODS,
    compute_baseline_schedule,
)
from src.simulator.cache_simulator import compute_directional_matrix


def _random_workload(
    n: int, universe: int = 100, size: int = 30, seed: int = 0
) -> tuple[list[frozenset[int]], list[str]]:
    rng = random.Random(seed)
    page_sets = [
        frozenset(rng.sample(range(universe), size)) for _ in range(n)
    ]
    return page_sets, [f"q{i}" for i in range(n)]


CACHE = 40
GA_KW = dict(seed=42, population=12, generations=8)


class TestComputeBaselineSchedule:
    def test_all_methods_produce_valid_schedules(self):
        page_sets, qids = _random_workload(n=9)
        d = compute_directional_matrix(page_sets, CACHE)
        for method in BASELINE_METHODS:
            schedule = compute_baseline_schedule(
                method, page_sets, qids, d, CACHE, **GA_KW
            )
            assert is_valid_permutation(schedule, 9), method

    def test_deterministic_given_seed(self):
        page_sets, qids = _random_workload(n=8)
        d = compute_directional_matrix(page_sets, CACHE)
        for method in BASELINE_METHODS:
            s1 = compute_baseline_schedule(
                method, page_sets, qids, d, CACHE, **GA_KW
            )
            s2 = compute_baseline_schedule(
                method, page_sets, qids, d, CACHE, **GA_KW
            )
            assert s1 == s2, method

    def test_unknown_method_raises(self):
        page_sets, qids = _random_workload(n=4)
        d = compute_directional_matrix(page_sets, CACHE)
        with pytest.raises(ValueError):
            compute_baseline_schedule(
                "nope", page_sets, qids, d, CACHE, **GA_KW
            )


class TestBaselineCli:
    def test_cli_end_to_end(self, tmp_path):
        from src.bayesopt.run_baselines import main

        page_sets, qids = _random_workload(n=6)
        d = tmp_path / "page_access"
        d.mkdir()
        for qid, pages in zip(qids, page_sets):
            lines = ["table,block"] + [f"t{p % 3},{p}" for p in sorted(pages)]
            (d / f"{qid}.csv").write_text("\n".join(lines) + "\n")
        out = tmp_path / "out"

        main([
            "--workload", "tpch",
            "--cache-pages", str(CACHE),
            "--page-access-dir", str(d),
            "--seed", "42",
            "--ga-population", "10",
            "--ga-generations", "5",
            "--out-dir", str(out),
        ])

        for method in BASELINE_METHODS:
            path = out / f"tpch_c{CACHE}_baseline_{method}_s42.summary.json"
            assert path.is_file(), method
            payload = json.loads(path.read_text())
            assert payload["mode"] == "baseline"
            assert payload["method"] == method
            assert is_valid_permutation(payload["schedule"], 6)
            assert 0.0 <= payload["sim_hit_ratio"] <= 1.0
            assert payload["cost"] == pytest.approx(
                1.0 - payload["sim_hit_ratio"]
            )
            assert payload["sim_page_reads"] == (
                payload["sim_requests"] - payload["sim_hits"]
            )
