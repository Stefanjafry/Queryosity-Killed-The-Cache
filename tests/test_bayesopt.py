"""
Tests for the Mode B priority-vector BO consumer.

All tests run on synthetic page sets — no database, no real profiles.
The SMAC and BoTorch backend tests are skipped automatically when the
corresponding library is not installed, so the base suite stays green
on environments without the BO dependencies.
"""

from __future__ import annotations

import json
import random

import pytest

from src.bayesopt.data import load_workload_pages
from src.bayesopt.objective import ExactSimObjective
from src.bayesopt.priority import decode_priority_vector, is_valid_permutation
from src.bayesopt.search import (
    BudgetExhausted,
    PrioritySpace,
    TrialRecorder,
    default_n_init,
    run_search,
)
from src.simulator.cache_simulator import simulate_schedule_page_level


def _tiny_workload() -> tuple[list[frozenset[int]], list[str]]:
    """Four queries with hand-checkable overlap at capacity 100."""
    page_sets = [
        frozenset(range(0, 10)),   # q0
        frozenset(range(5, 15)),   # q1: shares 5 pages with q0
        frozenset(range(20, 25)),  # q2: disjoint
        frozenset(range(0, 5)),    # q3: subset of q0
    ]
    return page_sets, ["q0", "q1", "q2", "q3"]


def _random_workload(
    n: int, universe: int = 100, size: int = 30, seed: int = 0
) -> tuple[list[frozenset[int]], list[str]]:
    """n random page sets whose ordering matters under a small cache."""
    rng = random.Random(seed)
    page_sets = [
        frozenset(rng.sample(range(universe), size)) for _ in range(n)
    ]
    return page_sets, [f"q{i}" for i in range(n)]


class TestDecodePriorityVector:
    def test_descending_order(self):
        assert decode_priority_vector([0.1, 0.9, 0.5]) == [1, 2, 0]

    def test_ties_break_on_lower_index(self):
        assert decode_priority_vector([0.5, 0.5, 0.2]) == [0, 1, 2]
        assert decode_priority_vector([0.0, 0.0, 0.0, 0.0]) == [0, 1, 2, 3]

    def test_always_a_permutation(self):
        rng = random.Random(7)
        for _ in range(100):
            n = rng.randint(1, 30)
            vec = [rng.random() for _ in range(n)]
            schedule = decode_priority_vector(vec)
            assert is_valid_permutation(schedule, n)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            decode_priority_vector([])


class TestExactSimObjective:
    def test_matches_exact_simulator(self):
        page_sets, _ = _tiny_workload()
        obj = ExactSimObjective(page_sets, cache_capacity_pages=100)
        schedule = [0, 1, 2, 3]
        metrics, memo_hit = obj.evaluate_schedule(schedule)
        oracle = simulate_schedule_page_level(page_sets, schedule, 100)
        assert memo_hit is False
        assert metrics.hit_ratio == oracle.hit_ratio
        assert metrics.total_hits == oracle.total_hits == 10
        assert metrics.total_requests == oracle.total_requests == 30
        assert metrics.page_reads == 20
        assert metrics.cost == pytest.approx(1.0 - oracle.hit_ratio)

    def test_memoization_returns_identical_metrics(self):
        page_sets, _ = _tiny_workload()
        obj = ExactSimObjective(page_sets, cache_capacity_pages=100)
        first, hit1 = obj.evaluate_schedule([3, 0, 1, 2])
        second, hit2 = obj.evaluate_schedule([3, 0, 1, 2])
        assert hit1 is False and hit2 is True
        assert first == second

    def test_edge_sum_diagnostic(self):
        page_sets, _ = _tiny_workload()
        d = [
            [0, 5, 0, 5],
            [5, 0, 0, 0],
            [0, 0, 0, 0],
            [5, 0, 0, 0],
        ]
        obj = ExactSimObjective(page_sets, 100, d_matrix=d)
        metrics, _ = obj.evaluate_schedule([0, 1, 2, 3])
        assert metrics.edge_sum_d == d[0][1] + d[1][2] + d[2][3]

    def test_d_matrix_size_mismatch_raises(self):
        page_sets, _ = _tiny_workload()
        with pytest.raises(ValueError):
            ExactSimObjective(page_sets, 100, d_matrix=[[0]])


class TestRecorderAndRandomBackend:
    def test_budget_is_enforced(self):
        page_sets, qids = _tiny_workload()
        obj = ExactSimObjective(page_sets, 100)
        recorder = TrialRecorder(obj, qids, budget=2, static_fields={})
        recorder([0.1, 0.2, 0.3, 0.4])
        recorder([0.4, 0.3, 0.2, 0.1])
        with pytest.raises(BudgetExhausted):
            recorder([0.5, 0.5, 0.5, 0.5])

    def test_random_search_run(self, tmp_path):
        page_sets, qids = _random_workload(n=8)
        obj = ExactSimObjective(page_sets, cache_capacity_pages=40)
        trials_path = tmp_path / "trials.jsonl"
        with open(trials_path, "w") as stream:
            run = run_search(
                "random",
                PrioritySpace(8),
                obj,
                qids,
                budget=12,
                seed=11,
                static_fields={"workload": "synthetic", "cache_pages": 40},
                trials_stream=stream,
            )

        assert run.n_trials == 12
        assert len(run.best_so_far) == 12
        assert all(
            a >= b for a, b in zip(run.best_so_far, run.best_so_far[1:])
        )
        assert is_valid_permutation(run.best_schedule, 8)
        assert run.best_schedule_qids == [qids[i] for i in run.best_schedule]

        records = [
            json.loads(line) for line in trials_path.read_text().splitlines()
        ]
        assert len(records) == 12
        assert [r["trial"] for r in records] == list(range(1, 13))
        assert all(r["valid"] for r in records)
        assert all(r["mode"] == "priority_vector" for r in records)
        for key in (
            "workload", "cache_pages", "backend", "seed", "priorities",
            "schedule", "schedule_qids", "cost", "sim_hit_ratio",
            "sim_page_reads", "best_cost_so_far", "memo_hit",
        ):
            assert key in records[0], f"missing log field {key}"
        assert run.best_cost == pytest.approx(min(r["cost"] for r in records))

    def test_random_search_deterministic(self):
        page_sets, qids = _random_workload(n=8)
        runs = []
        for _ in range(2):
            obj = ExactSimObjective(page_sets, cache_capacity_pages=40)
            runs.append(
                run_search(
                    "random", PrioritySpace(8), obj, qids,
                    budget=10, seed=3,
                )
            )
        assert runs[0].best_schedule == runs[1].best_schedule
        assert runs[0].best_so_far == runs[1].best_so_far

    def test_n_init_validation(self):
        page_sets, qids = _tiny_workload()
        obj = ExactSimObjective(page_sets, 100)
        with pytest.raises(ValueError):
            run_search(
                "random", PrioritySpace(4), obj, qids,
                budget=4, seed=0, n_init=10,
            )

    def test_default_n_init(self):
        assert default_n_init(25) == 6
        assert default_n_init(50) == 12
        assert default_n_init(100) == 25
        assert default_n_init(200) == 25
        assert default_n_init(8) == 5


class TestSmacBackend:
    def test_smac_run_and_determinism(self, tmp_path):
        pytest.importorskip("smac")
        page_sets, qids = _random_workload(n=6)
        runs = []
        for attempt in range(2):
            obj = ExactSimObjective(page_sets, cache_capacity_pages=40)
            runs.append(
                run_search(
                    "smac",
                    PrioritySpace(6),
                    obj,
                    qids,
                    budget=8,
                    seed=21,
                    n_init=4,
                    workdir=tmp_path / f"smac_{attempt}",
                )
            )
        for run in runs:
            assert run.n_trials == 8
            assert len(run.best_so_far) == 8
            assert is_valid_permutation(run.best_schedule, 6)
        assert runs[0].best_cost == runs[1].best_cost
        assert runs[0].best_schedule == runs[1].best_schedule


class TestBotorchBackend:
    def test_botorch_gp_run(self):
        pytest.importorskip("botorch")
        page_sets, qids = _random_workload(n=5)
        obj = ExactSimObjective(page_sets, cache_capacity_pages=40)
        run = run_search(
            "botorch_gp", PrioritySpace(5), obj, qids,
            budget=6, seed=5, n_init=4,
        )
        assert run.n_trials == 6
        assert len(run.best_so_far) == 6
        assert is_valid_permutation(run.best_schedule, 5)
        assert all(
            a >= b for a, b in zip(run.best_so_far, run.best_so_far[1:])
        )


class TestDataLoadingAndCli:
    @staticmethod
    def _write_csv_dir(tmp_path, page_sets, qids):
        d = tmp_path / "page_access"
        d.mkdir()
        for qid, pages in zip(qids, page_sets):
            lines = ["table,block"]
            lines += [f"t{p % 3},{p}" for p in sorted(pages)]
            (d / f"{qid}.csv").write_text("\n".join(lines) + "\n")
        return d

    def test_load_workload_pages_natural_order(self, tmp_path):
        page_sets, _ = _random_workload(n=3)
        d = self._write_csv_dir(tmp_path, page_sets, ["q2", "q10", "q1"])
        wp = load_workload_pages(d)
        assert wp.query_ids == ["q1", "q2", "q10"]
        assert wp.n == 3
        assert all(isinstance(ps, frozenset) for ps in wp.page_sets)

    def test_load_workload_pages_exclude(self, tmp_path):
        page_sets, qids = _tiny_workload()
        d = self._write_csv_dir(tmp_path, page_sets, qids)
        wp = load_workload_pages(d, exclude={"q2"})
        assert wp.query_ids == ["q0", "q1", "q3"]
        with pytest.raises(ValueError):
            load_workload_pages(d, exclude={"nope"})

    def test_cli_end_to_end_random(self, tmp_path):
        from src.bayesopt.run_bayesopt import main

        page_sets, qids = _random_workload(n=5)
        d = self._write_csv_dir(tmp_path, page_sets, qids)
        out = tmp_path / "out"
        main([
            "--workload", "tpch",
            "--cache-pages", "40",
            "--page-access-dir", str(d),
            "--backend", "random",
            "--budget", "7",
            "--seed", "9",
            "--out-dir", str(out),
        ])
        trials = out / "tpch_c40_random_b7_s9.trials.jsonl"
        summary = out / "tpch_c40_random_b7_s9.summary.json"
        assert trials.is_file() and summary.is_file()
        records = [json.loads(x) for x in trials.read_text().splitlines()]
        assert len(records) == 7
        assert all(r["edge_sum_d"] is not None for r in records)
        payload = json.loads(summary.read_text())
        assert payload["n_trials"] == 7
        assert payload["backend"] == "random"
        assert is_valid_permutation(payload["best_schedule"], 5)
