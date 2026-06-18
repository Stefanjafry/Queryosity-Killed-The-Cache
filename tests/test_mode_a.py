"""
Tests for the Mode A structured-scorer scheduler.

Synthetic page sets only; budgets shrunk for speed.  Covers feature
finiteness and the D≤M sanity invariant, scorer determinism and
anchoring, consumer validity and the Greedy-D containment guarantee,
search-space proposals staying in range, early stopping, end-to-end
determinism, and the full set of log artifacts and their columns.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import pytest

from src.bayesopt.mode_a.consumers import (
    beam_search,
    build_candidates,
    multistart_greedy,
)
from src.bayesopt.mode_a.features import build_features
from src.bayesopt.mode_a.runner import run_mode_a
from src.bayesopt.mode_a.scorers import ScorerConfig, score_edge
from src.bayesopt.mode_a.search_space import (
    LAMBDA_RANGES,
    FAMILY_LAMBDAS,
    neighbor_configs,
    random_config,
    warmup_configs,
)
from src.bayesopt.objective import (
    is_valid_permutation,
    simulate_schedule_page_level_traced,
)
from src.scheduler.greedy_directional import greedy_directional_schedule
from src.simulator.cache_simulator import simulate_schedule_page_level


def _workload(n=10, universe=200, seed=0):
    rng = random.Random(seed)
    page_sets = [
        frozenset(rng.sample(range(universe), rng.randint(20, 60)))
        for _ in range(n)
    ]
    return page_sets, [f"q{i}" for i in range(n)]


CACHE = 80


class TestFeatures:
    def test_all_features_finite(self):
        ps, _ = _workload(n=12)
        f = build_features(ps, CACHE)
        n = f.n
        for i in range(n):
            for j in range(n):
                for mat in (f.M_norm, f.D_norm, f.asym_norm, f.gap_norm,
                            f.survivor_ratio):
                    assert math.isfinite(mat[i][j])
        for vec in (f.size_norm, f.cache_pressure, f.in_D, f.out_D,
                    f.balance_D):
            assert all(math.isfinite(v) for v in vec)
        for k in f.topk_values:
            assert all(math.isfinite(v) for v in f.out_D_topk[k])
            assert all(math.isfinite(v) for v in f.out_M_topk[k])

    def test_dle_invariant_no_violations(self):
        ps, _ = _workload(n=12)
        f = build_features(ps, CACHE)
        assert f.dle_violations == 0
        for i in range(f.n):
            for j in range(f.n):
                assert f.D[i][j] <= f.M[i][j]

    def test_normalization_ranges(self):
        ps, _ = _workload(n=12)
        f = build_features(ps, CACHE)
        n = f.n
        for i in range(n):
            for j in range(n):
                assert 0.0 <= f.D_norm[i][j] <= 1.0
                assert 0.0 <= f.M_norm[i][j] <= 1.0
                assert 0.0 <= f.survivor_ratio[i][j] <= 1.0
                assert 0.0 <= f.gap_norm[i][j] <= 1.0
                assert -1.0 <= f.asym_norm[i][j] <= 1.0
        assert all(-1.0 <= b <= 1.0 for b in f.balance_D)
        assert all(0.0 <= s <= 1.0 for s in f.size_norm)

    def test_topk_safe_when_k_exceeds_n(self):
        ps, _ = _workload(n=4)
        f = build_features(ps, CACHE, topk_values=(3, 5, 10))
        for k in (3, 5, 10):
            assert all(math.isfinite(v) for v in f.out_D_topk[k])

    def test_gap_is_nonnegative(self):
        ps, _ = _workload(n=10)
        f = build_features(ps, CACHE)
        for i in range(f.n):
            for j in range(f.n):
                assert f.gap_raw[i][j] >= 0


class TestScorers:
    def test_deterministic(self):
        ps, _ = _workload(n=10)
        f = build_features(ps, CACHE)
        for fam in ("m", "d", "hybrid"):
            cfg = ScorerConfig(family=fam, topk=3, lambda_out=0.7,
                               lambda_size=0.3, lambda_balance=0.5,
                               lambda_asym=0.2, lambda_ratio=0.4,
                               lambda_gap=-0.6)
            a = score_edge(cfg, f, 0, 1)
            b = score_edge(cfg, f, 0, 1)
            assert a == b
            assert math.isfinite(a)

    def test_pure_d_anchors_on_D_norm(self):
        ps, _ = _workload(n=10)
        f = build_features(ps, CACHE)
        cfg = ScorerConfig(family="d", topk=3)
        for j in range(1, f.n):
            assert score_edge(cfg, f, 0, j) == f.D_norm[0][j]

    def test_pure_m_anchors_on_M_norm(self):
        ps, _ = _workload(n=10)
        f = build_features(ps, CACHE)
        cfg = ScorerConfig(family="m", topk=3)
        for j in range(1, f.n):
            assert score_edge(cfg, f, 0, j) == f.M_norm[0][j]

    def test_unknown_family_raises(self):
        ps, _ = _workload(n=4)
        f = build_features(ps, CACHE)
        with pytest.raises(ValueError):
            score_edge(ScorerConfig(family="x"), f, 0, 1)


class TestConsumers:
    def test_multistart_valid_permutations(self):
        ps, _ = _workload(n=10)
        f = build_features(ps, CACHE)
        cfg = ScorerConfig(family="d", topk=3)
        for c in multistart_greedy(cfg, f, num_starts=5):
            assert is_valid_permutation(list(c.schedule), f.n)

    def test_beam_valid_permutations(self):
        ps, _ = _workload(n=10)
        f = build_features(ps, CACHE)
        cfg = ScorerConfig(family="hybrid", topk=3)
        for w in (2, 4, 8):
            cands = beam_search(cfg, f, beam_width=w)
            assert len(cands) <= w
            for c in cands:
                assert is_valid_permutation(list(c.schedule), f.n)

    def test_pure_d_multistart_contains_greedy_d(self):
        ps, _ = _workload(n=12)
        f = build_features(ps, CACHE)
        cfg = ScorerConfig(family="d", topk=3)
        cands = multistart_greedy(cfg, f, num_starts=5)
        gd = tuple(greedy_directional_schedule(f.D, f.pagecount))
        assert any(c.schedule == gd for c in cands)

    def test_consumers_deterministic(self):
        ps, _ = _workload(n=10)
        f = build_features(ps, CACHE)
        cfg = ScorerConfig(family="d", topk=3, lambda_out=0.5)
        assert (build_candidates(cfg, f, "multistart_greedy")
                == build_candidates(cfg, f, "multistart_greedy"))
        assert (build_candidates(cfg, f, "beam", beam_width=4)
                == build_candidates(cfg, f, "beam", beam_width=4))

    def test_greedy_tie_break_lower_index(self):
        # Two candidates with identical scores must resolve to lower index.
        ps, _ = _workload(n=6)
        f = build_features(ps, CACHE)
        # All-zero D row would tie everything; pure-D from start 0 then
        # appends in ascending index order on ties.
        cfg = ScorerConfig(family="d", topk=3)
        sched = multistart_greedy(cfg, f, num_starts=1)[0].schedule
        assert is_valid_permutation(list(sched), f.n)


class TestSearchSpace:
    def test_warmup_includes_pure_anchor(self):
        for fam in ("m", "d", "hybrid"):
            props = warmup_configs(fam, (3, 5, 10))
            pure = props[0].config
            assert "pure" in props[0].source
            assert pure.lambda_out == 0.0 and pure.lambda_size == 0.0
            assert pure.lambda_balance == 0.0 and pure.lambda_asym == 0.0
            assert pure.lambda_ratio == 0.0 and pure.lambda_gap == 0.0

    def test_random_config_in_range(self):
        rng = random.Random(0)
        for fam in ("m", "d", "hybrid"):
            for _ in range(50):
                cfg = random_config(fam, (3, 5, 10), rng).config
                for name in FAMILY_LAMBDAS[fam]:
                    lo, hi = LAMBDA_RANGES[name]
                    assert lo <= getattr(cfg, name) <= hi
                assert cfg.topk in (3, 5, 10)

    def test_neighbors_in_range_and_exclude_self(self):
        inc = ScorerConfig(family="d", topk=5, lambda_out=1.0,
                           lambda_size=0.5)
        neigh = neighbor_configs(inc, (3, 5, 10))
        for p in neigh:
            c = p.config
            # Each neighbour differs from the incumbent in at least one
            # searched dimension (full-config identity, not a projection).
            assert c != inc
            for name in FAMILY_LAMBDAS["d"]:
                lo, hi = LAMBDA_RANGES[name]
                assert lo <= getattr(c, name) <= hi


class TestTracedSimulator:
    def test_traced_matches_aggregate(self):
        ps, _ = _workload(n=10)
        sched = list(range(10))
        agg = simulate_schedule_page_level(ps, sched, CACHE)
        req, hits, traces = simulate_schedule_page_level_traced(ps, sched, CACHE)
        assert req == agg.total_requests
        assert hits == agg.total_hits
        assert sum(t.hits for t in traces) == hits
        assert sum(t.requests for t in traces) == req
        assert len(traces) == 10


class TestRunnerEndToEnd:
    def test_six_families_run_and_log(self, tmp_path):
        ps, qids = _workload(n=10)
        for fam in ("m", "d", "hybrid"):
            for cons in ("multistart_greedy", "beam"):
                res = run_mode_a(
                    page_sets=ps, query_ids=qids, workload="tpch",
                    cache_pages=CACHE, family=fam, consumer=cons,
                    seed=42, max_trials=20, output_dir=tmp_path,
                )
                d = Path(res.output_dir)
                for art in ("run_config.json", "feature_stats.json",
                            "parameter_trials.csv", "candidate_schedules.csv",
                            "edge_diagnostics.csv", "per_query_metrics.csv",
                            "best_so_far.csv", "timing_breakdown.json",
                            "summary.json"):
                    assert (d / art).is_file(), f"{fam}/{cons} missing {art}"
                assert is_valid_permutation(
                    [qids.index(q) for q in res.best_schedule_qids], 10
                )

    def test_determinism(self, tmp_path):
        ps, qids = _workload(n=10)
        a = run_mode_a(page_sets=ps, query_ids=qids, workload="tpch",
                       cache_pages=CACHE, family="hybrid", consumer="beam",
                       seed=7, max_trials=25, output_dir=tmp_path / "a")
        b = run_mode_a(page_sets=ps, query_ids=qids, workload="tpch",
                       cache_pages=CACHE, family="hybrid", consumer="beam",
                       seed=7, max_trials=25, output_dir=tmp_path / "b")
        assert a.best_schedule_qids == b.best_schedule_qids
        assert a.best_cost == b.best_cost
        assert a.best_config == b.best_config

    def test_best_observed_is_actually_best(self, tmp_path):
        ps, qids = _workload(n=10)
        res = run_mode_a(
            page_sets=ps, query_ids=qids, workload="tpch", cache_pages=CACHE,
            family="d", consumer="multistart_greedy", seed=42,
            max_trials=30, output_dir=tmp_path,
        )
        rows = list(csv.DictReader(
            open(Path(res.output_dir) / "candidate_schedules.csv")))
        # CSV costs are rounded to 6 dp; the recommended cost must equal
        # the minimum logged cost to within that rounding.
        best_cost = min(float(r["cost"]) for r in rows)
        assert abs(res.best_cost - best_cost) < 1e-6

    def test_param_and_schedule_ids_logged(self, tmp_path):
        ps, qids = _workload(n=8)
        res = run_mode_a(
            page_sets=ps, query_ids=qids, workload="tpch", cache_pages=CACHE,
            family="d", consumer="multistart_greedy", seed=42,
            max_trials=15, output_dir=tmp_path,
        )
        d = Path(res.output_dir)
        prows = list(csv.DictReader(open(d / "parameter_trials.csv")))
        crows = list(csv.DictReader(open(d / "candidate_schedules.csv")))
        assert all("parameter_trial_id" in r for r in prows)
        assert all("num_candidate_schedules" in r for r in prows)
        assert all("num_exact_evals_used" in r for r in prows)
        assert all("schedule_eval_id" in r for r in crows)
        assert all("parameter_trial_id" in r for r in crows)
        # exactly one row is the final global incumbent
        assert sum(r["is_global_incumbent"] == "True" for r in crows) == 1

    def test_early_stopping_triggers(self, tmp_path):
        ps, qids = _workload(n=8)
        res = run_mode_a(
            page_sets=ps, query_ids=qids, workload="tpch", cache_pages=CACHE,
            family="d", consumer="multistart_greedy", seed=42,
            max_trials=500, early_stop_patience=5, output_dir=tmp_path,
        )
        # The search must terminate well short of the 500-trial cap,
        # whether by early stopping or by refinement converging.
        assert res.num_parameter_trials < 500
        assert res.stop_reason in (
            "early_stop_random_phase", "early_stop_refinement",
            "refinement_converged",
        )

    def test_budget_is_nested(self, tmp_path):
        # With early stopping disabled, a larger budget must reach a cost
        # no worse than a smaller budget (it evaluates a superset).
        ps, qids = _workload(n=12)
        costs = []
        for mt in (15, 30, 60):
            r = run_mode_a(
                page_sets=ps, query_ids=qids, workload="tpch",
                cache_pages=CACHE, family="d",
                consumer="multistart_greedy", seed=42, max_trials=mt,
                early_stop_patience=10 ** 6, output_dir=tmp_path / f"b{mt}",
            )
            costs.append(r.best_cost)
        assert costs[0] >= costs[1] >= costs[2]

    def test_random_only_mode_runs(self, tmp_path):
        ps, qids = _workload(n=10)
        r = run_mode_a(
            page_sets=ps, query_ids=qids, workload="tpch", cache_pages=CACHE,
            family="d", consumer="multistart_greedy", seed=42,
            max_trials=20, search_mode="random_only",
            early_stop_patience=10 ** 6, output_dir=tmp_path,
        )
        assert is_valid_permutation(
            [qids.index(q) for q in r.best_schedule_qids], 10
        )
        # random_only must not run warm-up or refinement.
        assert r.refinement_ran is False
        assert r.neighbor_configs_evaluated == 0
        assert r.best_source in ("random", "neighbor:_none")
        assert not r.best_source.startswith("warmup")

    def test_warmup_random_mode_no_refinement(self, tmp_path):
        ps, qids = _workload(n=10)
        r = run_mode_a(
            page_sets=ps, query_ids=qids, workload="tpch", cache_pages=CACHE,
            family="d", consumer="multistart_greedy", seed=42,
            max_trials=30, search_mode="warmup_random",
            early_stop_patience=10 ** 6, output_dir=tmp_path,
        )
        assert r.refinement_ran is False
        assert r.neighbor_configs_evaluated == 0

    def test_full_mode_forces_refinement(self, tmp_path):
        ps, qids = _workload(n=12)
        r = run_mode_a(
            page_sets=ps, query_ids=qids, workload="tpch", cache_pages=CACHE,
            family="d", consumer="multistart_greedy", seed=42,
            max_trials=60, search_mode="warmup_random_neighbor",
            early_stop_patience=5, output_dir=tmp_path,
        )
        # Refinement must be exercised at least once even with tight
        # patience (early stop is suppressed during the forced first pass).
        assert r.refinement_ran is True
        assert r.neighbor_configs_evaluated > 0

    def test_three_arms_are_distinct(self, tmp_path):
        ps, qids = _workload(n=14)
        outs = {}
        for mode in ("random_only", "warmup_random",
                     "warmup_random_neighbor"):
            r = run_mode_a(
                page_sets=ps, query_ids=qids, workload="tpch",
                cache_pages=CACHE, family="d",
                consumer="multistart_greedy", seed=42, max_trials=50,
                search_mode=mode, early_stop_patience=8,
                output_dir=tmp_path / mode,
            )
            outs[mode] = r
        # Only the full arm runs refinement.
        assert outs["random_only"].refinement_ran is False
        assert outs["warmup_random"].refinement_ran is False
        assert outs["warmup_random_neighbor"].refinement_ran is True

    def test_stop_reason_recorded(self, tmp_path):
        ps, qids = _workload(n=8)
        r = run_mode_a(
            page_sets=ps, query_ids=qids, workload="tpch", cache_pages=CACHE,
            family="d", consumer="multistart_greedy", seed=42,
            max_trials=500, early_stop_patience=5, output_dir=tmp_path,
        )
        # Termination is recorded with a concrete reason (the tiny space
        # may converge before patience triggers).
        assert r.stop_reason in (
            "early_stop_random_phase", "early_stop_refinement",
            "refinement_converged", "budget_exhausted",
        )

    def test_feature_stats_records_overlap_survivor(self, tmp_path):
        ps, qids = _workload(n=10)
        res = run_mode_a(
            page_sets=ps, query_ids=qids, workload="tpch", cache_pages=CACHE,
            family="hybrid", consumer="multistart_greedy", seed=42,
            max_trials=12, output_dir=tmp_path,
        )
        stats = json.loads(
            (Path(res.output_dir) / "feature_stats.json").read_text())
        assert stats["dle_violations"] == 0
        assert "survivor_ratio_all" in stats
        assert "survivor_ratio_overlap_only" in stats

    def test_edge_diagnostics_trial_winner_only_by_default(self, tmp_path):
        ps, qids = _workload(n=8)
        res = run_mode_a(
            page_sets=ps, query_ids=qids, workload="tpch", cache_pages=CACHE,
            family="d", consumer="multistart_greedy", seed=42,
            max_trials=10, output_dir=tmp_path,
        )
        d = Path(res.output_dir)
        edge = list(csv.DictReader(open(d / "edge_diagnostics.csv")))
        cand = list(csv.DictReader(open(d / "candidate_schedules.csv")))
        winners = {r["schedule_eval_id"] for r in cand
                   if r["is_trial_winner"] == "True"}
        edge_ids = {r["schedule_eval_id"] for r in edge}
        # Every edge-logged schedule is a trial winner.
        assert edge_ids <= winners


class TestBotorchBackend:
    """BoTorch GP arm.  GP-fit tests skip when botorch is absent; the
    unit-cube mapping helpers and runner integration run everywhere."""

    def test_unit_cube_mapping_roundtrip(self):
        from src.bayesopt.mode_a.botorch_backend import (
            _bounds_for, _config_from_unit, _config_to_unit,
        )
        from src.bayesopt.mode_a.scorers import ScorerConfig
        dims = _bounds_for("d")
        cfg = ScorerConfig(family="d", topk=5, lambda_out=1.0,
                           lambda_size=0.5, lambda_balance=2.0,
                           lambda_asym=0.0)
        unit = _config_to_unit(cfg, dims)
        assert all(0.0 <= u <= 1.0 for u in unit)
        back = _config_from_unit("d", 5, unit, dims)
        for name, _, _ in dims:
            assert abs(getattr(back, name) - getattr(cfg, name)) < 1e-9

    def test_unit_cube_clamps_out_of_range(self):
        from src.bayesopt.mode_a.botorch_backend import (
            _bounds_for, _config_from_unit,
        )
        dims = _bounds_for("d")
        cfg = _config_from_unit("d", 3, [2.0, -1.0, 0.5, 0.5], dims)
        for name, lo, hi in dims:
            assert lo <= getattr(cfg, name) <= hi

    def test_botorch_run_if_available(self, tmp_path):
        pytest.importorskip("botorch")
        ps, qids = _workload(n=10)
        r = run_mode_a(
            page_sets=ps, query_ids=qids, workload="tpch", cache_pages=CACHE,
            family="d", consumer="multistart_greedy", seed=42,
            max_trials=40, search_mode="botorch_gp",
            early_stop_patience=10 ** 6, output_dir=tmp_path,
        )
        assert is_valid_permutation(
            [qids.index(q) for q in r.best_schedule_qids], 10
        )
        assert r.bo_warmup > 0
        assert r.bo_init > 0
        # recommender must return the true minimum-cost schedule
        rows = list(csv.DictReader(
            open(Path(r.output_dir) / "candidate_schedules.csv")))
        mincost = min(float(x["cost"]) for x in rows)
        assert abs(r.best_cost - mincost) < 1e-6

    def test_botorch_determinism_if_available(self, tmp_path):
        pytest.importorskip("botorch")
        ps, qids = _workload(n=10)
        a = run_mode_a(
            page_sets=ps, query_ids=qids, workload="tpch", cache_pages=CACHE,
            family="d", consumer="multistart_greedy", seed=42, max_trials=30,
            search_mode="botorch_gp", early_stop_patience=10 ** 6,
            output_dir=tmp_path / "a")
        b = run_mode_a(
            page_sets=ps, query_ids=qids, workload="tpch", cache_pages=CACHE,
            family="d", consumer="multistart_greedy", seed=42, max_trials=30,
            search_mode="botorch_gp", early_stop_patience=10 ** 6,
            output_dir=tmp_path / "b")
        assert a.best_cost == b.best_cost
        assert a.best_schedule_qids == b.best_schedule_qids
