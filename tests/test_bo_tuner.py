"""Tests for the BO tuning harness."""

from __future__ import annotations

from src.bayesopt.bo_tuner import (
    DEFAULT_SPACE,
    run_search,
    warmup_vectors,
    weights_from_vec,
)
from src.bayesopt.step_scorer import ScorerWeights


def test_warmup_includes_greedy_equiv_and_each_term() -> None:
    vs = warmup_vectors()
    # greedy-equivalent: all aux weights zero
    assert vs[0]["w_regret"] == 0.0 and vs[0]["w_cache"] == 0.0
    assert vs[0]["w_connector"] == 0.0
    # one seed activates each term, plus a full seed
    activated = {k for v in vs for k in ("w_regret", "w_cache", "w_connector")
                 if v[k] > 0}
    assert activated == {"w_regret", "w_cache", "w_connector"}


def test_weights_anchor_immediate_at_one() -> None:
    w = weights_from_vec({"w_regret": 0.3, "w_cache": 0.2, "w_connector": 0.1,
                          "connector_rho": 0.25}, "fit")
    assert isinstance(w, ScorerWeights)
    assert w.w_immediate == 1.0
    assert w.w_regret == 0.3


def test_eval_counting_respects_budget_and_K() -> None:
    calls = {"n": 0}

    def objective(w: ScorerWeights) -> tuple[float, int]:
        calls["n"] += 1
        return 0.5, 4  # consumer spends K=4 evals per trial

    res = run_search(objective, "random", eval_budget=20, seed=1)
    assert res.n_evals >= 20
    # budget is in evals, so ~ ceil(20/4) trials, not 20 trials
    assert res.n_trials <= 7


def test_greedy_equiv_is_floor_best_cannot_be_below_warmup() -> None:
    # objective rewards exactly the greedy-equiv point (all aux = 0).
    def objective(w: ScorerWeights) -> tuple[float, int]:
        penalty = w.w_regret + w.w_cache + w.w_connector
        return 0.9 - 0.1 * penalty, 1

    res = run_search(objective, "neighbor", eval_budget=40, seed=2, patience=50)
    assert abs(res.best_fhit - 0.9) < 1e-9  # found the aux=0 optimum
    assert res.best_vec["w_regret"] < 1e-6


def test_neighbor_finds_an_interior_optimum_random_may_miss() -> None:
    # optimum at w_regret=1.0; neighbor refinement should home in.
    def objective(w: ScorerWeights) -> tuple[float, int]:
        return 1.0 - (w.w_regret - 1.0) ** 2, 1

    res = run_search(objective, "neighbor", eval_budget=80, seed=3, patience=100)
    assert res.best_fhit > 0.98
    assert abs(res.best_vec["w_regret"] - 1.0) < 0.1


def test_different_seeds_explore_different_points() -> None:
    # random backend must actually depend on the seed
    def objective(w: ScorerWeights) -> tuple[float, int]:
        return 0.5 - 0.01 * w.w_regret, 1  # smooth, non-degenerate

    r42 = run_search(objective, "random", eval_budget=30, seed=42)
    r43 = run_search(objective, "random", eval_budget=30, seed=43)
    # post-warmup sampled vectors should differ between seeds
    s42 = [t.vec["w_regret"] for t in r42.history if not t.is_warmup]
    s43 = [t.vec["w_regret"] for t in r43.history if not t.is_warmup]
    assert s42 != s43
    # but the same seed reproduces exactly
    r42b = run_search(objective, "random", eval_budget=30, seed=42)
    s42b = [t.vec["w_regret"] for t in r42b.history if not t.is_warmup]
    assert s42 == s42b


def test_patience_early_stops_and_logs_reason() -> None:
    def flat(w: ScorerWeights) -> tuple[float, int]:
        return 0.5, 1  # never improves after first

    res = run_search(flat, "neighbor", eval_budget=10_000, seed=4, patience=10)
    assert res.early_stop_reason == "patience"
    assert res.n_evals < 10_000  # stopped well before budget


def test_convergence_tracks_best_and_origin() -> None:
    from src.bayesopt.bo_tuner import convergence_of, history_records

    # objective peaks at w_regret=1; warm-up won't hit it, search should
    def objective(w: ScorerWeights) -> tuple[float, int]:
        return 1.0 - (w.w_regret - 1.0) ** 2, 1

    res = run_search(objective, "neighbor", eval_budget=60, seed=7, patience=100)
    conv = convergence_of(res)
    # best_so_far is monotonic non-decreasing and ends at the best
    assert conv.best_so_far == sorted(conv.best_so_far)
    assert abs(conv.best_so_far[-1] - res.best_fhit) < 1e-9
    # the winning vector leans on w_regret
    assert conv.dominant_term == "w_regret"
    # history has one record per trial with best_so_far populated
    recs = history_records(res)
    assert len(recs) == res.n_trials
    assert recs[-1]["best_so_far"] == res.best_fhit


def test_convergence_flags_warmup_origin_when_warmup_is_best() -> None:
    from src.bayesopt.bo_tuner import convergence_of

    # greedy-equiv (all aux 0) is optimal -> best comes from warm-up
    def objective(w: ScorerWeights) -> tuple[float, int]:
        return 0.9 - 0.1 * (w.w_regret + w.w_cache + w.w_connector), 1

    res = run_search(objective, "random", eval_budget=40, seed=8)
    conv = convergence_of(res)
    assert conv.best_from_warmup is True
    assert conv.dominant_term == "none(greedy-equiv)"
