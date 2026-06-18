"""
Mode A search runner: orchestrate scorer search and write deep logs.

Pipeline per run (one workload × cache × family × consumer):

1. Build features once.
2. Propose configs: warm-up first, then random-weight baseline, then
   neighbour refinement around the best-so-far, until a trial/eval
   budget or early-stop patience is hit.
3. For each config, the consumer builds candidate schedules; every
   candidate is scored by the exact simulator; the best candidate (the
   **parameter-trial winner**) defines the config's cost.
4. The incumbent is the best parameter-trial winner seen so far
   (deterministic objective ⇒ best-observed recommendation, no
   posterior mean).

All nine artifacts (run_config, feature_stats, parameter_trials,
candidate_schedules, edge_diagnostics, per_query_metrics, best_so_far,
timing_breakdown, summary) are written under
``<output_dir>/<run_id>/``.  Detailed per-edge / per-query logs cover
the trial winner only, unless ``log_all_candidates`` is set.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.bayesopt.mode_a.consumers import Candidate, build_candidates
from src.bayesopt.mode_a.features import FeatureTable, build_features
from src.bayesopt.mode_a.scorers import ScorerConfig, score_edge
from src.bayesopt.mode_a.search_space import (
    Proposal,
    neighbor_configs,
    random_config,
    warmup_configs,
)
from src.bayesopt.objective import simulate_schedule_page_level_traced
import random as _random
from uuid import uuid4


@dataclass
class _Eval:
    """Internal: one exact-sim evaluation of one candidate schedule."""

    schedule_eval_id: int
    parameter_trial_id: int
    source: str
    schedule: tuple[int, ...]
    hit_ratio: float
    cost: float
    total_hits: int
    total_requests: int
    page_reads: int


@dataclass
class ModeARunResult:
    """Summary of a completed Mode A run (also serialized to summary.json)."""

    run_id: str
    workload: str
    cache_pages: int
    family: str
    consumer: str
    seed: int
    best_schedule_qids: list[str] = field(default_factory=list)
    best_hit_ratio: float = 0.0
    best_cost: float = 1.0
    best_config: dict | None = None
    best_source: str = ""
    num_parameter_trials: int = 0
    num_exact_evals: int = 0
    early_stopped: bool = False
    stop_reason: str = ""
    refinement_ran: bool = False
    neighbor_configs_evaluated: int = 0
    bo_warmup: int = 0
    bo_init: int = 0
    bo_suggestions: int = 0
    output_dir: str = ""


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _matrix_offdiag(mat: list[list], n: int) -> list[float]:
    return [float(mat[i][j]) for i in range(n) for j in range(n) if i != j]


def _write_feature_stats(path: Path, feats: FeatureTable) -> None:
    n = feats.n
    survivor_all = _matrix_offdiag(feats.survivor_ratio, n)
    survivor_overlap = [
        feats.survivor_ratio[i][j]
        for i in range(n) for j in range(n)
        if i != j and feats.M[i][j] > 0
    ]
    zero_d_rows = sum(
        1 for i in range(n)
        if all(feats.D[i][j] == 0 for j in range(n) if j != i)
    )
    stats = {
        "n_queries": n,
        "cache_pages": feats.cache_pages,
        "pagecount": _summary_stats([float(c) for c in feats.pagecount]),
        "M_offdiag": _summary_stats(_matrix_offdiag(feats.M, n)),
        "D_offdiag": _summary_stats(_matrix_offdiag(feats.D, n)),
        "gap_raw_offdiag": _summary_stats(_matrix_offdiag(feats.gap_raw, n)),
        "survivor_ratio_all": _summary_stats(survivor_all),
        "survivor_ratio_overlap_only": _summary_stats(survivor_overlap),
        "asym_norm_offdiag": _summary_stats(_matrix_offdiag(feats.asym_norm, n)),
        "balance_D": _summary_stats(feats.balance_D),
        "zero_D_rows": zero_d_rows,
        "dle_violations": feats.dle_violations,
        "topk_values": list(feats.topk_values),
    }
    path.write_text(json.dumps(stats, indent=2))


def _config_dict(cfg: ScorerConfig) -> dict:
    return asdict(cfg)


class _Logger:
    """Owns the CSV/JSONL artifact writers for one run."""

    def __init__(self, outdir: Path) -> None:
        self.dir = outdir
        self.param_rows: list[dict] = []
        self.candidate_rows: list[dict] = []
        self.edge_rows: list[dict] = []
        self.perq_rows: list[dict] = []
        self.bsf_rows: list[dict] = []

    def _write_csv(self, name: str, rows: list[dict]) -> None:
        path = self.dir / name
        if not rows:
            path.write_text("")
            return
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    def flush(self) -> None:
        self._write_csv("parameter_trials.csv", self.param_rows)
        self._write_csv("candidate_schedules.csv", self.candidate_rows)
        self._write_csv("edge_diagnostics.csv", self.edge_rows)
        self._write_csv("per_query_metrics.csv", self.perq_rows)
        self._write_csv("best_so_far.csv", self.bsf_rows)


def _edge_rows_for(
    eval_id: int, cfg: ScorerConfig, feats: FeatureTable,
    schedule: tuple[int, ...], query_ids: list[str], consumer: str,
) -> list[dict]:
    rows = []
    for k in range(len(schedule) - 1):
        i, j = schedule[k], schedule[k + 1]
        rows.append({
            "schedule_eval_id": eval_id,
            "position": k,
            "prev_query": query_ids[i],
            "next_query": query_ids[j],
            "family": cfg.family,
            "consumer": consumer,
            "edge_score": round(score_edge(cfg, feats, i, j), 6),
            "M": feats.M[i][j],
            "D_ij": feats.D[i][j],
            "D_ji": feats.D[j][i],
            "M_norm": round(feats.M_norm[i][j], 6),
            "D_norm": round(feats.D_norm[i][j], 6),
            "asym_norm": round(feats.asym_norm[i][j], 6),
            "gap_raw": feats.gap_raw[i][j],
            "gap_norm": round(feats.gap_norm[i][j], 6),
            "survivor_ratio": round(feats.survivor_ratio[i][j], 6),
            "out_M_topK_j": round(feats.out_M_topk[cfg.topk][j], 6),
            "out_D_topK_j": round(feats.out_D_topk[cfg.topk][j], 6),
            "in_D_j": round(feats.in_D[j], 6),
            "balance_D_j": round(feats.balance_D[j], 6),
            "size_norm_j": round(feats.size_norm[j], 6),
            "cache_pressure_j": round(feats.cache_pressure[j], 6),
        })
    return rows


def _perq_rows_for(
    eval_id: int, schedule: tuple[int, ...], query_ids: list[str],
    page_sets: list[frozenset[int]], cache_pages: int,
) -> list[dict]:
    _, _, traces = simulate_schedule_page_level_traced(
        page_sets, list(schedule), cache_pages
    )
    rows = []
    cum_h = cum_m = cum_r = 0
    for t in traces:
        cum_h += t.hits
        cum_m += t.misses
        cum_r += t.requests
        rows.append({
            "schedule_eval_id": eval_id,
            "position": t.position,
            "query_id": query_ids[t.query_index],
            "hits": t.hits,
            "misses": t.misses,
            "requests": t.requests,
            "query_hit_ratio": round(t.hits / t.requests, 6) if t.requests else 0.0,
            "cum_hits": cum_h,
            "cum_misses": cum_m,
            "cum_requests": cum_r,
            "prefix_hit_ratio": round(cum_h / cum_r, 6) if cum_r else 0.0,
        })
    return rows


def run_mode_a(
    *,
    page_sets: list[frozenset[int]],
    query_ids: list[str],
    workload: str,
    cache_pages: int,
    family: str,
    consumer: str,
    seed: int = 42,
    max_trials: int = 200,
    max_schedule_evals: int = 100000,
    early_stop_patience: int = 15,
    num_starts: int = 5,
    beam_widths: tuple[int, ...] = (2, 4, 8),
    topk_values: tuple[int, ...] = (3, 5, 10),
    search_mode: str = "warmup_random_neighbor",
    output_dir: Path | None = None,
    run_id: str | None = None,
    log_all_candidates: bool = False,
) -> ModeARunResult:
    """
    Run one Mode A scorer search and write all artifacts.

    Parameters mirror the CLI; see module docstring for the pipeline.
    ``search_mode`` is one of:

    * ``"random_only"`` — random weights only, **no warm-up** (the true
      honesty floor: pure random sampling over the scorer space).
    * ``"warmup_random"`` — warm-up grid + random, **no refinement**.
    * ``"warmup_random_neighbor"`` — warm-up + random + neighbour
      refinement, with at least one forced refinement pass before early
      stopping can trigger (so the optimizer is always exercised).

    Returns
    -------
    ModeARunResult
        Best schedule/config and run metadata.
    """
    t_start = time.perf_counter()
    timing: dict[str, object] = {}

    run_id = run_id or (
        f"{workload}_c{cache_pages}_{family}_{consumer}_{search_mode}"
        f"_s{seed}_{int(t_start)}_{uuid4().hex[:6]}"
    )
    outdir = (output_dir or Path("experiment_logs/mode_a")) / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    feats = build_features(page_sets, cache_pages, topk_values=topk_values)
    timing["feature_construction"] = time.perf_counter() - t0
    _write_feature_stats(outdir / "feature_stats.json", feats)

    logger = _Logger(outdir)
    rng = _random.Random(seed)

    # The beam widths a single config explores: for the beam consumer we
    # treat each width as a separate candidate set merged per trial.
    def candidates_for(cfg: ScorerConfig) -> list[Candidate]:
        if consumer == "beam":
            out: list[Candidate] = []
            for w in beam_widths:
                out += [
                    Candidate(c.schedule, source=f"w{w}:{c.source}")
                    for c in build_candidates(
                        cfg, feats, "beam", beam_width=w
                    )
                ]
            return out
        return build_candidates(
            cfg, feats, "multistart_greedy", num_starts=num_starts
        )

    eval_id = 0
    trial_id = 0
    best_cost = float("inf")
    best_eval: _Eval | None = None
    best_config: ScorerConfig | None = None
    best_source = ""
    trials_since_improve = 0
    early_stopped = False
    warmup_done = False
    t_construct = 0.0
    t_sim = 0.0

    def evaluate_config(prop: Proposal) -> tuple[_Eval, list[_Eval], float, float]:
        """Build + sim all candidates; return winner, all evals, timings."""
        nonlocal eval_id
        tc = time.perf_counter()
        cands = candidates_for(prop.config)
        construct_s = time.perf_counter() - tc
        evals: list[_Eval] = []
        sim_s = 0.0
        seen_sched: set[tuple[int, ...]] = set()
        for cand in cands:
            if cand.schedule in seen_sched:
                continue
            seen_sched.add(cand.schedule)
            ts = time.perf_counter()
            req, hits, _ = simulate_schedule_page_level_traced(
                page_sets, list(cand.schedule), cache_pages
            )
            sim_s += time.perf_counter() - ts
            hr = hits / req if req else 0.0
            eval_id += 1
            evals.append(_Eval(
                schedule_eval_id=eval_id,
                parameter_trial_id=trial_id,
                source=cand.source,
                schedule=cand.schedule,
                hit_ratio=hr,
                cost=1.0 - hr,
                total_hits=hits,
                total_requests=req,
                page_reads=req - hits,
            ))
        winner = min(evals, key=lambda e: e.cost)
        return winner, evals, construct_s, sim_s

    def record_trial(prop: Proposal, winner: _Eval, evals: list[_Eval],
                     improved: bool, construct_s: float, sim_s: float) -> None:
        cfg = prop.config
        logger.param_rows.append({
            "parameter_trial_id": trial_id,
            "family": cfg.family,
            "consumer": consumer,
            "source": prop.source,
            "topk": cfg.topk,
            "lambda_out": cfg.lambda_out,
            "lambda_size": cfg.lambda_size,
            "lambda_balance": cfg.lambda_balance,
            "lambda_asym": cfg.lambda_asym,
            "lambda_ratio": cfg.lambda_ratio,
            "lambda_gap": cfg.lambda_gap,
            "seed": seed,
            "num_candidate_schedules": len(evals),
            "num_exact_evals_used": len(evals),
            "best_candidate_eval_id": winner.schedule_eval_id,
            "best_F_hit": round(winner.hit_ratio, 6),
            "best_cost": round(winner.cost, 6),
            "incumbent_F_hit": round(1 - best_cost, 6),
            "incumbent_cost": round(best_cost, 6),
            "improved": improved,
            "elapsed_s": round(time.perf_counter() - t_start, 4),
            "construct_s": round(construct_s, 6),
            "sim_s": round(sim_s, 6),
        })
        for e in evals:
            is_trial_winner = e.schedule_eval_id == winner.schedule_eval_id
            logger.candidate_rows.append({
                "schedule_eval_id": e.schedule_eval_id,
                "parameter_trial_id": e.parameter_trial_id,
                "source": e.source,
                "schedule_qids": "|".join(query_ids[i] for i in e.schedule),
                "F_hit": round(e.hit_ratio, 6),
                "cost": round(e.cost, 6),
                "total_hits": e.total_hits,
                "total_requests": e.total_requests,
                "page_reads": e.page_reads,
                "cache_pages": cache_pages,
                "is_trial_winner": is_trial_winner,
                "became_incumbent_this_trial": improved and is_trial_winner,
                "is_global_incumbent": False,  # set post-hoc to the final best
            })
        # Detailed logs: trial winner only, unless log_all_candidates.
        targets = evals if log_all_candidates else [winner]
        for e in targets:
            logger.edge_rows += _edge_rows_for(
                e.schedule_eval_id, cfg, feats, e.schedule,
                query_ids, consumer,
            )
            logger.perq_rows += _perq_rows_for(
                e.schedule_eval_id, e.schedule, query_ids,
                page_sets, cache_pages,
            )
        logger.bsf_rows.append({
            "parameter_trial_id": trial_id,
            "exact_evals_so_far": eval_id,
            "elapsed_s": round(time.perf_counter() - t_start, 4),
            "trial_best_F_hit": round(winner.hit_ratio, 6),
            "global_best_F_hit": round(1 - best_cost, 6),
            "global_best_cost": round(best_cost, 6),
            "source": prop.source,
            "family": cfg.family,
            "consumer": consumer,
        })

    last_trial_cost = [1.0]  # holder: cost of the most recent trial winner

    def process(prop: Proposal) -> None:
        nonlocal trial_id, best_cost, best_eval, best_config, best_source
        nonlocal trials_since_improve, t_construct, t_sim
        trial_id += 1
        winner, evals, construct_s, sim_s = evaluate_config(prop)
        last_trial_cost[0] = winner.cost
        t_construct += construct_s
        t_sim += sim_s
        improved = winner.cost < best_cost
        if improved:
            best_cost = winner.cost
            best_eval = winner
            best_config = prop.config
            best_source = prop.source
            trials_since_improve = 0
        elif warmup_done:
            trials_since_improve += 1
        record_trial(prop, winner, evals, improved, construct_s, sim_s)

    # Budget-independent random pool: a fixed, seed-determined sequence of
    # random configs.  Both short and long budgets consume a *prefix* of
    # this pool, so a longer budget evaluates a superset of a shorter
    # one's proposals (unless early stopping cuts it off first) — the
    # search is nested across budgets.  Pool size is generous; the trial
    # budget, not the pool, bounds how many are actually evaluated.
    RANDOM_POOL_SIZE = 1024
    random_pool = [
        random_config(family, topk_values, rng)
        for _ in range(RANDOM_POOL_SIZE)
    ]
    stop_reason = "budget_exhausted"
    refinement_ran = False
    neighbor_configs_evaluated = 0
    bo_counts = {"n_warmup": 0, "n_init": 0, "n_bo": 0}

    use_botorch = search_mode == "botorch_gp"
    use_warmup = search_mode in ("warmup_random", "warmup_random_neighbor")
    use_refine = search_mode == "warmup_random_neighbor"

    # --- Phase 1: warm-up (skipped in random_only — the true floor) ---
    t0 = time.perf_counter()
    if use_warmup and not use_botorch:
        for prop in warmup_configs(family, topk_values):
            if trial_id >= max_trials or eval_id >= max_schedule_evals:
                break
            process(prop)
    timing["warmup"] = time.perf_counter() - t0
    warmup_done = True

    # --- Phase 2: random-weight phase (prefix of the fixed pool) ---
    # full mode reserves budget for refinement (half); the other modes
    # spend the whole remaining budget on random.  Early stopping is
    # SUPPRESSED here in full mode so refinement is guaranteed at least
    # one pass — otherwise a lucky random draw could end the search
    # before the optimizer is ever exercised.
    t0 = time.perf_counter()
    pool_idx = 0
    if not use_botorch:
        if use_refine:
            random_cap = max(0, (max_trials - trial_id) // 2)
        else:
            random_cap = max_trials - trial_id
        for _ in range(random_cap):
            if eval_id >= max_schedule_evals or trial_id >= max_trials:
                break
            if pool_idx >= len(random_pool):
                break
            process(random_pool[pool_idx])
            pool_idx += 1
            # Early stop applies only when there is no refinement phase to
            # follow; in full mode we never early-stop during random.
            if not use_refine and trials_since_improve >= early_stop_patience:
                early_stopped = True
                stop_reason = "early_stop_random_phase"
                break
    timing["random_search"] = time.perf_counter() - t0

    # --- Phase 3: neighbour refinement around the incumbent ---
    # Full mode only.  At least one neighbour pass is forced (the random
    # phase does not early-stop), so refinement is always exercised when
    # budget allows; early stopping can trigger only from the second
    # pass onward.
    t0 = time.perf_counter()
    if use_refine and not use_botorch and best_config is not None:
        refine = True
        first_pass = True
        while (refine and trial_id < max_trials
               and eval_id < max_schedule_evals):
            assert best_config is not None
            anchor = best_config
            improved_in_pass = False
            for prop in neighbor_configs(anchor, topk_values):
                if trial_id >= max_trials or eval_id >= max_schedule_evals:
                    break
                refinement_ran = True
                neighbor_configs_evaluated += 1
                before = best_cost
                process(prop)
                if best_cost < before:
                    improved_in_pass = True
                # Suppress early stop during the first forced pass.
                if (not first_pass
                        and trials_since_improve >= early_stop_patience):
                    early_stopped = True
                    stop_reason = "early_stop_refinement"
                    break
            first_pass = False
            if early_stopped or not improved_in_pass or best_config is anchor:
                if not early_stopped:
                    stop_reason = "refinement_converged"
                refine = False
    timing["neighbor_refinement"] = time.perf_counter() - t0
    timing["refinement_ran"] = refinement_ran
    timing["neighbor_configs_evaluated"] = neighbor_configs_evaluated

    # --- Phase 4: BoTorch GP loop (botorch_gp mode only) ---
    # warm-up anchors + Sobol initial design + GP/LogEI acquisition.
    # Early stopping is suppressed until ALL warm-ups, ALL init points,
    # and at least `bo_min_suggestions` acquisition steps have run, so a
    # real BO attempt is always made before any cutoff.
    t0 = time.perf_counter()
    if use_botorch:
        from src.bayesopt.mode_a.botorch_backend import propose_botorch

        # Budget split: ~1/3 init, the rest acquisition, capped by trials.
        remaining = max_trials - trial_id
        n_warmup_expected = len(warmup_configs(family, topk_values))
        budget_after_warmup = max(0, remaining - n_warmup_expected)
        n_init = max(6, budget_after_warmup // 3)
        n_bo = max(0, budget_after_warmup - n_init)
        bo_min_suggestions = min(n_bo, 10)

        def _should_stop() -> bool:
            # Hard caps only; early-stop patience is enforced separately
            # after the forced minimum of BO suggestions.
            if eval_id >= max_schedule_evals or trial_id >= max_trials:
                return True
            if (bo_counts["n_bo"] >= bo_min_suggestions
                    and trials_since_improve >= early_stop_patience):
                return True
            return False

        def _evaluate_real(cfg: ScorerConfig, source: str) -> float:
            # Evaluate THIS config (records the trial) and return its own
            # trial-winner cost — not the incumbent's — so the GP learns
            # the realized response surface.
            process(Proposal(cfg, source))
            return last_trial_cost[0]

        bo_counts = propose_botorch(
            family=family,
            topk_values=topk_values,
            evaluate=_evaluate_real,
            n_init=n_init,
            n_bo=n_bo,
            seed=seed,
            should_stop=_should_stop,
        )
        if (bo_counts["n_bo"] >= bo_min_suggestions
                and trials_since_improve >= early_stop_patience):
            early_stopped = True
            stop_reason = "early_stop_botorch"
        else:
            stop_reason = "botorch_budget_exhausted"
    timing["botorch"] = time.perf_counter() - t0
    timing["bo_counts"] = bo_counts

    timing["schedule_construction"] = t_construct
    timing["exact_simulation"] = t_sim
    timing["total_wall"] = time.perf_counter() - t_start
    timing["time_to_best_trial"] = (
        best_eval.parameter_trial_id if best_eval else 0
    )
    timing["time_to_best_eval"] = (
        best_eval.schedule_eval_id if best_eval else 0
    )
    timing["early_stopped"] = early_stopped
    timing["stop_reason"] = stop_reason
    timing["parameter_trials_run"] = trial_id
    timing["exact_evals_run"] = eval_id

    # --- Write remaining artifacts ---
    t0 = time.perf_counter()
    if best_eval is not None:
        for row in logger.candidate_rows:
            if row["schedule_eval_id"] == best_eval.schedule_eval_id:
                row["is_global_incumbent"] = True
                break
    logger.flush()
    timing["logging"] = time.perf_counter() - t0
    (outdir / "timing_breakdown.json").write_text(json.dumps(timing, indent=2))

    run_config = {
        "run_id": run_id,
        "workload": workload,
        "cache_pages": cache_pages,
        "family": family,
        "consumer": consumer,
        "seed": seed,
        "topk_values": list(topk_values),
        "beam_widths": list(beam_widths),
        "num_starts": num_starts,
        "lambda_ranges": {
            "lambda_out": [0.0, 2.0], "lambda_size": [0.0, 2.0],
            "lambda_balance": [0.0, 2.0], "lambda_asym": [0.0, 2.0],
            "lambda_ratio": [0.0, 2.0], "lambda_gap": [-2.0, 2.0],
        },
        "early_stop_patience": early_stop_patience,
        "max_trials": max_trials,
        "max_schedule_evals": max_schedule_evals,
        "search_mode": search_mode,
        "log_all_candidates": log_all_candidates,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
    }
    (outdir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    assert best_eval is not None and best_config is not None
    result = ModeARunResult(
        run_id=run_id,
        workload=workload,
        cache_pages=cache_pages,
        family=family,
        consumer=consumer,
        seed=seed,
        best_schedule_qids=[query_ids[i] for i in best_eval.schedule],
        best_hit_ratio=best_eval.hit_ratio,
        best_cost=best_eval.cost,
        best_config=_config_dict(best_config),
        best_source=best_source,
        num_parameter_trials=trial_id,
        num_exact_evals=eval_id,
        early_stopped=early_stopped,
        stop_reason=stop_reason,
        refinement_ran=refinement_ran,
        neighbor_configs_evaluated=neighbor_configs_evaluated,
        bo_warmup=bo_counts["n_warmup"],
        bo_init=bo_counts["n_init"],
        bo_suggestions=bo_counts["n_bo"],
        output_dir=str(outdir),
    )
    (outdir / "summary.json").write_text(json.dumps(asdict(result), indent=2))
    return result


__all__ = ["ModeARunResult", "run_mode_a"]
