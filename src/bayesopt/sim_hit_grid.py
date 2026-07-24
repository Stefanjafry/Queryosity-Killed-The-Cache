"""
Simulated cache-hit ratio (F_hit) grid for all methods x all configs.

Uses the exact clock-sweep simulator only (no PostgreSQL). For each
(workload, cache) it builds the schedule each method commits to and reports
its exact-simulator F_hit, so you can count how often each directional method
beats GA_M. Deterministic; no seed sensitivity for the sweep methods.

Run from the repo root:
    ./venv/bin/python -m src.bayesopt.sim_hit_grid
Set QKC_WORKERS to parallelise exact-simulation selection, e.g.
``QKC_WORKERS=4``; 0 means all cores. Selection only — the schedules and
F_hit values are identical at any worker count.
"""

import csv
import os
from pathlib import Path

from src.bayesopt.data import load_workload_pages
from src.bayesopt.objective import ExactSimObjective
from src.bayesopt.run_baselines import compute_baseline_schedule
from src.bayesopt.selection import resolve_workers, score_pool
from src.bayesopt.step_scorer import (
    ScorerWeights, beam_search_schedule, good_start_set, make_step_scorer,
    multistart_greedy_schedules, row_normalize,
)
from src.simulator.cache_simulator import (
    compute_directional_matrix, compute_overlap_matrix,
)

# (workload, page_access_dir, cache_pages) — matches the runtime study.
CONFIGS = [
    ("tpch", "page_access/tpch", 102400),
    ("tpch", "page_access/tpch", 262144),
    ("tpch", "page_access/tpch", 524288),
    ("tpcds", "page_access/tpcds", 102400),
    ("tpcds", "page_access/tpcds", 262144),
    ("tpcds", "page_access/tpcds", 524288),
    ("job", "page_access/job", 102400),
    ("job", "page_access/job", 262144),
    ("job", "page_access/job", 524288),
]

# Same exclusions as the runtime comparison (so the sim grid matches).
# Note: sets differ across cache sizes within a workload — mirroring the
# runtime study — so cross-cache trends carry that caveat; per-cell method
# comparisons are unaffected (all methods share the set).
EXCLUDE = {
    ("tpch", 102400): {"q5", "q13"},
    ("tpch", 262144): {"q5"},
    ("tpch", 524288): {"q5"},
    ("tpcds", 102400): {"query17", "query25", "query29"},
    ("tpcds", 262144): {"query17", "query25", "query29"},
    ("tpcds", 524288): {"query25"},
}

REGRET_GRID = [round(0.1 * i, 1) for i in range(11)] + [1.5, 2.0]
BEAM_WIDTHS = (2, 3, 4)
GA_POP, GA_GEN, SEED = 100, 200, 42


def best_by_fhit(scheds, fhit):
    best_s, best_f = None, -1.0
    for s in scheds:
        f = fhit(s)
        if f > best_f:
            best_f, best_s = f, s
    return best_s, best_f


def best_pooled(scheds, page_sets, cap, D, workers):
    """Dedup + (optionally parallel) exact-score; strict > keeps build order."""
    cands = [tuple(s) for s in scheds]
    scores, n_distinct = score_pool(cands, page_sets, cap, D, workers)
    best_s, best_f = None, -1.0
    for c, f in zip(cands, scores):
        if f > best_f:
            best_f, best_s = f, list(c)
    return best_s, best_f, n_distinct


def main() -> None:
    workers = resolve_workers(int(os.environ.get("QKC_WORKERS", "1")))
    if workers > 1:
        print(f"selection workers: {workers}")
    rows_out = []
    print(f"{'workload':7s} {'cache':>7s} {'n':>4s} | "
          f"{'GA_M':>7s} {'GA_D':>7s} {'sweep_D':>8s} {'sweepBeam':>9s} | "
          f"{'D beats M?':>10s}")
    print("-" * 76)

    win_system = 0   # best directional method beats GA_M
    win_gad = 0      # GA_D alone beats GA_M

    for wl, dirn, cap in CONFIGS:
        exclude = EXCLUDE.get((wl, cap), set())
        wp = load_workload_pages(Path(dirn), exclude=exclude)
        pc = [len(ps) for ps in wp.page_sets]
        ints = [frozenset(ps) for ps in wp.page_sets]
        n = wp.n

        M = compute_overlap_matrix(wp.page_sets)
        D = compute_directional_matrix(wp.page_sets, cap)
        Dn = row_normalize(D)
        obj = ExactSimObjective(wp.page_sets, cap, d_matrix=D)

        def fhit(s, obj=obj):
            m, _ = obj.evaluate_schedule(s)
            return m.hit_ratio

        # GA_M and GA_D
        ga_m = fhit(compute_baseline_schedule("ga_m", ints, wp.query_ids, D,
                                              cap, SEED, GA_POP, GA_GEN))
        ga_d = fhit(compute_baseline_schedule("ga_d", ints, wp.query_ids, D,
                                              cap, SEED, GA_POP, GA_GEN))

        # sweep_D (multistart) and sweep_beam_D — independent consumers,
        # each deduplicated and exact-scored over its OWN candidate pool.
        starts = good_start_set(Dn, pc, 4)
        _, sweep_d, nd_ms = best_pooled(
            [s for wr in REGRET_GRID
             for s in multistart_greedy_schedules(
                 make_step_scorer(Dn, ScorerWeights(w_regret=wr), pc, cap),
                 n, pc, starts)],
            wp.page_sets, cap, D, workers)
        _, sweep_beam, nd_beam = best_pooled(
            [s for wr in REGRET_GRID for bw in BEAM_WIDTHS
             for s in beam_search_schedule(
                 make_step_scorer(Dn, ScorerWeights(w_regret=wr), pc, cap),
                 n, pc, bw)],
            wp.page_sets, cap, D, workers)

        best_dir = max(ga_d, sweep_d, sweep_beam)
        beats = best_dir > ga_m
        win_system += int(beats)
        win_gad += int(ga_d > ga_m)

        print(f"{wl:7s} {cap:>7d} {n:>4d} | "
              f"{ga_m:7.4f} {ga_d:7.4f} {sweep_d:8.4f} {sweep_beam:9.4f} | "
              f"{'YES' if beats else 'no':>10s}")
        rows_out.append({
            "workload": wl, "cache": cap, "n": n,
            "GA_M": round(ga_m, 4), "GA_D": round(ga_d, 4),
            "sweep_D": round(sweep_d, 4), "sweep_beam_D": round(sweep_beam, 4),
            "sims_sweep_D": nd_ms, "sims_sweep_beam_D": nd_beam,
            "best_directional_beats_GA_M": beats,
        })

    print("-" * 76)
    print(f"System (best directional) beats GA_M in {win_system} of {len(CONFIGS)} configs.")
    print(f"GA_D alone beats GA_M in {win_gad} of {len(CONFIGS)} configs.")

    out = Path("experiment_logs/sim_hit_grid.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
