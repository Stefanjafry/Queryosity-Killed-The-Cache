"""
Candidate-pool analysis for the regret sweep: what do the 117 beam and 52
multistart candidates actually contain, and what would pruning cost?

Two modes:

  COUNTS (default, fast, construction only — no exact sims):
    Per config: pool size -> distinct schedules for beam and multistart,
    joint distinct, and which regret weights contribute novel schedules
    (scanning the grid in ascending order). A weight contributing zero
    novel schedules at a config is provably removable there with no
    effect on selection.

  EXACT (--exact, ~30-40 min total on the VM for all 9 configs):
    Additionally exact-simulates every DISTINCT candidate once and writes
    experiment_logs/candidate_pool.csv with one row per (config,
    candidate): provenance (source pool, weight, width/start), F_hit,
    and gap to the config winner. This is the evidence base for any
    pruning decision (epsilon-loss subsets, width necessity,
    leave-one-workload-out validation).

Usage (repo root):
    PYTHONHASHSEED=0 ./venv/bin/python -m src.bayesopt.analyze_candidate_pool
    PYTHONHASHSEED=0 ./venv/bin/python -m src.bayesopt.analyze_candidate_pool --exact
    ... --only tpch --caches 102400,262144        # filters for smoke runs

Read-only investigation tool: changes nothing about the production
selection. The pinned grid/widths/exclusions are imported from
sim_hit_grid so this always analyzes exactly the paper configuration.
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

from src.bayesopt.data import load_workload_pages
from src.bayesopt.objective import ExactSimObjective
from src.bayesopt.sim_hit_grid import BEAM_WIDTHS, CONFIGS, EXCLUDE, REGRET_GRID
from src.bayesopt.step_scorer import (
    ScorerWeights,
    beam_search_schedule,
    good_start_set,
    make_step_scorer,
    multistart_greedy_schedules,
    row_normalize,
)
from src.simulator.cache_simulator import compute_directional_matrix


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exact", action="store_true",
                   help="exact-simulate every distinct candidate; write CSV")
    p.add_argument("--only", default="",
                   help="comma-separated workload filter (e.g. tpch,job)")
    p.add_argument("--caches", default="",
                   help="comma-separated cache filter (e.g. 102400)")
    p.add_argument("--out", type=Path,
                   default=Path("experiment_logs/candidate_pool.csv"))
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    only = {w.strip() for w in args.only.split(",") if w.strip()}
    caches = {int(c) for c in args.caches.split(",") if c.strip()}

    rows: list[dict[str, object]] = []
    print(f"{'config':16s} {'beam':>10s} {'mstart':>10s} {'joint':>6s}"
          f"  novel-weights(beam pool, ascending scan)")

    for wl, dirn, cap in CONFIGS:
        if only and wl not in only:
            continue
        if caches and cap not in caches:
            continue

        wp = load_workload_pages(Path(dirn),
                                 exclude=EXCLUDE.get((wl, cap), set()))
        pc = [len(p) for p in wp.page_sets]
        D = compute_directional_matrix(wp.page_sets, cap)
        Dn = row_normalize(D)

        # Provenance-tagged pools. first_seen maps schedule -> provenance
        # of the candidate that first produced it (ascending weight scan).
        first_seen: dict[tuple[int, ...], tuple[str, float, int]] = {}
        beam_total = 0
        novel_per_weight: dict[float, int] = {}
        for w in REGRET_GRID:
            sc = make_step_scorer(Dn, ScorerWeights(w_regret=w), pc, cap)
            novel = 0
            for bw in BEAM_WIDTHS:
                for s in beam_search_schedule(sc, wp.n, pc, bw):
                    beam_total += 1
                    t = tuple(s)
                    if t not in first_seen:
                        first_seen[t] = ("beam", w, bw)
                        novel += 1
            if novel:
                novel_per_weight[w] = novel
        beam_distinct = len(first_seen)

        starts = good_start_set(Dn, pc, 4)
        ms_total = 0
        ms_seen: set[tuple[int, ...]] = set()
        ms_only: dict[tuple[int, ...], tuple[str, float, int]] = {}
        for w in REGRET_GRID:
            sc = make_step_scorer(Dn, ScorerWeights(w_regret=w), pc, cap)
            for si, s in enumerate(
                    multistart_greedy_schedules(sc, wp.n, pc, starts)):
                ms_total += 1
                t = tuple(s)
                ms_seen.add(t)
                if t not in first_seen and t not in ms_only:
                    ms_only[t] = ("mstart", w, si)
        joint = beam_distinct + len(ms_only)

        nw = " ".join(f"{w}:{n}" for w, n in novel_per_weight.items())
        print(f"{wl}@{cap:<9d} {beam_total:>4d}->{beam_distinct:<4d} "
              f"{ms_total:>4d}->{len(ms_seen):<4d} "
              f"{joint:>5d}  {nw}")

        if args.exact:
            obj = ExactSimObjective(wp.page_sets, cap, d_matrix=D)
            all_cands = {**first_seen, **ms_only}
            t0 = time.perf_counter()
            fhits: dict[tuple[int, ...], float] = {}
            for t in all_cands:
                m, _ = obj.evaluate_schedule(list(t))
                fhits[t] = m.hit_ratio
            elapsed = time.perf_counter() - t0
            best = max(fhits.values())
            for t, (src, w, extra) in all_cands.items():
                rows.append({
                    "workload": wl, "cache": cap, "source": src,
                    "w_regret": w,
                    "width_or_start": extra,
                    "f_hit": round(fhits[t], 6),
                    "gap_to_best": round(best - fhits[t], 6),
                    "is_winner": fhits[t] == best,
                })
            n_win = sum(1 for t in all_cands if fhits[t] == best)
            print(f"    exact: {len(all_cands)} distinct sims in "
                  f"{elapsed:.1f}s; best F_hit={best:.4f} "
                  f"achieved by {n_win} candidate(s)")

        del wp, pc, D, Dn, first_seen, ms_only
        gc.collect()

    if args.exact and rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
