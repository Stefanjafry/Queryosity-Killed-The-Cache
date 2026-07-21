"""
End-to-end scheduling cost: Sweep+Beam-D vs GA_M.

Answers three questions, all Sweep_beam_D vs GA_M only:

  Q1  END-TO-END search cost as currently implemented.
        GA_M      = full evolutionary search (cheap approximate fitness)
                    + 1 final exact-sim validation.
        sweep_beam= construction of all candidates
                    + exact-sim scoring of every candidate (current selection).
      This is the honest "time to a committed, exact-scored schedule".

  Q2  FRAMING B identity check.
        Framing B = rank candidates by the CHEAP step-score, exact-sim ONLY
        the winner (like GA ranks by approximate fitness, exact-sims once).
        Does Framing B pick the SAME final schedule (exact query order) as the
        current exact-sim-selects-all method? If not, Framing B is a different
        (weaker-selecting) method and must be retired for honesty.

  Q3  APPLICATION deployment cost with GA hyperparameter tuning.
        GA has 6 knobs someone must choose. Deploying GA on a new workload
        realistically costs N full GA searches (the tuning sweep) + 1 final.
        Reported for N in {1 (untuned), 10, 27 (small grid)} against the
        sweep, which has no tuning phase.

Usage:
    python -m src.bayesopt.bench_end_to_end \\
        --workload tpch --cache-pages 262144 \\
        --page-access-dir page_access/tpch --reps 3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.bayesopt.data import load_workload_pages
from src.bayesopt.objective import ExactSimObjective
from src.bayesopt.run_baselines import compute_baseline_schedule
from src.bayesopt.step_scorer import (
    ScorerWeights,
    beam_search_schedule,
    make_step_scorer,
    row_normalize,
)
from src.simulator.cache_simulator import (
    compute_directional_matrix,
    compute_overlap_matrix,
)
from src.utilities.constants import WORKLOAD_DIRS

DEFAULT_REGRET_GRID = [round(0.1 * i, 1) for i in range(11)] + [1.5, 2.0]
DEFAULT_BEAM_WIDTHS = (2, 3, 4)


def _mean_std(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, v ** 0.5


def _fmt(x: float) -> str:
    return f"{x * 1000:.1f} ms" if x < 1 else f"{x:.2f} s"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=list(WORKLOAD_DIRS), required=True)
    p.add_argument("--cache-pages", type=int, required=True)
    p.add_argument("--page-access-dir", type=Path, required=True)
    p.add_argument("--exclude", default="")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ga-pop", type=int, default=100)
    p.add_argument("--ga-gen", type=int, default=200)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}

    wp = load_workload_pages(args.page_access_dir, exclude=exclude)
    pc = [len(ps) for ps in wp.page_sets]
    page_sets_int = [frozenset(ps) for ps in wp.page_sets]
    n = wp.n
    cap = args.cache_pages
    grid = DEFAULT_REGRET_GRID

    D = compute_directional_matrix(wp.page_sets, cap)
    _ = compute_overlap_matrix(wp.page_sets)  # M not needed beyond parity
    D_norm = row_normalize(D)
    obj = ExactSimObjective(wp.page_sets, cap, d_matrix=D)

    def exact_fhit(sched: list[int]) -> float:
        m, _ = obj.evaluate_schedule(sched)
        return m.hit_ratio

    def cheap_score(sched: list[int]) -> float:
        # cheap surrogate: sum of consecutive directional reuse along the order
        # (the same immediate signal the scorer uses; no clock-sweep sim).
        return sum(D_norm[sched[i]][sched[i + 1]] for i in range(len(sched) - 1))

    print(f"END-TO-END  Sweep+Beam-D vs GA_M  "
          f"(workload={args.workload}, cache={cap:,}, n={n}, reps={args.reps})")
    print("=" * 68)

    # ---------- build the beam candidate pool once (schedules only) ----------
    def build_candidates() -> list[list[int]]:
        cands = []
        for wr in grid:
            sc = make_step_scorer(D_norm, ScorerWeights(w_regret=wr), pc, cap)
            for bw in DEFAULT_BEAM_WIDTHS:
                cands.extend(beam_search_schedule(sc, n, pc, bw))
        return cands

    # ============ Q1: end-to-end search cost, as implemented ============
    ga_times, sweep_cur_times = [], []
    cands: list[list[int]] = []
    for _rep in range(args.reps):
        t0 = time.perf_counter()
        gsched = compute_baseline_schedule("ga_m", page_sets_int, wp.query_ids,
                                           D, cap, args.seed,
                                           args.ga_pop, args.ga_gen)
        exact_fhit(gsched)  # 1 final validation, as the paper does
        ga_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        cands = build_candidates()
        best_f, _bcur = -1.0, None
        for c in cands:
            f = exact_fhit(c)          # current selection: exact-sim ALL
            if f > best_f:
                best_f, _bcur = f, c
        sweep_cur_times.append(time.perf_counter() - t0)

    ga_m, ga_s = _mean_std(ga_times)
    sc_m, sc_s = _mean_std(sweep_cur_times)
    print("\nQ1  End-to-end (current implementation)")
    print(f"    GA_M            : {_fmt(ga_m)}  (±{_fmt(ga_s)})   "
          f"[search + 1 exact-sim]")
    print(f"    Sweep+Beam-D    : {_fmt(sc_m)}  (±{_fmt(sc_s)})   "
          f"[build + exact-sim ALL {len(cands)} candidates]")
    if sc_m > ga_m:
        print(f"    -> sweep is {sc_m / ga_m:.1f}x SLOWER end-to-end "
              f"(exact-sim-all selection is the cost)")
    else:
        print(f"    -> sweep is {ga_m / sc_m:.1f}x faster end-to-end")

    # ============ Q2: Framing B identity check ============
    # current method: winner by exact-sim
    best_f, win_exact = -1.0, cands[0]
    for c in cands:
        f = exact_fhit(c)
        if f > best_f:
            best_f, win_exact = f, c
    # framing B: winner by cheap score, exact-sim only that one
    best_c, win_cheap = -1e18, cands[0]
    for c in cands:
        s = cheap_score(c)
        if s > best_c:
            best_c, win_cheap = s, c
    identical = win_exact == win_cheap
    print("\nQ2  Framing B identity (rank cheap, exact-sim only winner)")
    print(f"    exact-selected winner F_hit : {best_f:.4f}")
    print(f"    cheap-selected winner F_hit : {exact_fhit(win_cheap):.4f}")
    print(f"    SAME EXACT SCHEDULE?         : "
          f"{'YES (Framing B is free)' if identical else 'NO -> retire Framing B (dishonest for our runtime)'}")

    # ============ Q3: application deployment cost with GA tuning ============
    # one GA search time (no final validation, that is amortized once)
    one_ga = ga_m
    # sweep end-to-end (current) is sc_m; sweep has NO tuning phase
    print("\nQ3  Application deployment cost (GA needs hyperparameter tuning)")
    print(f"    one GA search        : {_fmt(one_ga)}")
    print(f"    Sweep+Beam-D (total) : {_fmt(sc_m)}   [no tuning phase]")
    for N in (1, 10, 27):
        ga_deploy = N * one_ga + one_ga  # N tuning runs + 1 final run
        ratio = ga_deploy / sc_m
        tag = "untuned" if N == 1 else f"{N}-combo grid"
        print(f"    GA deploy (N={N:<2d} {tag:12s}): {_fmt(ga_deploy):>10s}   "
              f"= {ratio:.1f}x the sweep")


if __name__ == "__main__":
    main()
