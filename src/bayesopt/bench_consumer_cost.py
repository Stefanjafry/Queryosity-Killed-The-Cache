"""
Per-consumer scheduling cost: sweep_D and sweep_beam_D, measured independently.

The two sweep consumers are SEPARATE schedulers, not a pooled ensemble.
sweep_D's candidate pool is multistart-greedy only; sweep_beam_D's is beam
only. Each deduplicates and selects its own winner by exact simulation over
its own pool. This benchmark therefore never pools them and never uses one
consumer's result to justify a configuration choice in the other.

Measurement hygiene
-------------------
* A FRESH ``ExactSimObjective`` is constructed for every measurement, so no
  consumer ever inherits another's memo cache. Running both consumers in one
  process without this silently credits whichever runs second with the
  other's cached simulations.
* Candidates are deduplicated by schedule tuple BEFORE dispatch. Memoization
  is per-process, so parallel workers would otherwise re-simulate duplicates.
* Winner selection uses strict ``>`` over the deterministic construction
  order, so the selected schedule is identical regardless of worker count or
  completion order.
* The shared directional-matrix build is timed and reported separately
  (excluded from consumer totals, matching the Figure-15 framing).

Reported per consumer: construction time, raw and distinct candidate counts,
selection time at each worker count, total, and peak RSS.

GA_M is measured as evolutionary search plus one final exact-sim validation,
single-core, because the reference implementation is serial. Multi-worker
numbers for the sweep consumers are an ENGINEERING optimization, not an
algorithmic advantage: a GA's per-generation evaluations are independent and
parallelizable in principle too. Report both columns and say so.

Usage (repo root):
    PYTHONHASHSEED=0 ./venv/bin/python -m src.bayesopt.bench_consumer_cost \\
        --workload tpch --cache-pages 262144 \\
        --page-access-dir page_access/tpch --workers 1,2,4 --ga
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import resource
import time
from pathlib import Path

from src.bayesopt.data import load_workload_pages
from src.bayesopt.objective import ExactSimObjective
from src.bayesopt.run_baselines import compute_baseline_schedule
from src.bayesopt.sim_hit_grid import BEAM_WIDTHS, EXCLUDE, REGRET_GRID
from src.bayesopt.step_scorer import (
    ScorerWeights,
    beam_search_schedule,
    good_start_set,
    make_step_scorer,
    multistart_greedy_schedules,
    row_normalize,
)
from src.simulator.cache_simulator import compute_directional_matrix

Sched = tuple[int, ...]

_OBJ: ExactSimObjective | None = None


def _init_worker(page_sets: list[frozenset[int]], cap: int,
                 d_matrix: list[list[int]]) -> None:
    """Build one fresh objective per worker process (fork shares page_sets COW)."""
    global _OBJ
    _OBJ = ExactSimObjective(page_sets, cap, d_matrix=d_matrix)


def _score(sched: Sched) -> float:
    assert _OBJ is not None
    m, _ = _OBJ.evaluate_schedule(list(sched))
    return m.hit_ratio


def _fmt(x: float) -> str:
    return f"{x * 1000:.0f} ms" if x < 1 else f"{x:.2f} s"


def _peak_mb() -> float:
    s = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(s, c) / 1024.0  # ru_maxrss is KiB on Linux


def build_beam(D_norm: list[list[float]], n: int, pc: list[int],
               cap: int) -> list[Sched]:
    """sweep_beam_D candidate pool: every regret weight x every beam width."""
    out: list[Sched] = []
    for w in REGRET_GRID:
        sc = make_step_scorer(D_norm, ScorerWeights(w_regret=w), pc, cap)
        for bw in BEAM_WIDTHS:
            out.extend(tuple(s) for s in beam_search_schedule(sc, n, pc, bw))
    return out


def build_multistart(D_norm: list[list[float]], n: int, pc: list[int],
                     cap: int, k: int) -> list[Sched]:
    """sweep_D candidate pool: every regret weight x K greedy starts."""
    starts = good_start_set(D_norm, pc, k)
    out: list[Sched] = []
    for w in REGRET_GRID:
        sc = make_step_scorer(D_norm, ScorerWeights(w_regret=w), pc, cap)
        out.extend(tuple(s) for s in
                   multistart_greedy_schedules(sc, n, pc, starts))
    return out


def dedup(cands: list[Sched]) -> list[Sched]:
    """Order-preserving dedup; construction order fixes deterministic ties."""
    seen: set[Sched] = set()
    out: list[Sched] = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def select(distinct: list[Sched], page_sets: list[frozenset[int]], cap: int,
           d_matrix: list[list[int]], workers: int) -> tuple[Sched, float, float]:
    """
    Exact-simulate every distinct candidate; return (winner, F_hit, seconds).

    Fresh objective per call. Strict ``>`` over construction order means the
    winner is worker-count independent.
    """
    t0 = time.perf_counter()
    if workers == 1:
        obj = ExactSimObjective(page_sets, cap, d_matrix=d_matrix)
        scores = [obj.evaluate_schedule(list(s))[0].hit_ratio for s in distinct]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_init_worker,
                      initargs=(page_sets, cap, d_matrix)) as pool:
            scores = pool.map(_score, distinct, chunksize=1)
    elapsed = time.perf_counter() - t0

    best_i, best_f = 0, -1.0
    for i, f in enumerate(scores):
        if f > best_f:          # strict: earliest construction order wins ties
            best_f, best_i = f, i
    return distinct[best_i], best_f, elapsed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", required=True)
    p.add_argument("--cache-pages", type=int, required=True)
    p.add_argument("--page-access-dir", type=Path, required=True)
    p.add_argument("--exclude", default="",
                   help="blank = use the pinned per-config set from sim_hit_grid")
    p.add_argument("--workers", default="1,2,4")
    p.add_argument("--num-starts", type=int, default=4)
    p.add_argument("--ga", action="store_true", help="also measure GA_M (slow)")
    p.add_argument("--ga-pop", type=int, default=100)
    p.add_argument("--ga-gen", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cap = args.cache_pages
    workers = [int(w) for w in args.workers.split(",") if w.strip()]

    exclude = ({q.strip() for q in args.exclude.split(",") if q.strip()}
               if args.exclude else EXCLUDE.get((args.workload, cap), set()))

    wp = load_workload_pages(args.page_access_dir, exclude=exclude)
    pc = [len(p) for p in wp.page_sets]
    n = wp.n

    t0 = time.perf_counter()
    D = compute_directional_matrix(wp.page_sets, cap)
    D_norm = row_normalize(D)
    matrix_s = time.perf_counter() - t0

    print(f"PER-CONSUMER SCHEDULING COST  workload={args.workload} "
          f"cache={cap:,} n={n} excluded={sorted(exclude) or 'none'}")
    print(f"shared directional-matrix build: {_fmt(matrix_s)} "
          f"(excluded from consumer totals)")
    print("=" * 76)

    consumers = [
        ("sweep_D      (multistart)", lambda: build_multistart(
            D_norm, n, pc, cap, args.num_starts)),
        ("sweep_beam_D (beam)      ", lambda: build_beam(D_norm, n, pc, cap)),
    ]

    for label, builder in consumers:
        t0 = time.perf_counter()
        raw = builder()
        construct_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        distinct = dedup(raw)
        dedup_s = time.perf_counter() - t0

        print(f"\n{label}")
        print(f"  construction : {_fmt(construct_s)}   candidates "
              f"{len(raw)} raw -> {len(distinct)} distinct "
              f"({100 * (1 - len(distinct) / len(raw)):.0f}% dedup saving)")

        ref_sched, ref_f, base_s = None, None, None
        for w in workers:
            sched, f, sel_s = select(distinct, wp.page_sets, cap, D, w)
            if ref_sched is None:
                ref_sched, ref_f, base_s = sched, f, sel_s
            same = (sched == ref_sched)
            assert same, f"winner changed at {w} workers — determinism broken"
            assert base_s is not None
            total = construct_s + dedup_s + sel_s
            print(f"  selection W={w}: {_fmt(sel_s):>9s}  speedup "
                  f"{base_s / sel_s:>4.2f}x  eff {100 * base_s / (w * sel_s):>3.0f}%"
                  f"   TOTAL {_fmt(total):>9s}")
        assert ref_f is not None
        print(f"  winner F_hit : {ref_f:.4f}   (identical at every worker count)")
        print(f"  peak RSS     : {_peak_mb():.0f} MB")

    if args.ga:
        print(f"\nGA_M (reference baseline, serial implementation)")
        t0 = time.perf_counter()
        ints = [frozenset(p) for p in wp.page_sets]
        g = compute_baseline_schedule("ga_m", ints, wp.query_ids, D, cap,
                                      args.seed, args.ga_pop, args.ga_gen)
        obj = ExactSimObjective(wp.page_sets, cap, d_matrix=D)
        gf, _ = obj.evaluate_schedule(list(g))
        ga_s = time.perf_counter() - t0
        print(f"  search + 1 exact-sim validation: {_fmt(ga_s)}   "
              f"F_hit {gf.hit_ratio:.4f}")
        print(f"  NOTE: GA is measured single-core because the reference "
              f"implementation is serial.")
        print(f"        Its per-generation evaluations are independent and "
              f"parallelizable in principle;")
        print(f"        the multi-worker columns above are an engineering "
              f"optimization, not an")
        print(f"        algorithmic advantage. Compare W=1 against GA for the "
              f"algorithmic ratio.")


if __name__ == "__main__":
    main()
