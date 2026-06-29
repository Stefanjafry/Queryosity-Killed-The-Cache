"""
Candidate-set tuning — num_starts (multistart) and beam_width (beam).

Sweeps the two consumer knobs on the fixed single-step scorer (paired M/D)
and reports the F_hit-vs-exact-eval-cost frontier, so a knob value is chosen
where returns flatten rather than by feel.  A multistart-K run costs K exact
evals; a beam-B run costs B — both reported per row, since the fair currency
is exact simulations, not knob value.

topK is intentionally not swept: it only entered through the retired future
term, so after future was dropped it has nothing to tune.

This uses the immediate-only scorer (no regret/cache/connector) to isolate
the consumer-knob effect from weight tuning; BO over weights is a separate
layer.  If a knob shows real headroom here, it is worth pairing with BO.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.bayesopt.data import load_workload_pages
from src.bayesopt.objective import ExactSimObjective
from src.bayesopt.step_scorer import (
    ScorerWeights,
    beam_search_schedule,
    good_start_set,
    make_step_scorer,
    multistart_greedy_schedules,
    row_normalize,
)
from src.simulator.cache_simulator import (
    compute_directional_matrix,
    compute_overlap_matrix,
)
from src.utilities.constants import PROJECT_ROOT, WORKLOAD_DIRS

_KNEE_PP = 0.1  # gain below this from one step to the next = flattened


def _frontier_knee(points: list[tuple[int, float]]) -> int:
    """Smallest knob value past which each further step gains < _KNEE_PP."""
    best_k = points[0][0]
    best_f = points[0][1]
    for k, f in points[1:]:
        if (f - best_f) * 100.0 >= _KNEE_PP:
            best_k, best_f = k, f
        else:
            break
    return best_k


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=list(WORKLOAD_DIRS), required=True)
    p.add_argument("--cache-pages", type=int, required=True)
    p.add_argument("--page-access-dir", type=Path, required=True)
    p.add_argument("--exclude", default="")
    p.add_argument("--num-starts", default="1,4,8")
    p.add_argument("--beam-widths", default="1,2,4,8,16")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "experiment_logs" / "candidate_sets")
    args = p.parse_args(argv)
    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}
    ks = [int(x) for x in args.num_starts.split(",") if x.strip()]
    bs = [int(x) for x in args.beam_widths.split(",") if x.strip()]

    print(f"Loading page sets from {args.page_access_dir}…")
    wp = load_workload_pages(args.page_access_dir, exclude=exclude)
    pc = [len(ps) for ps in wp.page_sets]
    print(f"  {wp.n} queries, C = {args.cache_pages:,} pages")

    t0 = time.perf_counter()
    M = compute_overlap_matrix(wp.page_sets)
    D = compute_directional_matrix(wp.page_sets, args.cache_pages)
    M_norm = row_normalize(M)
    D_norm = row_normalize(D)
    print(f"  matrices in {time.perf_counter() - t0:.1f}s")

    obj = ExactSimObjective(wp.page_sets, args.cache_pages, d_matrix=D)

    def fhit(sched: list[int]) -> float:
        m, _ = obj.evaluate_schedule(sched)
        return m.hit_ratio

    def best_of(scheds: list[list[int]]) -> float:
        return max(fhit(s) for s in scheds)

    def multistart(immediate: list[list[float]], k: int) -> tuple[float, int]:
        starts = good_start_set(immediate, pc, k)
        score = make_step_scorer(immediate, ScorerWeights(), pc, args.cache_pages)
        scheds = multistart_greedy_schedules(score, wp.n, pc, starts)
        return best_of(scheds), len(scheds)

    def beam(immediate: list[list[float]], b: int) -> tuple[float, int]:
        score = make_step_scorer(immediate, ScorerWeights(), pc, args.cache_pages)
        scheds = beam_search_schedule(score, wp.n, pc, b)
        return best_of(scheds), len(scheds)

    families: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    for fam, knobs, fn in (("multistart", ks, multistart), ("beam", bs, beam)):
        families[fam] = {}
        for arm, immediate in (("M", M_norm), ("D", D_norm)):
            rows: list[dict[str, float | int]] = []
            for k in knobs:
                f, evals = fn(immediate, k)
                rows.append({"knob": k, "fhit": f, "evals": evals})
            families[fam][arm] = rows

    summary = {
        "workload": args.workload, "cache_pages": args.cache_pages,
        "n_queries": wp.n, "num_starts": ks, "beam_widths": bs,
        "families": families,
        "knee": {
            fam: {arm: _frontier_knee([(int(r["knob"]), float(r["fhit"]))
                                       for r in rows])
                  for arm, rows in arms.items()}
            for fam, arms in families.items()},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"candidate_sets_{args.workload}_{args.cache_pages}.json"
    out.write_text(json.dumps(summary, indent=2))

    print(f"\n=== candidate-set frontier: {args.workload} {args.cache_pages} ===")
    for fam, arms in families.items():
        print(f"  {fam}:")
        for arm, rows in arms.items():
            cells = "  ".join(f"k={int(r['knob'])}:{float(r['fhit']):.4f}"
                              f"(ev{int(r['evals'])})" for r in rows)
            knee = summary["knee"][fam][arm]
            print(f"    {arm}: {cells}   knee@{knee}")
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
