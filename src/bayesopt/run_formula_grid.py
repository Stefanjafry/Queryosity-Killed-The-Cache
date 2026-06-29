"""
Phase 1 — paired formula-grid direct evaluation (Greedy-1 consumer).

Every formula is run as an M/D pair through the same scorer and the same
fixed weights, so each row of the grid is a fair symmetric-vs-directional
comparison.  Hybrid (tunable M/D mix) is included as the both-matrices arm.

This is a SCREEN, not a verdict: weights are fixed at hand defaults, so it
answers (a) which terms change the schedule at all and (b) which terms are
collinear/redundant — NOT which terms 'help'.  A term that looks inert here
at a fixed weight may still earn its place once BO tunes it, so prune in
Phase 1 only for collinearity or true inertness, never for 'did not help'.

``exact_state_greedy`` is reported as the matrix-free ceiling both M and D
approximate; ``D_only`` / ``M_only`` reproduce Greedy-D / Greedy-M exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from src.bayesopt.data import load_workload_pages
from src.bayesopt.objective import ExactSimObjective
from src.bayesopt.step_scorer import (
    ScorerWeights,
    exact_state_greedy,
    greedy_schedule,
    hybrid_immediate,
    make_step_scorer,
    row_normalize,
)
from src.simulator.cache_simulator import (
    compute_directional_matrix,
    compute_overlap_matrix,
)
from src.utilities.constants import PROJECT_ROOT, WORKLOAD_DIRS

_TIE_PP = 0.1


def _diff(a: list[int], b: list[int]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    ma = sum(a) / n
    mb = sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va < 1e-12 or vb < 1e-12:
        return float("nan")
    cov = sum((a[k] - ma) * (b[k] - mb) for k in range(n))
    return cov / math.sqrt(va * vb)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=list(WORKLOAD_DIRS), required=True)
    p.add_argument("--cache-pages", type=int, required=True)
    p.add_argument("--page-access-dir", type=Path, required=True)
    p.add_argument("--exclude", default="")
    p.add_argument("--w-future", type=float, default=0.5)
    p.add_argument("--w-regret", type=float, default=0.5)
    p.add_argument("--w-cache", type=float, default=0.3)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--cache-mode", choices=["pressure", "fit"], default="fit")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "experiment_logs" / "formula_grid")
    args = p.parse_args(argv)
    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}

    print(f"Loading page sets from {args.page_access_dir}…")
    wp = load_workload_pages(args.page_access_dir, exclude=exclude)
    pc = [len(ps) for ps in wp.page_sets]
    print(f"  {wp.n} queries, {wp.total_unique_pages:,} unique pages, "
          f"C = {args.cache_pages:,} pages")

    t0 = time.perf_counter()
    M = compute_overlap_matrix(wp.page_sets)
    D = compute_directional_matrix(wp.page_sets, args.cache_pages)
    M_norm = row_normalize(M)
    D_norm = row_normalize(D)
    H_norm = hybrid_immediate(M_norm, D_norm, 0.5, 0.5)
    print(f"  matrices + norms in {time.perf_counter() - t0:.1f}s")

    obj = ExactSimObjective(wp.page_sets, args.cache_pages, d_matrix=D)

    def run(immediate: list[list[float]], w: ScorerWeights) -> tuple[float, list[int]]:
        sched = greedy_schedule(
            make_step_scorer(immediate, w, pc, args.cache_pages), wp.n, pc)
        m, _ = obj.evaluate_schedule(sched)
        return m.hit_ratio, sched

    base = ScorerWeights(topk=args.topk, cache_mode=args.cache_mode)
    wf = ScorerWeights(w_future=args.w_future, topk=args.topk, cache_mode=args.cache_mode)
    wr = ScorerWeights(w_regret=args.w_regret, topk=args.topk, cache_mode=args.cache_mode)
    wc = ScorerWeights(w_cache=args.w_cache, topk=args.topk, cache_mode=args.cache_mode)
    wfull = ScorerWeights(w_future=args.w_future, w_regret=args.w_regret,
                          w_cache=args.w_cache, topk=args.topk,
                          cache_mode=args.cache_mode)

    # immediate-only twins reproduce Greedy-M / Greedy-D.
    m_only_fhit, m_only_sched = run(M_norm, base)
    d_only_fhit, d_only_sched = run(D_norm, base)

    # Paired formula structures: each is run on M and on D.
    structures = [("only", base), ("future", wf), ("regret", wr),
                  ("cachefit", wc), ("full", wfull)]
    rows: list[dict[str, object]] = []
    print_rows: list[tuple[str, float, float, float, int, int]] = []
    for name, w in structures:
        mf, ms = run(M_norm, w)
        df, ds = run(D_norm, w)
        dm = (df - mf) * 100.0
        mchg = _diff(ms, m_only_sched)
        dchg = _diff(ds, d_only_sched)
        print_rows.append((name, mf, df, dm, mchg, dchg))
        rows.append({
            "structure": name,
            "M_fhit": mf, "D_fhit": df,
            "D_minus_M_pp": dm,
            "M_changed_vs_Monly": mchg,
            "D_changed_vs_Donly": dchg,
            "M_schedule": ms, "D_schedule": ds,
        })
    # Hybrid arm (both matrices).
    h_only_fhit, _ = run(H_norm, base)
    h_full_fhit, _ = run(H_norm, wfull)

    # Matrix-free ceiling.
    es_sched = exact_state_greedy(wp.page_sets, args.cache_pages)
    es_metrics, _ = obj.evaluate_schedule(es_sched)
    es_fhit = es_metrics.hit_ratio

    # Collinearity screen at the first-step context (U = all queries).
    allU = list(range(wp.n))
    from src.bayesopt.step_scorer import (
        cache_feature, connector_in, connector_out, future_value,
        net_reuse_gap, regret)
    tau = ScorerWeights().connector_rho
    cols = ["imm_M", "imm_D", "future_D", "regret_D", "cache",
            "connout_D", "connin_D", "netgap"]
    vec: dict[str, list[float]] = {k: [] for k in cols}
    for i in range(wp.n):
        for j in range(wp.n):
            if i == j:
                continue
            vec["imm_M"].append(M_norm[i][j])
            vec["imm_D"].append(D_norm[i][j])
            vec["future_D"].append(future_value(j, allU, D_norm, args.topk))
            vec["regret_D"].append(regret(i, j, allU, D_norm))
            vec["cache"].append(cache_feature(j, args.cache_pages, pc, args.cache_mode))
            # connectors are normalisation-invariant (relative threshold);
            # netgap MUST use raw matrices or it collapses to a constant.
            vec["connout_D"].append(connector_out(j, allU, D, tau))
            vec["connin_D"].append(connector_in(j, allU, D, tau))
            vec["netgap"].append(net_reuse_gap(j, allU, D, M))
    names = list(vec)
    redundancy = {a: {b: _pearson(vec[a], vec[b]) for b in names} for a in names}

    summary = {
        "workload": args.workload, "cache_pages": args.cache_pages,
        "n_queries": wp.n,
        "weights": {"w_future": args.w_future, "w_regret": args.w_regret,
                    "w_cache": args.w_cache, "topk": args.topk,
                    "cache_mode": args.cache_mode},
        "greedy_M_fhit": m_only_fhit, "greedy_D_fhit": d_only_fhit,
        "hybrid_only_fhit": h_only_fhit, "hybrid_full_fhit": h_full_fhit,
        "exact_state_ceiling_fhit": es_fhit,
        "paired_structures": rows,
        "term_redundancy": redundancy,
        "note": "fixed-weight SCREEN: prune only on collinearity/inertness, "
                "not on 'did not help'.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "formula_grid_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    print("\n=== paired formula grid (Greedy-1, exact F_hit) — SCREEN ONLY ===")
    print(f"  exact-state ceiling : {es_fhit:.4f}   (M and D both approximate this)")
    print(f"  {'structure':10s} {'M_fhit':>8s} {'D_fhit':>8s} {'D-M pp':>8s}"
          f"  {'Mchg':>4s} {'Dchg':>4s}")
    for name, mf, df, dm, mchg, dchg in print_rows:
        tag = "tie" if abs(dm) < _TIE_PP else ("D>M" if dm > 0 else "M>D")
        print(f"  {name:10s} {mf:8.4f} {df:8.4f}"
              f" {dm:+8.3f}  {mchg:4d} {dchg:4d}  [{tag}]")
    print(f"  hybrid_only={h_only_fhit:.4f}  hybrid_full={h_full_fhit:.4f}")
    print("  term redundancy (Pearson, first-step context):")
    print(f"    {'':9s}" + "".join(f"{n:>9s}" for n in names))
    for a in names:
        print(f"    {a:9s}" + "".join(f"{redundancy[a][b]:9.3f}" for b in names))
    print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
