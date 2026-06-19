#!/usr/bin/env python
"""
Stage-2 Dynamic Scorer v2 harness — direct evaluation only.

Direct-evaluates the Dynamic Scorer v2 grid (including the new
missed_opportunity penalty) by deterministic multistart greedy on the
exact simulator across the three TPC-H caches.  No BO, no neighbour
refinement — this stage measures whether the v2 *structures* are better.

For each variant it logs F_hit, deltas vs Greedy-D / GA+D / D-only, the
average edge features (D_norm, survivor, role_out, missed_opportunity,
cache_pressure), and — at the autopsy cache — q3's position, predecessor,
hits, and whether q3 landed after q1 (its best predecessor).

``--num-starts`` accepts an int or ``all`` (n starts); ``all`` gives the
candidate-generation upper bound for a later stage.

Usage:
    python -m src.bayesopt.run_dynamic_scorer \\
        --page-access-dir page_access/tpch_6gb \\
        --caches 102400,262144,524288 --seed 42 --num-starts 5 \\
        --out-dir experiment_logs/mode_a_dynamic
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.bayesopt.data import load_workload_pages
from src.bayesopt.mode_a.dynamic_scorer import (
    DynamicSpec,
    default_dynamic_grid,
    dynamic_score,
    multistart_greedy_dynamic,
    spec_name,
)
from src.bayesopt.mode_a.features import build_features
from src.bayesopt.objective import simulate_schedule_page_level_traced
from src.utilities.constants import PROJECT_ROOT


def _eval(page_sets, schedule, cache):
    req, hits, traces = simulate_schedule_page_level_traced(
        page_sets, list(schedule), cache)
    return (hits / req if req else 0.0), traces


def _baseline(baseline_dir: Path, cache: int, method: str):
    p = baseline_dir / f"tpch_c{cache}_baseline_{method}_s42.summary.json"
    return (json.loads(p.read_text())["sim_hit_ratio"]
            if p.is_file() else None)


def _avg_edge_features(spec, feats, schedule):
    """Average the dynamic features over the schedule's adjacent edges."""
    n = len(schedule)
    if n < 2:
        return {}
    acc = {"D_norm": 0.0, "survivor": 0.0, "role_out": 0.0,
           "missed_opportunity": 0.0, "cache_pressure": 0.0, "out_rem": 0.0}
    for kk in range(n - 1):
        i, j = schedule[kk], schedule[kk + 1]
        remaining = list(schedule[kk + 1:])  # j and everything after
        _, f = dynamic_score(spec, feats, i, j, remaining)
        acc["D_norm"] += feats.D_norm[i][j]
        acc["survivor"] += feats.survivor_ratio[i][j]
        acc["role_out"] += f["role_out"]
        acc["missed_opportunity"] += f["missed_opportunity"]
        acc["cache_pressure"] += f["cache_pressure"]
        acc["out_rem"] += f["out_rem"]
    return {kk: round(v / (n - 1), 4) for kk, v in acc.items()}


def _q3_info(feats, schedule, query_ids, page_sets, cache, target="q3"):
    if target not in query_ids:
        return None
    tgt = query_ids.index(target)
    pos = schedule.index(tgt)
    pred = schedule[pos - 1] if pos > 0 else None
    _, traces = _eval(page_sets, schedule, cache)
    tr = traces[pos]
    # best available predecessor for q3 over the whole workload
    best_pred = max(range(feats.n), key=lambda u: feats.D_norm[u][tgt]
                    if u != tgt else -1)
    return {
        "q3_position": pos,
        "q3_predecessor": query_ids[pred] if pred is not None else None,
        "q3_hits": tr.hits,
        "q3_misses": tr.misses,
        "D_norm_into_q3": round(feats.D_norm[pred][tgt], 4) if pred is not None else None,
        "survivor_into_q3": round(feats.survivor_ratio[pred][tgt], 4) if pred is not None else None,
        "q3_best_predecessor": query_ids[best_pred],
        "q3_after_best_pred": pred == best_pred,
        "q3_pagecount": feats.pagecount[tgt],
        "q3_cache_fit": round(min(1.0, cache / feats.pagecount[tgt]), 4),
    }


def _resolve_starts(num_starts_arg: str, n: int) -> int:
    return n if num_starts_arg == "all" else int(num_starts_arg)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--page-access-dir", type=Path, required=True)
    p.add_argument("--caches", default="102400,262144,524288")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-starts", default="5",
                   help="int, or 'all' for the candidate-gen upper bound")
    p.add_argument("--autopsy-cache", type=int, default=102400)
    p.add_argument("--autopsy-query", default="q3")
    p.add_argument("--baseline-dir", type=Path,
                   default=PROJECT_ROOT / "experiment_logs" / "bo")
    p.add_argument("--exclude", default="")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "experiment_logs" / "mode_a_dynamic")
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    caches = [int(c) for c in args.caches.split(",") if c.strip()]
    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}

    wp = load_workload_pages(args.page_access_dir, exclude=exclude)
    n_starts = _resolve_starts(args.num_starts, wp.n)
    grid = default_dynamic_grid()
    print(f"Loaded {wp.n} queries, {wp.total_unique_pages:,} unique pages")
    print(f"Dynamic v2 grid: {len(grid)} specs × {len(caches)} caches, "
          f"num_starts={n_starts}\n")

    all_rows = []
    per_variant: dict[str, dict[int, dict]] = {}
    q3_by_variant: dict[str, dict] = {}
    for cache in caches:
        feats = build_features(wp.page_sets, cache)
        gd = _baseline(args.baseline_dir, cache, "greedy_d")
        gad = _baseline(args.baseline_dir, cache, "ga_d")
        d_only = None
        rows = []
        print(f"=== cache {cache} ===")
        for spec in grid:
            cands = multistart_greedy_dynamic(spec, feats, num_starts=n_starts)
            best_f, best_sched = -1.0, None
            for c in cands:
                f, _ = _eval(wp.page_sets, c.schedule, cache)
                if f > best_f:
                    best_f, best_sched = f, c.schedule
            name = spec_name(spec)
            if spec.formula == "D_only" and spec.alpha_D == 1:
                d_only = best_f
            avg = _avg_edge_features(spec, feats, best_sched)
            row = {
                "variant": name, "formula": spec.formula,
                "alpha_D": spec.alpha_D, "cache": cache,
                "F_hit": round(best_f, 6),
                "avg_D_norm": avg.get("D_norm"),
                "avg_survivor": avg.get("survivor"),
                "avg_role_out": avg.get("role_out"),
                "avg_missed_opp": avg.get("missed_opportunity"),
                "avg_cache_pressure": avg.get("cache_pressure"),
                "vs_greedy_d": round(best_f - gd, 6) if gd is not None else None,
                "vs_ga_d": round(best_f - gad, 6) if gad is not None else None,
            }
            rows.append((row, best_sched, spec))
            if cache == args.autopsy_cache:
                qi = _q3_info(feats, best_sched, wp.query_ids,
                              wp.page_sets, cache, args.autopsy_query)
                if qi:
                    q3_by_variant[name] = qi
        for row, _, _ in rows:
            row["vs_d_only"] = (round(row["F_hit"] - d_only, 6)
                                if d_only is not None else None)
        # console: top 5 by F_hit
        ranked = sorted(rows, key=lambda r: -r[0]["F_hit"])
        print(f"  baselines: greedy_d={_r(gd)} ga_d={_r(gad)} D_only={_r(d_only)}")
        for row, _, _ in ranked[:6]:
            print(f"    {row['variant']:36s} F={row['F_hit']:.4f} "
                  f"vsGD={_f(row['vs_greedy_d'])} vsGAD={_f(row['vs_ga_d'])} "
                  f"missedOpp={row['avg_missed_opp']}")
        if cache == args.autopsy_cache:
            print(f"  q3 autopsy (best 3 by hits):")
            top_q3 = sorted(q3_by_variant.items(),
                            key=lambda kv: -kv[1]["q3_hits"])[:3]
            for name, qi in top_q3:
                print(f"    {name:36s} pos={qi['q3_position']:>2} "
                      f"pred={qi['q3_predecessor']} hits={qi['q3_hits']} "
                      f"afterBest={qi['q3_after_best_pred']}")
            worst = min(q3_by_variant.items(), key=lambda kv: kv[1]["q3_hits"])
            print(f"    WORST q3: {worst[0]} hits={worst[1]['q3_hits']} "
                  f"pred={worst[1]['q3_predecessor']}")
        for row, _, _ in rows:
            all_rows.append(row)
            per_variant.setdefault(row["variant"], {})[cache] = row
        (args.out_dir / f"dynamic_cache_{cache}.json").write_text(json.dumps(
            {"context": {"greedy_d": gd, "ga_d": gad, "d_only": d_only},
             "variants": [r for r, _, _ in rows]}, indent=2))
        print()

    # summary CSV
    keys = ["variant", "formula", "alpha_D", "cache", "F_hit", "avg_D_norm",
            "avg_survivor", "avg_role_out", "avg_missed_opp",
            "avg_cache_pressure", "vs_greedy_d", "vs_ga_d", "vs_d_only"]
    with open(args.out_dir / "dynamic_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in keys})
    if q3_by_variant:
        with open(args.out_dir / "dynamic_q3_autopsy.csv", "w", newline="") as f:
            qk = ["variant"] + list(next(iter(q3_by_variant.values())).keys())
            w = csv.DictWriter(f, fieldnames=qk)
            w.writeheader()
            for name, qi in q3_by_variant.items():
                w.writerow({"variant": name, **qi})

    # best-per-cache summary
    print("=" * 70)
    print("BEST DYNAMIC v2 PER CACHE")
    print("=" * 70)
    for cache in caches:
        cands = [r for r in all_rows if r["cache"] == cache]
        best = max(cands, key=lambda r: r["F_hit"])
        print(f"  cache {cache}: {best['variant']:36s} F={best['F_hit']:.4f} "
              f"vsGD={_f(best['vs_greedy_d'])} vsGAD={_f(best['vs_ga_d'])}")
    print(f"\nWrote dynamic_summary.csv, dynamic_q3_autopsy.csv, per-cache JSON "
          f"to {args.out_dir}")


def _f(v):
    return f"{v:+.4f}" if isinstance(v, (int, float)) else "n/a"


def _r(v):
    return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"


if __name__ == "__main__":
    main()
