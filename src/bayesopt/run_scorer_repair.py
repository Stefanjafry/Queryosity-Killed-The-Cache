#!/usr/bin/env python
"""
Stage-2 scorer-repair harness for Mode A Track 4 (direct evaluation).

Evaluates the controlled variant grid by deterministic multistart greedy
on the exact simulator — no optimizer, no neighbour refinement, no
BoTorch — so the result measures scorer *structure*, not search.  For
each TPC-H cache it builds features once, scores every variant's best
multistart schedule, and reports F_hit with deltas vs Greedy-D, GA+D,
the current Mode A neighbor best, and the D-only anchor; plus, for the
small cache, a q3 placement autopsy per variant.

Run protocol (per spec): TPC-H, caches 102400/262144/524288, multistart
only, seed 42 first; seeds 43/44 are a follow-up for survivors only.

Read inputs from the page-access profiles and the bo baselines; writes
``scorer_repair_summary.csv``, one detailed JSON per cache, and a q3
autopsy table for the small cache.

Usage:
    python -m src.bayesopt.run_scorer_repair \\
        --page-access-dir page_access/tpch_6gb \\
        --caches 102400,262144,524288 --seed 42 \\
        --out-dir experiment_logs/mode_a_repair
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.bayesopt.data import load_workload_pages
from src.bayesopt.mode_a.features import build_features
from src.bayesopt.mode_a.scorer_variants import (
    VariantSpec,
    default_variant_grid,
    multistart_greedy_variant,
    variant_score,
)
from src.bayesopt.objective import simulate_schedule_page_level_traced
from src.utilities.constants import PROJECT_ROOT


def _eval_schedule(page_sets, schedule, cache_pages):
    req, hits, traces = simulate_schedule_page_level_traced(
        page_sets, list(schedule), cache_pages
    )
    return (hits / req if req else 0.0), req, hits, traces


def _baseline(baseline_dir: Path, cache: int, method: str):
    p = baseline_dir / f"tpch_c{cache}_baseline_{method}_s42.summary.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())["sim_hit_ratio"]


def _modea_neighbor_best(arms_dir: Path, cache: int, family: str, tag: str):
    """Best F_hit of the warmup_random_neighbor arm for this cell, if present."""
    cell = f"c{cache}_{family}_{tag}"
    ds = [d for d in sorted(arms_dir.glob(f"{cell}_warmup_random_neighbor_s42*"))
          if (d / "summary.json").is_file()]
    if not ds:
        return None
    return json.loads((ds[-1] / "summary.json").read_text())["best_hit_ratio"]


def _q3_autopsy(spec, feats, schedule, query_ids, page_sets, cache_pages,
                target="q3"):
    """Position/predecessor/hits/D-into-target for one variant's schedule."""
    if target not in query_ids:
        return None
    tgt_idx = query_ids.index(target)
    pos = schedule.index(tgt_idx)
    pred_idx = schedule[pos - 1] if pos > 0 else None
    _, _, _, traces = _eval_schedule(page_sets, schedule, cache_pages)
    tr = traces[pos]
    d_into = feats.D_norm[pred_idx][tgt_idx] if pred_idx is not None else None
    surv_into = (feats.survivor_ratio[pred_idx][tgt_idx]
                 if pred_idx is not None else None)
    return {
        "variant": spec.name,
        "q3_position": pos,
        "q3_predecessor": query_ids[pred_idx] if pred_idx is not None else None,
        "q3_hits": tr.hits,
        "q3_misses": tr.misses,
        "q3_requests": tr.requests,
        "D_norm_into_q3": round(d_into, 4) if d_into is not None else None,
        "survivor_into_q3": round(surv_into, 4) if surv_into is not None else None,
    }


def _avg_d_norm(feats, schedule):
    """Average immediate D_norm over the schedule's adjacent edges."""
    if len(schedule) < 2:
        return 0.0
    vals = [feats.D_norm[schedule[k]][schedule[k + 1]]
            for k in range(len(schedule) - 1)]
    return round(sum(vals) / len(vals), 4)


def evaluate_cache(
    page_sets, query_ids, cache: int, grid: list[VariantSpec],
    baseline_dir: Path, arms_dir: Path, family: str, tag: str,
    num_starts: int, autopsy_target: str | None,
):
    feats = build_features(page_sets, cache)
    gd = _baseline(baseline_dir, cache, "greedy_d")
    gad = _baseline(baseline_dir, cache, "ga_d")
    neigh = _modea_neighbor_best(arms_dir, cache, family, tag)

    d_only_fhit = None
    rows = []
    autopsy = []
    for spec in grid:
        cands = multistart_greedy_variant(spec, feats, num_starts=num_starts)
        best_f, best_sched = -1.0, None
        for c in cands:
            f, *_ = _eval_schedule(page_sets, c.schedule, cache)
            if f > best_f:
                best_f, best_sched = f, c.schedule
        if spec.name == "D_only":
            d_only_fhit = best_f
        row = {
            "variant": spec.name,
            "alpha_D": spec.alpha_D,
            "cache_pages": cache,
            "F_hit": round(best_f, 6),
            "avg_D_norm": _avg_d_norm(feats, best_sched),
            "vs_greedy_d": round(best_f - gd, 6) if gd is not None else None,
            "vs_ga_d": round(best_f - gad, 6) if gad is not None else None,
            "vs_neighbor": round(best_f - neigh, 6) if neigh is not None else None,
        }
        rows.append((row, best_sched, best_f))
    # fill vs_d_only now that we have it
    for row, _, _ in rows:
        row["vs_d_only"] = (round(row["F_hit"] - d_only_fhit, 6)
                            if d_only_fhit is not None else None)

    # q3 autopsy for every variant (small cache only — caller decides)
    if autopsy_target and autopsy_target in query_ids:
        for (row, sched, _f) in rows:
            spec = next(s for s in grid if s.name == row["variant"])
            a = _q3_autopsy(spec, feats, sched, query_ids, page_sets,
                            cache, target=autopsy_target)
            if a:
                autopsy.append(a)

    return [r for r, _, _ in rows], autopsy, {"greedy_d": gd, "ga_d": gad,
                                              "neighbor": neigh,
                                              "d_only": d_only_fhit}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--page-access-dir", type=Path, required=True)
    p.add_argument("--caches", default="102400,262144,524288")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--family", default="d",
                   help="cell family tag for locating the neighbor baseline")
    p.add_argument("--consumer-tag", default="msg")
    p.add_argument("--num-starts", type=int, default=5)
    p.add_argument("--autopsy-query", default="q3")
    p.add_argument("--autopsy-cache", type=int, default=102400)
    p.add_argument("--baseline-dir", type=Path,
                   default=PROJECT_ROOT / "experiment_logs" / "bo")
    p.add_argument("--arms-dir", type=Path,
                   default=PROJECT_ROOT / "experiment_logs" / "mode_a_arms")
    p.add_argument("--exclude", default="")
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "experiment_logs" / "mode_a_repair")
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    caches = [int(c) for c in args.caches.split(",") if c.strip()]
    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}

    wp = load_workload_pages(args.page_access_dir, exclude=exclude)
    print(f"Loaded {wp.n} queries, {wp.total_unique_pages:,} unique pages")
    grid = default_variant_grid()
    print(f"Variant grid: {len(grid)} variants × {len(caches)} caches "
          f"= {len(grid) * len(caches)} direct evaluations\n")

    all_rows = []
    win_loss: dict[str, dict[str, int]] = {}
    per_variant_fhit: dict[str, list[float]] = {}
    for cache in caches:
        tgt = args.autopsy_query if cache == args.autopsy_cache else None
        print(f"=== cache {cache} ===")
        rows, autopsy, ctx = evaluate_cache(
            wp.page_sets, wp.query_ids, cache, grid,
            args.baseline_dir, args.arms_dir, args.family,
            args.consumer_tag, args.num_starts, tgt,
        )
        (args.out_dir / f"cache_{cache}_variants.json").write_text(
            json.dumps({"context": ctx, "variants": rows,
                        "q3_autopsy": autopsy}, indent=2))
        # console: top 5 and any q3-catastrophe
        ranked = sorted(rows, key=lambda r: -r["F_hit"])
        print(f"  baselines: greedy_d={ctx['greedy_d']} ga_d={ctx['ga_d']} "
              f"neighbor={ctx['neighbor']} D_only={ctx['d_only']}")
        for r in ranked[:5]:
            print(f"    {r['variant']:30s} F={r['F_hit']:.4f} "
                  f"vsGD={_fmt(r['vs_greedy_d'])} vsGAD={_fmt(r['vs_ga_d'])} "
                  f"vsNeigh={_fmt(r['vs_neighbor'])} avgDnorm={r['avg_D_norm']}")
        if autopsy:
            print(f"  q3 autopsy (cache {cache}):")
            for a in sorted(autopsy, key=lambda x: -x["q3_hits"])[:3]:
                print(f"    {a['variant']:30s} pos={a['q3_position']:>2} "
                      f"pred={a['q3_predecessor']} hits={a['q3_hits']} "
                      f"Dinto={a['D_norm_into_q3']}")
            worst = min(autopsy, key=lambda x: x["q3_hits"])
            print(f"    WORST q3: {worst['variant']} hits={worst['q3_hits']} "
                  f"pred={worst['q3_predecessor']} Dinto={worst['D_norm_into_q3']}")
        for r in rows:
            r["cache"] = cache
            all_rows.append(r)
            per_variant_fhit.setdefault(r["variant"], []).append(r["F_hit"])
            wl = win_loss.setdefault(r["variant"], {"win": 0, "tie": 0, "loss": 0})
            if r["vs_neighbor"] is not None:
                if r["vs_neighbor"] > 1e-6:
                    wl["win"] += 1
                elif r["vs_neighbor"] < -1e-6:
                    wl["loss"] += 1
                else:
                    wl["tie"] += 1
        print()

    # summary CSV
    keys = ["variant", "alpha_D", "cache", "F_hit", "avg_D_norm",
            "vs_greedy_d", "vs_ga_d", "vs_neighbor", "vs_d_only"]
    with open(args.out_dir / "scorer_repair_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in keys})

    # win/loss + largest regression per variant, survivor list
    print("=" * 70)
    print("CROSS-CACHE SUMMARY (vs neighbor)")
    print("=" * 70)
    survivors = []
    for variant in sorted(per_variant_fhit):
        wl = win_loss[variant]
        regs = [r["vs_neighbor"] for r in all_rows
                if r["variant"] == variant and r["vs_neighbor"] is not None]
        largest_reg = min(regs) if regs else None
        best_imp = max(regs) if regs else None
        # survivor: never catastrophically regresses and wins/ties somewhere
        survivor = (largest_reg is not None and largest_reg > -0.005
                    and wl["win"] >= 1)
        if survivor:
            survivors.append(variant)
        print(f"  {variant:30s} W/T/L={wl['win']}/{wl['tie']}/{wl['loss']} "
              f"bestΔ={_fmt(best_imp)} worstΔ={_fmt(largest_reg)}"
              f"{'  <-- survivor' if survivor else ''}")

    (args.out_dir / "survivors.json").write_text(json.dumps({
        "survivors": survivors,
        "criterion": "worst vs_neighbor > -0.005 AND >=1 win across caches",
    }, indent=2))
    print(f"\nSurvivors ({len(survivors)}): {survivors}")
    print(f"\nWrote scorer_repair_summary.csv, per-cache JSON, survivors.json "
          f"to {args.out_dir}")


def _fmt(v):
    return f"{v:+.4f}" if isinstance(v, (int, float)) else "n/a"


if __name__ == "__main__":
    main()
