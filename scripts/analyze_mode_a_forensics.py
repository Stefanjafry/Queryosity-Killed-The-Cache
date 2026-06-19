#!/usr/bin/env python
"""
Forensic diagnostic harness for Mode A arm artifacts (read-only).

Analyzes the existing ``experiment_logs/mode_a_arms`` runs — it never
reruns an experiment.  For each targeted cell it compares the best
schedule of each search arm (random_only / warmup_random /
warmup_random_neighbor / botorch_gp), and Greedy-D / GA+D where their
baseline summaries are available, and answers *why* the arms differ:

  1. schedule-difference audit  (position diffs, edge overlap, Kendall tau)
  2. per-query hit/miss audit    (where BoTorch loses hits vs neighbor)
  3. edge-level audit            (D_norm / survivor / out_D of the edges)
  4. config/source audit         (best by source, top-10, lambda spread)
  5. objective-surface audit     (distinct schedules/F_hit, jaggedness)
  6. timing audit                (wall, sim, overhead ratio, F_hit/min)

Edge and per-query diagnostics are logged only for each trial's winning
schedule, so the harness joins the global-incumbent schedule_eval_id to
recover that schedule's edge/per-query rows.  Greedy-D / GA+D have
schedules (from the bo baseline summaries) but no edge/per-query logs,
so only the schedule-difference audit covers them.

Usage:
    python scripts/analyze_mode_a_forensics.py \\
        --arms-dir experiment_logs/mode_a_arms \\
        --baseline-dir experiment_logs/bo \\
        --out-dir experiment_logs/mode_a_forensics
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ARMS = ["random_only", "warmup_random", "warmup_random_neighbor", "botorch_gp"]
ARM_SHORT = {"random_only": "rand", "warmup_random": "w+rand",
             "warmup_random_neighbor": "neighbor", "botorch_gp": "botorch"}

CELLS = [
    ("c102400_d_msg", 102400, "d", "multistart_greedy"),
    ("c262144_m_beam4", 262144, "m", "beam"),
    ("c524288_m_msg", 524288, "m", "multistart_greedy"),
    ("c524288_hybrid_beam4", 524288, "hybrid", "beam"),
]

EDGE_FEATURE_COLS = [
    "edge_score", "M", "D_ij", "D_ji", "M_norm", "D_norm", "asym_norm",
    "gap_raw", "gap_norm", "survivor_ratio", "out_M_topK_j", "out_D_topK_j",
    "in_D_j", "balance_D_j", "size_norm_j", "cache_pressure_j",
]


# ---------- loading ----------

def _latest(arms_dir: Path, cell: str, arm: str) -> Path | None:
    ds = [d for d in sorted(arms_dir.glob(f"{cell}_{arm}_s42*"))
          if (d / "summary.json").is_file()]
    return ds[-1] if ds else None


def _load_summary(d: Path) -> dict:
    return json.loads((d / "summary.json").read_text())


def _read_csv(d: Path, name: str) -> list[dict]:
    p = d / name
    if not p.is_file() or not p.read_text().strip():
        return []
    return list(csv.DictReader(open(p)))


def _incumbent_eval_id(d: Path) -> str | None:
    for r in _read_csv(d, "candidate_schedules.csv"):
        if r["is_global_incumbent"] == "True":
            return r["schedule_eval_id"]
    return None


def _schedule_qids(d: Path) -> list[str]:
    return _load_summary(d)["best_schedule_qids"]


def _baseline_schedule(baseline_dir: Path, cache: int, method: str):
    p = baseline_dir / f"tpch_c{cache}_baseline_{method}_s42.summary.json"
    if not p.is_file():
        return None, None
    s = json.loads(p.read_text())
    return s["schedule_qids"], s["sim_hit_ratio"]


# ---------- 1. schedule-difference ----------

def _kendall_tau(a: list[str], b: list[str]) -> float:
    """Pairwise order agreement in [-1, 1] over the shared query set."""
    common = [q for q in a if q in b]
    pos_b = {q: i for i, q in enumerate(b)}
    idx = [pos_b[q] for q in common]
    n = len(idx)
    if n < 2:
        return 1.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (idx[i] - idx[j])
            if s < 0:
                conc += 1
            elif s > 0:
                disc += 1
    total = conc + disc
    return (conc - disc) / total if total else 1.0


def _edges(sched: list[str]) -> set[tuple[str, str]]:
    return set(zip(sched, sched[1:]))


def schedule_diff(a: list[str], b: list[str]) -> dict:
    identical = a == b
    pos_diffs = sum(1 for x, y in zip(a, b) if x != y)
    first_div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    ea, eb = _edges(a), _edges(b)
    edge_overlap = len(ea & eb)
    return {
        "identical": identical,
        "positions_differing": pos_diffs,
        "first_divergence": first_div,
        "adjacent_edge_overlap": edge_overlap,
        "adjacent_edges_total": max(len(a) - 1, 0),
        "kendall_tau": round(_kendall_tau(a, b), 4),
        "shared_first3": a[:3] == b[:3],
        "shared_last3": a[-3:] == b[-3:],
    }


# ---------- 2. per-query ----------

def _perq_for(d: Path, eval_id: str) -> list[dict]:
    return [r for r in _read_csv(d, "per_query_metrics.csv")
            if r["schedule_eval_id"] == eval_id]


def per_query_audit(neigh_d: Path, bo_d: Path) -> dict:
    nid, bid = _incumbent_eval_id(neigh_d), _incumbent_eval_id(bo_d)
    if not nid or not bid:
        return {"available": False}
    npq = _perq_for(neigh_d, nid)
    bpq = _perq_for(bo_d, bid)
    # per-query hits keyed by query id
    nh = {r["query_id"]: int(r["hits"]) for r in npq}
    bh = {r["query_id"]: int(r["hits"]) for r in bpq}
    deltas = []
    for q in set(nh) | set(bh):
        d = bh.get(q, 0) - nh.get(q, 0)  # botorch − neighbor
        if d != 0:
            deltas.append((q, d))
    deltas.sort(key=lambda x: x[1])  # most-negative first = BO loses here
    # where does BO first fall behind on cumulative hits?
    nc = [(int(r["position"]), int(r["cum_hits"])) for r in npq]
    bc = {int(r["position"]): int(r["cum_hits"]) for r in bpq}
    first_behind = None
    for pos, ch in sorted(nc):
        if bc.get(pos, 0) < ch:
            first_behind = pos
            break
    total_gap = sum(d for _, d in deltas)
    bo_losses = [(q, d) for q, d in deltas if d < 0]
    concentration = None
    if bo_losses:
        worst = sum(abs(d) for _, d in bo_losses[:3])
        allneg = sum(abs(d) for _, d in bo_losses)
        concentration = round(worst / allneg, 3) if allneg else None
    return {
        "available": True,
        "net_hit_delta_bo_minus_neigh": total_gap,
        "first_position_bo_behind": first_behind,
        "queries_bo_loses_most": bo_losses[:5],
        "queries_bo_gains_most": [(q, d) for q, d in reversed(deltas) if d > 0][:5],
        "loss_concentration_top3_frac": concentration,
        "n_queries_bo_worse": len(bo_losses),
    }


# ---------- 3. edge-level ----------

def _edge_rows_for(d: Path, eval_id: str) -> list[dict]:
    return [r for r in _read_csv(d, "edge_diagnostics.csv")
            if r["schedule_eval_id"] == eval_id]


def _avg(rows: list[dict], col: str) -> float | None:
    vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
    return round(sum(vals) / len(vals), 4) if vals else None


def edge_audit(neigh_d: Path, bo_d: Path) -> dict:
    nid, bid = _incumbent_eval_id(neigh_d), _incumbent_eval_id(bo_d)
    if not nid or not bid:
        return {"available": False}
    ne = _edge_rows_for(neigh_d, nid)
    be = _edge_rows_for(bo_d, bid)
    if not ne or not be:
        return {"available": False}
    summary = {"available": True, "neighbor_avg": {}, "botorch_avg": {},
               "delta_bo_minus_neigh": {}}
    for col in ("D_norm", "M_norm", "survivor_ratio", "out_D_topK_j",
                "balance_D_j", "size_norm_j", "asym_norm", "cache_pressure_j"):
        na, ba = _avg(ne, col), _avg(be, col)
        summary["neighbor_avg"][col] = na
        summary["botorch_avg"][col] = ba
        if na is not None and ba is not None:
            summary["delta_bo_minus_neigh"][col] = round(ba - na, 4)
    # weakest edges (lowest D_norm) in each
    def weakest(rows, k=3):
        s = sorted(rows, key=lambda r: float(r["D_norm"]))
        return [{"pos": int(r["position"]), "edge": f'{r["prev_query"]}->{r["next_query"]}',
                 "D_norm": round(float(r["D_norm"]), 3),
                 "survivor": round(float(r["survivor_ratio"]), 3)} for r in s[:k]]
    summary["neighbor_weakest_edges"] = weakest(ne)
    summary["botorch_weakest_edges"] = weakest(be)
    # interpretation flags
    d = summary["delta_bo_minus_neigh"]
    summary["flags"] = {
        "bo_lower_immediate_D": d.get("D_norm", 0) < -0.02,
        "bo_higher_out_D": d.get("out_D_topK_j", 0) > 0.02,
        "bo_higher_asym": d.get("asym_norm", 0) > 0.02,
        "bo_ignores_size": d.get("size_norm_j", 0) > 0.05,
        "bo_lower_survivor": d.get("survivor_ratio", 0) < -0.02,
    }
    return summary


# ---------- 4. config/source ----------

def config_audit(d: Path) -> dict:
    pt = _read_csv(d, "parameter_trials.csv")
    if not pt:
        return {"available": False}
    for r in pt:
        r["_F"] = float(r["best_F_hit"])
    top = sorted(pt, key=lambda r: -r["_F"])[:10]
    src_dist: dict[str, int] = {}
    topk_dist: dict[str, int] = {}
    for r in top:
        s = r["source"].split(":")[0]
        src_dist[s] = src_dist.get(s, 0) + 1
        topk_dist[r["topk"]] = topk_dist.get(r["topk"], 0) + 1
    best = top[0]
    # did acquisition improve over init? (botorch only)
    init_best = max((r["_F"] for r in pt if r["source"].startswith("botorch_init")),
                    default=None)
    acq_best = max((r["_F"] for r in pt if r["source"].startswith("botorch_acq")),
                   default=None)
    acq_improved = (acq_best is not None and init_best is not None
                    and acq_best > init_best + 1e-9)
    # duplicate-config proposals (botorch proposing similar lambdas)
    lam_keys = [tuple(round(float(r[k]), 2) for k in
                ("lambda_out", "lambda_size", "lambda_balance",
                 "lambda_asym", "lambda_ratio", "lambda_gap"))
                for r in pt]
    dup_configs = len(lam_keys) - len(set(lam_keys))
    return {
        "available": True,
        "best_source": best["source"],
        "best_F_hit": round(best["_F"], 4),
        "top10_source_dist": src_dist,
        "top10_topk_dist": topk_dist,
        "best_config_lambdas": {k: round(float(best[k]), 3) for k in
                                ("lambda_out", "lambda_size", "lambda_balance",
                                 "lambda_asym", "lambda_ratio", "lambda_gap")},
        "botorch_init_best_F": round(init_best, 4) if init_best else None,
        "botorch_acq_best_F": round(acq_best, 4) if acq_best else None,
        "botorch_acq_improved_over_init": acq_improved,
        "duplicate_config_proposals": dup_configs,
        "n_trials": len(pt),
    }


# ---------- 5. objective surface ----------

def surface_audit(d: Path) -> dict:
    cand = _read_csv(d, "candidate_schedules.csv")
    pt = _read_csv(d, "parameter_trials.csv")
    if not cand:
        return {"available": False}
    scheds = [r["schedule_qids"] for r in cand]
    fhits = [round(float(r["F_hit"]), 6) for r in cand]
    distinct_sched = len(set(scheds))
    distinct_fhit = len(set(fhits))
    dup_rate = round(1 - distinct_sched / len(scheds), 3) if scheds else 0.0
    # jaggedness: among trial winners, do near-identical lambda configs give
    # different F_hit? approximate via spread of winner F_hit across trials.
    winner_fhits = sorted({round(float(r["best_F_hit"]), 4) for r in pt}) \
        if pt else []
    return {
        "available": True,
        "n_candidates": len(cand),
        "distinct_schedules": distinct_sched,
        "distinct_F_hit_values": distinct_fhit,
        "duplicate_schedule_rate": dup_rate,
        "F_hit_min": min(fhits),
        "F_hit_max": max(fhits),
        "F_hit_spread": round(max(fhits) - min(fhits), 4),
        "distinct_winner_F_hit": len(winner_fhits),
    }


# ---------- 6. timing ----------

def timing_audit(d: Path) -> dict:
    t = d / "timing_breakdown.json"
    if not t.is_file():
        return {"available": False}
    tb = json.loads(t.read_text())
    return {
        "available": True,
        "total_wall_s": round(tb.get("total_wall", 0), 1),
        "exact_simulation_s": round(tb.get("exact_simulation", 0), 1),
        "schedule_construction_s": round(tb.get("schedule_construction", 0), 1),
        "botorch_s": round(tb.get("botorch", 0), 1),
        "neighbor_refinement_s": round(tb.get("neighbor_refinement", 0), 1),
        "exact_evals": tb.get("exact_evals_run"),
    }


# ---------- orchestration ----------

def analyze_cell(arms_dir: Path, baseline_dir: Path, cell: str,
                 cache: int, family: str, consumer: str) -> dict:
    dirs = {arm: _latest(arms_dir, cell, arm) for arm in ARMS}
    report: dict = {"cell": cell, "cache_pages": cache, "family": family,
                    "consumer": consumer, "arms_present": {}}
    fhit = {}
    sched = {}
    for arm, d in dirs.items():
        if d is None:
            report["arms_present"][arm] = False
            continue
        report["arms_present"][arm] = True
        s = _load_summary(d)
        fhit[arm] = s["best_hit_ratio"]
        sched[arm] = s["best_schedule_qids"]
    report["best_F_hit"] = {a: round(v, 4) for a, v in fhit.items()}

    # baselines
    gd_s, gd_f = _baseline_schedule(baseline_dir, cache, "greedy_d")
    gad_s, gad_f = _baseline_schedule(baseline_dir, cache, "ga_d")
    report["baseline_F_hit"] = {
        "greedy_d": round(gd_f, 4) if gd_f else None,
        "ga_d": round(gad_f, 4) if gad_f else None,
    }

    neigh_d = dirs["warmup_random_neighbor"]
    bo_d = dirs["botorch_gp"]

    # 1. schedule diffs: every arm vs neighbor, plus neighbor-vs-botorch, plus baselines
    diffs = {}
    if neigh_d and bo_d:
        diffs["neighbor_vs_botorch"] = schedule_diff(
            sched["warmup_random_neighbor"], sched["botorch_gp"])
    for arm in ("random_only", "warmup_random"):
        if neigh_d and dirs[arm]:
            diffs[f"neighbor_vs_{arm}"] = schedule_diff(
                sched["warmup_random_neighbor"], sched[arm])
    if neigh_d and gd_s:
        diffs["neighbor_vs_greedy_d"] = schedule_diff(
            sched["warmup_random_neighbor"], gd_s)
    if bo_d and gd_s:
        diffs["botorch_vs_greedy_d"] = schedule_diff(sched["botorch_gp"], gd_s)
    report["schedule_diffs"] = diffs

    # 2-3. per-query and edge: neighbor vs botorch
    if neigh_d and bo_d:
        report["per_query_audit"] = per_query_audit(neigh_d, bo_d)
        report["edge_audit"] = edge_audit(neigh_d, bo_d)

    # 4-6. per-arm config/surface/timing
    report["config_audit"] = {a: config_audit(d) for a, d in dirs.items() if d}
    report["surface_audit"] = {a: surface_audit(d) for a, d in dirs.items() if d}
    report["timing_audit"] = {a: timing_audit(d) for a, d in dirs.items() if d}

    # overhead ratio + F_hit/min
    if neigh_d and bo_d:
        nt = report["timing_audit"]["warmup_random_neighbor"]["total_wall_s"]
        bt = report["timing_audit"]["botorch_gp"]["total_wall_s"]
        report["bo_overhead_ratio"] = round(bt / nt, 2) if nt else None
    return report


def _verdict_line(rep: dict) -> str:
    """One-line mechanistic read for the console."""
    cell = rep["cell"]
    f = rep["best_F_hit"]
    ng, bo = f.get("warmup_random_neighbor"), f.get("botorch_gp")
    if ng is None or bo is None:
        return f"{cell}: incomplete arms"
    delta = bo - ng
    bits = [f"{cell}: botorch {bo:.4f} vs neighbor {ng:.4f} ({delta:+.4f})"]
    ca = rep.get("config_audit", {}).get("botorch_gp", {})
    if ca.get("available"):
        if not ca.get("botorch_acq_improved_over_init"):
            bits.append("acq did NOT beat init")
        else:
            bits.append("acq beat init")
    ea = rep.get("edge_audit", {})
    if ea.get("available"):
        flags = [k for k, v in ea["flags"].items() if v]
        if flags:
            bits.append("edges: " + ",".join(flags))
    pq = rep.get("per_query_audit", {})
    if pq.get("available") and pq.get("queries_bo_loses_most"):
        worst = pq["queries_bo_loses_most"][0]
        bits.append(f"BO loses most hits on {worst[0]}({worst[1]})")
    sa = rep.get("surface_audit", {}).get("botorch_gp", {})
    if sa.get("available"):
        bits.append(f"dup_sched_rate={sa['duplicate_schedule_rate']}")
    ovr = rep.get("bo_overhead_ratio")
    if ovr:
        bits.append(f"{ovr}x wall")
    return " | ".join(bits)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms-dir", type=Path,
                   default=Path("experiment_logs/mode_a_arms"))
    p.add_argument("--baseline-dir", type=Path,
                   default=Path("experiment_logs/bo"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("experiment_logs/mode_a_forensics"))
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    print("=" * 78)
    print("MODE A FORENSIC AUDIT — botorch vs neighbor (and arms/baselines)")
    print("=" * 78)
    for cell, cache, family, consumer in CELLS:
        rep = analyze_cell(args.arms_dir, args.baseline_dir,
                           cell, cache, family, consumer)
        (args.out_dir / f"{cell}_forensics.json").write_text(
            json.dumps(rep, indent=2))
        print("\n" + _verdict_line(rep))

        # per-cell detail block
        sd = rep.get("schedule_diffs", {}).get("neighbor_vs_botorch")
        if sd:
            print(f"    schedule: identical={sd['identical']} "
                  f"pos_diff={sd['positions_differing']}/{sd['adjacent_edges_total']+1} "
                  f"edge_overlap={sd['adjacent_edge_overlap']}/{sd['adjacent_edges_total']} "
                  f"tau={sd['kendall_tau']} first_div={sd['first_divergence']}")
        ea = rep.get("edge_audit", {})
        if ea.get("available"):
            print(f"    edge avg D_norm  neigh={ea['neighbor_avg']['D_norm']} "
                  f"bo={ea['botorch_avg']['D_norm']} "
                  f"(Δ{ea['delta_bo_minus_neigh'].get('D_norm')})")
            print(f"    edge avg survivor neigh={ea['neighbor_avg']['survivor_ratio']} "
                  f"bo={ea['botorch_avg']['survivor_ratio']} "
                  f"(Δ{ea['delta_bo_minus_neigh'].get('survivor_ratio')})")
        ca = rep.get("config_audit", {}).get("botorch_gp", {})
        if ca.get("available"):
            print(f"    botorch: init_best={ca['botorch_init_best_F']} "
                  f"acq_best={ca['botorch_acq_best_F']} "
                  f"acq>init={ca['botorch_acq_improved_over_init']} "
                  f"dup_configs={ca['duplicate_config_proposals']}")

        # accumulate summary row
        f = rep["best_F_hit"]
        summary_rows.append({
            "cell": cell, "cache_pages": cache, "family": family,
            "consumer": consumer,
            "random_only": f.get("random_only"),
            "warmup_random": f.get("warmup_random"),
            "neighbor": f.get("warmup_random_neighbor"),
            "botorch": f.get("botorch_gp"),
            "bo_minus_neighbor": (round(f["botorch_gp"] - f["warmup_random_neighbor"], 4)
                                  if f.get("botorch_gp") and f.get("warmup_random_neighbor") else None),
            "greedy_d": rep["baseline_F_hit"]["greedy_d"],
            "ga_d": rep["baseline_F_hit"]["ga_d"],
            "bo_overhead_ratio": rep.get("bo_overhead_ratio"),
            "bo_acq_improved_over_init": ca.get("botorch_acq_improved_over_init"),
            "neighbor_vs_botorch_identical": (sd or {}).get("identical"),
            "neighbor_vs_botorch_pos_diff": (sd or {}).get("positions_differing"),
        })

    # summary CSV
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(args.out_dir / "forensics_summary.csv", "w", newline="") as fobj:
            w = csv.DictWriter(fobj, fieldnames=keys)
            w.writeheader()
            w.writerows(summary_rows)
    print("\n" + "=" * 78)
    print(f"Wrote {len(summary_rows)} cell reports + forensics_summary.csv "
          f"to {args.out_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()
