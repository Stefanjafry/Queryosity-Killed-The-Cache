#!/usr/bin/env python3
"""Generate paper-valid Queryosity demo schedules + explanation metadata.

This script is intentionally an *adapter* around the research implementation.
It imports the existing D(C), row-normalization, regret scorer, Sweep-D,
Sweep+Beam-D and complete-order simulator code from the Queryosity repository;
it does not implement a second scheduler in the demo.

Typical VM use, from Queryosity-Killed-The-Cache/queryosity-whatif-demo:

    export PYTHONHASHSEED=0
    python scripts/generate_demo_artifacts.py --all

Artifacts are written to data/generated/<workload>_<capacity>.json and are what
the live frontend reads.  PostgreSQL is not required: the research repository's
stored page-access profiles are the input.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

DEMO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_ROOT = DEMO_ROOT.parent
DEFAULT_OUT = DEMO_ROOT / "data" / "generated"
CAPACITIES = (102_400, 262_144, 524_288)
WORKLOADS = ("tpch", "tpcds", "job")
METHOD_LABELS = {"sweep_D": "Sweep-D", "sweep_beam_D": "Sweep+Beam-D"}


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _human_explanation(
    *, method: str, selected_q: str, predecessor_q: str, wr: float,
    regret_value: float, best_pred_q: str | None,
) -> str:
    """Build deterministic explanation text from recorded scheduler quantities.

    For Sweep+Beam-D we deliberately do not claim a local greedy rank. The
    completed path is retained by cumulative beam search and then selected by
    complete-order cache simulation, so a final-path edge must not be described
    as if it were independently chosen as the locally best successor.
    """
    if method == "sweep_beam_D":
        base = (
            f"{predecessor_q} → {selected_q} lies on the completed beam candidate "
            "retained after complete-order cache simulation. "
            "Sweep+Beam-D ranks partial paths cumulatively, so a transition on the "
            "retained path does not have to be the locally best successor at that step."
        )
        if wr == 0:
            return base + " With w_r = 0, edge scores use immediate normalized directional reuse only."
        if regret_value > 0 and best_pred_q is not None:
            return base + (
                f" The regret term is nonzero because {best_pred_q} provides a stronger "
                f"remaining predecessor pairing into {selected_q}."
            )
        return base

    if wr == 0:
        return (
            f"With w_r = 0, Sweep-D scores successors using immediate normalized "
            f"directional reuse only. {selected_q} is the greedy successor from "
            f"{predecessor_q} at this step."
        )
    if regret_value > 0 and best_pred_q is not None:
        return (
            f"{selected_q} is selected from {predecessor_q} after applying the regret "
            f"penalty. {best_pred_q} provides a stronger remaining predecessor pairing "
            f"into {selected_q}, so that missed opportunity is subtracted before the "
            "successor is chosen."
        )
    return (
        f"{selected_q} is selected from {predecessor_q}. Its construction score combines "
        "normalized directional reuse with the configured regret penalty; no stronger "
        "remaining predecessor creates a positive regret term here."
    )


def _build_transition_metadata(
    *, order: list[int], query_ids: list[str], d_raw: Sequence[Sequence[int]],
    d_norm: Sequence[Sequence[float]], pagecounts: Sequence[int], cap: int,
    wr: float, method: str, beam_width: int | None,
    make_step_scorer, ScorerWeights, regret_fn,
) -> dict[str, Any]:
    scorer = make_step_scorer(d_norm, ScorerWeights(w_regret=wr), pagecounts, cap)
    visited = {order[0]} if order else set()
    transitions: list[dict[str, Any]] = []

    for step, (i, j) in enumerate(zip(order, order[1:])):
        # This is exactly the state shape passed by greedy_schedule / beam search:
        # all query indices not yet visited, including the candidate successor j.
        unscheduled = [k for k in range(len(query_ids)) if k not in visited]
        candidates = []
        for cand in unscheduled:
            r = float(regret_fn(i, cand, unscheduled, d_norm))
            s = float(scorer(i, cand, unscheduled))
            best_pred_idx = None
            pred_pool = [k for k in unscheduled if k != cand and k != i]
            if pred_pool:
                best_pred_idx = max(pred_pool, key=lambda k: d_norm[k][cand])
            candidates.append({
                "successor": query_ids[cand],
                "successor_index": cand,
                "directional_reuse": int(d_raw[i][cand]),
                "normalized_reuse": float(d_norm[i][cand]),
                "regret": r,
                "transition_score": s,
                "best_remaining_predecessor": query_ids[best_pred_idx] if best_pred_idx is not None else None,
                "best_remaining_predecessor_reuse": (
                    float(d_norm[best_pred_idx][cand]) if best_pred_idx is not None else 0.0
                ),
                "selected": cand == j,
            })

        ranked = sorted(candidates, key=lambda x: (-x["transition_score"], x["successor_index"]))
        selected = next(x for x in candidates if x["selected"])
        alternatives = ranked[:6]
        if not any(x["selected"] for x in alternatives):
            alternatives.append(selected)

        transitions.append({
            "step": step + 1,
            "from": query_ids[i],
            "to": query_ids[j],
            "directional_reuse": selected["directional_reuse"],
            "normalized_reuse": selected["normalized_reuse"],
            "regret": selected["regret"],
            "transition_score": selected["transition_score"],
            "selected_w_regret": wr,
            "beam_width": beam_width,
            "remaining_count": len(unscheduled),
            "transition_role": ("retained_beam_transition" if method == "sweep_beam_D" else "greedy_transition"),
            "best_remaining_predecessor": selected["best_remaining_predecessor"],
            "alternatives": alternatives,
            "text": _human_explanation(
                method=method,
                selected_q=query_ids[j],
                predecessor_q=query_ids[i],
                wr=wr,
                regret_value=selected["regret"],
                best_pred_q=selected["best_remaining_predecessor"],
            ),
        })
        visited.add(j)

    return {
        "available": True,
        "semantics": {
            "construction_score": (
                "Directional reuse and regret guide candidate construction. "
                "This score does not select the final completed schedule."
            ),
            "complete_order_score": (
                "Complete-order page-level clock-sweep simulation produces F_hit; "
                "the completed candidate with the largest F_hit is retained."
            ),
        },
        "transitions": transitions,
    }


def generate_one(root: Path, out_dir: Path, workload: str, cap: int) -> Path:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.bayesopt.data import load_workload_pages
    from src.bayesopt.objective import ExactSimObjective
    from src.bayesopt.run_regret_sweep import (
        DEFAULT_BEAM_WIDTHS,
        DEFAULT_REGRET_GRID,
        _sweep_beam,
        _sweep_multistart,
    )
    from src.bayesopt.step_scorer import (
        ScorerWeights,
        beam_search_schedule,
        good_start_set,
        make_step_scorer,
        multistart_greedy_schedules,
        regret,
        row_normalize,
    )
    from src.simulator.cache_simulator import compute_directional_matrix

    profile_dir = root / "page_access" / workload
    if not profile_dir.exists():
        raise FileNotFoundError(f"profile directory not found: {profile_dir}")

    wp = load_workload_pages(profile_dir, exclude=set())
    page_sets = list(wp.page_sets)
    query_ids = [str(q) for q in wp.query_ids]
    n = len(query_ids)
    pagecounts = [len(ps) for ps in page_sets]

    print(f"[{workload} @ {cap:,}] loading {n} profiles; computing D(C)…")
    d_raw = compute_directional_matrix(page_sets, cap)
    d_norm = row_normalize(d_raw)
    objective = ExactSimObjective(page_sets, cap, d_matrix=d_raw)

    def fhit(schedule: list[int]) -> float:
        metrics, _ = objective.evaluate_schedule(schedule)
        return metrics.hit_ratio

    # Use the production sweep helpers to select the same w_r / beam width.
    sweep_d = _sweep_multistart(
        d_norm, pagecounts, cap, n, fhit, list(DEFAULT_REGRET_GRID),
        page_sets, d_raw, 1,
    )
    sweep_b = _sweep_beam(
        d_norm, pagecounts, cap, n, fhit, list(DEFAULT_REGRET_GRID),
        tuple(DEFAULT_BEAM_WIDTHS), page_sets, d_raw, 1,
    )

    def select_sweep_d() -> tuple[list[int], Any, dict[str, Any]]:
        wr = float(sweep_d.best_w_regret)
        scorer = make_step_scorer(d_norm, ScorerWeights(w_regret=wr), pagecounts, cap)
        starts = good_start_set(d_norm, pagecounts, k=4)
        candidates = multistart_greedy_schedules(scorer, n, pagecounts, starts)
        metrics = [objective.evaluate_schedule(c)[0] for c in candidates]
        best_f = max(m.hit_ratio for m in metrics)
        idx = next(i for i, m in enumerate(metrics) if m.hit_ratio == best_f)
        if abs(best_f - sweep_d.best_fhit) > 1e-12:
            raise RuntimeError("Sweep-D reconstruction does not match production sweep result")
        return candidates[idx], metrics[idx], {
            "selected_w_regret": wr,
            "start_query": query_ids[candidates[idx][0]],
            "deterministic_starts": [query_ids[s] for s in starts],
            "regret_grid": list(DEFAULT_REGRET_GRID),
            "raw_candidates": sweep_d.n_evals,
            "distinct_candidates_scored": sweep_d.n_distinct,
        }

    def select_beam() -> tuple[list[int], Any, dict[str, Any]]:
        wr = float(sweep_b.best_w_regret)
        bw = int(sweep_b.best_extra["beam_width"])
        scorer = make_step_scorer(d_norm, ScorerWeights(w_regret=wr), pagecounts, cap)
        candidates = beam_search_schedule(scorer, n, pagecounts, bw)
        metrics = [objective.evaluate_schedule(c)[0] for c in candidates]
        best_f = max(m.hit_ratio for m in metrics)
        idx = next(i for i, m in enumerate(metrics) if m.hit_ratio == best_f)
        if abs(best_f - sweep_b.best_fhit) > 1e-12:
            raise RuntimeError("Sweep+Beam-D reconstruction does not match production sweep result")
        return candidates[idx], metrics[idx], {
            "selected_w_regret": wr,
            "beam_width": bw,
            "start_query": query_ids[candidates[idx][0]],
            "regret_grid": list(DEFAULT_REGRET_GRID),
            "beam_width_grid": list(DEFAULT_BEAM_WIDTHS),
            "raw_candidates": sweep_b.n_evals,
            "distinct_candidates_scored": sweep_b.n_distinct,
        }

    methods: dict[str, Any] = {}
    for method, selector in (("sweep_D", select_sweep_d), ("sweep_beam_D", select_beam)):
        order_idx, metrics, construction = selector()
        beam_width = construction.get("beam_width")
        explanation = _build_transition_metadata(
            order=order_idx,
            query_ids=query_ids,
            d_raw=d_raw,
            d_norm=d_norm,
            pagecounts=pagecounts,
            cap=cap,
            wr=float(construction["selected_w_regret"]),
            method=method,
            beam_width=int(beam_width) if beam_width is not None else None,
            make_step_scorer=make_step_scorer,
            ScorerWeights=ScorerWeights,
            regret_fn=regret,
        )
        methods[method] = {
            "method": method,
            "label": METHOD_LABELS[method],
            "order": [query_ids[i] for i in order_idx],
            "query_count": len(order_idx),
            "total_requests": int(metrics.total_requests),
            "total_hits": int(metrics.total_hits),
            "total_misses": int(metrics.page_reads),
            "f_hit": float(metrics.hit_ratio),
            "construction": construction,
            "explanation": explanation,
            "source": {
                "kind": "Queryosity research implementation",
                "generator": "scripts/generate_demo_artifacts.py",
                "profile_dir": str(profile_dir.relative_to(root)),
            },
        }

    artifact = {
        "schema_version": 3,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "queryosity-whatif-demo/scripts/generate_demo_artifacts.py",
        "backend": "real",
        "research_commit": _git_commit(root),
        "workload": workload,
        "capacity_pages": cap,
        "page_size_bytes": 8192,
        "profile_dir": str(profile_dir.relative_to(root)),
        "methods": methods,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{workload}_{cap}.json"
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        f"  wrote {out.name}: Sweep-D={methods['sweep_D']['f_hit']:.6f}, "
        f"Sweep+Beam-D={methods['sweep_beam_D']['f_hit']:.6f}"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--workload", choices=WORKLOADS)
    p.add_argument("--capacity", type=int, choices=CAPACITIES)
    p.add_argument("--all", action="store_true", help="generate all 3 workloads × 3 capacities")
    args = p.parse_args()

    root = args.project_root.resolve()
    if not (root / "src").is_dir() or not (root / "page_access").is_dir():
        p.error(f"{root} does not look like Queryosity-Killed-The-Cache (src/ + page_access/ required)")

    if args.all:
        jobs = [(w, c) for w in WORKLOADS for c in CAPACITIES]
    else:
        if args.workload is None or args.capacity is None:
            p.error("use --all or provide both --workload and --capacity")
        jobs = [(args.workload, args.capacity)]

    for workload, cap in jobs:
        generate_one(root, args.out_dir.resolve(), workload, cap)


if __name__ == "__main__":
    main()
