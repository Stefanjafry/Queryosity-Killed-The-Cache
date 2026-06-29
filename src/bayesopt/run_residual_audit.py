"""
Phase 1 — residual decomposition audit (no BO, no search).

Decomposes the directional utility matrix ``D`` into a per-row-scaled
copy of the symmetric overlap matrix ``M`` plus an edge-specific
residual::

    alpha_i      = <D_i, M_i> / <M_i, M_i>          (least-squares, off-diag)
    E[i][j]      = D[i][j] - alpha_i * M[i][j]

and measures how much of ``D`` the row-rank-1 model ``alpha_i * M``
already explains.  This is the gate for the whole residual/windowed
plan: if ``E`` is negligible everywhere, ``D`` collapses to a row-scaled
``M`` and the directional contribution beyond symmetric overlap is
marginal — a clean negative — and Phase 2 should not be built.

The audit runs on the simulator-side matrices only (no live SQL), so
wall-clock query exclusions (e.g. TPC-H q13) do **not** apply here;
profile every query for which a page set exists.

Outputs (under ``--out-dir``):

* ``residual_audit_summary.json``  — headline metrics + mechanical verdict
* ``residual_audit_rows.csv``      — per-query row diagnostics
* ``residual_top_edges.csv``       — largest +/- residual edges
* ``feature_redundancy.csv``       — Pearson r among M/D/gap/ratio/E
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.bayesopt.data import load_workload_pages
from src.simulator.cache_simulator import (
    compute_directional_matrix,
    compute_overlap_matrix,
    compute_residual,
)
from src.utilities.constants import PROJECT_ROOT, WORKLOAD_DIRS

_EPS = 1e-9


@dataclass
class RowDiag:
    """Per-query (per predecessor row *i*) residual diagnostics."""

    index: int
    query_id: str
    pagecount: int
    residency: int  # |R(Q_i; C)|
    overflows: bool  # pagecount > C
    alpha_ls: float  # least-squares row scalar
    alpha_sum: float  # sum-ratio row scalar
    survivor_ratio_mean: float  # mean_j D[i][j]/(M[i][j]+eps); should track alpha
    d_row_norm: float
    e_row_norm: float
    e_row_frac: float  # ||E_i|| / ||D_i||
    row_r2: float  # variance of D_i explained by alpha_i * M_i


@dataclass
class AuditResult:
    """Everything the audit measures for one (workload, cache) pair."""

    n_queries: int
    cache_pages: int
    total_unique_pages: int
    overall_r2: float
    e_frac_of_d: float  # ||E||_F / ||D||_F   (off-diagonal)  <-- gate metric
    overflow_energy_frac: float  # share of sum(E^2) on overflowing predecessors
    d_gt_m_violations: int
    nonfinite: bool
    rows: list[RowDiag]
    top_pos_edges: list[tuple[int, int, float]]
    top_neg_edges: list[tuple[int, int, float]]
    redundancy: dict[str, dict[str, float]]
    verdict: dict[str, object] = field(default_factory=dict)


def _offdiag_mask(n: int) -> np.ndarray:
    """Boolean (n, n) mask that is True off the diagonal."""
    return ~np.eye(n, dtype=bool)


def _r2(target: np.ndarray, residual: np.ndarray) -> float:
    """Coefficient of determination of *target* given its *residual*."""
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    if ss_tot < _EPS:
        return 1.0  # target is constant; model explains it trivially
    ss_res = float(np.sum(residual ** 2))
    return 1.0 - ss_res / ss_tot


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two flattened vectors, NaN-safe."""
    if a.std() < _EPS or b.std() < _EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def audit_matrices(
    m_matrix: list[list[int]],
    d_matrix: list[list[int]],
    pagecounts: list[int],
    residencies: list[int],
    query_ids: list[str],
    cache_pages: int,
    total_unique_pages: int,
    top_k_edges: int = 25,
    continue_e_frac: float = 0.10,
    overflow_concentration: float = 0.50,
) -> AuditResult:
    """
    Pure-math core of the audit.  Decoupled from loading so it can be
    unit-tested on hand-built matrices with known ``alpha`` and ``E``.
    """
    M = np.asarray(m_matrix, dtype=float)
    D = np.asarray(d_matrix, dtype=float)
    n = M.shape[0]
    mask = _offdiag_mask(n)

    # Per-row least-squares and sum-ratio survival coefficients (off-diag).
    alpha_ls = np.zeros(n)
    alpha_sum = np.zeros(n)
    survivor_mean = np.zeros(n)
    E = np.zeros((n, n))
    for i in range(n):
        m_i = M[i][mask[i]]
        d_i = D[i][mask[i]]
        denom = float(np.dot(m_i, m_i))
        alpha_ls[i] = float(np.dot(d_i, m_i) / denom) if denom > _EPS else 0.0
        sm = float(m_i.sum())
        alpha_sum[i] = float(d_i.sum() / sm) if sm > _EPS else 0.0
        survivor_mean[i] = float(np.mean(d_i / (m_i + _EPS)))
        E[i] = D[i] - alpha_ls[i] * M[i]
    E[~mask] = 0.0  # residual is defined off-diagonal only

    d_off = D[mask]
    e_off = E[mask]
    overall_r2 = _r2(d_off, e_off)
    d_norm = float(np.sqrt(np.sum(d_off ** 2)))
    e_norm = float(np.sqrt(np.sum(e_off ** 2)))
    e_frac = e_norm / (d_norm + _EPS)

    # How much residual energy sits on overflowing predecessor rows.
    overflow_rows = np.array([pc > cache_pages for pc in pagecounts])
    total_e2 = float(np.sum(E[mask] ** 2))
    overflow_e2 = float(np.sum((E[overflow_rows] ** 2))) if overflow_rows.any() else 0.0
    overflow_energy_frac = overflow_e2 / (total_e2 + _EPS)

    # D <= M must hold elementwise (R(Q_i;C) subset of P(Q_i)).
    d_gt_m = int(np.sum((D > M + 1e-6) & mask))

    # Per-row diagnostics.
    rows: list[RowDiag] = []
    for i in range(n):
        m_i = M[i][mask[i]]
        d_i = D[i][mask[i]]
        e_i = E[i][mask[i]]
        d_row_norm = float(np.sqrt(np.sum(d_i ** 2)))
        e_row_norm = float(np.sqrt(np.sum(e_i ** 2)))
        rows.append(
            RowDiag(
                index=i,
                query_id=query_ids[i],
                pagecount=pagecounts[i],
                residency=residencies[i],
                overflows=bool(overflow_rows[i]),
                alpha_ls=alpha_ls[i],
                alpha_sum=alpha_sum[i],
                survivor_ratio_mean=survivor_mean[i],
                d_row_norm=d_row_norm,
                e_row_norm=e_row_norm,
                e_row_frac=e_row_norm / (d_row_norm + _EPS),
                row_r2=_r2(d_i, e_i),
            )
        )

    # Top signed residual edges.
    edges = [
        (i, j, float(E[i][j]))
        for i in range(n)
        for j in range(n)
        if i != j
    ]
    edges_by_val = sorted(edges, key=lambda e: e[2])
    top_neg = edges_by_val[:top_k_edges]
    top_pos = list(reversed(edges_by_val[-top_k_edges:]))

    # Feature redundancy among flattened off-diagonal vectors.
    gap = (M - D)[mask]
    survivor = (D / (M + _EPS))[mask]
    feats = {"M": M[mask], "D": d_off, "gap": gap,
             "survivor_ratio": survivor, "E": e_off}
    names = list(feats)
    redundancy = {
        a: {b: _pearson(feats[a], feats[b]) for b in names} for a in names
    }

    nonfinite = bool(
        not np.isfinite(alpha_ls).all()
        or not np.isfinite(E).all()
        or not math.isfinite(overall_r2)
    )

    result = AuditResult(
        n_queries=n,
        cache_pages=cache_pages,
        total_unique_pages=total_unique_pages,
        overall_r2=overall_r2,
        e_frac_of_d=e_frac,
        overflow_energy_frac=overflow_energy_frac,
        d_gt_m_violations=d_gt_m,
        nonfinite=nonfinite,
        rows=rows,
        top_pos_edges=top_pos,
        top_neg_edges=top_neg,
        redundancy=redundancy,
    )

    # Pre-registered, mechanical verdict — decided by thresholds, not by eye.
    e_gate = e_frac >= continue_e_frac
    overflow_gate = overflow_energy_frac >= overflow_concentration
    result.verdict = {
        "continue_e_frac_threshold": continue_e_frac,
        "overflow_concentration_threshold": overflow_concentration,
        "e_frac_of_d": e_frac,
        "overflow_energy_frac": overflow_energy_frac,
        "e_gate_passed": e_gate,
        "overflow_gate_passed": overflow_gate,
        "continue_to_phase2": bool(e_gate and overflow_gate and not nonfinite
                                   and d_gt_m == 0),
        "rationale": (
            "E carries meaningful directional energy concentrated on "
            "overflowing predecessors — residual/windowed scorer is "
            "worth building."
            if (e_gate and overflow_gate)
            else "D is largely explained by row-scaled M and/or residual "
            "energy is diffuse — directional contribution beyond M looks "
            "marginal; treat as a candidate clean negative before building "
            "Phase 2."
        ),
    }
    return result


def _write_outputs(result: AuditResult, workload: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    summary = {
        "workload": workload,
        "cache_pages": result.cache_pages,
        "n_queries": result.n_queries,
        "total_unique_pages": result.total_unique_pages,
        "overall_r2_D_given_alphaM": result.overall_r2,
        "e_frac_of_d": result.e_frac_of_d,
        "overflow_energy_frac": result.overflow_energy_frac,
        "d_gt_m_violations": result.d_gt_m_violations,
        "nonfinite": result.nonfinite,
        "verdict": result.verdict,
    }
    p = out_dir / "residual_audit_summary.json"
    p.write_text(json.dumps(summary, indent=2))
    written.append(p)

    p = out_dir / "residual_audit_rows.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "index", "query_id", "pagecount", "residency", "overflows",
            "alpha_ls", "alpha_sum", "survivor_ratio_mean",
            "d_row_norm", "e_row_norm", "e_row_frac", "row_r2",
        ])
        for r in result.rows:
            w.writerow([
                r.index, r.query_id, r.pagecount, r.residency, int(r.overflows),
                f"{r.alpha_ls:.6f}", f"{r.alpha_sum:.6f}",
                f"{r.survivor_ratio_mean:.6f}", f"{r.d_row_norm:.3f}",
                f"{r.e_row_norm:.3f}", f"{r.e_row_frac:.6f}", f"{r.row_r2:.6f}",
            ])
    written.append(p)

    p = out_dir / "residual_top_edges.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sign", "pred_i", "succ_j", "residual_E"])
        for i, j, v in result.top_pos_edges:
            w.writerow(["pos", i, j, f"{v:.3f}"])
        for i, j, v in result.top_neg_edges:
            w.writerow(["neg", i, j, f"{v:.3f}"])
    written.append(p)

    p = out_dir / "feature_redundancy.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh)
        names = list(result.redundancy)
        w.writerow([""] + names)
        for a in names:
            w.writerow([a] + [f"{result.redundancy[a][b]:.4f}" for b in names])
    written.append(p)

    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=list(WORKLOAD_DIRS), required=True)
    p.add_argument("--cache-pages", type=int, required=True)
    p.add_argument(
        "--page-access-dir", type=Path, required=True,
        help="Per-query page-access CSV directory (use the _6gb dirs).",
    )
    p.add_argument(
        "--exclude", default="",
        help="Comma-separated query ids to drop (usually empty: the audit "
             "is simulator-only, so wall-clock exclusions do not apply).",
    )
    p.add_argument("--top-k-edges", type=int, default=25)
    p.add_argument(
        "--continue-e-frac", type=float, default=0.10,
        help="Pre-registered gate: min ||E||/||D|| to justify Phase 2.",
    )
    p.add_argument(
        "--overflow-concentration", type=float, default=0.50,
        help="Pre-registered gate: min share of residual energy on "
             "overflowing (pagecount>C) predecessor rows.",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=PROJECT_ROOT / "experiment_logs" / "residual_audit",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}

    print(f"Loading page sets from {args.page_access_dir}…")
    wp = load_workload_pages(args.page_access_dir, exclude=exclude)
    print(f"  {wp.n} queries, {wp.total_unique_pages:,} unique pages, "
          f"C = {args.cache_pages:,} pages")

    t0 = time.perf_counter()
    M = compute_overlap_matrix(wp.page_sets)
    D = compute_directional_matrix(wp.page_sets, args.cache_pages)
    pagecounts = [len(ps) for ps in wp.page_sets]
    residencies = [len(compute_residual(ps, args.cache_pages))
                   for ps in wp.page_sets]
    print(f"  M, D, residuals in {time.perf_counter() - t0:.1f}s")

    result = audit_matrices(
        M, D, pagecounts, residencies, wp.query_ids,
        args.cache_pages, wp.total_unique_pages,
        top_k_edges=args.top_k_edges,
        continue_e_frac=args.continue_e_frac,
        overflow_concentration=args.overflow_concentration,
    )

    written = _write_outputs(result, args.workload, args.out_dir)

    v = result.verdict
    print("\n=== residual audit ===")
    print(f"  overall R²(D | alpha*M)   : {result.overall_r2:.4f}")
    print(f"  ||E|| / ||D||             : {result.e_frac_of_d:.4f}"
          f"  (gate >= {args.continue_e_frac})")
    print(f"  residual energy on overflow: {result.overflow_energy_frac:.4f}"
          f"  (gate >= {args.overflow_concentration})")
    print(f"  D > M violations          : {result.d_gt_m_violations}")
    print(f"  non-finite                : {result.nonfinite}")
    print(f"  --> continue_to_phase2    : {v['continue_to_phase2']}")
    print(f"      {v['rationale']}")
    print("  wrote:")
    for p in written:
        print(f"    {p}")


if __name__ == "__main__":
    main()
