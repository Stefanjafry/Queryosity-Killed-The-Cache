# directional vs symmetric utility matrix

"""
Plot: Directional vs Symmetric Utility Matrix

Side-by-side heatmaps of:
  M[i][j] = |P(Qi) ∩ P(Qj)|          (symmetric; existing baseline)
  D[i][j] = |R(Qi; C) ∩ P(Qj)|        (directional; this work)

Cells where M and D differ are exactly the cells where eviction
under clock-sweep destroys reusable overlap.  Asymmetric streaks
(``D[i][j]`` ≠ ``D[j][i]``) visually motivate the contribution:
the symmetric matrix cannot represent these differences.

This visualization is independent of the GA — it characterises the
workload + cache size pair, and is useful even before any scheduling
results are run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.simulator.cache_simulator import (
    compute_directional_matrix,
    compute_overlap_matrix,
)
from src.visualization.style import apply_style, OUTPUT_DIR

apply_style()


def plot_directional_matrix(
    page_sets: list[frozenset[int]],
    query_ids: list[str],
    cache_capacity_pages: int,
    ga_schedule: list[int] | None = None,
    workload: str = "tpch",
) -> Path:
    """
    Side-by-side heatmaps of the symmetric overlap and directional matrices.

    Both matrices use a shared color scale so the loss of utility from
    clock-sweep eviction is visually apparent: cells where ``M[i][j]``
    is hot but ``D[i][j]`` is cool indicate that Qᵢ's overlap with Qⱼ
    does not survive Qᵢ running first under the given cache capacity.

    Parameters
    ----------
    page_sets : list[frozenset[int]]
        Integer-encoded page sets (from ``encode_page_sets``).
    query_ids : list[str]
        Human-readable labels for each query, aligned with *page_sets*.
    cache_capacity_pages : int
        Cache capacity used to build the directional matrix.  This must
        match the capacity the scheduler is targetting; otherwise the
        residuals will not reflect simulator behaviour.
    ga_schedule : list[int] | None
        Optional GA-optimised permutation.  When supplied, axes are
        reordered to follow the schedule and a thin red box overlays
        the consecutive pairs (i, i+1) actually realised by the GA.
        Brightness along the upper sub-diagonal of D under that
        ordering is the schedule's predicted hit profile.
    workload : str
        Workload name; used in the title and filename.

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    n = len(page_sets)
    if n == 0:
        raise ValueError("Cannot plot empty page_sets")

    M = np.asarray(compute_overlap_matrix(page_sets), dtype=float)
    D = np.asarray(
        compute_directional_matrix(page_sets, cache_capacity_pages),
        dtype=float,
    )

    order = ga_schedule if ga_schedule is not None else list(range(n))
    M_ord = M[np.ix_(order, order)]
    D_ord = D[np.ix_(order, order)]
    labels = [query_ids[i] for i in order]

    # Mask diagonal so the colour scale focuses on inter-query overlaps.
    diag_mask = np.eye(n, dtype=bool)
    M_ord = np.ma.masked_where(diag_mask, M_ord)
    D_ord = np.ma.masked_where(diag_mask, D_ord)

    vmax = float(max(M_ord.max(), D_ord.max())) or 1.0
    vmin = 0.0

    fig, axes = plt.subplots(
        1, 2,
        figsize=(max(14, n * 0.8), max(7, n * 0.4)),
        sharey=True,
    )

    for ax, mat, name in (
        (axes[0], M_ord, "M[i][j] = |P(Qi) ∩ P(Qj)|  (symmetric)"),
        (axes[1], D_ord, f"D[i][j] = |R(Qi;C) ∩ P(Qj)|  (C = {cache_capacity_pages})"),
    ):
        im = ax.imshow(
            mat,
            cmap="YlOrRd",
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Current  Qj")
        ax.set_title(name, fontsize=10)
        ax.grid(False)

    axes[0].set_ylabel("Previous  Qi")

    # Single shared colorbar on the right.
    cbar = fig.colorbar(
        im, ax=axes.ravel().tolist(),
        shrink=0.85, label="Shared pages",
    )

    # When a GA schedule is supplied, highlight the realised edges
    # (consecutive pairs in the ordering) on the directional plot.
    if ga_schedule is not None and n >= 2:
        for k in range(n - 1):
            axes[1].add_patch(
                plt.Rectangle(  # type: ignore[attr-defined]
                    (k + 1 - 0.5, k - 0.5),
                    1, 1, fill=False,
                    edgecolor="black", linewidth=1.2,
                )
            )

    suffix = "_ga" if ga_schedule is not None else ""
    fig.suptitle(
        f"Utility Matrices — {workload.upper()}",
        fontsize=12, y=1.02,
    )

    out = OUTPUT_DIR / f"directional_matrix_{workload}{suffix}.png"
    fig.savefig(str(out.resolve()), bbox_inches="tight")
    plt.close(fig)
    return out


__all__ = ["plot_directional_matrix"]
