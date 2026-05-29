"""
Plot wall-clock experiment results produced by ``run_sweep``.

Reads the tidy long CSV and produces, for a chosen OS-cache mode:

  1. cumulative shared-buffer reads vs. query position (one line per
     schedule, averaged across reps)
  2. workload completion time per schedule (bar + std error bars)
  3. total shared-buffer reads per schedule (bar + std error bars)
  4. overall hit ratio per schedule (bar)

Figures are written to plots/.

Example
-------
    python -m src.experiment.plot_sweep \
        --results experiment_logs/results_tpch_100k_warm.csv \
        --os-cache warm \
        --out-prefix plots/tpch_100k_warm
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

# Stable display order + labels.
SCHEDULE_ORDER = ["random", "ga_m", "greedy_d", "ga_d"]
SCHEDULE_LABELS = {
    "random": "Random",
    "ga_m": "GA+M",
    "greedy_d": "Greedy-D",
    "ga_d": "GA+D",
}


def load_rows(path: Path, os_cache: str) -> list[dict]:
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["os_cache"] == os_cache]
    if not rows:
        raise SystemExit(f"No rows with os_cache={os_cache!r} in {path}")
    return rows


def present_schedules(rows: list[dict]) -> list[str]:
    seen = {r["schedule"] for r in rows}
    ordered = [s for s in SCHEDULE_ORDER if s in seen]
    ordered += sorted(seen - set(ordered))
    return ordered


def cumulative_reads_by_position(rows, schedules):
    """
    Return {schedule: [mean cumulative reads at position 1..N]} averaged
    over reps.  Uses per-query rows (position > 0) only.
    """
    # per (schedule, rep) -> list of (position, reads)
    by_sched_rep: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for r in rows:
        pos = int(r["position"])
        if pos == 0:
            continue
        key = (r["schedule"], r["rep"])
        by_sched_rep[key].append((pos, int(r["shared_read_blocks"])))

    out: dict[str, list[float]] = {}
    for sched in schedules:
        rep_curves = []
        for (s, _rep), seq in by_sched_rep.items():
            if s != sched:
                continue
            seq.sort(key=lambda t: t[0])
            cum, acc = [], 0
            for _pos, reads in seq:
                acc += reads
                cum.append(acc)
            rep_curves.append(cum)
        if not rep_curves:
            continue
        n = min(len(c) for c in rep_curves)
        mean_curve = [
            sum(c[i] for c in rep_curves) / len(rep_curves) for i in range(n)
        ]
        out[sched] = mean_curve
    return out


def totals_by_schedule(rows, schedules):
    """
    Return {schedule: {'time': [..], 'reads': [..], 'hits': [..]}} from the
    TOTAL rows (one per rep).
    """
    agg = {s: {"time": [], "reads": [], "hits": []} for s in schedules}
    for r in rows:
        if int(r["position"]) != 0:
            continue
        s = r["schedule"]
        if s not in agg:
            continue
        agg[s]["time"].append(float(r["elapsed_ms"]))
        agg[s]["reads"].append(int(r["shared_read_blocks"]))
        agg[s]["hits"].append(int(r["shared_hit_blocks"]))
    return agg


def _mean_std(xs):
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var ** 0.5


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--os-cache", default="warm", choices=["warm", "cold"])
    parser.add_argument("--out-prefix", default="plots/sweep")
    parser.add_argument("--cache-label", default="100K")
    args = parser.parse_args(argv)

    rows = load_rows(Path(args.results), args.os_cache)
    schedules = present_schedules(rows)
    colors = {s: c for s, c in zip(SCHEDULE_ORDER, ["#888888", "#d62728", "#1f77b4", "#2ca02c"])}
    labels = [SCHEDULE_LABELS.get(s, s) for s in schedules]
    Path(args.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    tag = f"C={args.cache_label}, {args.os_cache} OS cache"

    # ---- Figure 1: cumulative reads vs position ----
    curves = cumulative_reads_by_position(rows, schedules)
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in schedules:
        if s not in curves:
            continue
        y = curves[s]
        ax.plot(range(1, len(y) + 1), y, marker="o", ms=3,
                color=colors.get(s), label=SCHEDULE_LABELS.get(s, s))
    ax.set_xlabel("Query position in schedule")
    ax.set_ylabel("Cumulative shared-buffer reads (pages)")
    ax.set_title(f"Cumulative shared-buffer reads — {tag}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1 = f"{args.out_prefix}_cumulative_reads.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ---- aggregates ----
    agg = totals_by_schedule(rows, schedules)
    x = range(len(schedules))

    # ---- Figure 2: completion time ----
    means = [_mean_std(agg[s]["time"])[0] / 1000.0 for s in schedules]  # seconds
    stds = [_mean_std(agg[s]["time"])[1] / 1000.0 for s in schedules]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, means, yerr=stds, capsize=5,
           color=[colors.get(s, "#777") for s in schedules])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Workload completion time (s)")
    ax.set_title(f"Workload completion time — {tag}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = f"{args.out_prefix}_completion_time.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # ---- Figure 3: total reads ----
    means = [_mean_std(agg[s]["reads"])[0] for s in schedules]
    stds = [_mean_std(agg[s]["reads"])[1] for s in schedules]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, means, yerr=stds, capsize=5,
           color=[colors.get(s, "#777") for s in schedules])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total shared-buffer reads (pages)")
    ax.set_title(f"Total shared-buffer reads — {tag}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p3 = f"{args.out_prefix}_total_reads.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)

    # ---- Figure 4: hit ratio (aggregate AND per-query mean) ----
    # Aggregate hit ratio: Σ hits / Σ (hits + reads) across the whole
    # workload.  Per-query mean: average over queries of each query's
    # individual hit ratio.  The paper reports the latter as "Average
    # Cache Hit Ratio"; we show both so the comparison against the
    # paper is unambiguous.
    hit_pct_agg: list[float] = []
    hit_pct_perq: list[float] = []
    for s in schedules:
        h = sum(agg[s]["hits"])
        r = sum(agg[s]["reads"])
        hit_pct_agg.append(100.0 * h / (h + r) if (h + r) else 0.0)
        # Per-query average over per-query rows of this schedule (any
        # rep).  Compute by recomputing from the raw rows so we use
        # exactly the same definition as run_sweep.
        per_q_ratios = []
        for r_row in rows:
            if r_row["schedule"] != s or int(r_row["position"]) == 0:
                continue
            hh = int(r_row["shared_hit_blocks"])
            rr = int(r_row["shared_read_blocks"])
            if hh + rr > 0:
                per_q_ratios.append(100.0 * hh / (hh + rr))
            else:
                per_q_ratios.append(0.0)
        hit_pct_perq.append(
            sum(per_q_ratios) / len(per_q_ratios)
            if per_q_ratios
            else 0.0
        )

    width = 0.4
    xs = list(x)
    xs_agg = [v - width / 2 for v in xs]
    xs_perq = [v + width / 2 for v in xs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        xs_agg, hit_pct_agg, width,
        label="Aggregate (Σ hits / Σ blocks)",
        color=[colors.get(s, "#777") for s in schedules],
        edgecolor="black",
    )
    ax.bar(
        xs_perq, hit_pct_perq, width,
        label="Per-query mean (paper's metric)",
        color=[colors.get(s, "#777") for s in schedules],
        hatch="//",
        edgecolor="black",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Shared-buffer hit ratio (%)")
    ax.set_title(f"Hit ratio — {tag}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p4 = f"{args.out_prefix}_hit_ratio.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)

    print("Wrote:")
    for p in (p1, p2, p3, p4):
        print(f"  {p}")


if __name__ == "__main__":
    main()
