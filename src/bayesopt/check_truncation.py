"""
Profile truncation / correctness check.

A page profiler run with a finite shared-buffer clamps any query whose true
working set exceeds the buffer: its recorded page set saturates at the cap.
The signature is a cluster of near-identical large page counts sitting just
under a power-of-two cap (e.g. several queries at ~262 092 under the 2GB /
262 144 cap).  Genuinely large queries do not produce near-identical counts.

This decides three things per workload:
  1. Did the 6GB re-profiling take effect, or are queries still clamped at
     the old 2GB cap (262 144)?  (the open TODO)
  2. Which queries sit at the 6GB ceiling — true working set >= 6GB, an
     accepted limitation that shifts absolute F_hit but not the relative
     method comparison.
  3. THE one that corrupts the experiment: any query truncated at or below
     the largest cache tested (524 288).  A cap <= the test cache means the
     simulator can't even fill the cache for that query, so the tested range
     itself is distorted, not just the absolute numbers.

Verdict: CLEAN (no cap), OK_ABOVE_RANGE (cap > largest test cache), or
CORRUPT_IN_RANGE (cap <= largest test cache -> re-profile at >= 6GB).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from src.bayesopt.data import load_workload_pages
from src.utilities.constants import WORKLOAD_DIRS

# candidate profiler caps in pages (8KB pages): 2/4/6/8 GB
CANDIDATE_CAPS = [
    ("2GB", 262144),
    ("4GB", 524288),
    ("6GB", 786432),
    ("8GB", 1048576),
]
DEFAULT_TEST_CACHES = [102400, 262144, 524288]
_NEAR = 0.01    # within 1% at/below a cap counts as "at the cap"
_TIGHT = 0.002  # top counts within 0.2% of each other => clamped, not natural


def _percentile(sorted_desc: list[int], q: float) -> int:
    if not sorted_desc:
        return 0
    asc = sorted_desc[::-1]
    idx = min(len(asc) - 1, max(0, int(round(q * (len(asc) - 1)))))
    return asc[idx]


@dataclass
class TruncationResult:
    effective_cap: int | None
    effective_label: str
    cap_hits: list[tuple[str, int, int]]
    top_spread: float
    clamped_top: bool
    n_at_cap: int
    verdict: str
    note: str


def classify_counts(
    counts_desc: list[int], test_caches: list[int],
) -> TruncationResult:
    """
    Pure truncation classifier over page counts (descending).  Returns the
    effective cap (if any), the per-cap pile-up scan, top-spread, and the
    verdict.  Kept free of I/O so it can be unit-tested.
    """
    mx = counts_desc[0]
    largest_test = sorted(test_caches)[-1]
    cap_hits: list[tuple[str, int, int]] = []
    for label, cap in CANDIDATE_CAPS:
        near = sum(1 for c in counts_desc if cap * (1 - _NEAR) <= c <= cap * 1.005)
        cap_hits.append((label, cap, near))

    effective_cap: int | None = None
    effective_label = "none"
    for label, cap, near in cap_hits:
        if cap * (1 - _NEAR) <= mx <= cap * 1.005 and near >= 2:
            effective_cap = cap
            effective_label = label
            break

    top = counts_desc[:max(2, min(8, len(counts_desc)))]
    top_spread = (top[0] - top[-1]) / top[0] if top[0] else 1.0
    clamped_top = effective_cap is not None and top_spread <= _TIGHT
    n_at_cap = (sum(1 for c in counts_desc if c >= mx * (1 - _NEAR))
                if effective_cap is not None else 0)

    if effective_cap is None:
        verdict = "CLEAN"
        note = "no cap pile-up; max page count is a genuine query size."
    elif effective_cap > largest_test:
        verdict = "OK_ABOVE_RANGE"
        note = (f"cap {effective_cap:,} ({effective_label}) is above the largest "
                f"tested cache {largest_test:,}; clamped queries still exceed the "
                f"cache, so the tested range is faithful (absolute F_hit shifted, "
                f"relative method comparison unaffected).")
    else:
        verdict = "CORRUPT_IN_RANGE"
        note = (f"cap {effective_cap:,} ({effective_label}) is at/below the largest "
                f"tested cache {largest_test:,}; clamped queries cannot fill the "
                f"cache, so cache tests up to {largest_test:,} are distorted. "
                f"Re-profile at >= 6GB.")

    return TruncationResult(effective_cap, effective_label, cap_hits, top_spread,
                            clamped_top, n_at_cap, verdict, note)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", choices=list(WORKLOAD_DIRS), required=True)
    p.add_argument("--page-access-dir", type=Path, required=True)
    p.add_argument("--exclude", default="")
    p.add_argument("--test-caches", default="102400,262144,524288")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    exclude = {q.strip() for q in args.exclude.split(",") if q.strip()}
    test_caches = sorted(int(c) for c in args.test_caches.split(",") if c.strip())
    largest_test = test_caches[-1]

    print(f"Loading page sets from {args.page_access_dir}…")
    wp = load_workload_pages(args.page_access_dir, exclude=exclude)
    pairs = sorted(((len(ps), qid) for ps, qid in zip(wp.page_sets, wp.query_ids)),
                   reverse=True)
    counts = [c for c, _ in pairs]
    mx = counts[0]

    # cap scan + classification (pure, tested separately)
    cls = classify_counts(counts, test_caches)
    cap_hits = cls.cap_hits
    effective_cap = cls.effective_cap
    effective_label = cls.effective_label
    top_spread = cls.top_spread
    clamped_top = cls.clamped_top
    n_at_cap = cls.n_at_cap
    verdict = cls.verdict
    note = cls.note
    top = counts[:max(2, min(8, len(counts)))]
    overflow = {tc: sum(1 for c in counts if c > tc) for tc in test_caches}

    print(f"\n=== truncation check: {args.workload} ({args.page_access_dir}) ===")
    print(f"  n={wp.n} queries, total unique pages={wp.total_unique_pages:,}")
    print(f"  page counts: min={counts[-1]:,}  p50={_percentile(counts, 0.5):,}"
          f"  p90={_percentile(counts, 0.9):,}  max={mx:,}")
    print("  top counts by query:")
    for c, qid in pairs[:8]:
        print(f"    {qid:>10s}  {c:,}")
    print("  cap pile-up scan (queries within 1% at/below cap):")
    for label, cap, near in cap_hits:
        flag = "  <-- pile-up" if near >= 2 and cap * (1 - _NEAR) <= mx * 1.005 else ""
        print(f"    {label:4s} ({cap:,}): {near}{flag}")
    print(f"  top-{len(top)} spread={top_spread * 100:.2f}% "
          f"({'clamped/near-identical' if clamped_top else 'naturally varying'})")
    print(f"  effective profiling cap: "
          f"{('~%d (%s)' % (effective_cap, effective_label)) if effective_cap else 'none detected'}")
    print(f"  queries truncated at cap: {n_at_cap}")
    print("  queries exceeding each tested cache (where D vs M matters):")
    for tc in test_caches:
        print(f"    > {tc:,}: {overflow[tc]}")
    print(f"  largest tested cache: {largest_test:,}")
    print(f"  VERDICT: {verdict}")
    print(f"           {note}")

    if args.out_dir is not None:
        summary = {
            "workload": args.workload, "page_access_dir": str(args.page_access_dir),
            "n_queries": wp.n, "total_unique_pages": wp.total_unique_pages,
            "max_page_count": mx, "min_page_count": counts[-1],
            "p50": _percentile(counts, 0.5), "p90": _percentile(counts, 0.9),
            "top_counts": [{"query": qid, "pages": c} for c, qid in pairs[:8]],
            "cap_scan": [{"label": l, "cap": cap, "n_near": n} for l, cap, n in cap_hits],
            "top_spread_frac": top_spread, "clamped_top": clamped_top,
            "effective_cap": effective_cap, "effective_cap_label": effective_label,
            "n_at_cap": n_at_cap, "test_caches": test_caches,
            "largest_test_cache": largest_test, "overflow_per_cache": overflow,
            "verdict": verdict, "note": note,
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out = args.out_dir / f"truncation_{args.workload}.json"
        out.write_text(json.dumps(summary, indent=2))
        print(f"  wrote: {out}")


if __name__ == "__main__":
    main()
