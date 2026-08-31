"""
Deterministic synthetic page-access profiles for the MOCK backend.

These are NOT the TPC-H / TPC-DS / JOB benchmarks. They are small, seeded,
synthetic workloads that reproduce the *structure* the real ones have --
cross-query page overlap plus per-query private pages -- so that reordering a
schedule changes the simulated hit ratio in a sensible way and the whole UI
can be exercised locally.

Sizes are tuned so that at the smallest preset capacity (102,400 pages) the
working set overflows the buffer (reordering matters), while at the largest
preset (524,288 pages) most of the footprint fits (hit ratio saturates) --
mirroring the qualitative regime change in the real results.

Everything produced from here is surfaced as ``backend: "mock"`` and must
never be shown as a real benchmark measurement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class MockSpec:
    label: str
    n_queries: int
    n_bands: int          # number of shared "relation" bands
    band_pages: int       # pages per shared band
    private_pages: int    # size of each query's private range
    seed: int


# Rough analogues of the real workloads (query counts match the real ones so
# the UI's large-workload handling is exercised for tpcds/job).
MOCK_SPECS: dict[str, MockSpec] = {
    "tpch": MockSpec("TPC-H (SF10) - MOCK", n_queries=22, n_bands=12,
                     band_pages=30_000, private_pages=1_200, seed=1_071),
    "tpcds": MockSpec("TPC-DS - MOCK", n_queries=93, n_bands=40,
                      band_pages=9_000, private_pages=600, seed=262_144),
    "job": MockSpec("JOB / IMDB - MOCK", n_queries=113, n_bands=48,
                    band_pages=8_000, private_pages=500, seed=524_288),
}


def _build(spec: MockSpec) -> list[tuple[int, ...]]:
    """Build one workload's page sets as sorted tuples of page ids."""
    rng = random.Random(spec.seed)

    # Lay out shared bands and a private region back to back in page-id space.
    bands: list[range] = []
    cursor = 0
    for _ in range(spec.n_bands):
        bands.append(range(cursor, cursor + spec.band_pages))
        cursor += spec.band_pages
    private_base = cursor

    page_sets: list[tuple[int, ...]] = []
    for q in range(spec.n_queries):
        pages: set[int] = set()

        # Each query touches 1-3 shared bands, taking a contiguous slice of
        # each. The home band is scattered across query indices (not q %
        # n_bands), so page-sharing queries are NOT adjacent in natural order.
        # A good schedule has to bring them together -- which is exactly the
        # reuse a scheduler exploits and what the what-if editor lets you probe.
        home = rng.randrange(spec.n_bands)
        chosen = {home}
        if rng.random() < 0.7:
            chosen.add(rng.randrange(spec.n_bands))
        if rng.random() < 0.35:
            chosen.add(rng.randrange(spec.n_bands))

        for b in chosen:
            band = bands[b]
            span = rng.randint(spec.band_pages // 3, spec.band_pages)
            start = rng.randint(band.start, max(band.start, band.stop - span))
            step = rng.choice((1, 1, 1, 2))  # occasional strided scan
            pages.update(range(start, min(start + span, band.stop), step))

        # Private pages, unique to this query (one-off dimension scans, etc.).
        priv = spec.private_pages
        p0 = private_base + q * priv
        pages.update(range(p0, p0 + priv))

        page_sets.append(tuple(sorted(pages)))

    return page_sets


# Build once at import; deterministic, so repeated builds are identical.
_CACHE: dict[str, list[tuple[int, ...]]] = {}


def workload_names() -> list[str]:
    return list(MOCK_SPECS.keys())


def label(name: str) -> str:
    return MOCK_SPECS[name].label


def query_ids(name: str) -> list[str]:
    return [str(i) for i in range(1, MOCK_SPECS[name].n_queries + 1)]


def page_sets(name: str) -> list[tuple[int, ...]]:
    if name not in _CACHE:
        _CACHE[name] = _build(MOCK_SPECS[name])
    return _CACHE[name]


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    from reference_sim import simulate_clock_sweep

    for wl in workload_names():
        ps = page_sets(wl)
        distinct = len(set().union(*ps))
        total = sum(len(s) for s in ps)
        nat = list(range(len(ps)))
        rev = list(reversed(nat))
        for cap in (102_400, 262_144, 524_288):
            h_nat, r = simulate_clock_sweep([ps[i] for i in nat], cap)
            h_rev, _ = simulate_clock_sweep([ps[i] for i in rev], cap)
            print(f"{wl:6s} cap={cap:>7d} distinct={distinct:>7d} "
                  f"requests={r:>8d} hit%(nat)={100*h_nat/r:5.1f} "
                  f"hit%(rev)={100*h_rev/r:5.1f} Δ={100*(h_nat-h_rev)/r:+5.1f}pp")
