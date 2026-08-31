"""
Simulator adapter.

The API scores every user schedule through exactly one call to
``Adapter.score(workload, order, cap)`` -> one page-level clock-sweep simulation.
That is the *only* thing the /api/score path does; it never runs Sweep-D or
Sweep+Beam-D search. The two are completely different costs and are
kept apart deliberately.

Two backends satisfy the same tiny interface:

    * RealSimulatorBackend  -- imports the project's load_workload_pages and
      simulate_schedule_page_level and runs the real clock-sweep simulator.
      Used on the VM. Source of truth.

    * MockSimulatorBackend  -- synthetic workloads + bundled illustrative artifacts.
      Used for local UI work and tests. Every result it returns is tagged
      ``backend == "mock"`` so it can never be mistaken for a real measurement.

Selection is driven by config.SIM_BACKEND (auto / real / mock).
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol, Sequence

import config


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WorkloadInfo:
    name: str
    label: str
    query_ids: list[str]          # available queries, natural (numeric) order
    page_counts: dict[str, int]   # query id -> page-set size
    excluded: list[str]           # query ids excluded at load time


@dataclass(frozen=True)
class ScoreResult:
    total_requests: int
    total_hits: int
    total_misses: int
    hit_ratio: float


class SimulatorBackend(Protocol):
    kind: str                                 # "real" | "mock"

    def workloads(self) -> dict[str, WorkloadInfo]: ...

    def simulate_sequence(
        self, sequence_page_sets: Sequence, cap: int
    ) -> tuple[int, int]:
        """Return (total_hits, total_requests) for page sets in schedule order."""
        ...


# --------------------------------------------------------------------------
# Mock backend
# --------------------------------------------------------------------------

class MockSimulatorBackend:
    kind = "mock"

    def __init__(self) -> None:
        import mock_workloads
        from reference_sim import simulate_clock_sweep

        self._mw = mock_workloads
        self._sim = simulate_clock_sweep
        self._page_sets: dict[str, list] = {}
        self._info: dict[str, WorkloadInfo] = {}

        for name in mock_workloads.workload_names():
            ps = mock_workloads.page_sets(name)
            qids = mock_workloads.query_ids(name)
            self._page_sets[name] = ps
            self._info[name] = WorkloadInfo(
                name=name,
                label=mock_workloads.label(name),
                query_ids=qids,
                page_counts={q: len(s) for q, s in zip(qids, ps)},
                excluded=[],
            )

    def workloads(self) -> dict[str, WorkloadInfo]:
        return self._info

    def page_sets(self, workload: str) -> list:
        return self._page_sets[workload]

    def simulate_sequence(self, sequence_page_sets, cap):
        return self._sim(sequence_page_sets, cap)


# --------------------------------------------------------------------------
# Real backend
# --------------------------------------------------------------------------

def _load_callable(locator: str):
    """Import ``module:function`` and return the callable."""
    module_name, _, attr = locator.partition(":")
    if not attr:
        raise ValueError(f"locator must be 'module:function', got {locator!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


class RealSimulatorBackend:
    kind = "real"

    def __init__(self) -> None:
        from pathlib import Path

        if config.PROJECT_ROOT and config.PROJECT_ROOT not in sys.path:
            sys.path.insert(0, config.PROJECT_ROOT)

        self._load_pages = _load_callable(config.LOAD_WORKLOAD_PAGES)
        self._simulate = _load_callable(config.SIMULATE_SCHEDULE)

        profile_root = Path(config.PROFILE_ROOT)
        if not profile_root.is_absolute() and config.PROJECT_ROOT:
            profile_root = Path(config.PROJECT_ROOT) / profile_root

        self._page_sets: dict[str, list] = {}
        self._index: dict[str, dict[str, int]] = {}
        self._info: dict[str, WorkloadInfo] = {}

        for name, cfg in config.REAL_WORKLOADS.items():
            directory = profile_root / cfg["subdir"]
            exclude = set(str(x) for x in cfg.get("exclude", []))
            wp = self._load_pages(directory, exclude)
            query_ids = [str(q) for q in wp.query_ids]
            page_sets = list(wp.page_sets)
            if len(query_ids) != len(page_sets):
                raise RuntimeError(
                    f"{name}: query_ids ({len(query_ids)}) and page_sets "
                    f"({len(page_sets)}) length mismatch from load_workload_pages"
                )
            self._page_sets[name] = page_sets
            self._index[name] = {q: i for i, q in enumerate(query_ids)}
            self._info[name] = WorkloadInfo(
                name=name,
                label=cfg["label"],
                query_ids=query_ids,
                page_counts={q: len(page_sets[i]) for i, q in enumerate(query_ids)},
                excluded=sorted(exclude),
            )

    def workloads(self) -> dict[str, WorkloadInfo]:
        return self._info

    def index(self, workload: str) -> dict[str, int]:
        return self._index[workload]

    def page_sets(self, workload: str) -> list:
        return self._page_sets[workload]

    def simulate_sequence(self, sequence_page_sets, cap):
        # Robust translation: the page sets are already arranged in schedule
        # order, so we pass an identity permutation. This is correct whether the
        # real simulator iterates `for t in order` or `for t in range(len)`, and
        # it makes subset schedules work without assuming anything about how the
        # simulator interprets a partial permutation of the full workload.
        base = config.ORDER_BASE
        order = list(range(base, base + len(sequence_page_sets)))
        result = self._simulate(list(sequence_page_sets), order, cap)
        return int(result.total_hits), int(result.total_requests)


# --------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------

class UnknownWorkloadError(KeyError):
    pass


class Adapter:
    """Owns the active backend and the score cache; exposes score()."""

    def __init__(self, backend: SimulatorBackend) -> None:
        self.backend = backend
        self.kind = backend.kind
        self._info = backend.workloads()
        # id -> index map per workload (mock/real both expose page_sets ordered
        # parallel to WorkloadInfo.query_ids).
        self._index: dict[str, dict[str, int]] = {
            name: {q: i for i, q in enumerate(info.query_ids)}
            for name, info in self._info.items()
        }
        self._cache: "OrderedDict[str, ScoreResult]" = OrderedDict()

    # -- introspection ------------------------------------------------------

    def workload_names(self) -> list[str]:
        return list(self._info.keys())

    def info(self, workload: str) -> WorkloadInfo:
        if workload not in self._info:
            raise UnknownWorkloadError(workload)
        return self._info[workload]

    def known_query_ids(self, workload: str) -> set[str]:
        return set(self._index[workload].keys())

    # -- scoring ------------------------------------------------------------

    @staticmethod
    def cache_key(workload: str, cap: int, order: Sequence[str]) -> str:
        raw = f"{workload}|{cap}|{','.join(order)}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def score(self, workload: str, order: Sequence[str], cap: int) -> tuple[ScoreResult, str, float, bool]:
        """Score one schedule with one page-level simulation.

        Returns (result, cache_key, elapsed_ms, cache_hit). Assumes the request
        has already been validated (known workload, known ids, no duplicates,
        positive cap, non-empty order).
        """
        if workload not in self._info:
            raise UnknownWorkloadError(workload)

        key = self.cache_key(workload, cap, order)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached, key, 0.0, True

        index = self._index[workload]
        page_sets = self.backend.page_sets(workload)  # type: ignore[attr-defined]
        sequence = [page_sets[index[q]] for q in order]

        started = time.perf_counter()
        hits, requests = self.backend.simulate_sequence(sequence, cap)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        misses = requests - hits
        hit_ratio = (hits / requests) if requests else 0.0
        result = ScoreResult(
            total_requests=requests,
            total_hits=hits,
            total_misses=misses,
            hit_ratio=hit_ratio,
        )

        self._cache[key] = result
        if len(self._cache) > config.SCORE_CACHE_SIZE:
            self._cache.popitem(last=False)
        return result, key, elapsed_ms, False


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

def build_adapter() -> tuple[Adapter, str | None]:
    """Build the adapter per config.SIM_BACKEND.

    Returns (adapter, fallback_reason). fallback_reason is None on success and
    a human-readable string when 'auto' fell back to mock.
    """
    want = config.SIM_BACKEND
    if want == "mock":
        return Adapter(MockSimulatorBackend()), None
    if want == "real":
        return Adapter(RealSimulatorBackend()), None

    # auto
    try:
        return Adapter(RealSimulatorBackend()), None
    except Exception as exc:  # noqa: BLE001 - intentional fallback
        reason = f"{type(exc).__name__}: {exc}"
        return Adapter(MockSimulatorBackend()), reason
