"""Configuration for the Queryosity conference demo."""

from __future__ import annotations

import os
from pathlib import Path

PAGE_SIZE_BYTES = 8192
PRESET_CAPACITIES = [102_400, 262_144, 524_288]
SCORE_CACHE_SIZE = int(os.environ.get("QKC_SCORE_CACHE", "512"))
EXACT_SUBSET_MAX = int(os.environ.get("QKC_EXACT_SUBSET_MAX", "5"))
SIM_BACKEND = os.environ.get("QKC_SIM_BACKEND", "auto").strip().lower()

_HERE = Path(__file__).resolve().parent
DEMO_ROOT = _HERE.parent
FRONTEND_DIR = DEMO_ROOT / "frontend"
DATA_DIR = DEMO_ROOT / "data"

# When the demo is extracted directly inside Queryosity-Killed-The-Cache (the
# normal VM layout), discover the research repo automatically.  An explicit
# QKC_PROJECT_ROOT always wins.
_parent = DEMO_ROOT.parent
_auto_project = _parent if (_parent / "src").is_dir() and (_parent / "page_access").is_dir() else None
PROJECT_ROOT = os.environ.get("QKC_PROJECT_ROOT", str(_auto_project or ""))

LOAD_WORKLOAD_PAGES = os.environ.get(
    "QKC_LOAD_WORKLOAD_PAGES", "src.bayesopt.data:load_workload_pages"
)
SIMULATE_SCHEDULE = os.environ.get(
    "QKC_SIMULATE_SCHEDULE", "src.simulator.cache_simulator:simulate_schedule_page_level"
)
PROFILE_ROOT = os.environ.get("QKC_PROFILE_ROOT", "page_access")
ORDER_BASE = int(os.environ.get("QKC_ORDER_BASE", "0"))

REAL_WORKLOADS: dict[str, dict] = {
    "tpch":  {"label": "TPC-H (SF10)",  "subdir": "tpch",  "exclude": []},
    "tpcds": {"label": "TPC-DS",        "subdir": "tpcds", "exclude": []},
    "job":   {"label": "JOB / IMDB",    "subdir": "job",   "exclude": []},
}

GENERATED_DIR_REAL = Path(os.environ.get(
    "QKC_GENERATED_DIR", str(DATA_DIR / "generated")
))
GENERATED_DIR_MOCK = Path(os.environ.get(
    "QKC_GENERATED_DIR_MOCK", str(DATA_DIR / "generated_mock")
))


def generated_dir(backend_kind: str) -> Path:
    return GENERATED_DIR_MOCK if backend_kind == "mock" else GENERATED_DIR_REAL
