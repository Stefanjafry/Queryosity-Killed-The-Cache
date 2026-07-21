"""
Project-wide constants for workload paths and database names.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]

WORKLOAD_ROOT = PROJECT_ROOT / "workloads"

WORKLOAD_DIRS = {
    "tpch": WORKLOAD_ROOT / "tpch",
    "tpcds": WORKLOAD_ROOT / "tpcds",
    "job": WORKLOAD_ROOT / "job",
}

DB_DEFAULTS = {
    "tpch": "tpch",
    "tpcds": "tpcds",
    "job": "imdb",
}
