"""Shared fixtures. Mock mode is forced before importing the Flask app."""

import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

os.environ["QKC_SIM_BACKEND"] = "mock"

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def app_module():
    import app
    return app


@pytest.fixture()
def client(app_module):
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


@pytest.fixture()
def tpch_ids(client):
    data = client.get("/api/workloads").get_json()
    for w in data["workloads"]:
        if w["name"] == "tpch":
            return w["query_ids"]
    raise AssertionError("tpch workload not found")
