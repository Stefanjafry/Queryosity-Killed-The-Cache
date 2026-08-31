"""Conference-demo endpoint behaviour and paper/demo consistency (mock mode)."""


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["backend"] == "mock"
    assert body["wired"] is False
    assert body["editing_reruns_scheduler"] is False
    assert body["postgresql_validation_present"] is False
    assert body["generated_artifacts"]["loaded"] is True


def test_workloads_shape(client):
    body = client.get("/api/workloads").get_json()
    assert body["backend"] == "mock"
    assert body["page_size_bytes"] == 8192
    assert body["methods"] == ["sweep_D", "sweep_beam_D"]
    assert len(body["presets"]) == 3
    for p in body["presets"]:
        assert {"pages", "bytes", "human"} <= set(p)
        assert p["bytes"] == p["pages"] * 8192

    names = {w["name"] for w in body["workloads"]}
    assert {"tpch", "tpcds", "job"} <= names
    for w in body["workloads"]:
        assert len(w["query_ids"]) == w["query_count"]
        assert set(w["page_counts"].keys()) == set(w["query_ids"])
        assert w["generated_capacities"] == [102400, 262144, 524288]


def test_score_schema_and_invariants(client, tpch_ids):
    r = client.post("/api/score", json={"workload": "tpch", "cap": 102400, "order": tpch_ids})
    assert r.status_code == 200
    body = r.get_json()
    for key in ("workload", "capacity_pages", "capacity_bytes", "capacity_human",
                "order", "query_count", "total_requests", "total_hits",
                "total_misses", "hit_ratio", "f_hit", "cache_key", "cached",
                "elapsed_ms", "backend"):
        assert key in body, f"missing key {key}"
    assert body["backend"] == "mock"
    assert body["query_count"] == len(tpch_ids)
    assert body["total_hits"] + body["total_misses"] == body["total_requests"]
    assert abs(body["total_hits"] / body["total_requests"] - body["f_hit"]) < 1e-9
    assert body["f_hit"] == body["hit_ratio"]
    assert body["capacity_bytes"] == 102400 * 8192


def test_score_cache_flag_flips(client, tpch_ids):
    payload = {"workload": "tpch", "cap": 262144, "order": tpch_ids}
    first = client.post("/api/score", json=payload).get_json()
    second = client.post("/api/score", json=payload).get_json()
    # It may already be cached by another test in the session, but after one
    # request the same key must definitely be cached.
    assert second["cached"] is True
    assert first["cache_key"] == second["cache_key"]
    assert second["total_hits"] == first["total_hits"]


def test_generated_schedule_endpoint_only_exposes_paper_methods(client):
    r = client.get("/api/generated-schedules?workload=tpch&cap=102400")
    assert r.status_code == 200
    body = r.get_json()
    assert body["workload"] == "tpch"
    assert body["capacity_pages"] == 102400
    assert set(body["methods"]) == {"sweep_D", "sweep_beam_D"}
    for method, entry in body["methods"].items():
        assert entry["label"] in {"Sweep-D", "Sweep+Beam-D"}
        assert entry["total_hits"] + entry["total_misses"] == entry["total_requests"]
        assert entry["query_count"] == len(entry["order"])
        assert entry["f_hit"] == entry["hit_ratio"]
        assert method not in {"GA_M", "GA_D", "baseline"}


def test_explanation_is_capacity_and_method_scoped(client):
    r = client.get("/api/explanation?workload=tpch&cap=262144&method=sweep_D")
    assert r.status_code == 200
    body = r.get_json()
    assert body["workload"] == "tpch"
    assert body["capacity_pages"] == 262144
    assert body["method"] == "sweep_D"
    assert body["label"] == "Sweep-D"
    # Mock mode deliberately refuses to fabricate research construction traces.
    assert body["explanation"]["available"] is False


def test_invalid_explanation_method(client):
    r = client.get("/api/explanation?workload=tpch&cap=102400&method=GA_D")
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "invalid_method"


def test_unknown_generated_capacity_404(client):
    r = client.get("/api/generated-schedules?workload=tpch&cap=99999")
    assert r.status_code == 404
    body = r.get_json()
    assert body["error"]["code"] == "generated_schedule_unavailable"
    assert "regenerate" in body["error"]["detail"]


def test_compare_scores_multiple_orders(client, tpch_ids):
    payload = {
        "workload": "tpch", "cap": 102400,
        "schedules": [
            {"label": "natural", "order": tpch_ids[:5]},
            {"label": "reverse", "order": list(reversed(tpch_ids[:5]))},
        ],
    }
    r = client.post("/api/compare", json=payload)
    assert r.status_code == 200
    body = r.get_json()
    assert [x["label"] for x in body["schedules"]] == ["natural", "reverse"]
    for x in body["schedules"]:
        assert x["total_hits"] + x["total_misses"] == x["total_requests"]


def test_score_path_does_not_read_generated_artifacts(client, app_module, tpch_ids, monkeypatch):
    """A drag/edit score request has no scheduler/artifact lookup in its path."""
    def fail(*_args, **_kwargs):
        raise AssertionError("generated scheduler artifacts were consulted during /api/score")

    monkeypatch.setattr(app_module.ARTIFACTS, "for_capacity", fail)
    r = client.post("/api/score", json={"workload": "tpch", "cap": 102400, "order": tpch_ids[:4]})
    assert r.status_code == 200


def test_compat_scheduler_alias_is_sweep_only(client):
    body = client.get("/api/schedulers").get_json()
    assert body["methods"] == ["sweep_D", "sweep_beam_D"]
    data = client.get("/api/schedulers/tpch/102400").get_json()
    assert set(data["methods"]) == {"sweep_D", "sweep_beam_D"}


def test_unknown_api_path_is_json_404(client):
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404
    assert r.get_json()["error"]["code"] == "not_found"
