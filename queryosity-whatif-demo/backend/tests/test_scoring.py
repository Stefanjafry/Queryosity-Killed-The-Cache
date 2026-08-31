"""Scoring semantics, artifact arithmetic, and exact-subset separation."""

import math


def _score(client, order, cap=102400, workload="tpch"):
    return client.post("/api/score",
                       json={"workload": workload, "cap": cap, "order": order}).get_json()


def test_reorder_changes_hits_but_not_requests(client, tpch_ids):
    natural = _score(client, tpch_ids)
    reversed_ = _score(client, list(reversed(tpch_ids)))
    assert natural["total_requests"] == reversed_["total_requests"]
    assert natural["total_hits"] != reversed_["total_hits"]


def test_subset_schedule_scores(client, tpch_ids):
    subset = tpch_ids[:5]
    body = _score(client, subset)
    assert body["query_count"] == 5
    assert body["total_requests"] > 0
    assert body["total_hits"] + body["total_misses"] == body["total_requests"]
    full = _score(client, tpch_ids)
    assert body["total_requests"] < full["total_requests"]


def test_larger_cache_never_fewer_hits(client, tpch_ids):
    small = _score(client, tpch_ids, cap=102400)
    large = _score(client, tpch_ids, cap=524288)
    assert large["total_requests"] == small["total_requests"]
    assert large["total_hits"] >= small["total_hits"]


def test_mock_generated_artifacts_are_schema_consistent(app_module):
    import generated

    results = generated.DemoArtifacts(app_module.config.GENERATED_DIR_MOCK)
    assert results.loaded, results.error
    assert results.warnings == [], results.warnings
    assert set(results.availability()) >= {"tpch", "tpcds", "job"}
    for workload, caps in results.availability().items():
        for cap in caps:
            data = results.for_capacity(workload, cap)
            assert data
            assert set(data["methods"]) == {"sweep_D", "sweep_beam_D"}
            for entry in data["methods"].values():
                assert entry["total_hits"] + entry["total_misses"] == entry["total_requests"]
                assert entry["query_count"] == len(entry["order"])
                assert abs(entry["f_hit"] - entry["total_hits"] / entry["total_requests"]) < 1e-8
                assert entry["explanation"]["available"] is False


def test_exact_subset_reference_is_explicitly_separate(client, tpch_ids):
    subset = tpch_ids[:5]
    r = client.post("/api/optimize", json={"workload": "tpch", "cap": 102400, "order": subset})
    assert r.status_code == 200
    body = r.get_json()
    assert body["scope"] == "selected_subset_only"
    assert body["query_count"] == len(subset)
    assert body["permutations_evaluated"] == math.factorial(len(subset))
    assert body["global_optimum_claimed"] is False
    assert set(body["order"]) == set(subset)
    assert body["f_hit"] == body["hit_ratio"]


def test_exact_subset_reference_rejects_large_order(client, tpch_ids):
    r = client.post("/api/optimize", json={"workload": "tpch", "cap": 102400, "order": tpch_ids[:8]})
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "subset_too_large"
