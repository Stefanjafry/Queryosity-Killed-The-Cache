"""All five structured validation error codes, exercised through /api/score."""


def _code(client, payload):
    r = client.post("/api/score", json=payload)
    assert r.status_code == 400, r.get_json()
    return r.get_json()["error"]["code"]


def test_invalid_capacity_zero(client, tpch_ids):
    assert _code(client, {"workload": "tpch", "cap": 0, "order": tpch_ids}) == "invalid_capacity"


def test_invalid_capacity_type(client, tpch_ids):
    assert _code(client, {"workload": "tpch", "cap": "lots", "order": tpch_ids}) == "invalid_capacity"


def test_invalid_capacity_bool(client, tpch_ids):
    # bool is a subclass of int but must be rejected
    assert _code(client, {"workload": "tpch", "cap": True, "order": tpch_ids}) == "invalid_capacity"


def test_invalid_order_empty(client):
    assert _code(client, {"workload": "tpch", "cap": 102400, "order": []}) == "invalid_order"


def test_unknown_workload(client, tpch_ids):
    assert _code(client, {"workload": "nope", "cap": 102400, "order": tpch_ids}) == "unknown_workload"


def test_unknown_query_ids(client):
    assert _code(client, {"workload": "tpch", "cap": 102400, "order": ["999999"]}) == "unknown_query_ids"


def test_duplicate_query_ids(client, tpch_ids):
    dup = [tpch_ids[0], tpch_ids[0], tpch_ids[1]]
    assert _code(client, {"workload": "tpch", "cap": 102400, "order": dup}) == "duplicate_query_ids"


def test_validate_endpoint_ok(client, tpch_ids):
    r = client.post("/api/schedules/validate", json={"workload": "tpch", "order": tpch_ids})
    assert r.status_code == 200
    body = r.get_json()
    assert body["valid"] is True
    assert body["query_count"] == len(tpch_ids)
