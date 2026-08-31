"""Queryosity conference-demo Flask application.

The live request path has two distinct responsibilities:
* `/api/score` replays one user-supplied order through the shared page-level
  cache simulator.
* generated Sweep-D / Sweep+Beam-D schedules and their explanations are read
  from artifacts produced offline by `scripts/generate_demo_artifacts.py`.

A schedule edit never invokes scheduler search.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

import config
import generated
from simulator import UnknownWorkloadError, build_adapter
from validation import ApiError, validate_capacity, validate_schedule

log = logging.getLogger("queryosity")

app = Flask(__name__, static_folder=str(config.FRONTEND_DIR), static_url_path="")

ADAPTER, FALLBACK_REASON = build_adapter()
ARTIFACTS = generated.DemoArtifacts(config.generated_dir(ADAPTER.kind))

if FALLBACK_REASON:
    log.warning("Real simulator unavailable; using MOCK backend: %s", FALLBACK_REASON)
if ARTIFACTS.error:
    log.warning("Generated schedule artifacts unavailable: %s", ARTIFACTS.error)
for warning in ARTIFACTS.warnings:
    log.warning("artifact: %s", warning)


def _capacity_bytes(pages: int) -> int:
    return pages * config.PAGE_SIZE_BYTES


def _human_bytes(num: int) -> str:
    val = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if val < 1024.0 or unit == "TB":
            if unit in ("B", "KB"):
                return f"{val:.0f} {unit}"
            return f"{val:.2f} {unit}".replace(".00", "")
        val /= 1024.0
    return f"{num} B"


def _score_payload(workload: str, cap: int, order: list[str]) -> dict[str, Any]:
    result, key, elapsed_ms, cached = ADAPTER.score(workload, order, cap)
    return {
        "workload": workload,
        "capacity_pages": cap,
        "capacity_bytes": _capacity_bytes(cap),
        "capacity_human": _human_bytes(_capacity_bytes(cap)),
        "order": order,
        "query_count": len(order),
        "total_requests": result.total_requests,
        "total_hits": result.total_hits,
        "total_misses": result.total_misses,
        "hit_ratio": result.hit_ratio,  # compatibility
        "f_hit": result.hit_ratio,
        "cache_key": key,
        "cached": cached,
        "elapsed_ms": round(elapsed_ms, 3),
        "backend": ADAPTER.kind,
    }


def _validate_workload(value: Any) -> str:
    if not isinstance(value, str) or value not in set(ADAPTER.workload_names()):
        raise ApiError(400, "unknown_workload", "unknown or missing workload",
                       {"received": value, "supported": ADAPTER.workload_names()})
    return value


def _validate_method(value: Any) -> str:
    if not isinstance(value, str) or value not in generated.METHODS:
        raise ApiError(400, "invalid_method", "unknown or missing scheduling method",
                       {"received": value, "supported": list(generated.METHODS)})
    return value


@app.errorhandler(ApiError)
def _handle_api_error(err: ApiError):
    return jsonify(err.to_dict()), err.status


@app.errorhandler(404)
def _handle_404(_err):
    if request.path.startswith("/api/"):
        return jsonify({"error": {"code": "not_found", "message": f"no such endpoint: {request.path}"}}), 404
    return send_from_directory(config.FRONTEND_DIR, "index.html")


@app.errorhandler(Exception)
def _handle_unexpected(err: Exception):
    if isinstance(err, ApiError):
        return _handle_api_error(err)
    log.exception("unhandled error")
    return jsonify({"error": {"code": "internal_error", "message": str(err)}}), 500


@app.get("/")
def index():
    return send_from_directory(config.FRONTEND_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "backend": ADAPTER.kind,
        "wired": ADAPTER.kind == "real",
        "fallback_reason": FALLBACK_REASON,
        "project_root": config.PROJECT_ROOT or None,
        "workloads_loaded": ADAPTER.workload_names(),
        "generated_artifacts": ARTIFACTS.summary(),
        "editing_reruns_scheduler": False,
        "postgresql_validation_present": False,
    })


@app.get("/api/workloads")
def workloads():
    avail = ARTIFACTS.availability()
    payload = []
    for name in ADAPTER.workload_names():
        info = ADAPTER.info(name)
        payload.append({
            "name": name,
            "label": info.label,
            "query_ids": info.query_ids,
            "query_count": len(info.query_ids),
            "excluded": info.excluded,
            "page_counts": info.page_counts,
            "profile_available": True,
            "generated_capacities": avail.get(name, []),
        })
    return jsonify({
        "backend": ADAPTER.kind,
        "wired": ADAPTER.kind == "real",
        "page_size_bytes": config.PAGE_SIZE_BYTES,
        "presets": [
            {"pages": c, "bytes": _capacity_bytes(c), "human": _human_bytes(_capacity_bytes(c))}
            for c in config.PRESET_CAPACITIES
        ],
        "workloads": payload,
        "methods": list(generated.METHODS),
        "method_labels": generated.METHOD_LABELS,
    })


@app.post("/api/score")
def score():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ApiError(400, "invalid_body", "request body must be a JSON object")
    workload, order = validate_schedule(ADAPTER, payload.get("workload"), payload.get("order"))
    cap = validate_capacity(payload.get("cap"))
    try:
        return jsonify(_score_payload(workload, cap, order))
    except UnknownWorkloadError as exc:
        raise ApiError(400, "unknown_workload", "unknown workload", {"received": str(exc)})


@app.post("/api/schedules/validate")
def validate_schedule_endpoint():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ApiError(400, "invalid_body", "request body must be a JSON object")
    workload, order = validate_schedule(ADAPTER, payload.get("workload"), payload.get("order"))
    return jsonify({"valid": True, "workload": workload, "order": order, "query_count": len(order)})


@app.get("/api/generated-schedules")
def generated_schedules():
    workload = _validate_workload(request.args.get("workload"))
    try:
        cap = validate_capacity(int(request.args.get("cap", "")))
    except (TypeError, ValueError):
        raise ApiError(400, "invalid_capacity", "capacity must be a positive integer (pages)")
    data = ARTIFACTS.for_capacity(workload, cap)
    if data is None:
        raise ApiError(404, "generated_schedule_unavailable",
                       "no prepared Queryosity schedules for this workload/capacity",
                       {"workload": workload, "capacity_pages": cap,
                        "regenerate": "python scripts/generate_demo_artifacts.py --all"})
    methods = {}
    for method in generated.METHODS:
        entry = (data.get("methods") or {}).get(method)
        if not entry:
            continue
        methods[method] = {
            "method": method,
            "label": generated.METHOD_LABELS[method],
            "order": entry["order"],
            "query_count": entry["query_count"],
            "total_requests": entry["total_requests"],
            "total_hits": entry["total_hits"],
            "total_misses": entry["total_misses"],
            "f_hit": entry["f_hit"],
            "hit_ratio": entry["f_hit"],
            "construction": entry.get("construction", {}),
            "source": entry.get("source", {}),
        }
    return jsonify({
        "backend": data.get("backend", ADAPTER.kind),
        "workload": workload,
        "capacity_pages": cap,
        "capacity_human": _human_bytes(_capacity_bytes(cap)),
        "methods": methods,
    })


@app.get("/api/explanation")
def explanation():
    workload = _validate_workload(request.args.get("workload"))
    method = _validate_method(request.args.get("method"))
    try:
        cap = validate_capacity(int(request.args.get("cap", "")))
    except (TypeError, ValueError):
        raise ApiError(400, "invalid_capacity", "capacity must be a positive integer (pages)")
    data = ARTIFACTS.for_capacity(workload, cap)
    entry = ARTIFACTS.method_entry(workload, cap, method)
    if data is None or entry is None:
        raise ApiError(404, "explanation_unavailable",
                       "no prepared explanation metadata for this selection",
                       {"workload": workload, "capacity_pages": cap, "method": method})
    return jsonify({
        "backend": data.get("backend", ADAPTER.kind),
        "workload": workload,
        "capacity_pages": cap,
        "method": method,
        "label": generated.METHOD_LABELS[method],
        "order": entry["order"],
        "total_requests": entry["total_requests"],
        "total_hits": entry["total_hits"],
        "total_misses": entry["total_misses"],
        "f_hit": entry["f_hit"],
        "construction": entry.get("construction", {}),
        "explanation": entry.get("explanation", {}),
        "source": entry.get("source", {}),
    })


# Compatibility aliases for the earlier frontend/API.  They now expose only
# the two paper schedulers, never GA_M/GA_D.
@app.get("/api/schedulers")
def schedulers_alias():
    return jsonify({
        "backend": ADAPTER.kind,
        "loaded": ARTIFACTS.loaded,
        "availability": ARTIFACTS.availability(),
        "methods": list(generated.METHODS),
        "method_labels": generated.METHOD_LABELS,
    })


@app.get("/api/schedulers/<workload>/<int:capacity>")
def schedulers_for_alias(workload: str, capacity: int):
    _validate_workload(workload)
    validate_capacity(capacity)
    data = ARTIFACTS.for_capacity(workload, capacity)
    if data is None:
        raise ApiError(404, "generated_schedule_unavailable", "no prepared schedules",
                       {"workload": workload, "capacity_pages": capacity})
    methods = {}
    for m in generated.METHODS:
        e = (data.get("methods") or {}).get(m)
        if e:
            methods[m] = {
                "method": m, "label": generated.METHOD_LABELS[m], "order": e["order"],
                "query_count": e["query_count"], "total_requests": e["total_requests"],
                "total_hits": e["total_hits"], "total_misses": e["total_misses"],
                "f_hit": e["f_hit"], "hit_ratio": e["f_hit"],
                "construction": e.get("construction", {}), "source": e.get("source", {}),
            }
    return jsonify({"workload": workload, "capacity_pages": capacity,
                    "capacity_human": _human_bytes(_capacity_bytes(capacity)), "methods": methods})


@app.post("/api/compare")
def compare():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ApiError(400, "invalid_body", "request body must be a JSON object")
    workload = _validate_workload(payload.get("workload"))
    cap = validate_capacity(payload.get("cap"))
    schedules = payload.get("schedules")
    if not isinstance(schedules, list) or not schedules:
        raise ApiError(400, "invalid_schedules", "schedules must be a non-empty list")
    out = []
    for idx, item in enumerate(schedules):
        if not isinstance(item, dict):
            raise ApiError(400, "invalid_schedules", "each schedule must be an object")
        _, order = validate_schedule(ADAPTER, workload, item.get("order"))
        scored = _score_payload(workload, cap, order)
        scored["label"] = str(item.get("label") or f"Schedule {idx + 1}")
        out.append(scored)
    return jsonify({"workload": workload, "capacity_pages": cap, "schedules": out})


@app.post("/api/optimize")
def optimize_small_subset():
    """Enumerate all permutations of a deliberately small selected subset."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ApiError(400, "invalid_body", "request body must be a JSON object")
    workload, order = validate_schedule(ADAPTER, payload.get("workload"), payload.get("order"))
    cap = validate_capacity(payload.get("cap"))
    if len(order) > config.EXACT_SUBSET_MAX:
        raise ApiError(400, "subset_too_large",
                       f"exact subset reference is limited to {config.EXACT_SUBSET_MAX} queries",
                       {"received": len(order), "max": config.EXACT_SUBSET_MAX})

    best_order: list[str] | None = None
    best_score = None
    evaluated = 0
    for perm in itertools.permutations(order):
        evaluated += 1
        result, _, _, _ = ADAPTER.score(workload, list(perm), cap)
        if best_score is None or result.hit_ratio > best_score.hit_ratio:
            best_score = result
            best_order = list(perm)

    assert best_order is not None and best_score is not None
    return jsonify({
        "workload": workload,
        "capacity_pages": cap,
        "scope": "selected_subset_only",
        "query_count": len(order),
        "permutations_evaluated": evaluated,
        "order": best_order,
        "total_requests": best_score.total_requests,
        "total_hits": best_score.total_hits,
        "total_misses": best_score.total_misses,
        "f_hit": best_score.hit_ratio,
        "hit_ratio": best_score.hit_ratio,
        "global_optimum_claimed": False,
        "note": "Exact only for the selected small subset; not the full benchmark workload.",
    })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app.run(host="127.0.0.1", port=8000, debug=True)
