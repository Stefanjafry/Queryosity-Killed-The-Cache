"""Request validation and structured error responses."""

from __future__ import annotations

from typing import Any, Sequence

import config


class ApiError(Exception):
    """An error that maps to a structured JSON response and HTTP status."""

    def __init__(self, status: int, code: str, message: str, detail: Any = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict:
        body: dict[str, Any] = {"error": {"code": self.code, "message": self.message}}
        if self.detail is not None:
            body["error"]["detail"] = self.detail
        return body


def validate_capacity(cap: Any) -> int:
    if isinstance(cap, bool) or not isinstance(cap, int):
        raise ApiError(400, "invalid_capacity",
                       "capacity must be a positive integer (pages)",
                       {"received": cap})
    if cap <= 0:
        raise ApiError(400, "invalid_capacity",
                       "capacity must be greater than zero",
                       {"received": cap})
    return cap


def validate_order(order: Any) -> list[str]:
    if not isinstance(order, list) or not order:
        raise ApiError(400, "invalid_order",
                       "order must be a non-empty list of query ids")
    normalized: list[str] = []
    for item in order:
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise ApiError(400, "invalid_order",
                           "each order entry must be a query id (string or int)",
                           {"received": item})
        normalized.append(str(item))
    return normalized


def validate_schedule(adapter, workload: Any, order: Any,
                      allow_duplicates: bool = False) -> tuple[str, list[str]]:
    """Validate workload + order together. Returns (workload, normalized_order)."""
    if not isinstance(workload, str) or workload not in set(adapter.workload_names()):
        raise ApiError(400, "unknown_workload",
                       "unknown or missing workload",
                       {"received": workload,
                        "supported": adapter.workload_names()})

    normalized = validate_order(order)

    known = adapter.known_query_ids(workload)
    unknown = [q for q in normalized if q not in known]
    if unknown:
        raise ApiError(400, "unknown_query_ids",
                       "order references queries not in this workload",
                       {"unknown": sorted(set(unknown)),
                        "workload": workload})

    if not allow_duplicates:
        seen: set[str] = set()
        dupes: set[str] = set()
        for q in normalized:
            if q in seen:
                dupes.add(q)
            seen.add(q)
        if dupes:
            raise ApiError(400, "duplicate_query_ids",
                           "order contains duplicate query ids",
                           {"duplicates": sorted(dupes)})

    return workload, normalized
