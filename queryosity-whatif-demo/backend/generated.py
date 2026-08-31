"""Load prepared Queryosity demo artifacts.

Real artifacts are generated offline from the research implementation by
``scripts/generate_demo_artifacts.py``.  Request-time code only reads them;
editing a schedule therefore never invokes Sweep-D or Sweep+Beam-D.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3
METHODS = ("sweep_D", "sweep_beam_D")
METHOD_LABELS = {
    "sweep_D": "Sweep-D",
    "sweep_beam_D": "Sweep+Beam-D",
}


class DemoArtifacts:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.loaded = False
        self.error: str | None = None
        self.warnings: list[str] = []
        self._entries: dict[tuple[str, int], dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.directory.exists():
            self.error = f"not found: {self.directory}"
            return
        files = sorted(self.directory.glob("*.json"))
        if not files:
            self.error = f"no artifact JSON files in {self.directory}"
            return

        for path in files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                self.warnings.append(f"{path.name}: parse error: {exc}")
                continue

            version = data.get("schema_version")
            if version != SCHEMA_VERSION:
                self.warnings.append(
                    f"{path.name}: schema_version {version!r} != {SCHEMA_VERSION}")
                continue

            workload = data.get("workload")
            cap = data.get("capacity_pages")
            methods = data.get("methods")
            if not isinstance(workload, str) or not isinstance(cap, int) or not isinstance(methods, dict):
                self.warnings.append(f"{path.name}: missing workload/capacity/methods")
                continue

            missing = [m for m in METHODS if m not in methods]
            if missing:
                self.warnings.append(f"{path.name}: missing methods {missing}")

            for method, entry in methods.items():
                if method not in METHODS:
                    continue
                self._check_entry(path.name, method, entry)

            self._entries[(workload, cap)] = data

        self.loaded = bool(self._entries)
        if not self.loaded and self.error is None:
            self.error = "no valid artifact files loaded"

    def _check_entry(self, filename: str, method: str, entry: Any) -> None:
        if not isinstance(entry, dict):
            self.warnings.append(f"{filename}/{method}: entry is not an object")
            return
        required = (
            "order", "query_count", "total_requests", "total_hits",
            "total_misses", "f_hit", "construction", "explanation",
        )
        miss = [k for k in required if k not in entry]
        if miss:
            self.warnings.append(f"{filename}/{method}: missing fields {miss}")
            return
        order = entry.get("order") or []
        if entry.get("query_count") != len(order):
            self.warnings.append(f"{filename}/{method}: query_count != len(order)")
        h = entry.get("total_hits")
        m = entry.get("total_misses")
        r = entry.get("total_requests")
        if all(isinstance(x, int) for x in (h, m, r)) and h + m != r:
            self.warnings.append(f"{filename}/{method}: hits + misses != requests")
        if isinstance(r, int) and r > 0 and isinstance(h, int):
            expected = h / r
            if abs(float(entry.get("f_hit", 0.0)) - expected) > 1e-8:
                self.warnings.append(f"{filename}/{method}: f_hit != hits/requests")

    def availability(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for workload, cap in sorted(self._entries):
            out.setdefault(workload, []).append(cap)
        return out

    def for_capacity(self, workload: str, cap: int) -> dict[str, Any] | None:
        return self._entries.get((workload, cap))

    def method_entry(self, workload: str, cap: int, method: str) -> dict[str, Any] | None:
        data = self.for_capacity(workload, cap)
        if not data:
            return None
        return (data.get("methods") or {}).get(method)

    def summary(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "directory": str(self.directory),
            "error": self.error,
            "warnings": self.warnings,
            "availability": self.availability(),
        }
