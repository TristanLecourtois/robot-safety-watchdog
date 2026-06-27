"""Audit logging for harness decisions."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class JsonlAuditLogger:
    def __init__(self, path: str | Path = "harness_events.jsonl"):
        self.path = Path(path)

    def log_event(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = _json_safe({"logged_at": time.time(), **event})
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


def _json_safe(value):
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
