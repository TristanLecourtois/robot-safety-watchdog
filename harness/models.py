"""Shared data models for the robot safety harness."""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


DecisionValue = Literal["ALLOW", "WARN", "BLOCK", "PAUSE", "STOP", "RESUME"]
Severity = Literal["none", "low", "medium", "high", "critical"]


def utc_timestamp_s() -> float:
    return time.time()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class PlannedAction:
    """Robot intent known before execution."""

    type: str
    object: str
    id: str = field(default_factory=lambda: new_id("act"))
    arm: str = "both_arms"
    target_zone: str = "workspace"
    source: str = "unknown"
    expected_duration_seconds: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlannedAction":
        known = {
            "id",
            "type",
            "object",
            "arm",
            "target_zone",
            "source",
            "expected_duration_seconds",
            "payload",
        }
        payload = dict(data.get("payload") or {})
        payload.update({k: v for k, v in data.items() if k not in known})
        return cls(
            id=data.get("id") or new_id("act"),
            type=data["type"],
            object=data.get("object", "unknown"),
            arm=data.get("arm", "both_arms"),
            target_zone=data.get("target_zone", "workspace"),
            source=data.get("source", "unknown"),
            expected_duration_seconds=data.get("expected_duration_seconds"),
            payload=payload,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Policy:
    id: str
    name: str
    enabled: bool
    severity: Severity
    decision_on_violation: DecisionValue
    conditions: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Policy":
        return cls(
            id=data["id"],
            name=data["name"],
            enabled=bool(data.get("enabled", True)),
            severity=data.get("severity", "critical"),
            decision_on_violation=data.get("decision_on_violation", "BLOCK"),
            conditions=dict(data.get("conditions") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Decision:
    decision: DecisionValue
    severity: Severity
    rule: str
    reason: str
    mitigation: str
    id: str = field(default_factory=lambda: new_id("dec"))
    timestamp: float = field(default_factory=utc_timestamp_s)
    planned_action_id: str | None = None
    evidence_frame_id: str | None = None
    affected_arm: str = "both_arms"
    policy_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RobotCommandResult:
    command: str
    executed: bool
    affected_arm: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
