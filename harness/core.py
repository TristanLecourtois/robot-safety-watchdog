"""Harness orchestration helpers."""
from __future__ import annotations

import time
from typing import Any

from harness.logger import JsonlAuditLogger
from harness.models import Decision, PlannedAction, RobotCommandResult, new_id
from harness.policy import PolicyEngine, default_policies


def safe_execute(
    planned_action: PlannedAction | dict[str, Any],
    *,
    camera=None,
    frame=None,
    policies=None,
    perception_adapter,
    robot_adapter,
    logger: JsonlAuditLogger | None = None,
    policy_engine: PolicyEngine | None = None,
) -> Decision:
    """Gate an OpenARM action behind perception and deterministic policies."""
    action = planned_action if isinstance(planned_action, PlannedAction) else PlannedAction.from_dict(planned_action)
    logger = logger or JsonlAuditLogger()
    policy_engine = policy_engine or PolicyEngine()
    policies = policies or default_policies()
    evidence_frame_id = new_id("frame")

    if frame is None:
        if camera is None:
            raise ValueError("safe_execute requires either `frame` or `camera`.")
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("Could not read a frame for safety check.")

    scene_context = perception_adapter.parse_frame(frame, now=time.time())
    decision = policy_engine.evaluate(action, scene_context, policies, evidence_frame_id)

    base_event = {
        "kind": "harness_decision",
        "planned_action": action.to_dict(),
        "scene_context": scene_context,
        "decision": decision.to_dict(),
    }
    logger.log_event({**base_event, "stage": "pre_robot_command"})

    robot_result = _apply_decision(decision, action, robot_adapter)
    logger.log_event({**base_event, "stage": "post_robot_command", "robot_result": robot_result.to_dict()})
    return decision


def _apply_decision(decision: Decision, action: PlannedAction, robot_adapter) -> RobotCommandResult:
    if decision.decision == "ALLOW":
        return robot_adapter.execute(action)
    if decision.decision == "PAUSE":
        return robot_adapter.pause(decision.affected_arm)
    if decision.decision == "STOP":
        return robot_adapter.stop(decision.affected_arm)
    if decision.decision == "RESUME":
        return robot_adapter.resume(decision.affected_arm)
    return RobotCommandResult(
        command=decision.decision.lower(),
        executed=False,
        affected_arm=decision.affected_arm,
        detail="robot action skipped by harness decision",
    )
