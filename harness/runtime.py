"""Runtime watchdog supervision with pause/resume state."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from harness.logger import JsonlAuditLogger
from harness.models import Decision, RobotCommandResult, new_id
from harness.policy import PolicyEngine, default_policies


@dataclass
class RuntimeSupervisorState:
    mode: str = "RUNNING"  # RUNNING | PAUSED | STOPPED
    clear_frames: int = 0
    last_decision: Decision | None = None
    last_robot_result: RobotCommandResult | None = None


class RuntimeWatchdogSupervisor:
    """Keep cameras running while mapping watchdog decisions to robot control.

    This class is for the "robot is already moving" case. It does not use
    `BLOCK` to interrupt motion. It calls `pause`, `stop`, and `resume` on the
    robot adapter while perception continues to run on every frame.
    """

    def __init__(
        self,
        *,
        perception_adapter,
        robot_adapter,
        policies=None,
        policy_engine: PolicyEngine | None = None,
        logger: JsonlAuditLogger | None = None,
        clear_frames_before_resume: int = 10,
        auto_resume: bool = True,
        stop_is_terminal: bool = True,
    ):
        self.perception_adapter = perception_adapter
        self.robot_adapter = robot_adapter
        self.policies = policies or default_policies()
        self.policy_engine = policy_engine or PolicyEngine()
        self.logger = logger or JsonlAuditLogger()
        self.clear_frames_before_resume = clear_frames_before_resume
        self.auto_resume = auto_resume
        self.stop_is_terminal = stop_is_terminal
        self.state = RuntimeSupervisorState()

    def step(self, frame, now: float | None = None) -> RuntimeSupervisorState:
        """Process one camera frame and apply pause/stop/resume transitions."""
        now = now if now is not None else time.time()
        evidence_frame_id = new_id("frame")
        scene_context = self.perception_adapter.parse_frame(frame, now=now)
        decision = self.policy_engine.evaluate(
            planned_action=None,
            scene_context=scene_context,
            policies=self.policies,
            evidence_frame_id=evidence_frame_id,
        )

        robot_result = self._apply_runtime_decision(decision)
        self.state.last_decision = decision
        self.state.last_robot_result = robot_result

        self.logger.log_event(
            {
                "kind": "runtime_watchdog",
                "mode": self.state.mode,
                "scene_context": scene_context,
                "decision": decision.to_dict(),
                "robot_result": robot_result.to_dict() if robot_result else None,
            }
        )
        return self.state

    def _apply_runtime_decision(self, decision: Decision) -> RobotCommandResult | None:
        if self.state.mode == "STOPPED" and self.stop_is_terminal:
            return None

        if decision.decision == "STOP":
            self.state.mode = "STOPPED"
            self.state.clear_frames = 0
            return self.robot_adapter.stop(decision.affected_arm)

        if decision.decision == "PAUSE":
            self.state.clear_frames = 0
            if self.state.mode != "PAUSED":
                self.state.mode = "PAUSED"
                return self.robot_adapter.pause(decision.affected_arm)
            return None

        safe_now = decision.decision in ("ALLOW", "WARN")
        if self.state.mode == "PAUSED" and safe_now:
            self.state.clear_frames += 1
            if self.auto_resume and self.state.clear_frames >= self.clear_frames_before_resume:
                self.state.mode = "RUNNING"
                self.state.clear_frames = 0
                return self.robot_adapter.resume(decision.affected_arm)
            return None

        if self.state.mode == "RUNNING" and safe_now:
            self.state.clear_frames = min(self.state.clear_frames + 1, self.clear_frames_before_resume)
            return None

        return None


def run_runtime_watchdog(camera, supervisor: RuntimeWatchdogSupervisor, sleep_s: float = 0.02):
    """Run continuous camera supervision until the camera ends or interrupted."""
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        supervisor.step(frame, now=time.time())
        time.sleep(sleep_s)
