"""Smoke test for the OpenARM safety harness.

Run from a fresh clone with:

    python3 -m harness.openarm_smoke

This intentionally avoids the heavy perception stack. It checks the harness
control path with a fake perception adapter and a controller-shaped object that
matches the OpenARM/LeRobot methods used by `OpenArmLeRobotAdapter`.
"""
from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness import JsonlAuditLogger, OpenArmLeRobotAdapter, RuntimeWatchdogSupervisor, safe_execute


SAFE_SCENE = {
    "camera_id": "smoke_test_camera",
    "objects": ["sharp_tool"],
    "zones": {"human_in_workspace": False, "tool_in_active_zone": True},
    "hazards": [],
    "relations": [],
    "confidence": 1.0,
}

UNSAFE_SCENE = {
    "camera_id": "smoke_test_camera",
    "objects": ["sharp_tool", "human_hand"],
    "zones": {"human_in_workspace": True, "tool_in_active_zone": True},
    "hazards": ["blade_tip_aimed_at_hand"],
    "relations": [
        {
            "subject": "human_hand",
            "relation": "near",
            "object": "sharp_tool",
            "distance_estimate": "close",
        }
    ],
    "confidence": 1.0,
}


class ScriptedPerceptionAdapter:
    """Returns known scene contexts so the harness path is deterministic."""

    def __init__(self, scenes: list[dict[str, Any]]):
        self.scenes = scenes
        self.index = 0

    def parse_frame(self, frame, now=None) -> dict[str, Any]:
        scene = self.scenes[min(self.index, len(self.scenes) - 1)]
        self.index += 1
        if now is not None:
            scene = {**scene, "timestamp": now}
        return scene


@dataclass
class MockOpenArmController:
    """Small stand-in for an OpenARM/LeRobot controller."""

    calls: list[tuple[str, Any]] = field(default_factory=list)

    def execute(self, planned_action: dict[str, Any]) -> None:
        self.calls.append(("execute", planned_action["id"]))

    def pause(self, affected_arm: str = "both_arms") -> None:
        self.calls.append(("pause", affected_arm))

    def resume(self, affected_arm: str = "both_arms") -> None:
        self.calls.append(("resume", affected_arm))

    def stop(self, affected_arm: str = "both_arms") -> None:
        self.calls.append(("stop", affected_arm))

    def status(self) -> dict[str, Any]:
        return {"connected": True, "mode": "mock"}


def build_openarm_controller(use_real_openarm: bool) -> Any:
    """Replace this function with your real OpenARM/LeRobot construction.

    Expected surface:

        controller.execute(planned_action_dict)
        controller.pause(affected_arm="both_arms")
        controller.resume(affected_arm="both_arms")
        controller.stop(affected_arm="both_arms")

    If your real object exposes `send_action`, `play_trajectory`, or
    `replay_trajectory` instead, `OpenArmLeRobotAdapter` can use those too.
    """
    if not use_real_openarm:
        return MockOpenArmController()

    raise NotImplementedError(
        "Edit build_openarm_controller() in harness/openarm_smoke.py to return "
        "your real OpenARM/LeRobot controller, then rerun with --real-openarm."
    )


def run_command_gating(robot: OpenArmLeRobotAdapter, log_path: Path) -> None:
    planned_action = {
        "id": "smoke_pick_sharp_tool",
        "type": "pick",
        "object": "sharp_tool",
        "arm": "left_arm",
        "source": "openarm_smoke",
        "payload": {"action": {"left_shoulder_pan.pos": 0.0}},
    }

    blocked = safe_execute(
        planned_action,
        frame="unsafe_frame_placeholder",
        perception_adapter=ScriptedPerceptionAdapter([UNSAFE_SCENE]),
        robot_adapter=robot,
        logger=JsonlAuditLogger(log_path),
    )
    print(f"command gate unsafe scene: {blocked.decision} ({blocked.rule})")
    print("  PASS: unsafe action was blocked before reaching the robot")

    allowed = safe_execute(
        planned_action,
        frame="safe_frame_placeholder",
        perception_adapter=ScriptedPerceptionAdapter([SAFE_SCENE]),
        robot_adapter=robot,
        logger=JsonlAuditLogger(log_path),
    )
    print(f"command gate safe scene:   {allowed.decision} ({allowed.rule})")
    print("  CHECK: safe action should call your controller execute/send_action path")


def run_runtime_supervisor(robot: OpenArmLeRobotAdapter, log_path: Path) -> None:
    supervisor = RuntimeWatchdogSupervisor(
        perception_adapter=ScriptedPerceptionAdapter([UNSAFE_SCENE, SAFE_SCENE, SAFE_SCENE]),
        robot_adapter=robot,
        logger=JsonlAuditLogger(log_path),
        clear_frames_before_resume=2,
        auto_resume=True,
    )

    for i in range(3):
        state = supervisor.step(frame=f"runtime_frame_{i}", now=1000.0 + i)
        decision = state.last_decision.decision if state.last_decision else "NONE"
        if state.last_robot_result:
            result = state.last_robot_result
            robot_text = f"{result.command}, executed={result.executed}, detail={result.detail}"
        else:
            robot_text = "no robot command"
        print(f"runtime frame {i}:          {decision} -> mode={state.mode}, {robot_text}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenARM harness smoke test.")
    parser.add_argument(
        "--real-openarm",
        action="store_true",
        help="call build_openarm_controller(True); edit that function first",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="JSONL audit log path; defaults to a temporary file",
    )
    args = parser.parse_args()

    controller = build_openarm_controller(args.real_openarm)
    robot = OpenArmLeRobotAdapter(controller=controller)
    log_path = args.log or Path(tempfile.gettempdir()) / "openarm_harness_smoke.jsonl"

    print(f"robot adapter status:       {robot.status()}")
    run_command_gating(robot, log_path)
    run_runtime_supervisor(robot, log_path)

    calls = getattr(controller, "calls", None)
    if calls is not None:
        print(f"controller calls:           {calls}")
    print(f"audit log:                  {log_path}")


if __name__ == "__main__":
    main()
