"""Smoke test for the OpenARM safety harness.

Run from a fresh clone with:

    python3 -m harness.openarm_smoke

This intentionally avoids the heavy perception stack. It checks the harness
control path with a fake perception adapter and a controller-shaped object that
matches the OpenARM/LeRobot methods used by `OpenArmLeRobotAdapter`.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness import (
    JsonlAuditLogger,
    LeRobotOpenArmController,
    OpenArmLeRobotAdapter,
    RuntimeWatchdogSupervisor,
    safe_execute,
)


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


@dataclass
class MockLeRobotRobot:
    """Small stand-in for a LeRobot robot object, without OpenARM hardware."""

    calls: list[tuple[str, Any]] = field(default_factory=list)
    action_features: dict[str, Any] = field(
        default_factory=lambda: {
            "left_shoulder_pan.pos": {},
            "right_shoulder_pan.pos": {},
        }
    )

    def send_action(self, action: dict[str, Any]) -> None:
        self.calls.append(("send_action", action))

    def get_observation(self) -> dict[str, float]:
        self.calls.append(("get_observation", None))
        return {
            "left_shoulder_pan.pos": 0.1,
            "right_shoulder_pan.pos": -0.1,
        }

    def disconnect(self) -> None:
        self.calls.append(("disconnect", None))


@dataclass
class MockLeRobotInference:
    """Stand-in for a LeRobot inference engine with pause/resume hooks."""

    calls: list[tuple[str, Any]] = field(default_factory=list)

    def pause(self) -> None:
        self.calls.append(("pause", None))

    def resume(self) -> None:
        self.calls.append(("resume", None))

    def reset(self) -> None:
        self.calls.append(("reset", None))

    def stop(self) -> None:
        self.calls.append(("stop", None))


@dataclass
class SmokeControllerBundle:
    controller: Any
    debug_objects: dict[str, Any] = field(default_factory=dict)


def build_real_lerobot_bundle() -> SmokeControllerBundle:
    """Import the real LeRobot package and run the harness on a fake Robot subclass."""
    try:
        import lerobot
        from lerobot.robots import Robot, RobotConfig
    except Exception as exc:
        raise RuntimeError(
            "LeRobot is not importable in this Python environment. Install LeRobot first, "
            "then rerun: python3 -m harness.openarm_smoke --real-lerobot"
        ) from exc

    @dataclass
    class SmokeLeRobotConfig(RobotConfig):
        type: str = "smoke_lerobot"

    class SmokeLeRobotRobot(Robot):
        config_class = SmokeLeRobotConfig
        name = "smoke_lerobot"

        def __init__(self, config: SmokeLeRobotConfig):
            super().__init__(config)
            self.calls: list[tuple[str, Any]] = []
            self._connected = False

        @property
        def observation_features(self) -> dict:
            return {
                "left_shoulder_pan.pos": float,
                "right_shoulder_pan.pos": float,
            }

        @property
        def action_features(self) -> dict:
            return {
                "left_shoulder_pan.pos": float,
                "right_shoulder_pan.pos": float,
            }

        @property
        def is_connected(self) -> bool:
            return self._connected

        @property
        def is_calibrated(self) -> bool:
            return True

        def connect(self, calibrate: bool = True) -> None:
            self.calls.append(("connect", {"calibrate": calibrate}))
            self._connected = True

        def calibrate(self) -> None:
            self.calls.append(("calibrate", None))

        def configure(self) -> None:
            self.calls.append(("configure", None))

        def get_observation(self) -> dict[str, float]:
            self.calls.append(("get_observation", None))
            return {
                "left_shoulder_pan.pos": 0.1,
                "right_shoulder_pan.pos": -0.1,
            }

        def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
            self.calls.append(("send_action", action))
            return action

        def disconnect(self) -> None:
            self.calls.append(("disconnect", None))
            self._connected = False

    try:
        config = SmokeLeRobotConfig(id="smoke_lerobot")
    except TypeError:
        config = SmokeLeRobotConfig()

    robot = SmokeLeRobotRobot(config)
    robot.connect(calibrate=False)
    inference = MockLeRobotInference()
    controller = LeRobotOpenArmController(
        robot=robot,
        inference_engine=inference,
        stop_mode="hold",
    )
    version = getattr(lerobot, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version("lerobot")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"

    return SmokeControllerBundle(
        controller=controller,
        debug_objects={
            "real_lerobot_module": {"version": version, "module": str(lerobot)},
            "real_lerobot_robot": robot,
            "real_lerobot_inference": inference,
        },
    )


def build_openarm_controller(
    use_real_openarm: bool,
    use_mock_lerobot: bool = False,
    use_real_lerobot: bool = False,
) -> SmokeControllerBundle:
    """Replace this function with your real OpenARM/LeRobot construction.

    Expected surface:

        controller.execute(planned_action_dict)
        controller.pause(affected_arm="both_arms")
        controller.resume(affected_arm="both_arms")
        controller.stop(affected_arm="both_arms")

    If your real object exposes `send_action`, `play_trajectory`, or
    `replay_trajectory` instead, `OpenArmLeRobotAdapter` can use those too.
    """
    if use_real_lerobot:
        return build_real_lerobot_bundle()

    if use_mock_lerobot:
        robot = MockLeRobotRobot()
        inference = MockLeRobotInference()
        controller = LeRobotOpenArmController(
            robot=robot,
            inference_engine=inference,
            stop_mode="hold",
        )
        return SmokeControllerBundle(
            controller=controller,
            debug_objects={"mock_lerobot_robot": robot, "mock_lerobot_inference": inference},
        )

    if not use_real_openarm:
        controller = MockOpenArmController()
        return SmokeControllerBundle(controller=controller, debug_objects={"mock_controller": controller})

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
        "--mock-lerobot",
        action="store_true",
        help="test the LeRobotOpenArmController path with fake send_action/get_observation objects",
    )
    parser.add_argument(
        "--real-lerobot",
        action="store_true",
        help="import the real lerobot package and test the harness with a fake lerobot.robots.Robot subclass",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="JSONL audit log path; defaults to a temporary file",
    )
    args = parser.parse_args()
    if sum([args.real_openarm, args.mock_lerobot, args.real_lerobot]) > 1:
        parser.error("choose only one of --real-openarm, --mock-lerobot, or --real-lerobot")

    try:
        bundle = build_openarm_controller(
            args.real_openarm,
            use_mock_lerobot=args.mock_lerobot,
            use_real_lerobot=args.real_lerobot,
        )
    except RuntimeError as exc:
        parser.exit(2, f"error: {exc}\n")
    controller = bundle.controller
    robot = OpenArmLeRobotAdapter(controller=controller)
    log_path = args.log or Path(tempfile.gettempdir()) / "openarm_harness_smoke.jsonl"

    print(f"robot adapter status:       {robot.status()}")
    run_command_gating(robot, log_path)
    run_runtime_supervisor(robot, log_path)

    calls = getattr(controller, "calls", None)
    if calls is not None:
        print(f"controller calls:           {calls}")
    for name, obj in bundle.debug_objects.items():
        if isinstance(obj, dict):
            print(f"{name}:     {obj}")
            continue
        debug_calls = getattr(obj, "calls", None)
        if debug_calls is not None:
            print(f"{name} calls:    {debug_calls}")
    print(f"audit log:                  {log_path}")


if __name__ == "__main__":
    main()
