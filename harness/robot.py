"""Robot adapters for OpenARM control through LeRobot."""
from __future__ import annotations

import time
from typing import Any

from harness.models import PlannedAction, RobotCommandResult


class RobotAdapter:
    def execute(self, planned_action: PlannedAction) -> RobotCommandResult:
        raise NotImplementedError

    def pause(self, affected_arm: str = "both_arms") -> RobotCommandResult:
        raise NotImplementedError

    def resume(self, affected_arm: str = "both_arms") -> RobotCommandResult:
        raise NotImplementedError

    def stop(self, affected_arm: str = "both_arms") -> RobotCommandResult:
        raise NotImplementedError

    def status(self) -> dict[str, Any]:
        return {}


class OpenArmLeRobotAdapter(RobotAdapter):
    """Adapter for a two-arm OpenARM cell controlled by LeRobot.

    Pass the actual LeRobot/OpenARM controller as `controller`. The adapter uses
    duck-typing so it can work with teleop replay objects, scripted action
    runners, or a direct robot instance.
    """

    def __init__(self, controller: Any | None = None, dry_run: bool = False):
        self.controller = controller
        self.dry_run = dry_run or controller is None

    def execute(self, planned_action: PlannedAction) -> RobotCommandResult:
        if self.dry_run:
            return RobotCommandResult("execute", False, planned_action.arm, "dry_run: action not sent")

        payload = planned_action.payload
        for method_name in ("execute", "run_action"):
            if _has_method(self.controller, method_name):
                getattr(self.controller, method_name)(planned_action.to_dict())
                return RobotCommandResult("execute", True, planned_action.arm, method_name)

        trajectory = payload.get("trajectory") or payload.get("teleop_replay")
        if trajectory is not None:
            for method_name in ("play_trajectory", "replay_trajectory"):
                if _has_method(self.controller, method_name):
                    getattr(self.controller, method_name)(trajectory)
                    return RobotCommandResult("execute", True, planned_action.arm, method_name)

        action_vector = payload.get("action")
        if action_vector is not None and _has_method(self.controller, "send_action"):
            self.controller.send_action(action_vector)
            return RobotCommandResult("execute", True, planned_action.arm, "send_action")

        if _has_method(self.controller, "send_action"):
            self.controller.send_action(planned_action.to_dict())
            return RobotCommandResult("execute", True, planned_action.arm, "send_action")

        raise AttributeError(
            "OpenARM controller has no supported execute method. "
            "Expose execute/run_action/send_action or provide trajectory replay."
        )

    def pause(self, affected_arm: str = "both_arms") -> RobotCommandResult:
        return self._call_control("pause", affected_arm)

    def resume(self, affected_arm: str = "both_arms") -> RobotCommandResult:
        return self._call_control("resume", affected_arm)

    def stop(self, affected_arm: str = "both_arms") -> RobotCommandResult:
        return self._call_control("stop", affected_arm, fallbacks=("stop", "emergency_stop", "disconnect"))

    def status(self) -> dict[str, Any]:
        if self.dry_run:
            return {"connected": False, "mode": "dry_run"}
        if _has_method(self.controller, "status"):
            return dict(self.controller.status())
        return {"connected": True, "mode": "unknown"}

    def _call_control(
        self,
        command: str,
        affected_arm: str,
        fallbacks: tuple[str, ...] | None = None,
    ) -> RobotCommandResult:
        if self.dry_run:
            return RobotCommandResult(command, False, affected_arm, f"dry_run: {command} not sent")

        names = fallbacks or (command,)
        for name in names:
            if _has_method(self.controller, name):
                method = getattr(self.controller, name)
                try:
                    method(affected_arm=affected_arm)
                except TypeError:
                    method()
                return RobotCommandResult(command, True, affected_arm, name)
        return RobotCommandResult(command, False, affected_arm, f"controller has no {command} method")


def _has_method(obj: Any, name: str) -> bool:
    return obj is not None and callable(getattr(obj, name, None))


class LeRobotOpenArmController:
    """Shim around LeRobot OpenARM robots and rollout inference engines.

    LeRobot `main` exposes OpenARM control primarily through `send_action`.
    It does not currently provide a standard robot-level `pause/resume` API for
    OpenArmFollower/BiOpenArmFollower, so this shim creates harness semantics:

    - pause: pause inference if available, then hold current joint positions.
    - resume: reset/resume inference if available.
    - stop: stop inference, then hold or disconnect depending on configuration.
    """

    def __init__(
        self,
        robot: Any,
        *,
        inference_engine: Any | None = None,
        interpolator: Any | None = None,
        hold_hz: float = 20.0,
        stop_mode: str = "hold",  # hold | disconnect
    ):
        self.robot = robot
        self.inference_engine = inference_engine
        self.interpolator = interpolator
        self.hold_hz = hold_hz
        self.stop_mode = stop_mode
        self._hold_action: dict[str, Any] | None = None
        self._paused = False

    def execute(self, planned_action: dict) -> None:
        payload = planned_action.get("payload") or {}
        trajectory = payload.get("trajectory") or payload.get("teleop_replay")
        if trajectory is not None:
            self.play_trajectory(trajectory)
            return
        action = payload.get("action")
        if action is not None:
            self.robot.send_action(action)
            return
        self.robot.send_action(planned_action)

    def play_trajectory(self, trajectory) -> None:
        for action in trajectory:
            if self._paused:
                self._send_hold_once()
                continue
            self.robot.send_action(action)

    def pause(self, affected_arm: str = "both_arms") -> None:
        _call_if_present(self.inference_engine, "pause")
        self._hold_action = self._current_hold_action(affected_arm)
        self._paused = True
        self._send_hold_once()

    def resume(self, affected_arm: str = "both_arms") -> None:
        if self.interpolator is not None:
            _call_if_present(self.interpolator, "reset")
        if self.inference_engine is not None:
            _call_if_present(self.inference_engine, "reset")
            _call_if_present(self.inference_engine, "resume")
        self._paused = False

    def stop(self, affected_arm: str = "both_arms") -> None:
        _call_if_present(self.inference_engine, "stop")
        if self.stop_mode == "disconnect" and _has_method(self.robot, "disconnect"):
            self.robot.disconnect()
            self._paused = False
            return
        self._hold_action = self._current_hold_action(affected_arm)
        self._paused = True
        self._send_hold_once()

    def _send_hold_once(self) -> None:
        if self._hold_action:
            self.robot.send_action(self._hold_action)
            if self.hold_hz > 0:
                time.sleep(1.0 / self.hold_hz)

    def _current_hold_action(self, affected_arm: str) -> dict[str, Any]:
        obs = self.robot.get_observation()
        action_keys = set(getattr(self.robot, "action_features", {}) or {})
        hold = {}
        for key, value in obs.items():
            if not key.endswith(".pos"):
                continue
            if action_keys and key not in action_keys:
                continue
            if affected_arm == "left_arm" and not key.startswith("left_"):
                continue
            if affected_arm == "right_arm" and not key.startswith("right_"):
                continue
            hold[key] = value
        if not hold:
            raise RuntimeError("Could not derive a hold action from LeRobot observation .pos keys.")
        return hold


def _call_if_present(obj: Any, name: str) -> bool:
    if _has_method(obj, name):
        getattr(obj, name)()
        return True
    return False
