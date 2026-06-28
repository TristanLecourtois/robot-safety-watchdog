"""Direct OpenARM pause/resume/stop timing test.

This script intentionally does not use perception, policies, VLA, or audit logs.
It only validates the LeRobot/OpenARM control surface used by the harness.

Example:

    python -m harness.scripts.openarm_pause_resume --robot-factory my_setup:build_robot
"""
from __future__ import annotations

import argparse
import importlib
import math
import time
from typing import Any, Callable

from harness import LeRobotOpenArmController


def build_openarm_robot() -> Any:
    """Return your real LeRobot/OpenARM robot object.

    Edit this function if you prefer a local hard-coded setup instead of passing
    --robot-factory module:function on the command line.
    """
    raise NotImplementedError(
        "Provide --robot-factory module:function, or edit build_openarm_robot() "
        "in harness/scripts/openarm_pause_resume.py to return openarm_robot."
    )


def _load_factory(factory_ref: str | None) -> Callable[[], Any]:
    if factory_ref is None:
        return build_openarm_robot
    if ":" not in factory_ref:
        raise ValueError("--robot-factory must look like module:function")
    module_name, function_name = factory_ref.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, function_name)
    if not callable(factory):
        raise TypeError(f"{factory_ref} is not callable")
    return factory


def _call_if_present(obj: Any, method_name: str, *args: Any, **kwargs: Any) -> bool:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return False
    method(*args, **kwargs)
    return True


def _maybe_connect(robot: Any, calibrate: bool) -> None:
    is_connected = getattr(robot, "is_connected", None)
    if is_connected is True:
        return
    connect = getattr(robot, "connect", None)
    if not callable(connect):
        return
    try:
        connect(calibrate=calibrate)
    except TypeError:
        connect()


def _sleep_countdown(label: str, seconds: float) -> None:
    if seconds <= 0:
        return
    print(f"{label}: waiting {seconds:g}s")
    end = time.monotonic() + seconds
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        print(f"  {label}: {remaining:0.1f}s remaining")
        time.sleep(min(1.0, remaining))


def _run_motion(
    controller: LeRobotOpenArmController,
    seconds: float,
    *,
    affected_arm: str,
    motion_joints: list[str],
    amplitude_deg: float,
    period: float,
    hz: float,
    telemetry: bool,
) -> None:
    if seconds <= 0:
        return
    if hz <= 0:
        raise ValueError("--motion-hz must be > 0")
    if period <= 0:
        raise ValueError("--motion-period must be > 0")

    base_action = controller._current_hold_action(affected_arm)
    keys = _select_motion_keys(base_action, motion_joints)
    if not keys:
        raise RuntimeError(
            "Could not find a motion joint in the current OpenARM action keys. "
            "Pass --motion-joint with a key from robot.action_features, for example left_joint_6.pos."
        )

    print(
        "motion: moving "
        f"{', '.join(keys)} for {seconds:g}s at +/-{amplitude_deg:g} deg, {hz:g} Hz"
    )
    for key in keys:
        print(f"  motion: base {key}={base_action[key]:0.3f}")
    start = time.monotonic()
    end = start + seconds
    interval = 1.0 / hz
    next_status = start
    while True:
        now = time.monotonic()
        remaining = end - now
        if remaining <= 0:
            break
        if now >= next_status:
            telemetry_text = ""
            if telemetry:
                obs = controller.robot.get_observation()
                telemetry_text = " | " + ", ".join(
                    f"obs {key}={float(obs.get(key, float('nan'))):0.3f}" for key in keys
                )
            print(f"  motion: {remaining:0.1f}s remaining{telemetry_text}")
            next_status = now + 1.0
        phase = 2.0 * math.pi * ((now - start) / period)
        offset = amplitude_deg * math.sin(phase)
        action = dict(base_action)
        for key in keys:
            action[key] = base_action[key] + offset
        controller.robot.send_action(action)
        time.sleep(interval)


def _select_motion_keys(base_action: dict[str, Any], requested: list[str]) -> list[str]:
    if requested:
        missing = [key for key in requested if key not in base_action]
        if missing:
            available = ", ".join(sorted(base_action))
            raise KeyError(f"Motion joint(s) not in hold action: {missing}. Available keys: {available}")
        return requested

    preferred_suffixes = (
        "joint_6.pos",
        "joint_5.pos",
        "joint_7.pos",
        "joint_1.pos",
    )
    for suffix in preferred_suffixes:
        matches = [key for key in sorted(base_action) if key.endswith(suffix)]
        if matches:
            return [matches[0]]
    return [key for key in sorted(base_action) if key.endswith(".pos")][:1]


def _maintain_hold(controller: LeRobotOpenArmController, seconds: float, hold_hz: float) -> None:
    if seconds <= 0:
        return
    if hold_hz <= 0:
        _sleep_countdown("hold", seconds)
        return

    print(f"hold: maintaining current pose for {seconds:g}s at {hold_hz:g} Hz")
    end = time.monotonic() + seconds
    interval = 1.0 / hold_hz
    next_status = time.monotonic()
    while True:
        now = time.monotonic()
        remaining = end - now
        if remaining <= 0:
            break
        if now >= next_status:
            print(f"  hold: {remaining:0.1f}s remaining")
            next_status = now + 1.0
        controller._send_hold_once()
        time.sleep(max(0.0, interval))


def run_sequence(args: argparse.Namespace) -> None:
    factory = _load_factory(args.robot_factory)
    robot = factory()

    if args.connect:
        print(f"connect: calling robot.connect(calibrate={args.calibrate}) if available")
        _maybe_connect(robot, calibrate=args.calibrate)

    controller = LeRobotOpenArmController(
        robot=robot,
        inference_engine=None,
        interpolator=None,
        hold_hz=args.hold_hz,
        stop_mode=args.stop_mode,
    )

    print("ready: sequence is pause -> resume -> stop")
    if args.motion:
        _run_motion(
            controller,
            args.before_pause,
            affected_arm=args.affected_arm,
            motion_joints=args.motion_joint,
            amplitude_deg=args.motion_amplitude_deg,
            period=args.motion_period,
            hz=args.motion_hz,
            telemetry=args.telemetry,
        )
    else:
        _sleep_countdown("before pause", args.before_pause)

    print("pause: calling controller.pause()")
    controller.pause(affected_arm=args.affected_arm)
    _maintain_hold(controller, args.before_resume, args.hold_hz)

    print("resume: calling controller.resume()")
    controller.resume(affected_arm=args.affected_arm)
    if args.motion:
        _run_motion(
            controller,
            args.before_stop,
            affected_arm=args.affected_arm,
            motion_joints=args.motion_joint,
            amplitude_deg=args.motion_amplitude_deg,
            period=args.motion_period,
            hz=args.motion_hz,
            telemetry=args.telemetry,
        )
    else:
        _sleep_countdown("before stop", args.before_stop)

    print("stop: calling controller.stop()")
    controller.stop(affected_arm=args.affected_arm)
    if args.hold_after_stop > 0 and args.stop_mode == "hold":
        _maintain_hold(controller, args.hold_after_stop, args.hold_hz)

    if args.disconnect:
        print("disconnect: calling robot.disconnect() if available")
        _call_if_present(robot, "disconnect")

    print("done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct OpenARM pause/resume/stop timing test.")
    parser.add_argument(
        "--robot-factory",
        default=None,
        help="Python factory returning openarm_robot, formatted as module:function",
    )
    parser.add_argument("--before-pause", type=float, default=10.0)
    parser.add_argument("--before-resume", type=float, default=10.0)
    parser.add_argument("--before-stop", type=float, default=10.0)
    parser.add_argument("--hold-after-stop", type=float, default=0.0)
    parser.add_argument("--hold-hz", type=float, default=20.0)
    parser.add_argument("--affected-arm", default="both_arms")
    parser.add_argument("--stop-mode", choices=("hold", "disconnect"), default="hold")
    parser.add_argument("--motion", action="store_true", help="send a tiny sinusoidal movement before pause and after resume")
    parser.add_argument(
        "--motion-joint",
        action="append",
        default=[],
        help="action key to move, e.g. left_joint_6.pos; can be passed multiple times",
    )
    parser.add_argument("--motion-amplitude-deg", type=float, default=2.0)
    parser.add_argument("--motion-period", type=float, default=4.0)
    parser.add_argument("--motion-hz", type=float, default=20.0)
    parser.add_argument("--telemetry", action="store_true", help="print observed joint position during motion")
    parser.add_argument("--connect", action="store_true", help="call robot.connect() before the test")
    parser.add_argument("--calibrate", action="store_true", help="pass calibrate=True when --connect is used")
    parser.add_argument("--disconnect", action="store_true", help="call robot.disconnect() after the test")
    args = parser.parse_args()

    try:
        run_sequence(args)
    except KeyboardInterrupt:
        raise SystemExit("\ninterrupted by operator\n")


if __name__ == "__main__":
    main()
