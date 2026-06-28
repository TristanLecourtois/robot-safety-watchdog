"""Factory for creating a real LeRobot OpenARM robot.

Use this as the --robot-factory target for the OpenARM harness demo scripts.

Environment variables:

    OPENARM_MODE=bi|single          default: bi
    OPENARM_LEFT_PORT=can0          default: can0
    OPENARM_RIGHT_PORT=can1         default: can1
    OPENARM_PORT=can0               used only for OPENARM_MODE=single
    OPENARM_SIDE=left|right         used only for OPENARM_MODE=single
    OPENARM_ID=bi_openarm_follower  robot id/calibration prefix
    OPENARM_CALIBRATION_DIR=/path   optional calibration directory
    OPENARM_CAN_INTERFACE=socketcan default: socketcan
    OPENARM_USE_CAN_FD=1            default: 1
    OPENARM_CAN_BITRATE=1000000     default: 1000000
    OPENARM_CAN_DATA_BITRATE=5000000 default: 5000000
"""
from __future__ import annotations

import os
from pathlib import Path


def build_robot():
    """Build and return the real OpenARM LeRobot robot object.

    The pause/resume test can connect it itself with --connect. If your teleop
    setup already connected a robot inside the same Python process, return that
    existing object instead of constructing a new one here.
    """
    mode = os.getenv("OPENARM_MODE", "bi").strip().lower()
    if mode in {"bi", "bimanual", "dual"}:
        return _build_bimanual_robot()
    if mode in {"single", "mono"}:
        return _build_single_robot()
    raise ValueError("OPENARM_MODE must be one of: bi, bimanual, dual, single, mono")


def _build_bimanual_robot():
    from lerobot.robots.bi_openarm_follower import BiOpenArmFollower, BiOpenArmFollowerConfig
    from lerobot.robots.openarm_follower import OpenArmFollowerConfigBase

    robot_id = os.getenv("OPENARM_ID", "bi_openarm_follower")
    common = _common_openarm_kwargs()
    config = BiOpenArmFollowerConfig(
        id=robot_id,
        calibration_dir=_optional_path("OPENARM_CALIBRATION_DIR"),
        left_arm_config=OpenArmFollowerConfigBase(
            port=os.getenv("OPENARM_LEFT_PORT", "can0"),
            side="left",
            **common,
        ),
        right_arm_config=OpenArmFollowerConfigBase(
            port=os.getenv("OPENARM_RIGHT_PORT", "can1"),
            side="right",
            **common,
        ),
    )
    return BiOpenArmFollower(config)


def _build_single_robot():
    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig

    side = os.getenv("OPENARM_SIDE", "left").strip().lower()
    if side not in {"left", "right"}:
        raise ValueError("OPENARM_SIDE must be 'left' or 'right'")

    config = OpenArmFollowerConfig(
        id=os.getenv("OPENARM_ID", f"openarm_follower_{side}"),
        calibration_dir=_optional_path("OPENARM_CALIBRATION_DIR"),
        port=os.getenv("OPENARM_PORT", os.getenv("OPENARM_LEFT_PORT", "can0")),
        side=side,
        **_common_openarm_kwargs(),
    )
    return OpenArmFollower(config)


def _common_openarm_kwargs() -> dict:
    return {
        "can_interface": os.getenv("OPENARM_CAN_INTERFACE", "socketcan"),
        "use_can_fd": _env_bool("OPENARM_USE_CAN_FD", default=True),
        "can_bitrate": _env_int("OPENARM_CAN_BITRATE", 1000000),
        "can_data_bitrate": _env_int("OPENARM_CAN_DATA_BITRATE", 5000000),
        "disable_torque_on_disconnect": _env_bool("OPENARM_DISABLE_TORQUE_ON_DISCONNECT", default=True),
    }


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    if not value:
        return None
    return Path(value).expanduser()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
