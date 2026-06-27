"""Minimal 'make Reachy Mini move' script.

Prerequisite: the daemon must be running in another terminal (`reachy-mini-daemon`).
Run this in a venv where `reachy-mini` is installed:

    python scripts/hello_reachy.py

Docs: https://huggingface.co/docs/reachy_mini/SDK/quickstart
"""
import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

# The constructor auto-detects Lite (USB/localhost) vs Wireless (network) and
# connects to the running daemon.
with ReachyMini() as mini:
    print("Connected to Reachy Mini!")

    # 1) Wiggle the antennas (simplest move)
    print("Wiggling antennas...")
    mini.goto_target(antennas=[0.5, -0.5], duration=0.5)
    mini.goto_target(antennas=[-0.5, 0.5], duration=0.5)
    mini.goto_target(antennas=[0, 0], duration=0.5)

    # 2) Move the head: look up 10mm + tilt (roll) 15 degrees
    print("Moving head...")
    mini.goto_target(
        head=create_head_pose(z=10, roll=15, degrees=True, mm=True),
        duration=1.0,
    )

    # 3) Combined move: head up + antennas + rotate body, smooth profile
    mini.goto_target(
        head=create_head_pose(z=10, mm=True),
        antennas=np.deg2rad([45, 45]),
        body_yaw=np.deg2rad(30),
        duration=2.0,
        method="minjerk",
    )

    # Back to neutral
    mini.goto_target(
        head=create_head_pose(),
        antennas=[0, 0],
        body_yaw=0.0,
        duration=1.0,
    )
    print("Done!")
