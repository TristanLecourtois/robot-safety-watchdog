"""Central configuration for the vision watchdog.

Tune thresholds here — they are the difference between a noisy demo and a
convincing one. All distances are in pixels (we work in image space; for a
real rig you'd calibrate to cm with a known reference object).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


# --- Objects we care about in a kitchen scene (COCO class names from YOLO) ---
# COCO doesn't have "stove" or "robot arm", so for the hackathon we proxy:
#   - knife/scissors      -> sharp hazard
#   - person/hand(proxy)  -> the human at risk
#   - cup/bowl/wine glass -> fragile / spill hazard
#   - oven/microwave      -> hot zone (static)
SHARP_CLASSES = {"knife", "scissors", "fork"}
HUMAN_CLASSES = {"person"}
FRAGILE_CLASSES = {"cup", "wine glass", "bowl", "bottle"}
HOT_ZONE_CLASSES = {"oven", "microwave", "toaster"}
# Treat any of these as "the thing the robot is manipulating" if held near a person.
HANDHELD_CLASSES = SHARP_CLASSES | FRAGILE_CLASSES


@dataclass
class Thresholds:
    # Hard rule: blade TIP this close (px) to a fingertip/hand -> immediate danger.
    blade_tip_to_hand_px: float = 90.0
    # Coarser fallback when we only have boxes: sharp box near person box.
    sharp_to_human_px: float = 120.0
    # The blade is "pointing at" the hand if the angle between the blade axis
    # and the tip->hand vector is below this (degrees). Small = aimed straight at.
    blade_aim_angle_deg: float = 30.0
    # Robot/object approaching a hot zone.
    object_to_hot_px: float = 90.0
    # Fragile object being moved fast (px/frame) -> risk of drop/shatter.
    fragile_speed_px: float = 45.0
    # Min detection confidence to consider a box at all.
    min_confidence: float = 0.35


@dataclass
class WatchdogConfig:
    # --- capture ---
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720

    # --- perception (layer 1) ---
    # SEGMENTATION model (-seg) so we get masks, not just boxes. Masks let us
    # recover the blade's orientation and tip via PCA. Bump to yolov8s-seg for
    # accuracy at some fps cost.
    yolo_model: str = "yolov8n-seg.pt"
    use_hand_landmarks: bool = True  # MediaPipe Hands for precise fingertip positions

    # --- reasoning (layer 2) — the GENERALIST detector -----------------------
    # The VLM is open-vocabulary: it judges ANY dangerous situation, not just the
    # ones the rules hard-code. So we run it continuously as the primary detector.
    # For a continuous cadence, claude-sonnet-4-6 / claude-haiku-4-5 are faster &
    # cheaper than opus and usually the right call (set WATCHDOG_VLM_MODEL).
    vlm_model: str = field(default_factory=lambda: os.getenv("WATCHDOG_VLM_MODEL", "claude-opus-4-8"))
    # Steady cadence: judge the scene roughly this often even with no rule hit.
    # This is what makes it generalist. ~1.5s ≈ as real-time as a VLM gets.
    vlm_interval_s: float = 1.5
    # Floor between calls when a rule escalates (call ASAP, but don't spam the API).
    vlm_min_interval_s: float = 0.8
    # Run the VLM in a background thread so the fast layer never stalls waiting
    # on the API (the overlay keeps the last verdict until a new one arrives).
    vlm_async: bool = True

    # --- alerting ---
    alert_cooldown_s: float = 3.0  # don't spam the same alert
    log_path: str = "watchdog_events.jsonl"

    thresholds: Thresholds = field(default_factory=Thresholds)


CONFIG = WatchdogConfig()
