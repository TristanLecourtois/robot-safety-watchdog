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


@dataclass
class CameraConfig:
    """Per-camera configuration for multi-camera setups.

    `source` is a camera device index (int) or any path/URL that OpenCV
    can open (video file, RTSP stream, etc.).

    For 3-D triangulation supply both `intrinsics` and `extrinsics`:
      intrinsics  — 3×3 camera matrix K as 9 floats, row-major
                    [fx, 0, cx,  0, fy, cy,  0, 0, 1]
      extrinsics  — 4×4 world-to-camera transform as 16 floats, row-major
    Leave both None to skip triangulation (cross-camera correlation still works).
    """
    source: int | str = 0
    label: str = ""
    frame_width: int = 1280
    frame_height: int = 720
    intrinsics: list[float] | None = None
    extrinsics: list[float] | None = None

    def __post_init__(self):
        if not self.label:
            self.label = f"cam_{self.source}"


# --- Open-vocabulary prompts (YOLOE backend) --------------------------------
# With the open-vocab backend we describe what to segment/classify in plain
# words — so we can target DANGEROUS objects directly and even detect "hand"
# without MediaPipe. Edit this list freely; YOLOE will segment + classify each.
OPEN_VOCAB_PROMPTS = [
    # people & pets at risk
    "person", "hand", "child", "dog", "cat",
    # sharp hazards
    "knife", "kitchen knife", "cleaver", "scissors", "fork",
    # fragile / spill
    "cup", "mug", "glass", "bottle", "bowl", "plate", "wine glass",
    # hot zones / fire
    "pot", "pan", "frying pan", "stove", "gas stove", "oven", "microwave",
    "kettle", "flame", "fire", "boiling water",
    # robot
    "robot arm", "robotic gripper", "robot",
]

# --- Hazard categories (match the class/label strings above) ----------------
SHARP_CLASSES = {"knife", "kitchen knife", "cleaver", "scissors", "fork"}
HUMAN_CLASSES = {"person", "child"}
HAND_CLASSES = {"hand"}  # open-vocab hand box -> precise-ish target without MediaPipe
FRAGILE_CLASSES = {"cup", "mug", "glass", "wine glass", "bowl", "plate", "bottle"}
HOT_ZONE_CLASSES = {"pot", "pan", "frying pan", "stove", "gas stove", "oven",
                    "microwave", "kettle", "flame", "fire", "boiling water"}
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
    # --- capture (single-camera defaults; ignored when `cameras` is set) ---
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720

    # --- multi-camera: set this to run N camera streams in parallel ----------
    # When empty the system falls back to single-camera mode using camera_index.
    cameras: list[CameraConfig] = field(default_factory=list)

    # --- perception (layer 1) ------------------------------------------------
    # backend:
    #   "yoloe" -> open-vocabulary segmentation (recommended): segments &
    #             classifies arbitrary objects from OPEN_VOCAB_PROMPTS, incl.
    #             dangerous ones and "hand". Needs CLIP + MobileCLIP (one-time
    #             ~572MB download; text embeddings are cached afterward).
    #   "yolo"  -> classic COCO segmentation (80 fixed classes). Lighter, no
    #             extra downloads. Use a LARGE model for usable knife masks.
    detector_backend: str = "yoloe"
    yoloe_model: str = "yoloe-11s-seg.pt"   # -> 11m/11l-seg for more accuracy
    open_vocab_prompts: list[str] = field(default_factory=lambda: list(OPEN_VOCAB_PROMPTS))
    # Used only when detector_backend == "yolo". Nano misses thin objects like
    # knives — default to a large seg model for real detection quality.
    yolo_model: str = "yolo11l-seg.pt"
    textpe_cache: str = ".textpe_cache.pt"  # cached open-vocab text embeddings
    use_hand_landmarks: bool = True  # MediaPipe Hands (arm64/Linux/Win only)

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

    @property
    def camera_configs(self) -> list[CameraConfig]:
        """Canonical list of cameras: explicit multi-camera list or single-camera fallback."""
        if self.cameras:
            return self.cameras
        return [CameraConfig(
            source=self.camera_index,
            label="camera_0",
            frame_width=self.frame_width,
            frame_height=self.frame_height,
        )]


CONFIG = WatchdogConfig()
