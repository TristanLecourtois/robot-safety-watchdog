"""Precise hand landmarks via MediaPipe Hands.

Person/knife boxes are coarse. To answer "is the blade tip near a *fingertip*"
we need the actual finger positions, not a box. MediaPipe gives 21 landmarks
per hand; we expose the 5 fingertips (and all points) in image pixels.

Degrades gracefully: if mediapipe isn't installed, returns no hands and the
system falls back to person-box proximity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import mediapipe as mp
    _HANDS = mp.solutions.hands
    _AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _AVAILABLE = False


# MediaPipe Hands landmark indices for the five fingertips.
FINGERTIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky


@dataclass
class Hand:
    points: np.ndarray            # 21x2 pixel coords
    fingertips: np.ndarray        # 5x2 pixel coords
    handedness: str               # "Left" / "Right"


class HandTracker:
    def __init__(self, max_hands: int = 4, min_conf: float = 0.4):
        self.available = _AVAILABLE
        if self.available:
            self._hands = _HANDS.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=min_conf,
                min_tracking_confidence=0.4,
            )

    def detect(self, frame_bgr: np.ndarray) -> list[Hand]:
        if not self.available:
            return []
        h, w = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]
        result = self._hands.process(rgb)
        hands: list[Hand] = []
        if not result.multi_hand_landmarks:
            return hands
        labels = result.multi_handedness or []
        for idx, lms in enumerate(result.multi_hand_landmarks):
            pts = np.array([[lm.x * w, lm.y * h] for lm in lms.landmark], dtype=np.float32)
            label = "?"
            if idx < len(labels) and labels[idx].classification:
                label = labels[idx].classification[0].label
            hands.append(Hand(points=pts, fingertips=pts[FINGERTIP_IDS], handedness=label))
        return hands


def nearest_fingertip(
    tip: tuple[float, float], hands: list[Hand]
) -> tuple[float, tuple[float, float]] | None:
    """Return (distance_px, fingertip_xy) for the closest fingertip to a point."""
    best = None
    p = np.asarray(tip, dtype=np.float32)
    for hand in hands:
        d = np.linalg.norm(hand.fingertips - p, axis=1)
        j = int(np.argmin(d))
        dist = float(d[j])
        if best is None or dist < best[0]:
            best = (dist, (float(hand.fingertips[j][0]), float(hand.fingertips[j][1])))
    return best
