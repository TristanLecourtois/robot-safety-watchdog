"""Layer 1 — fast per-frame perception.

Wraps YOLO (object detection) so the rest of the system sees a clean list of
`Detection` objects with stable track IDs (so we can measure speed across
frames). This runs every frame at ~15-30 fps on CPU with the nano model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    track_id: int          # stable across frames (from the tracker), -1 if untracked
    label: str             # COCO class name, e.g. "knife"
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    # Polygon mask in image coords (Nx2), when using a -seg model. Enables
    # orientation/tip recovery (see src/orientation.py). None for box-only models.
    mask: np.ndarray | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_distance(a: Detection, b: Detection) -> float:
    """Closest-edge distance between two boxes (0 if overlapping)."""
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    dx = max(bx1 - ax2, ax1 - bx2, 0.0)
    dy = max(by1 - ay2, ay1 - by2, 0.0)
    return float(np.hypot(dx, dy))


class Detector:
    def __init__(self, model_path: str, min_confidence: float):
        self.model = YOLO(model_path)
        self.min_confidence = min_confidence
        self.names = self.model.names  # id -> class name

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # persist=True keeps track IDs stable across calls (ByteTrack under the hood).
        results = self.model.track(
            frame, persist=True, verbose=False, conf=self.min_confidence
        )
        out: list[Detection] = []
        if not results:
            return out
        res = results[0]
        boxes = res.boxes
        if boxes is None:
            return out
        ids = boxes.id
        masks = res.masks  # may be None for non-seg models
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            track_id = int(ids[i].item()) if ids is not None else -1
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            mask_xy = None
            if masks is not None and masks.xy is not None and i < len(masks.xy):
                poly = masks.xy[i]
                if poly is not None and len(poly) >= 3:
                    mask_xy = np.asarray(poly, dtype=np.float32)
            out.append(
                Detection(
                    track_id=track_id,
                    label=self.names[cls_id],
                    confidence=float(boxes.conf[i].item()),
                    box=(x1, y1, x2, y2),
                    mask=mask_xy,
                )
            )
        return out
