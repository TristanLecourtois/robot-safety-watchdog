"""Layer 1 — per-frame perception.

Wraps a segmentation model so the rest of the system sees a clean list of
`Detection` objects (label, box, mask, track id). Two backends:

  - "yoloe": open-vocabulary segmentation. We hand it text prompts
    (config.OPEN_VOCAB_PROMPTS) and it segments + classifies *those* concepts,
    so we can target dangerous objects directly and detect "hand" without
    MediaPipe. Text-prompt embeddings are computed once (needs MobileCLIP) and
    cached to disk so later runs start fast.
  - "yolo": classic COCO segmentation (80 fixed classes), lighter.

Both produce masks, which is what enables blade orientation/tip recovery.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
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
    """Backend-agnostic detector. Construct via `Detector.build(config)`."""

    def __init__(self, model, min_confidence: float):
        self.model = model
        self.min_confidence = min_confidence
        self.names = model.names
        self._use_track = True  # flipped off if the backend can't track

    # ----- construction ------------------------------------------------------
    @classmethod
    def build(cls, config) -> "Detector":
        backend = getattr(config, "detector_backend", "yolo")
        if backend == "yoloe":
            try:
                return cls._build_yoloe(config)
            except Exception as e:  # fall back to classic seg if open-vocab fails
                print(f"[detector] YOLOE unavailable ({e}); falling back to {config.yolo_model}")
        return cls(YOLO(config.yolo_model), config.thresholds.min_confidence)

    @classmethod
    def _build_yoloe(cls, config) -> "Detector":
        from ultralytics import YOLOE  # imported lazily; pulls CLIP on first use

        model = YOLOE(config.yoloe_model)
        prompts = list(config.open_vocab_prompts)
        pe = cls._text_embeddings(model, prompts, config.textpe_cache)
        model.set_classes(prompts, pe)
        det = cls(model, config.thresholds.min_confidence)
        print(f"[detector] YOLOE open-vocab with {len(prompts)} prompts")
        return det

    @staticmethod
    def _text_embeddings(model, prompts: list[str], cache_path: str):
        """Compute text-prompt embeddings once, then cache. Loading the cache
        avoids reloading MobileCLIP (the 572MB encoder) on every startup."""
        key = "|".join(prompts)
        if os.path.exists(cache_path):
            blob = torch.load(cache_path, map_location="cpu")
            if blob.get("key") == key:
                return blob["pe"]
        pe = model.get_text_pe(prompts)
        try:
            torch.save({"key": key, "pe": pe}, cache_path)
        except Exception:
            pass
        return pe

    # ----- inference ---------------------------------------------------------
    def _infer(self, frame: np.ndarray):
        # Prefer tracking (stable IDs -> speed-based rules). Some open-vocab
        # configs don't support .track; fall back to .predict once and remember.
        if self._use_track:
            try:
                return self.model.track(
                    frame, persist=True, verbose=False, conf=self.min_confidence
                )
            except Exception:
                self._use_track = False
                print("[detector] tracking unavailable; using predict (no track IDs)")
        return self.model.predict(frame, verbose=False, conf=self.min_confidence)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._infer(frame)
        out: list[Detection] = []
        if not results:
            return out
        res = results[0]
        boxes = res.boxes
        if boxes is None:
            return out
        names = res.names if res.names else self.names  # reflects open-vocab set_classes
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
                    label=names[cls_id],
                    confidence=float(boxes.conf[i].item()),
                    box=(x1, y1, x2, y2),
                    mask=mask_xy,
                )
            )
        return out
