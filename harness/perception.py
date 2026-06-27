"""Adapter from the existing watchdog outputs to harness scene context."""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import config as cfg
from src.orientation import aim_angle_deg
from src.pose import nearest_fingertip
from src.rules import FrameAnalysis
from src.watchdog import Watchdog


class WatchdogPerceptionAdapter:
    """Convert `Watchdog.process_frame` output into the harness JSON shape."""

    def __init__(self, watchdog: Watchdog | None = None, camera_id: str = "external_webcam_1"):
        self.watchdog = watchdog or Watchdog()
        self.camera_id = camera_id

    def parse_frame(self, frame, now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        detections, hands, analysis = self.watchdog.process_frame(frame, now)
        return scene_context_from_watchdog(
            detections=detections,
            hands=hands,
            analysis=analysis,
            camera_id=self.camera_id,
            timestamp=now,
        )


def scene_context_from_watchdog(
    detections,
    hands,
    analysis: FrameAnalysis,
    camera_id: str,
    timestamp: float | None = None,
) -> dict[str, Any]:
    objects = sorted({normalize_object_label(d.label) for d in detections})
    hand_detected = any(d.label in cfg.HAND_CLASSES for d in detections)
    if (hands or hand_detected) and "human_hand" not in objects:
        objects.append("human_hand")

    hazards = [hit.category for hit in analysis.hits]
    geometry = _sharp_tool_geometry(analysis, hands)
    zones = {
        "human_in_workspace": bool(hands) or any(
            d.label in cfg.HUMAN_CLASSES or d.label in cfg.HAND_CLASSES for d in detections
        ),
        "tool_in_active_zone": any(d.label in cfg.SHARP_CLASSES for d in detections),
    }
    confidence = max([d.confidence for d in detections], default=0.0)

    return {
        "timestamp": timestamp if timestamp is not None else time.time(),
        "camera_id": camera_id,
        "objects": objects,
        "detections": [_serialize_detection(d) for d in detections],
        "hands": [_serialize_hand(h) for h in hands],
        "geometry": geometry,
        "relations": _relations_from_hazards(hazards),
        "zones": zones,
        "hazards": hazards,
        "rule_hits": [asdict(hit) for hit in analysis.hits],
        "max_rule_severity": analysis.max_severity,
        "confidence": confidence,
    }


def normalize_object_label(label: str) -> str:
    if label in cfg.SHARP_CLASSES:
        return "sharp_tool"
    if label in cfg.HUMAN_CLASSES:
        return "person"
    if label in cfg.HAND_CLASSES:
        return "human_hand"
    if label in cfg.FRAGILE_CLASSES:
        return "fragile_object"
    if label in cfg.HOT_ZONE_CLASSES:
        return "hot_zone"
    return label


def _sharp_tool_geometry(analysis: FrameAnalysis, hands) -> dict[str, Any]:
    tools = []
    for track_id, blade in analysis.blades.items():
        item = {
            "track_id": track_id,
            "blade_axis": blade.axis,
            "blade_axis_degrees": blade.angle_deg,
            "blade_tip_px": blade.tip,
            "blade_handle_px": blade.handle,
            "blade_length_px": blade.length_px,
            "blade_elongation": blade.elongation,
        }
        nf = nearest_fingertip(blade.tip, hands) if hands else None
        if nf is not None:
            dist, fingertip = nf
            aim = aim_angle_deg(blade.tip, blade.axis, fingertip)
            item.update(
                {
                    "nearest_fingertip_px": fingertip,
                    "tip_to_fingertip_distance_px": dist,
                    "tip_aim_angle_deg": aim,
                    "tip_aimed_at_hand": aim < cfg.CONFIG.thresholds.blade_aim_angle_deg,
                }
            )
        tools.append(item)
    return {"sharp_tools": tools}


def _relations_from_hazards(hazards: list[str]) -> list[dict[str, Any]]:
    relations = []
    if any(h in hazards for h in ("blade_tip_near_hand", "blade_tip_aimed_at_hand")):
        relations.append(
            {
                "subject": "human_hand",
                "relation": "near",
                "object": "sharp_tool",
                "distance_estimate": "close",
            }
        )
    if "sharp_near_person" in hazards:
        relations.append(
            {
                "subject": "person",
                "relation": "near",
                "object": "sharp_tool",
                "distance_estimate": "coarse_box_close",
            }
        )
    return relations


def _serialize_detection(d) -> dict[str, Any]:
    return {
        "track_id": d.track_id,
        "label": d.label,
        "object": normalize_object_label(d.label),
        "confidence": d.confidence,
        "box": d.box,
        "has_mask": d.mask is not None,
    }


def _serialize_hand(hand) -> dict[str, Any]:
    return {
        "handedness": hand.handedness,
        "fingertips": hand.fingertips.tolist(),
        "points": hand.points.tolist(),
    }
