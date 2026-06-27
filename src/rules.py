"""Layer 1.5 — deterministic safety rules over fine geometry.

These run every frame, are fast, and protect in real time (a kill-switch
shouldn't wait on an LLM). They consume the precise geometry: blade tip,
blade orientation, and fingertip positions. The VLM (layer 2) then adds
judgment for the fuzzy cases.

Each fired rule is a `RuleHit` with a severity and a human-readable reason
that doubles as context for the VLM prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import config as cfg
from src.detector import Detection, box_distance
from src.orientation import BladeGeometry, aim_angle_deg, blade_geometry
from src.pose import Hand


@dataclass
class RuleHit:
    category: str          # e.g. "blade_at_hand"
    severity: str          # "critical" | "warning"
    reason: str
    detail: dict = field(default_factory=dict)


@dataclass
class FrameAnalysis:
    hits: list[RuleHit] = field(default_factory=list)
    # Per-blade geometry we computed, keyed by track_id, for overlay drawing.
    blades: dict[int, BladeGeometry] = field(default_factory=dict)

    @property
    def max_severity(self) -> str | None:
        if any(h.severity == "critical" for h in self.hits):
            return "critical"
        if self.hits:
            return "warning"
        return None


class RuleEngine:
    def __init__(self, thresholds: cfg.Thresholds):
        self.t = thresholds
        self._prev_centers: dict[int, tuple[float, float]] = {}

    def analyze(self, detections: list[Detection], hands: list[Hand]) -> FrameAnalysis:
        out = FrameAnalysis()
        humans = [d for d in detections if d.label in cfg.HUMAN_CLASSES]
        hand_dets = [d for d in detections if d.label in cfg.HAND_CLASSES]
        sharps = [d for d in detections if d.label in cfg.SHARP_CLASSES]
        fragiles = [d for d in detections if d.label in cfg.FRAGILE_CLASSES]
        hot_zones = [d for d in detections if d.label in cfg.HOT_ZONE_CLASSES]

        # Precise target points for the blade rule, best source first:
        #   1. MediaPipe fingertips (when available)
        #   2. open-vocab "hand" detection centers (works on Intel Mac, no MP)
        target_points: list[tuple[float, float]] = []
        for h in hands:
            target_points.extend((float(ft[0]), float(ft[1])) for ft in h.fingertips)
        target_points.extend(d.center for d in hand_dets)

        # Boxes for the coarse fallback when we have no fine target points.
        proximity_boxes = humans + hand_dets

        self._blade_rules(sharps, target_points, proximity_boxes, out)
        self._fragile_speed_rules(fragiles, out)
        self._hot_zone_rules(detections, hot_zones, out)
        self._update_speed_cache(detections)
        return out

    # ----- the precise one: blade tip + orientation vs hand points -----------
    def _blade_rules(self, sharps, target_points, proximity_boxes, out: FrameAnalysis):
        for blade in sharps:
            # Hint target disambiguates which mask end is the tip: the end nearest
            # a hand is assumed to be the sharp/dangerous end.
            hint = self._nearest_point(blade.center, target_points)
            if hint is None and proximity_boxes:
                hint = min((b.center for b in proximity_boxes),
                           key=lambda p: (p[0] - blade.center[0]) ** 2 + (p[1] - blade.center[1]) ** 2)
            geo = blade_geometry(blade.mask, hint_toward=hint) if blade.mask is not None else None
            if geo is not None:
                out.blades[blade.track_id] = geo

            # Precise path: blade tip -> nearest hand point, plus aim direction.
            if geo is not None and target_points:
                nearest = self._nearest_point(geo.tip, target_points)
                dist = ((geo.tip[0] - nearest[0]) ** 2 + (geo.tip[1] - nearest[1]) ** 2) ** 0.5
                aim = aim_angle_deg(geo.tip, geo.axis, nearest)
                close = dist < self.t.blade_tip_to_hand_px
                aimed = aim < self.t.blade_aim_angle_deg
                if close and aimed:
                    out.hits.append(RuleHit(
                        "blade_tip_aimed_at_hand", "critical",
                        f"Blade tip {dist:.0f}px from a hand and pointing at it "
                        f"({aim:.0f}° off-axis, blade @ {geo.angle_deg:.0f}°).",
                        {"dist_px": dist, "aim_deg": aim, "blade_angle_deg": geo.angle_deg},
                    ))
                elif close:
                    out.hits.append(RuleHit(
                        "blade_tip_near_hand", "warning",
                        f"Blade tip {dist:.0f}px from a hand (not aimed, {aim:.0f}° off).",
                        {"dist_px": dist, "aim_deg": aim},
                    ))
                continue  # handled with the precise path

            # Fallback path: no mask or no hand points -> coarse box proximity.
            for box in proximity_boxes:
                d = box_distance(blade, box)
                if d < self.t.sharp_to_human_px:
                    out.hits.append(RuleHit(
                        "sharp_near_person", "warning",
                        f"{blade.label} {d:.0f}px from {box.label} (box-level).",
                        {"dist_px": d},
                    ))

    def _fragile_speed_rules(self, fragiles, out: FrameAnalysis):
        for d in fragiles:
            speed = self._speed(d)
            if speed is not None and speed > self.t.fragile_speed_px:
                out.hits.append(RuleHit(
                    "fragile_fast_motion", "warning",
                    f"{d.label} moving fast ({speed:.0f}px/frame) — drop/shatter risk.",
                    {"speed_px": speed},
                ))

    def _hot_zone_rules(self, detections, hot_zones, out: FrameAnalysis):
        movers = [d for d in detections if d.label in cfg.HANDHELD_CLASSES]
        for hz in hot_zones:
            for m in movers:
                d = box_distance(m, hz)
                if d < self.t.object_to_hot_px:
                    out.hits.append(RuleHit(
                        "object_near_hot_zone", "warning",
                        f"{m.label} {d:.0f}px from {hz.label} (hot zone).",
                        {"dist_px": d},
                    ))

    # ----- helpers -----------------------------------------------------------
    @staticmethod
    def _nearest_point(p, points):
        if not points:
            return None
        return min(points, key=lambda q: (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2)

    def _speed(self, d: Detection):
        if d.track_id < 0:
            return None
        prev = self._prev_centers.get(d.track_id)
        if prev is None:
            return None
        cx, cy = d.center
        return ((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5

    def _update_speed_cache(self, detections):
        self._prev_centers = {d.track_id: d.center for d in detections if d.track_id >= 0}
