"""Debug/demo overlay — draw the precise geometry on the frame so the
hackathon audience can *see* why the watchdog fired: blade axis, tip, the
line to the nearest fingertip, and the current verdict banner.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.detector import Detection
from src.orientation import BladeGeometry
from src.pose import Hand
from src.rules import FrameAnalysis

GREEN = (0, 200, 0)
AMBER = (0, 190, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def draw(frame, detections: list[Detection], hands: list[Hand],
         analysis: FrameAnalysis, verdict_text: str | None, severity: str | None,
         latent=None):
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
        cv2.putText(frame, f"{d.label} {d.confidence:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Fingertips
    for hand in hands:
        for ft in hand.fingertips:
            cv2.circle(frame, (int(ft[0]), int(ft[1])), 4, GREEN, -1)

    # Blade geometry: axis + tip + handle
    for geo in analysis.blades.values():
        tip = (int(geo.tip[0]), int(geo.tip[1]))
        handle = (int(geo.handle[0]), int(geo.handle[1]))
        cv2.line(frame, handle, tip, AMBER, 2)
        cv2.circle(frame, tip, 6, RED, -1)            # sharp end
        cv2.circle(frame, handle, 5, (255, 120, 0), -1)  # handle end
        cv2.putText(frame, f"{geo.angle_deg:.0f}deg", (tip[0] + 8, tip[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMBER, 1)

    banner_color = {"critical": RED, "warning": AMBER}.get(severity or "", GREEN)
    label = (severity or "clear").upper()
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), banner_color, -1)
    cv2.putText(frame, f"WATCHDOG: {label}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    if verdict_text:
        cv2.putText(frame, verdict_text[:90], (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

    if latent is not None:
        _draw_latent_panel(frame, latent)
    return frame


def _draw_latent_panel(frame, latent):
    """Top-right panel: V-JEPA latent OOD status, danger z-score, and a 2D PCA
    map (green = learned 'normal' cloud, moving dot = current situation)."""
    h, w = frame.shape[:2]
    pw, ph = 220, 200
    x0, y0 = w - pw - 10, 44
    panel = frame[y0:y0 + ph, x0:x0 + pw]
    panel[:] = (panel * 0.35).astype(panel.dtype)  # darken backdrop
    cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (90, 90, 90), 1)

    status = latent.status
    cv2.putText(frame, f"V-JEPA: {status}", (x0 + 8, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    if status == "calibrating":
        cv2.putText(frame, f"normal {latent.progress}", (x0 + 8, y0 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, AMBER, 1)
    elif status == "loading":
        cv2.putText(frame, "model ~80s...", (x0 + 8, y0 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, AMBER, 1)
    elif status == "monitoring":
        z = latent.danger_score
        col = RED if latent.is_anomaly else GREEN
        cv2.putText(frame, f"danger z={z:.1f}", (x0 + 8, y0 + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        # score bar
        frac = max(0.0, min(1.0, z / 6.0))
        cv2.rectangle(frame, (x0 + 8, y0 + 46), (x0 + 8 + int(frac * (pw - 16)), y0 + 54), col, -1)

    # 2D PCA map
    map_top = y0 + 62
    _scatter(frame, latent, x0 + 8, map_top, pw - 16, ph - (map_top - y0) - 8)


def _scatter(frame, latent, mx, my, mw, mh):
    import numpy as np

    pts = []
    if latent.normal_2d is not None and len(latent.normal_2d):
        pts.append(np.asarray(latent.normal_2d))
    if latent.recent_2d:
        pts.append(np.asarray(latent.recent_2d))
    if not pts:
        return
    allp = np.concatenate(pts, axis=0)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    rng = np.maximum(hi - lo, 1e-6)

    def to_px(p):
        u = (p - lo) / rng
        return int(mx + u[0] * mw), int(my + (1 - u[1]) * mh)

    if latent.normal_2d is not None:
        for p in latent.normal_2d:
            cv2.circle(frame, to_px(p), 2, GREEN, -1)
    for i, p in enumerate(latent.recent_2d):
        last = i == len(latent.recent_2d) - 1
        col = (RED if latent.is_anomaly else AMBER) if last else (150, 150, 150)
        cv2.circle(frame, to_px(np.asarray(p)), 4 if last else 2, col, -1)
