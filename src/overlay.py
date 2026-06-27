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
         analysis: FrameAnalysis, verdict_text: str | None, severity: str | None):
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
    return frame
