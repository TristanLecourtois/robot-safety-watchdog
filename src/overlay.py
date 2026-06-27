"""Debug/demo overlay — draw the precise geometry on the frame so the
hackathon audience can *see* why the watchdog fired: blade axis, tip, the
line to the nearest fingertip, and the current verdict banner.

For multi-camera mode `draw_grid` tiles N annotated camera frames into a
single display window and adds a cross-camera summary strip at the bottom.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from src.detector import Detection
from src.orientation import BladeGeometry
from src.pose import Hand
from src.rules import FrameAnalysis

GREEN = (0, 200, 0)
AMBER = (0, 190, 255)
RED = (0, 0, 255)
BLUE = (255, 100, 0)
WHITE = (255, 255, 255)
DARK = (30, 30, 30)


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


# ---------------------------------------------------------------------------
# Multi-camera grid display
# ---------------------------------------------------------------------------

def _resize_cell(frame: np.ndarray | None, w: int, h: int) -> np.ndarray:
    if frame is None:
        cell = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(cell, "NO SIGNAL", (w // 2 - 60, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
        return cell
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def draw_grid(
    frames: list[np.ndarray | None],
    cam_labels: list[str],
    detections_list: list[list[Detection]],
    hands_list: list[list[Hand]],
    analyses_list: list[FrameAnalysis],
    rationales_list: list[str | None],
    severities_list: list[str | None],
    cross_objects: list,          # list[CrossCameraObject] from multi_camera.py
    multi_verdict_text: str | None,
    multi_severity: str | None,
    cell_w: int = 640,
    cell_h: int = 360,
) -> np.ndarray:
    """Render a grid of annotated camera frames plus a multi-camera summary strip."""
    n = len(frames)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = max(1, math.ceil(n / cols))

    cells: list[np.ndarray] = []
    for frame, label, dets, hands, analysis, rationale, sev in zip(
        frames, cam_labels, detections_list, hands_list,
        analyses_list, rationales_list, severities_list
    ):
        cell = _resize_cell(frame, cell_w, cell_h)
        # Scale overlay coordinates to cell size
        if frame is not None:
            scale_x = cell_w / frame.shape[1]
            scale_y = cell_h / frame.shape[0]
            cell = _draw_scaled(cell, dets, hands, analysis, rationale, sev,
                                scale_x, scale_y)
        # Camera name badge (bottom-left of cell, above verdict bar)
        cv2.rectangle(cell, (0, cell_h - 52), (len(label) * 11 + 14, cell_h - 36),
                      DARK, -1)
        cv2.putText(cell, label, (6, cell_h - 39),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cells.append(cell)

    # Pad grid to full rows×cols
    blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    while len(cells) < rows * cols:
        cells.append(blank.copy())

    row_imgs = [np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)]
    grid = np.vstack(row_imgs)

    # --- Multi-camera summary strip ------------------------------------------
    strip_h = 54
    strip = np.zeros((strip_h, grid.shape[1], 3), dtype=np.uint8)
    banner_color = {"critical": RED, "warning": AMBER}.get(multi_severity or "", GREEN)
    strip[:4, :] = banner_color   # coloured top border

    # Left: overall verdict
    overall_label = (multi_severity or "clear").upper()
    cv2.putText(strip, f"MULTI-CAM: {overall_label}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, banner_color, 2)

    # Right: cross-camera objects and triangulated positions
    x_right = grid.shape[1] // 2
    y = 16
    for cco in cross_objects[:4]:
        cams = ", ".join(cco.cameras)
        if cco.position_3d:
            px, py, pz = cco.position_3d
            txt = f"{cco.label} @({px:.2f},{py:.2f},{pz:.2f})m  [{cams}]"
            color = AMBER
        else:
            txt = f"{cco.label}  [{cams}]"
            color = (180, 220, 255)
        cv2.putText(strip, txt, (x_right, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        y += 18

    # Bottom text: multi-camera VLM rationale
    if multi_verdict_text:
        cv2.putText(strip, multi_verdict_text[:110], (10, strip_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1)

    return np.vstack([grid, strip])


def _draw_scaled(
    cell: np.ndarray,
    dets: list[Detection],
    hands: list[Hand],
    analysis: FrameAnalysis,
    rationale: str | None,
    severity: str | None,
    sx: float,
    sy: float,
) -> np.ndarray:
    """Draw overlay on a resized cell frame, scaling pixel coordinates."""
    for d in dets:
        x1, y1, x2, y2 = int(d.box[0]*sx), int(d.box[1]*sy), int(d.box[2]*sx), int(d.box[3]*sy)
        cv2.rectangle(cell, (x1, y1), (x2, y2), (180, 180, 180), 1)
        cv2.putText(cell, f"{d.label} {d.confidence:.2f}", (x1, max(y1 - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    for hand in hands:
        for ft in hand.fingertips:
            cv2.circle(cell, (int(ft[0]*sx), int(ft[1]*sy)), 3, GREEN, -1)

    for geo in analysis.blades.values():
        tip = (int(geo.tip[0]*sx), int(geo.tip[1]*sy))
        handle = (int(geo.handle[0]*sx), int(geo.handle[1]*sy))
        cv2.line(cell, handle, tip, AMBER, 1)
        cv2.circle(cell, tip, 4, RED, -1)
        cv2.circle(cell, handle, 3, (255, 120, 0), -1)

    banner_color = {"critical": RED, "warning": AMBER}.get(severity or "", GREEN)
    lbl = (severity or "clear").upper()
    cv2.rectangle(cell, (0, 0), (cell.shape[1], 26), banner_color, -1)
    cv2.putText(cell, lbl, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    if rationale:
        cv2.putText(cell, rationale[:70], (6, cell.shape[0] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)
    return cell
