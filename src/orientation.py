"""Fine geometry — recover the *orientation* and *tip* of an elongated object
(a knife) from its segmentation mask.

A bounding box can't tell you which way a knife points. The mask can: a blade
is a long, thin blob, so its principal axis (first PCA component) is the blade
direction, and the two extreme points along that axis are the tip and the
handle end. We then disambiguate tip-vs-handle and can ask the precise
questions safety cares about:

  - exactly where is the tip (px)?
  - which way does the blade point (unit vector / angle)?
  - is it pointing *at* a given target (hand, person)?
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class BladeGeometry:
    centroid: tuple[float, float]
    axis: tuple[float, float]        # unit vector along the blade (tip direction)
    angle_deg: float                 # blade orientation, 0 = +x, CCW positive
    tip: tuple[float, float]         # estimated sharp end (px)
    handle: tuple[float, float]      # estimated blunt/handle end (px)
    length_px: float
    elongation: float                # major/minor axis ratio; high => confidently a blade


def _pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centroid, major_axis_unit, eigvals_desc)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    major = eigvecs[:, 0]
    return centroid, major, eigvals


def blade_geometry(
    mask_xy: np.ndarray,
    hint_toward: tuple[float, float] | None = None,
) -> BladeGeometry | None:
    """Compute blade orientation/tip from a polygon mask (Nx2 in image coords).

    `hint_toward`: if given (e.g. the nearest hand), the extreme end closest to
    that point is assumed to be the TIP. This matches the safety intuition
    "the dangerous end is the one near the human". Without a hint we fall back
    to a thinness heuristic (the end where the mask is narrower is the tip).
    """
    if mask_xy is None or len(mask_xy) < 3:
        return None
    pts = mask_xy.astype(np.float64)
    centroid, major, eigvals = _pca_axes(pts)
    if eigvals[1] <= 1e-6:
        return None
    elongation = float(math.sqrt(eigvals[0] / max(eigvals[1], 1e-6)))

    # Project all mask points onto the major axis; the two extremes are the ends.
    t = (pts - centroid) @ major
    end_a = centroid + major * t.min()
    end_b = centroid + major * t.max()
    length_px = float(t.max() - t.min())

    if hint_toward is not None:
        h = np.asarray(hint_toward, dtype=np.float64)
        tip, handle = (end_a, end_b) if np.linalg.norm(end_a - h) <= np.linalg.norm(end_b - h) else (end_b, end_a)
    else:
        # Tip = the narrower end. Measure mask width in a slab near each end.
        tip, handle = _tip_by_thinness(pts, centroid, major, end_a, end_b)

    axis = (tip - handle)
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return None
    axis = axis / n
    angle = math.degrees(math.atan2(axis[1], axis[0]))
    return BladeGeometry(
        centroid=(float(centroid[0]), float(centroid[1])),
        axis=(float(axis[0]), float(axis[1])),
        angle_deg=angle,
        tip=(float(tip[0]), float(tip[1])),
        handle=(float(handle[0]), float(handle[1])),
        length_px=length_px,
        elongation=elongation,
    )


def _tip_by_thinness(pts, centroid, major, end_a, end_b):
    """The blade tip is the thinner end. Compare mask spread perpendicular to
    the axis within a slab near each extreme."""
    minor = np.array([-major[1], major[0]])
    t = (pts - centroid) @ major
    span = t.max() - t.min()
    slab = 0.2 * span  # near-end window
    def width_near(t_end):
        sel = np.abs(t - t_end) < slab
        if sel.sum() < 2:
            return np.inf
        perp = (pts[sel] - centroid) @ minor
        return perp.max() - perp.min()
    wa = width_near(t.min())
    wb = width_near(t.max())
    return (end_a, end_b) if wa <= wb else (end_b, end_a)


def aim_angle_deg(tip: tuple[float, float], axis: tuple[float, float], target: tuple[float, float]) -> float:
    """Angle (deg) between the blade direction and the tip->target vector.
    0 = pointing straight at the target."""
    tx, ty = target[0] - tip[0], target[1] - tip[1]
    n = math.hypot(tx, ty)
    if n < 1e-6:
        return 0.0
    dot = (axis[0] * tx + axis[1] * ty) / n
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))
