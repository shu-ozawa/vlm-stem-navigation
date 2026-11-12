"""Where the targets of a specimen are: points or line segments, in specimen pixels (row, col).

Both kinds answer the same three questions about a square box around a FOV centre, which is all the
task rules (tasks/env.py) and the oracle need:
  nearest(center)              offset (d_row, d_col) from the centre to the nearest target point
  visible_box(center, half)    does a target lie in the box of half-width `half`?
  success_box(center, half)    the same, for the success square
A point lies in the visibility box if -half <= d < half on both axes and in the success box if
|d| < half; a line lies in either box if the segment intersects the closed box.
"""

from __future__ import annotations

import math

import numpy as np


class PointTargets:
    def __init__(self, rcs):
        self.rcs = np.asarray(rcs, dtype=float).reshape(-1, 2)

    def _offs(self, center) -> np.ndarray:
        return self.rcs - np.array(center, dtype=float)

    def nearest(self, center) -> tuple[float, float]:
        d = self._offs(center)
        k = int(np.argmin(np.hypot(d[:, 0], d[:, 1])))
        return float(d[k, 0]), float(d[k, 1])

    def visible_box(self, center, half: float) -> bool:
        d = self._offs(center)
        return bool(np.any((-half <= d[:, 0]) & (d[:, 0] < half) & (-half <= d[:, 1]) & (d[:, 1] < half)))

    def success_box(self, center, half: float) -> bool:
        d = self._offs(center)
        return bool(np.any((np.abs(d[:, 0]) < half) & (np.abs(d[:, 1]) < half)))


class LineTargets:
    def __init__(self, segments):
        self.segments = [(tuple(p0), tuple(p1)) for p0, p1 in segments]

    def nearest(self, center) -> tuple[float, float]:
        best = None
        for a, b in self.segments:
            d = seg_nearest(a, b, center)
            if best is None or math.hypot(*d) < math.hypot(*best):
                best = d
        return best

    def visible_box(self, center, half: float) -> bool:
        cy, cx = center
        return any(seg_hits_box(a, b, cy - half, cy + half, cx - half, cx + half) for a, b in self.segments)

    success_box = visible_box


# ---------------------------------------------------------------- segment helpers
def seg_hits_box(p0, p1, r0, r1, c0, c1) -> bool:
    """Does segment p0-p1 ((row, col)) intersect the closed box [r0, r1] x [c0, c1]? (Liang-Barsky)"""
    (y0, x0), (y1, x1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - c0), (dx, c1 - x0), (-dy, y0 - r0), (dy, r1 - y0)):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                if t > t1:
                    return False
                t0 = max(t0, t)
            else:
                if t < t0:
                    return False
                t1 = min(t1, t)
    return t0 <= t1


def seg_nearest(p0, p1, q) -> tuple[float, float]:
    """Offset (drow, dcol) from point q to the nearest point of segment p0-p1."""
    a, b, q = np.asarray(p0, float), np.asarray(p1, float), np.asarray(q, float)
    d = b - a
    L2 = float(d @ d)
    t = 0.0 if L2 == 0 else float(np.clip((q - a) @ d / L2, 0, 1))
    n = a + t * d
    return float(n[0] - q[0]), float(n[1] - q[1])


def clip_segment(a, b, H, W):
    """Clip segment a-b to the image rectangle [0, H-1] x [0, W-1] (Liang-Barsky); None if outside."""
    (y0, x0), (y1, x1) = a, b
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0), (dx, W - 1 - x0), (-dy, y0), (dy, H - 1 - y0)):
        if p == 0:
            if q < 0:
                return None
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    if t0 > t1:
        return None
    return ((y0 + t0 * dy, x0 + t0 * dx), (y0 + t1 * dy, x0 + t1 * dx))
