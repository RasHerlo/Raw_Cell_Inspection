"""Polygon ROI geometry and trace extraction.

Coordinates follow the image display: x = column, y = row, and pixel (r, c)
covers [c, c+1] x [r, r+1]. A pixel belongs to an ROI when its centre is inside
the polygon.
"""

from __future__ import annotations

import numpy as np

MAX_VERTICES = 60


def points_in_polygon(px: np.ndarray, py: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Even-odd rule point-in-polygon test, vectorised over the points."""
    vx = vertices[:, 0]
    vy = vertices[:, 1]
    inside = np.zeros(px.shape, dtype=bool)
    n = len(vertices)
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = vx[i], vy[i], vx[j], vy[j]
        crosses = (yi > py) != (yj > py)
        if np.any(crosses):
            x_cross = (xj - xi) * (py - yi) / ((yj - yi) if yj != yi else 1e-12) + xi
            inside ^= crosses & (px < x_cross)
        j = i
    return inside


def polygon_area(vertices: np.ndarray) -> float:
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def contains_point(vertices: np.ndarray, x: float, y: float) -> bool:
    return bool(points_in_polygon(np.array([x]), np.array([y]), vertices)[0])


def clip_vertices(vertices: np.ndarray, height: int, width: int) -> np.ndarray:
    out = np.asarray(vertices, dtype=float).copy()
    out[:, 0] = np.clip(out[:, 0], 0, width)
    out[:, 1] = np.clip(out[:, 1], 0, height)
    return out


def simplify_stroke(points: np.ndarray, tolerance: float = 0.7) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification, then cap the vertex count for editing."""
    pts = np.asarray(points, dtype=float)
    if len(pts) > 3:
        keep = np.zeros(len(pts), dtype=bool)
        keep[0] = keep[-1] = True
        stack = [(0, len(pts) - 1)]
        while stack:
            a, b = stack.pop()
            if b <= a + 1:
                continue
            seg = pts[b] - pts[a]
            seg_len = np.hypot(*seg)
            rel = pts[a + 1 : b] - pts[a]
            if seg_len < 1e-9:
                dist = np.hypot(rel[:, 0], rel[:, 1])
            else:
                dist = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / seg_len
            k = int(np.argmax(dist))
            if dist[k] > tolerance:
                idx = a + 1 + k
                keep[idx] = True
                stack.append((a, idx))
                stack.append((idx, b))
        pts = pts[keep]
    if len(pts) > MAX_VERTICES:
        idx = np.linspace(0, len(pts) - 1, MAX_VERTICES).round().astype(int)
        pts = pts[np.unique(idx)]
    return pts


def polygon_mask(vertices: np.ndarray, height: int, width: int) -> dict | None:
    """Return {'bbox': (r0, r1, c0, c1), 'mask': bool[r1-r0, c1-c0]} or None if empty."""
    v = np.asarray(vertices, dtype=float)
    c0 = max(0, int(np.floor(v[:, 0].min())))
    c1 = min(width, int(np.ceil(v[:, 0].max())) + 1)
    r0 = max(0, int(np.floor(v[:, 1].min())))
    r1 = min(height, int(np.ceil(v[:, 1].max())) + 1)
    if c1 <= c0 or r1 <= r0:
        return None
    cols, rows = np.meshgrid(np.arange(c0, c1) + 0.5, np.arange(r0, r1) + 0.5)
    mask = points_in_polygon(cols.ravel(), rows.ravel(), v).reshape(rows.shape)
    if not mask.any():
        return None
    rr, cc = np.nonzero(mask)
    mask = mask[rr.min() : rr.max() + 1, cc.min() : cc.max() + 1]
    return {
        "bbox": (r0 + int(rr.min()), r0 + int(rr.max()) + 1, c0 + int(cc.min()), c0 + int(cc.max()) + 1),
        "mask": mask,
    }


def roi_trace(stack: np.ndarray, mask_info: dict) -> np.ndarray:
    """Mean intensity inside the mask for every frame (float32)."""
    r0, r1, c0, c1 = mask_info["bbox"]
    mask = mask_info["mask"]
    crop = np.asarray(stack[:, r0:r1, c0:c1])
    return crop[:, mask].mean(axis=1, dtype=np.float64).astype(np.float32)
