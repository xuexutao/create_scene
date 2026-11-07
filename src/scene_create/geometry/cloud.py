from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _arrays(points, colors, weights):
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or colors.shape != points.shape
        or weights.shape != (len(points),)
    ):
        raise ValueError("Expected points/colors Nx3 and weights N")
    return points, colors, weights


def clean_cloud(points, colors, weights, neighbors=16, std_ratio=2.5):
    points, colors, weights = _arrays(points, colors, weights)
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(colors).all(axis=1)
        & np.isfinite(weights)
        & (weights > 0)
    )
    indices = np.flatnonzero(valid)
    if len(indices) > max(4, neighbors):
        distances = (
            cKDTree(points[indices])
            .query(points[indices], k=min(neighbors + 1, len(indices)))[0][:, 1:]
            .mean(axis=1)
        )
        threshold = distances.mean() + float(std_ratio) * distances.std()
        indices = indices[distances <= max(threshold, 1e-12)]
    return {
        "points": points[indices],
        "colors": np.clip(colors[indices], 0, 255).astype(np.uint8),
        "weights": weights[indices],
        "indices": indices,
    }


def voxel_fuse(points, colors, weights, voxel_size):
    points, colors, weights = _arrays(points, colors, weights)
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if (
        not np.isfinite(points).all()
        or not np.isfinite(colors).all()
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("Voxel fusion requires finite data and positive weights")
    if not len(points):
        return {
            "points": points,
            "colors": colors.astype(np.uint8),
            "weights": weights,
            "counts": np.zeros(0, dtype=np.int64),
        }
    cells = np.floor((points - points.min(axis=0)) / voxel_size).astype(np.int64)
    _, assignments, counts = np.unique(
        cells, axis=0, return_inverse=True, return_counts=True
    )
    totals = np.bincount(assignments, weights=weights)
    fused_points = np.column_stack(
        [
            np.bincount(assignments, weights=points[:, axis] * weights) / totals
            for axis in range(3)
        ]
    )
    fused_colors = np.column_stack(
        [
            np.bincount(assignments, weights=colors[:, axis].astype(float) * weights)
            / totals
            for axis in range(3)
        ]
    )
    return {
        "points": fused_points,
        "colors": np.clip(np.rint(fused_colors), 0, 255).astype(np.uint8),
        "weights": totals,
        "counts": counts,
    }


def cloud_overlap(a, b, distance):
    a, b = np.asarray(a), np.asarray(b)
    if min(len(a), len(b)) == 0:
        return 0.0
    if distance <= 0:
        raise ValueError("Overlap distance must be positive")
    forward = np.mean(cKDTree(b).query(a, k=1)[0] <= distance)
    backward = np.mean(cKDTree(a).query(b, k=1)[0] <= distance)
    return float(min(forward, backward))
