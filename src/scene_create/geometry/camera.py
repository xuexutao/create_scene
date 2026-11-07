from __future__ import annotations

import numpy as np


def validate_transform(matrix, allow_scale=False):
    matrix = np.asarray(matrix, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
    ):
        raise ValueError("Expected a finite homogeneous 4x4 transform")
    linear = matrix[:3, :3]
    determinant = np.linalg.det(linear)
    if determinant <= 0:
        raise ValueError("Transform must preserve handedness and have a positive scale")
    scale = np.cbrt(determinant) if allow_scale else 1.0
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(
            "Transform contains shear, anisotropic scale or an invalid rotation"
        )
    return matrix


def depth_to_points(depth, intrinsics, world_from_camera):
    depth = np.asarray(depth, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2 or not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError(
            "Depth must be a finite nonnegative HxW optical-axis depth map"
        )
    if (
        intrinsics.shape != (3, 3)
        or not np.isfinite(intrinsics).all()
        or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0
    ):
        raise ValueError("Invalid camera intrinsics")
    transform = validate_transform(world_from_camera)
    y, x = np.indices(depth.shape)
    pixels = np.stack([x, y, np.ones_like(x)], axis=-1)
    rays = pixels @ np.linalg.inv(intrinsics).T
    points = rays * depth[..., None]
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_points(points, intrinsics, world_from_camera):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected Nx3 points")
    world_from_camera = validate_transform(world_from_camera)
    camera = (points - world_from_camera[:3, 3]) @ world_from_camera[:3, :3]
    projection = camera @ np.asarray(intrinsics, dtype=np.float64).T
    uv = np.full((len(points), 2), np.nan)
    valid = np.isfinite(projection).all(axis=1) & (camera[:, 2] > 1e-9)
    uv[valid] = projection[valid, :2] / projection[valid, 2:3]
    return uv, camera[:, 2]
