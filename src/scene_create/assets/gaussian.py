from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from scene_create.geometry.camera import validate_transform

SH_C0 = 0.28209479177387814


def _dtype(degree):
    names = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
    names += [f"f_rest_{i}" for i in range(3 * ((degree + 1) ** 2 - 1))]
    names += (
        ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    )
    return np.dtype([(name, "f4") for name in names])


def _read(path):
    vertices = PlyData.read(str(path))["vertex"].data
    required = set(_dtype(0).names) - {"nx", "ny", "nz"}
    if not required.issubset(vertices.dtype.names):
        raise ValueError(
            f"Not a standard 3DGS PLY (XYZRGB is not Gaussian data): {path}"
        )
    rest = sorted(
        (name for name in vertices.dtype.names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    degree = int(round(np.sqrt(len(rest) / 3 + 1) - 1))
    if (
        degree not in range(4)
        or len(rest) != 3 * ((degree + 1) ** 2 - 1)
        or rest != [f"f_rest_{i}" for i in range(len(rest))]
    ):
        raise ValueError("Only complete standard SH degrees 0 through 3 are supported")
    result = np.zeros(len(vertices), dtype=_dtype(degree))
    for name in result.dtype.names:
        if name in vertices.dtype.names:
            result[name] = vertices[name]
        if not np.isfinite(result[name]).all():
            raise ValueError(f"Non-finite Gaussian field: {name}")
    rotations = np.column_stack([result[f"rot_{i}"] for i in range(4)])
    if len(rotations) and (np.linalg.norm(rotations, axis=1) < 1e-8).any():
        raise ValueError("Gaussian rotation quaternion cannot be zero")
    scales = np.column_stack([result[f"scale_{i}"] for i in range(3)])
    if len(scales) and ((scales < -80) | (scales > 80)).any():
        raise ValueError("Gaussian log scales are numerically invalid")
    return result, degree


def _write(path, vertices):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def inspect_gaussians(path):
    vertices, degree = _read(path)
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")])
    return {
        "count": len(vertices),
        "sh_degree": degree,
        "bounds": [points.min(axis=0).tolist(), points.max(axis=0).tolist()]
        if len(points)
        else None,
        "scale_encoding": "log",
        "opacity_encoding": "logit",
        "quaternion_order": "wxyz",
        "sh_layout": "channel_major",
    }


def initialize_gaussians(points, colors, output_path, opacity=0.1, sh_degree=0):
    points, colors = np.asarray(points, dtype=float), np.asarray(colors, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or colors.shape != points.shape
        or not len(points)
        or not np.isfinite(points).all()
        or not np.isfinite(colors).all()
    ):
        raise ValueError(
            "Gaussian initialization requires nonempty finite Nx3 points and colors"
        )
    if not 0 < opacity < 1 or sh_degree not in range(4):
        raise ValueError("Invalid opacity or SH degree")
    vertices = np.zeros(len(points), dtype=_dtype(sh_degree))
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, axis]
        vertices[f"f_dc_{axis}"] = (
            np.clip(colors[:, axis], 0, 255) / 255 - 0.5
        ) / SH_C0
    fallback = max(float(np.linalg.norm(np.ptp(points, axis=0))) * 0.001, 1e-6)
    spacing = (
        cKDTree(points).query(points, k=2)[0][:, 1]
        if len(points) > 1
        else np.array([fallback])
    )
    spacing = np.where(np.isfinite(spacing) & (spacing > 0), spacing, fallback)
    for axis in range(3):
        vertices[f"scale_{axis}"] = np.log(np.maximum(spacing * 0.5, 1e-8))
    vertices["rot_0"] = 1
    vertices["opacity"] = np.log(opacity / (1 - opacity))
    _write(output_path, vertices)
    return inspect_gaussians(output_path)


def sh_basis(directions, degree):
    directions = np.asarray(directions, dtype=float)
    x, y, z = directions.T
    values = [np.full_like(x, SH_C0)]
    if degree >= 1:
        c = 0.4886025119029199
        values += [-c * y, c * z, -c * x]
    if degree >= 2:
        values += [
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * z,
            0.31539156525252005 * (2 * z * z - x * x - y * y),
            -1.0925484305920792 * x * z,
            0.5462742152960396 * (x * x - y * y),
        ]
    if degree >= 3:
        values += [
            -0.5900435899266435 * y * (3 * x * x - y * y),
            2.890611442640554 * x * y * z,
            -0.4570457994644658 * y * (4 * z * z - x * x - y * y),
            0.3731763325901154 * z * (2 * z * z - 3 * x * x - 3 * y * y),
            -0.4570457994644658 * x * (4 * z * z - x * x - y * y),
            1.445305721320277 * z * (x * x - y * y),
            -0.5900435899266435 * x * (x * x - 3 * y * y),
        ]
    return np.stack(values, axis=1)


def rotate_sh(coefficients, rotation):
    coefficients = np.asarray(coefficients, dtype=float)
    degree = int(round(np.sqrt(coefficients.shape[-1]) - 1))
    if coefficients.shape[-1] != (degree + 1) ** 2 or degree not in range(4):
        raise ValueError("Unsupported SH coefficient count")
    if degree == 0:
        return coefficients.copy()
    index = np.arange(128, dtype=float) + 0.5
    z = 1 - 2 * index / len(index)
    angle = index * (np.pi * (3 - np.sqrt(5)))
    radius = np.sqrt(1 - z * z)
    directions = np.column_stack([radius * np.cos(angle), radius * np.sin(angle), z])
    basis = sh_basis(directions, degree)
    rotated_basis = sh_basis(directions @ rotation, degree)
    mapping = np.linalg.lstsq(basis, rotated_basis, rcond=None)[0]
    return np.einsum("lk,nck->ncl", mapping, coefficients)


def transform_gaussians(input_path, output_path, transform):
    transform = validate_transform(transform, allow_scale=True)
    vertices, degree = _read(input_path)
    scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
    rotation = transform[:3, :3] / scale
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")])
    transformed = points @ transform[:3, :3].T + transform[:3, 3]
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = transformed[:, axis]
        vertices[f"scale_{axis}"] += np.log(scale)
    if len(vertices):
        quaternions = np.column_stack([vertices[f"rot_{i}"] for i in (1, 2, 3, 0)])
        rotated = (
            Rotation.from_matrix(rotation) * Rotation.from_quat(quaternions)
        ).as_quat()[:, [3, 0, 1, 2]]
        for axis in range(4):
            vertices[f"rot_{axis}"] = rotated[:, axis]
        normals = (
            np.column_stack([vertices[name] for name in ("nx", "ny", "nz")])
            @ rotation.T
        )
        for axis, name in enumerate(("nx", "ny", "nz")):
            vertices[name] = normals[:, axis]
        if degree:
            count = (degree + 1) ** 2 - 1
            coefficients = np.empty((len(vertices), 3, count + 1))
            coefficients[:, :, 0] = np.column_stack(
                [vertices[f"f_dc_{i}"] for i in range(3)]
            )
            coefficients[:, :, 1:] = np.column_stack(
                [vertices[f"f_rest_{i}"] for i in range(3 * count)]
            ).reshape(-1, 3, count)
            coefficients = rotate_sh(coefficients, rotation)
            for channel in range(3):
                vertices[f"f_dc_{channel}"] = coefficients[:, channel, 0]
            flattened = coefficients[:, :, 1:].reshape(-1, 3 * count)
            for index in range(3 * count):
                vertices[f"f_rest_{index}"] = flattened[:, index]
    _write(output_path, vertices)
    return inspect_gaussians(output_path)


def merge_gaussians(paths, output_path):
    if not paths:
        raise ValueError("No Gaussian assets supplied for merging")
    assets = [_read(path) for path in paths]
    degree = max(item[1] for item in assets)
    vertices = np.zeros(sum(len(item[0]) for item in assets), dtype=_dtype(degree))
    ranges, start = [], 0
    target_count = (degree + 1) ** 2 - 1
    for path, (source, source_degree) in zip(paths, assets):
        target = vertices[start : start + len(source)]
        for name in source.dtype.names:
            if not name.startswith("f_rest_"):
                target[name] = source[name]
        source_count = (source_degree + 1) ** 2 - 1
        for channel in range(3):
            for index in range(source_count):
                target[f"f_rest_{channel * target_count + index}"] = source[
                    f"f_rest_{channel * source_count + index}"
                ]
        ranges.append({"path": str(path), "start": start, "count": len(source)})
        start += len(source)
    _write(output_path, vertices)
    return ranges
