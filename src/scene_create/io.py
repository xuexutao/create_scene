from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def atomic_json(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                data,
                stream,
                ensure_ascii=False,
                indent=2,
                default=json_value,
                allow_nan=False,
            )
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_npz(path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_npz(path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def save_image(path, array) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (
        array if isinstance(array, Image.Image) else Image.fromarray(np.asarray(array))
    )
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_mask(path, size=None) -> np.ndarray:
    with Image.open(path) as image:
        if size is not None and image.size != tuple(size):
            raise ValueError(f"Mask size {image.size} does not match {size}: {path}")
        return np.asarray(image.convert("L")) > 127


def safe_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_").lower()
    return name[:80] or "object"


def validate_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value
    ):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def file_digest(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, default=json_value, allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_cloud(path, points, colors=None) -> None:
    path = Path(path)
    xyz = np.asarray(points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Point cloud must contain finite Nx3 coordinates")
    rgb = (
        np.full(xyz.shape, 127, dtype=np.uint8)
        if colors is None
        else np.asarray(colors)
    )
    if rgb.shape != xyz.shape or not np.isfinite(rgb).all():
        raise ValueError("Point cloud colors must be finite Nx3")
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    vertices = np.empty(
        len(xyz),
        dtype=[(n, "f4") for n in ("x", "y", "z")]
        + [(n, "u1") for n in ("red", "green", "blue")],
    )
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        vertices[name] = rgb[:, index]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    try:
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_cloud(path) -> dict:
    vertices = PlyData.read(str(path))["vertex"].data
    points = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(
        np.float64
    )
    if {"red", "green", "blue"}.issubset(vertices.dtype.names):
        colors = np.column_stack(
            [vertices[name] for name in ("red", "green", "blue")]
        ).astype(np.uint8)
    else:
        colors = np.full(points.shape, 127, dtype=np.uint8)
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite points: {path}")
    return {"points": points, "colors": colors}
