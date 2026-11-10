"""原子化文件读写、标识校验与内容指纹。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

PathLike = Union[str, Path]

# -------------------------------------------
# 配置常量
# -------------------------------------------
# fmt: off
DIGEST_BLOCK_BYTES = 1024 * 1024  # 计算 SHA-256 时每次读取的字节数
MAX_SAFE_NAME_LENGTH = 80         # safe_name 生成的文件名最大字符数
DEFAULT_CLOUD_GRAY = 127          # 点云缺少颜色时使用的灰度值
# fmt: on

# 标识只允许字母数字开头，可含下划线和连字符，防止路径穿越。
VALID_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}")
UNSAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9_-]+")
XYZ_FIELDS = ("x", "y", "z")
RGB_FIELDS = ("red", "green", "blue")


# -------------------------------------------
# JSON 与原子写入
# -------------------------------------------
def json_value(value: Any) -> Any:
    """把 Path 与 numpy 值转换为 JSON 可序列化的值。

    Raises:
        TypeError: 值类型无法序列化。
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


@contextmanager
def atomic_path(path: PathLike) -> Iterator[str]:
    """提供同目录临时文件路径，只有成功退出时才替换 ``path``。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        yield temporary_name
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: PathLike, data: Any) -> None:
    """原子写入 UTF-8 JSON，拒绝 NaN。"""
    with atomic_path(path) as temporary:
        with Path(temporary).open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, default=json_value, allow_nan=False)
            stream.write("\n")


def read_json(path: PathLike) -> Any:
    """读取 UTF-8 JSON 文件。"""
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


# -------------------------------------------
# 数组与图像
# -------------------------------------------
def atomic_npz(path: PathLike, **arrays: np.ndarray) -> None:
    """原子写入压缩 NPZ。"""
    with atomic_path(path) as temporary:
        with Path(temporary).open("wb") as stream:
            np.savez_compressed(stream, **arrays)


def load_npz(path: PathLike) -> dict[str, np.ndarray]:
    """读取 NPZ 中的全部数组，禁止 pickle。"""
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def save_image(path: PathLike, array: Union[Image.Image, np.ndarray]) -> None:
    """原子写入 PNG 图像。"""
    if isinstance(array, Image.Image):
        image = array
    else:
        image = Image.fromarray(np.asarray(array))
    with atomic_path(path) as temporary:
        image.save(temporary, format="PNG")


def load_mask(path: PathLike, size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """读取单通道 mask 并二值化。

    Args:
        path: mask 图像路径。
        size: 期望的 ``(width, height)``；为 ``None`` 时不检查。

    Raises:
        ValueError: 图像尺寸与 ``size`` 不一致。
    """
    with Image.open(path) as image:
        if size is not None and image.size != tuple(size):
            raise ValueError(f"Mask size {image.size} does not match {size}: {path}")
        return np.asarray(image.convert("L")) > 127


# -------------------------------------------
# 标识与指纹
# -------------------------------------------
def safe_name(value: str) -> str:
    """把任意文本转换为安全的小写文件名片段。"""
    name = UNSAFE_NAME_PATTERN.sub("_", str(value)).strip("_").lower()
    return name[:MAX_SAFE_NAME_LENGTH] or "object"


def validate_id(value: str) -> str:
    """校验视图或物体标识并原样返回。

    Raises:
        ValueError: 标识为空、过长或包含不安全字符。
    """
    if not isinstance(value, str) or not VALID_ID_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def file_digest(path: PathLike) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(DIGEST_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    """计算 JSON 值的稳定 SHA-256 指纹。"""
    payload = json.dumps(value, sort_keys=True, default=json_value, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


# -------------------------------------------
# PLY 点云
# -------------------------------------------
def write_ply(path: PathLike, vertices: np.ndarray) -> None:
    """原子写入二进制 PLY 顶点表。"""
    with atomic_path(path) as temporary:
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(temporary)


def write_cloud(path: PathLike, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    """写入带 RGB 的点云 PLY。

    Raises:
        ValueError: 坐标或颜色不是有限的 Nx3 数组。
    """
    xyz = np.asarray(points, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Point cloud must contain finite Nx3 coordinates")
    if colors is None:
        rgb = np.full(xyz.shape, DEFAULT_CLOUD_GRAY, dtype=np.uint8)
    else:
        rgb = np.asarray(colors)
    if rgb.shape != xyz.shape or not np.isfinite(rgb).all():
        raise ValueError("Point cloud colors must be finite Nx3")
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    vertex_type = [(name, "f4") for name in XYZ_FIELDS] + [(name, "u1") for name in RGB_FIELDS]
    vertices = np.empty(len(xyz), dtype=vertex_type)
    for index, name in enumerate(XYZ_FIELDS):
        vertices[name] = xyz[:, index]
    for index, name in enumerate(RGB_FIELDS):
        vertices[name] = rgb[:, index]
    write_ply(path, vertices)


def read_cloud(path: PathLike) -> dict[str, np.ndarray]:
    """读取点云 PLY，返回 ``points``（float64）与 ``colors``（uint8）。

    Raises:
        ValueError: 坐标包含非有限值。
    """
    vertices = PlyData.read(str(path))["vertex"].data
    points = np.column_stack([vertices[name] for name in XYZ_FIELDS]).astype(np.float64)
    if set(RGB_FIELDS).issubset(vertices.dtype.names):
        colors = np.column_stack([vertices[name] for name in RGB_FIELDS]).astype(np.uint8)
    else:
        colors = np.full(points.shape, DEFAULT_CLOUD_GRAY, dtype=np.uint8)
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite points: {path}")
    return {"points": points, "colors": colors}
