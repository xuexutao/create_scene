"""标准 3DGS PLY 资产：读取校验、初始化、Sim(3) 变换（含 SH 旋转）与合并。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np
from numpy.typing import ArrayLike
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from scene_create.geometry import validate_transform
from scene_create.io import write_ply

# -------------------------------------------
# 配置常量
# -------------------------------------------
# fmt: off
POSITION_FIELDS = ("x", "y", "z")     # 高斯中心坐标字段
NORMAL_FIELDS = ("nx", "ny", "nz")    # 3DGS PLY 保留的法线字段（可选，缺失时补零）
MAX_SH_DEGREE = 3                     # 支持的最高球谐阶数
MIN_QUATERNION_NORM = 1e-8            # 旋转四元数视为零的范数阈值
MAX_ABS_LOG_SCALE = 80                # 对数尺度绝对值上限，超出视为数值损坏

DEFAULT_OPACITY = 0.1                 # 初始化高斯的不透明度（sigmoid 之前的概率值）
FALLBACK_SPACING_RATIO = 0.001        # 单点或零间距时初始尺度占点云对角线的比例
MIN_FALLBACK_SPACING = 1e-6           # 回退初始间距下限（场景单位）
INITIAL_SCALE_RATIO = 0.5             # 初始高斯半径占最近邻间距的比例
MIN_INITIAL_SCALE = 1e-8              # 初始高斯半径下限（场景单位）
SH_FIT_DIRECTION_COUNT = 128          # 拟合 SH 旋转矩阵使用的球面采样方向数
# fmt: on

# 3DGS 使用的实球谐归一化常数（0 至 3 阶）。
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


# -------------------------------------------
# 读取与检查
# -------------------------------------------
def read_gaussians(path: Union[str, Path]) -> tuple[np.ndarray, int]:
    """读取标准 3DGS PLY 并统一为完整字段的结构化数组。

    Returns:
        ``(vertices, sh_degree)``；缺失的法线字段补零。

    Raises:
        ValueError: 不是 3DGS PLY、SH 阶数不完整，或字段含非有限/非法值。
    """
    vertices = PlyData.read(str(path))["vertex"].data
    required = set(_gaussian_dtype(0).names) - set(NORMAL_FIELDS)
    if not required.issubset(vertices.dtype.names):
        raise ValueError(f"Not a standard 3DGS PLY (XYZRGB is not Gaussian data): {path}")
    degree = _detect_sh_degree(vertices.dtype.names)
    gaussians = np.zeros(len(vertices), dtype=_gaussian_dtype(degree))
    for name in gaussians.dtype.names:
        if name in vertices.dtype.names:
            gaussians[name] = vertices[name]
        if not np.isfinite(gaussians[name]).all():
            raise ValueError(f"Non-finite Gaussian field: {name}")
    rotations = _stack_fields(gaussians, _rotation_fields())
    if len(rotations) and (np.linalg.norm(rotations, axis=1) < MIN_QUATERNION_NORM).any():
        raise ValueError("Gaussian rotation quaternion cannot be zero")
    scales = _stack_fields(gaussians, _scale_fields())
    if len(scales) and (np.abs(scales) > MAX_ABS_LOG_SCALE).any():
        raise ValueError("Gaussian log scales are numerically invalid")
    return gaussians, degree


def inspect_gaussians(path: Union[str, Path]) -> dict[str, Any]:
    """返回 Gaussian 资产的数量、SH 阶数、包围盒和编码约定。"""
    vertices, degree = read_gaussians(path)
    points = _stack_fields(vertices, POSITION_FIELDS)
    bounds = None
    if len(points):
        bounds = [points.min(axis=0).tolist(), points.max(axis=0).tolist()]
    return {
        "count": len(vertices),
        "sh_degree": degree,
        "bounds": bounds,
        "scale_encoding": "log",
        "opacity_encoding": "logit",
        "quaternion_order": "wxyz",
        "sh_layout": "channel_major",
    }


# -------------------------------------------
# 初始化
# -------------------------------------------
def initialize_gaussians(
    points: ArrayLike,
    colors: ArrayLike,
    output_path: Union[str, Path],
    opacity: float = DEFAULT_OPACITY,
    sh_degree: int = 0,
) -> dict[str, Any]:
    """由彩色点云创建各向同性初始高斯并写入 ``output_path``。

    Args:
        points: Nx3 中心坐标。
        colors: Nx3 RGB（0-255）。
        output_path: 输出 PLY 路径。
        opacity: (0, 1) 之间的初始不透明度。
        sh_degree: 0 至 3 的 SH 阶数，高阶系数初始化为零。

    Returns:
        写出文件的 ``inspect_gaussians`` 结果。

    Raises:
        ValueError: 点或颜色为空、形状不一致、非有限，或不透明度/SH 阶数非法。
    """
    points, colors = np.asarray(points, dtype=float), np.asarray(colors, dtype=float)
    is_valid_cloud = points.ndim == 2 and points.shape[1] == 3 and colors.shape == points.shape and len(points) > 0
    is_finite_cloud = is_valid_cloud and np.isfinite(points).all() and np.isfinite(colors).all()
    if not is_finite_cloud:
        raise ValueError("Gaussian initialization requires nonempty finite Nx3 points and colors")
    if not 0 < opacity < 1 or sh_degree not in range(MAX_SH_DEGREE + 1):
        raise ValueError("Invalid opacity or SH degree")
    vertices = np.zeros(len(points), dtype=_gaussian_dtype(sh_degree))
    for axis, name in enumerate(POSITION_FIELDS):
        vertices[name] = points[:, axis]
        vertices[f"f_dc_{axis}"] = (np.clip(colors[:, axis], 0, 255) / 255 - 0.5) / SH_C0
    log_scales = np.log(np.maximum(_nearest_neighbor_spacing(points) * INITIAL_SCALE_RATIO, MIN_INITIAL_SCALE))
    for name in _scale_fields():
        vertices[name] = log_scales
    vertices["rot_0"] = 1
    vertices["opacity"] = np.log(opacity / (1 - opacity))
    write_ply(output_path, vertices)
    return inspect_gaussians(output_path)


# -------------------------------------------
# 球谐旋转
# -------------------------------------------
def sh_basis(directions: ArrayLike, degree: int) -> np.ndarray:
    """计算单位方向上 0 至 ``degree`` 阶的 3DGS 实球谐基，形状为 N x (degree+1)^2。"""
    directions = np.asarray(directions, dtype=float)
    x, y, z = directions.T
    values = [np.full_like(x, SH_C0)]
    if degree >= 1:
        values += [-SH_C1 * y, SH_C1 * z, -SH_C1 * x]
    if degree >= 2:
        values += [
            SH_C2[0] * x * y,
            SH_C2[1] * y * z,
            SH_C2[2] * (2 * z * z - x * x - y * y),
            SH_C2[3] * x * z,
            SH_C2[4] * (x * x - y * y),
        ]
    if degree >= 3:
        values += [
            SH_C3[0] * y * (3 * x * x - y * y),
            SH_C3[1] * x * y * z,
            SH_C3[2] * y * (4 * z * z - x * x - y * y),
            SH_C3[3] * z * (2 * z * z - 3 * x * x - 3 * y * y),
            SH_C3[4] * x * (4 * z * z - x * x - y * y),
            SH_C3[5] * z * (x * x - y * y),
            SH_C3[6] * x * (x * x - 3 * y * y),
        ]
    return np.stack(values, axis=1)


def rotate_sh(coefficients: ArrayLike, rotation: np.ndarray) -> np.ndarray:
    """把 N x 3 x K 的 SH 系数随物体旋转 ``rotation``，使颜色随视线方向一起旋转。

    Raises:
        ValueError: 系数个数不是 0 至 3 阶的完整 SH。
    """
    coefficients = np.asarray(coefficients, dtype=float)
    degree = int(round(np.sqrt(coefficients.shape[-1]) - 1))
    if coefficients.shape[-1] != (degree + 1) ** 2 or degree not in range(MAX_SH_DEGREE + 1):
        raise ValueError("Unsupported SH coefficient count")
    if degree == 0:
        return coefficients.copy()
    # 在近似均匀的球面方向上最小二乘拟合基变换，等价于各阶 Wigner 旋转。
    directions = _fibonacci_sphere(SH_FIT_DIRECTION_COUNT)
    basis = sh_basis(directions, degree)
    rotated_basis = sh_basis(directions @ rotation, degree)
    mapping = np.linalg.lstsq(basis, rotated_basis, rcond=None)[0]
    return np.einsum("lk,nck->ncl", mapping, coefficients)


# -------------------------------------------
# 变换与合并
# -------------------------------------------
def transform_gaussians(input_path: Union[str, Path], output_path: Union[str, Path], transform: ArrayLike) -> dict[str, Any]:
    """对 Gaussian 资产施加均匀尺度 Sim(3)，同步更新中心、尺度、朝向、法线和 SH。

    Returns:
        写出文件的 ``inspect_gaussians`` 结果。

    Raises:
        ValueError: 变换不是均匀尺度相似变换，或输入不是合法 3DGS PLY。
    """
    transform = validate_transform(transform, allow_scale=True)
    vertices, degree = read_gaussians(input_path)
    scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
    rotation = transform[:3, :3] / scale
    points = _stack_fields(vertices, POSITION_FIELDS)
    _assign_fields(vertices, POSITION_FIELDS, points @ transform[:3, :3].T + transform[:3, 3])
    for name in _scale_fields():
        vertices[name] += np.log(scale)
    if len(vertices):
        _rotate_orientations_in_place(vertices, rotation)
        if degree:
            _rotate_sh_fields_in_place(vertices, degree, rotation)
    write_ply(output_path, vertices)
    return inspect_gaussians(output_path)


def merge_gaussians(paths: Sequence[Union[str, Path]], output_path: Union[str, Path]) -> list[dict[str, Any]]:
    """按顺序合并多个 Gaussian 资产，SH 统一提升到其中最高阶（缺失系数补零）。

    Returns:
        每个输入的 ``path``、``start`` 与 ``count``，对应合并文件中的连续范围。

    Raises:
        ValueError: 未提供输入，或任一输入不是合法 3DGS PLY。
    """
    if not paths:
        raise ValueError("No Gaussian assets supplied for merging")
    assets = [read_gaussians(path) for path in paths]
    degree = max(source_degree for _, source_degree in assets)
    vertices = np.zeros(sum(len(source) for source, _ in assets), dtype=_gaussian_dtype(degree))
    ranges, start = [], 0
    for path, (source, source_degree) in zip(paths, assets):
        target = vertices[start : start + len(source)]
        for name in source.dtype.names:
            if not name.startswith("f_rest_"):
                target[name] = source[name]
        _copy_sh_rest(source, source_degree, target, degree)
        ranges.append({"path": str(path), "start": start, "count": len(source)})
        start += len(source)
    write_ply(output_path, vertices)
    return ranges


# -------------------------------------------
# 字段辅助函数
# -------------------------------------------
def _rest_per_channel(degree: int) -> int:
    """每个颜色通道除 DC 外的 SH 系数个数。"""
    return (degree + 1) ** 2 - 1


def _scale_fields() -> tuple[str, ...]:
    return tuple(f"scale_{axis}" for axis in range(3))


def _rotation_fields() -> tuple[str, ...]:
    return tuple(f"rot_{index}" for index in range(4))


def _gaussian_dtype(degree: int) -> np.dtype:
    names = [*POSITION_FIELDS, *NORMAL_FIELDS, *(f"f_dc_{i}" for i in range(3))]
    names += [f"f_rest_{i}" for i in range(3 * _rest_per_channel(degree))]
    names += ["opacity", *_scale_fields(), *_rotation_fields()]
    return np.dtype([(name, "f4") for name in names])


def _detect_sh_degree(field_names: Sequence[str]) -> int:
    rest = sorted((name for name in field_names if name.startswith("f_rest_")), key=lambda name: int(name.rsplit("_", 1)[1]))
    degree = int(round(np.sqrt(len(rest) / 3 + 1) - 1))
    is_complete = degree in range(MAX_SH_DEGREE + 1) and len(rest) == 3 * _rest_per_channel(degree) and rest == [f"f_rest_{i}" for i in range(len(rest))]
    if not is_complete:
        raise ValueError("Only complete standard SH degrees 0 through 3 are supported")
    return degree


def _stack_fields(vertices: np.ndarray, names: Sequence[str]) -> np.ndarray:
    return np.column_stack([vertices[name] for name in names])


def _assign_fields(vertices: np.ndarray, names: Sequence[str], values: np.ndarray) -> None:
    for index, name in enumerate(names):
        vertices[name] = values[:, index]


def _nearest_neighbor_spacing(points: np.ndarray) -> np.ndarray:
    """每个点到最近邻点的距离；单点或零距离时使用基于点云尺寸的回退值。"""
    fallback = max(float(np.linalg.norm(np.ptp(points, axis=0))) * FALLBACK_SPACING_RATIO, MIN_FALLBACK_SPACING)
    if len(points) > 1:
        spacing = cKDTree(points).query(points, k=2)[0][:, 1]
    else:
        spacing = np.array([fallback])
    return np.where(np.isfinite(spacing) & (spacing > 0), spacing, fallback)


def _fibonacci_sphere(count: int) -> np.ndarray:
    """黄金角螺旋上近似均匀分布的单位方向。"""
    index = np.arange(count, dtype=float) + 0.5
    z = 1 - 2 * index / len(index)
    angle = index * (np.pi * (3 - np.sqrt(5)))
    radius = np.sqrt(1 - z * z)
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle), z])


def _rotate_orientations_in_place(vertices: np.ndarray, rotation: np.ndarray) -> None:
    """原地旋转 wxyz 四元数与法线。"""
    quaternions_xyzw = np.column_stack([vertices[f"rot_{i}"] for i in (1, 2, 3, 0)])
    rotated = (Rotation.from_matrix(rotation) * Rotation.from_quat(quaternions_xyzw)).as_quat()[:, [3, 0, 1, 2]]
    _assign_fields(vertices, _rotation_fields(), rotated)
    _assign_fields(vertices, NORMAL_FIELDS, _stack_fields(vertices, NORMAL_FIELDS) @ rotation.T)


def _rotate_sh_fields_in_place(vertices: np.ndarray, degree: int, rotation: np.ndarray) -> None:
    """原地旋转通道优先布局的 DC 与高阶 SH 系数。"""
    count = _rest_per_channel(degree)
    coefficients = np.empty((len(vertices), 3, count + 1))
    coefficients[:, :, 0] = _stack_fields(vertices, [f"f_dc_{i}" for i in range(3)])
    coefficients[:, :, 1:] = _stack_fields(vertices, [f"f_rest_{i}" for i in range(3 * count)]).reshape(-1, 3, count)
    coefficients = rotate_sh(coefficients, rotation)
    for channel in range(3):
        vertices[f"f_dc_{channel}"] = coefficients[:, channel, 0]
    _assign_fields(vertices, [f"f_rest_{i}" for i in range(3 * count)], coefficients[:, :, 1:].reshape(-1, 3 * count))


def _copy_sh_rest(source: np.ndarray, source_degree: int, target: np.ndarray, target_degree: int) -> None:
    """把低阶通道优先 SH 系数写入高阶布局对应位置，其余保持为零。"""
    source_count, target_count = _rest_per_channel(source_degree), _rest_per_channel(target_degree)
    for channel in range(3):
        for index in range(source_count):
            target[f"f_rest_{channel * target_count + index}"] = source[f"f_rest_{channel * source_count + index}"]
