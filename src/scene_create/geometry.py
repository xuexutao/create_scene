"""相机、像素网格、点云、网格栅格化/采样与 Sim(3) 配准的几何基础函数。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    import trimesh

# -------------------------------------------
# 配置常量
# -------------------------------------------
# fmt: off
HOMOGENEOUS_ROW_TOLERANCE = 1e-7      # 齐次变换最后一行允许偏离 [0, 0, 0, 1] 的绝对误差
ORTHONORMAL_TOLERANCE = 1e-5          # 去除均匀尺度后旋转矩阵正交性的绝对误差
MIN_PROJECTION_DEPTH = 1e-9           # 投影时视为位于相机前方的最小光轴深度（场景单位）
MIN_MAPPING_DETERMINANT = 1e-12       # 像素映射矩阵可逆所需的最小行列式绝对值
MIN_HOMOGENEOUS_W = 1e-12             # 反投影像素齐次坐标 w 的最小绝对值

EXTENT_LOW_QUANTILE = 0.05            # 稳健包围盒对角线的下分位数
EXTENT_HIGH_QUANTILE = 0.95           # 稳健包围盒对角线的上分位数
DEFAULT_MIN_EXTENT = 1e-6             # extent() 的默认下限，避免后续除零（场景单位）

MIN_OUTLIER_FILTER_POINTS = 4         # 点数不超过该值或近邻数时不做统计离群点过滤
MIN_OUTLIER_THRESHOLD = 1e-12         # 离群点平均近邻距离阈值的下限（场景单位）

RASTER_NEAR_DEPTH = 1e-6              # 诊断栅格化时顶点视为可见的最小深度（场景单位）
BARYCENTRIC_TOLERANCE = 1e-8          # 像素中心落在三角形边上的重心坐标容差
MIN_TRIANGLE_AREA_TERM = 1e-12        # 屏幕空间三角形视为退化的重心分母绝对值
MIN_MESH_SAMPLES = 3                  # 网格表面采样的最小点数

MIN_SINGULAR_VALUE = 1e-15            # 协方差最大奇异值的下限，避免全零输入
SINGULAR_RANK_RATIO = 1e-10           # 判定奇异值有效的相对阈值
MIN_SOURCE_VARIANCE = 1e-15           # 相似变换源点加权方差下限
# fmt: on

# fmt: off
MIN_REGISTRATION_POINTS = 6           # 配准源/目标各自至少需要的有限 3D 点数
RANK_TOLERANCE = 1e-8                 # 判定点集至少张成平面的矩阵秩容差
MIN_REGISTRATION_SIZE = 1e-9          # 配准点集稳健尺寸下限（场景单位）
DEFAULT_COARSE_POINTS = 800           # 多初值粗配准使用的采样点数
DEFAULT_FINE_POINTS = 6000            # 精配准使用的采样点数
DEFAULT_COARSE_ITERATIONS = 10        # 每个初值的粗 ICP 迭代次数
DEFAULT_FINE_ITERATIONS = 60          # 带尺度精 ICP 迭代次数
RIGID_POLISH_ITERATIONS = 20          # 精配准后固定尺度的刚体 ICP 迭代次数
DEFAULT_REFINE_CANDIDATES = 4         # 进入精配准的粗配准最优候选数
DEFAULT_TRIM_FRACTION = 0.9           # ICP 保留的最近距离分位数
MIN_TRIM_FRACTION = 0.5               # trim_fraction 允许的最小值
DEFAULT_SCALE_CANDIDATES = (0.75, 1.0, 1.4)  # 相对 PCA 尺寸比例的初始尺度倍数
MIN_SCALE_FACTOR = 0.4                # ICP 尺度相对初始尺寸比例的下界倍数
MAX_SCALE_FACTOR = 3.5                # ICP 尺度相对初始尺寸比例的上界倍数
DEFAULT_DISTANCE_RATIO = 0.06         # 对应点覆盖半径占目标尺寸的比例
DEFAULT_MIN_FITNESS = 0.5             # 接受配准所需的观测表面覆盖率
DEFAULT_MAX_RMSE_RATIO = 0.04         # 接受配准允许的截断 RMSE 与目标尺寸之比
DEFAULT_MIN_SOURCE_SUPPORT = 0.03     # 接受配准所需的资产表面支撑比例
RANK_FITNESS_WEIGHT = 0.02            # 候选排序中未覆盖观测比例的权重
RANK_SUPPORT_WEIGHT = 0.005           # 候选排序中资产支撑不足的权重
MIN_ICP_MATCHES = 6                   # ICP 单次迭代所需的最少截断对应点
MIN_TRIM_DISTANCE = 1e-10             # ICP 截断距离阈值下限（场景单位）
MIN_MEDIAN_DISTANCE = 1e-8            # 鲁棒权重中位距离下限（场景单位）
QUALITY_TRIM_EPSILON = 1e-12          # 计算截断 RMSE 时的分位数容差
CONVERGENCE_ABSOLUTE = 1e-14          # ICP 均方误差变化的绝对收敛阈值
CONVERGENCE_RELATIVE = 1e-7           # ICP 均方误差变化的相对收敛阈值
REGISTRATION_METHOD = "pca_multistart_sim3_trimmed_observation_to_asset_icp"
# fmt: on


# -------------------------------------------
# 相机与像素网格
# -------------------------------------------
def validate_transform(matrix: ArrayLike, allow_scale: bool = False) -> np.ndarray:
    """校验 4x4 齐次变换为刚体变换，或在允许时为均匀尺度的相似变换。

    Args:
        matrix: 待校验的齐次变换。
        allow_scale: 是否允许正的均匀尺度。

    Returns:
        float64 的 4x4 变换矩阵。

    Raises:
        ValueError: 矩阵非有限、非齐次、改变手性，或含剪切/各向异性尺度。
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    is_homogeneous = matrix.shape == (4, 4) and np.isfinite(matrix).all() and np.allclose(matrix[3], [0, 0, 0, 1], atol=HOMOGENEOUS_ROW_TOLERANCE)
    if not is_homogeneous:
        raise ValueError("Expected a finite homogeneous 4x4 transform")
    linear = matrix[:3, :3]
    determinant = np.linalg.det(linear)
    if determinant <= 0:
        raise ValueError("Transform must preserve handedness and have a positive scale")
    scale = np.cbrt(determinant) if allow_scale else 1.0
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=ORTHONORMAL_TOLERANCE):
        raise ValueError("Transform contains shear, anisotropic scale or an invalid rotation")
    return matrix


def depth_to_points(depth: ArrayLike, intrinsics: ArrayLike, world_from_camera: ArrayLike) -> np.ndarray:
    """把 HxW 光轴深度图反投影为 HxWx3 世界坐标点图。

    Raises:
        ValueError: 深度非有限或为负、内参非法或位姿不是刚体变换。
    """
    depth = np.asarray(depth, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2 or not np.isfinite(depth).all() or (depth < 0).any():
        raise ValueError("Depth must be a finite nonnegative HxW optical-axis depth map")
    has_valid_intrinsics = intrinsics.shape == (3, 3) and np.isfinite(intrinsics).all() and min(intrinsics[0, 0], intrinsics[1, 1]) > 0
    if not has_valid_intrinsics:
        raise ValueError("Invalid camera intrinsics")
    transform = validate_transform(world_from_camera)
    y, x = np.indices(depth.shape)
    pixels = np.stack([x, y, np.ones_like(x)], axis=-1)
    rays = pixels @ np.linalg.inv(intrinsics).T
    points = rays * depth[..., None]
    return points @ transform[:3, :3].T + transform[:3, 3]


def project_points(points: ArrayLike, intrinsics: ArrayLike, world_from_camera: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """把世界坐标点投影到像素平面。

    Returns:
        ``(uv, z)``：Nx2 像素坐标（相机后方或非有限为 NaN）与 N 个光轴深度。

    Raises:
        ValueError: 点不是 Nx3 或位姿不是刚体变换。
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected Nx3 points")
    world_from_camera = validate_transform(world_from_camera)
    camera = (points - world_from_camera[:3, 3]) @ world_from_camera[:3, :3]
    projection = camera @ np.asarray(intrinsics, dtype=np.float64).T
    uv = np.full((len(points), 2), np.nan)
    is_projectable = np.isfinite(projection).all(axis=1) & (camera[:, 2] > MIN_PROJECTION_DEPTH)
    uv[is_projectable] = projection[is_projectable, :2] / projection[is_projectable, 2:3]
    return uv, camera[:, 2]


def map_mask(mask: np.ndarray, original_to_processed: ArrayLike, shape: tuple[int, int]) -> np.ndarray:
    """通过像素中心单应把原图 mask 最近邻重采样到处理网格。

    Raises:
        ValueError: 映射矩阵不是有限可逆的 3x3 矩阵。
    """
    matrix = np.asarray(original_to_processed, dtype=float)
    is_invertible = matrix.shape == (3, 3) and np.isfinite(matrix).all() and abs(np.linalg.det(matrix)) >= MIN_MAPPING_DETERMINANT
    if not is_invertible:
        raise ValueError("Invalid original_to_processed mapping")
    y, x = np.indices(shape)
    homogeneous = np.stack([x.ravel(), y.ravel(), np.ones(x.size)])
    original = np.linalg.inv(matrix) @ homogeneous
    is_finite = np.isfinite(original).all(axis=0) & (np.abs(original[2]) > MIN_HOMOGENEOUS_W)
    original[:2, is_finite] /= original[2:3, is_finite]
    u, v = np.zeros(x.size, dtype=np.int64), np.zeros(x.size, dtype=np.int64)
    u[is_finite], v[is_finite] = np.rint(original[:2, is_finite]).astype(np.int64)
    is_inside = is_finite & (u >= 0) & (v >= 0) & (u < mask.shape[1]) & (v < mask.shape[0])
    resampled = np.zeros(x.size, dtype=bool)
    resampled[is_inside] = mask[v[is_inside], u[is_inside]]
    return resampled.reshape(shape)


# -------------------------------------------
# 点云
# -------------------------------------------
def point_extent(points: ArrayLike) -> float:
    """返回 5%-95% 分位数稳健包围盒的对角线长度。"""
    low = np.quantile(points, EXTENT_LOW_QUANTILE, axis=0)
    high = np.quantile(points, EXTENT_HIGH_QUANTILE, axis=0)
    return float(np.linalg.norm(high - low))


def extent(points: ArrayLike, minimum: float = DEFAULT_MIN_EXTENT) -> float:
    """返回不小于 ``minimum`` 的稳健尺寸；少于两个点时直接返回 ``minimum``。"""
    if len(points) < 2:
        return minimum
    return max(point_extent(points), minimum)


def clean_cloud(points: ArrayLike, colors: ArrayLike, weights: ArrayLike, neighbors: int = 16, std_ratio: float = 2.5) -> dict[str, np.ndarray]:
    """去除非有限/非正权重点，并按平均近邻距离做统计离群点过滤。

    Returns:
        ``points``、``colors``（uint8）、``weights`` 和保留点在输入中的 ``indices``。

    Raises:
        ValueError: 点、颜色和权重形状不一致。
    """
    points, colors, weights = _validate_cloud_arrays(points, colors, weights)
    is_valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1) & np.isfinite(weights) & (weights > 0)
    indices = np.flatnonzero(is_valid)
    if len(indices) > max(MIN_OUTLIER_FILTER_POINTS, neighbors):
        valid_points = points[indices]
        distances = cKDTree(valid_points).query(valid_points, k=min(neighbors + 1, len(indices)))[0][:, 1:].mean(axis=1)
        threshold = distances.mean() + float(std_ratio) * distances.std()
        indices = indices[distances <= max(threshold, MIN_OUTLIER_THRESHOLD)]
    return {
        "points": points[indices],
        "colors": np.clip(colors[indices], 0, 255).astype(np.uint8),
        "weights": weights[indices],
        "indices": indices,
    }


def voxel_fuse(points: ArrayLike, colors: ArrayLike, weights: ArrayLike, voxel_size: float) -> dict[str, np.ndarray]:
    """按体素做置信度加权平均融合。

    Returns:
        融合后的 ``points``、``colors``（uint8）、每体素总权重 ``weights`` 和点数 ``counts``。

    Raises:
        ValueError: 体素尺寸非正，或输入含非有限值/非正权重。
    """
    points, colors, weights = _validate_cloud_arrays(points, colors, weights)
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    is_fusable = np.isfinite(points).all() and np.isfinite(colors).all() and np.isfinite(weights).all() and not (weights <= 0).any()
    if not is_fusable:
        raise ValueError("Voxel fusion requires finite data and positive weights")
    if not len(points):
        return {"points": points, "colors": colors.astype(np.uint8), "weights": weights, "counts": np.zeros(0, dtype=np.int64)}
    cells = np.floor((points - points.min(axis=0)) / voxel_size).astype(np.int64)
    _, assignments, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    totals = np.bincount(assignments, weights=weights)
    fused_colors = _weighted_voxel_mean(assignments, colors, weights, totals)
    return {
        "points": _weighted_voxel_mean(assignments, points, weights, totals),
        "colors": np.clip(np.rint(fused_colors), 0, 255).astype(np.uint8),
        "weights": totals,
        "counts": counts,
    }


def cloud_overlap(a: ArrayLike, b: ArrayLike, distance: float) -> float:
    """返回两点云在 ``distance`` 内相互覆盖比例的较小值；任一为空时为 0。

    Raises:
        ValueError: ``distance`` 非正。
    """
    a, b = np.asarray(a), np.asarray(b)
    if min(len(a), len(b)) == 0:
        return 0.0
    if distance <= 0:
        raise ValueError("Overlap distance must be positive")
    forward = np.mean(cKDTree(b).query(a, k=1)[0] <= distance)
    backward = np.mean(cKDTree(a).query(b, k=1)[0] <= distance)
    return float(min(forward, backward))


def _validate_cloud_arrays(points: ArrayLike, colors: ArrayLike, weights: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors)
    weights = np.asarray(weights, dtype=np.float64)
    has_matching_shapes = points.ndim == 2 and points.shape[1] == 3 and colors.shape == points.shape and weights.shape == (len(points),)
    if not has_matching_shapes:
        raise ValueError("Expected points/colors Nx3 and weights N")
    return points, colors, weights


def _weighted_voxel_mean(assignments: np.ndarray, values: np.ndarray, weights: np.ndarray, totals: np.ndarray) -> np.ndarray:
    columns = [np.bincount(assignments, weights=values[:, axis].astype(float) * weights) / totals for axis in range(3)]
    return np.column_stack(columns)


# -------------------------------------------
# 网格栅格化与采样
# -------------------------------------------
def rasterize_depth(
    vertices: ArrayLike,
    faces: ArrayLike,
    intrinsics: ArrayLike,
    world_from_camera: ArrayLike,
    width: int,
    height: int,
    max_side: int = 256,
) -> dict[str, Any]:
    """CPU 针孔 z-buffer，仅用于对齐诊断，不是外观渲染。

    输出网格按 ``max_side`` 等比缩小，``pixel_map`` 把原始像素中心映射到输出网格。

    Returns:
        ``depth``（不可见处为 inf）、``mask``、``pixel_map``、``width`` 和 ``height``。

    Raises:
        ValueError: 网格形状或索引非法，或位姿不是刚体变换。
    """
    vertices, faces = _validate_raster_mesh(vertices, faces)
    matrix = validate_transform(world_from_camera)
    pixel_map, target_width, target_height = _diagnostic_pixel_map(width, height, max_side)
    intrinsic = pixel_map @ np.asarray(intrinsics)
    camera = (vertices - matrix[:3, 3]) @ matrix[:3, :3]
    homogeneous = camera @ intrinsic.T
    pixels = np.zeros((len(vertices), 2))
    is_in_front = camera[:, 2] > RASTER_NEAR_DEPTH
    pixels[is_in_front] = homogeneous[is_in_front, :2] / homogeneous[is_in_front, 2:3]
    depth = np.full((target_height, target_width), np.inf, dtype=np.float64)
    for triangle in faces:
        if is_in_front[triangle].all():
            _draw_triangle_depth(depth, pixels[triangle], camera[triangle, 2])
    return {"depth": depth, "mask": np.isfinite(depth), "pixel_map": pixel_map, "width": target_width, "height": target_height}


def load_triangle_mesh(path: Union[str, Path], transform: Optional[ArrayLike] = None) -> trimesh.Trimesh:
    """把 GLB 场景图展开为资产坐标系下的单个三角网格，可再施加 ``transform``。

    Raises:
        ValueError: 资产中没有三角网格。
    """
    # trimesh 只在处理网格的阶段需要，延迟导入以缩短 CLI 启动时间。
    import trimesh

    scene = trimesh.load(str(path), force="scene", process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        local, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        mesh.apply_transform(local)
        if transform is not None:
            mesh.apply_transform(transform)
        parts.append(mesh)
    if not parts:
        raise ValueError("Asset contains no triangle mesh")
    return trimesh.util.concatenate(parts)


def sample_mesh(mesh_path: Union[str, Path], count: int = 20000, seed: int = 0) -> dict[str, np.ndarray]:
    """按面积在网格表面均匀采样，结果对 ``seed`` 可复现。

    Returns:
        Nx3 ``points`` 与所在面的 ``normals``。

    Raises:
        ValueError: 采样数过少，或网格为空、退化或含非有限值。
    """
    if count < MIN_MESH_SAMPLES:
        raise ValueError("At least three mesh samples are required")
    mesh = load_triangle_mesh(mesh_path)
    areas = np.asarray(mesh.area_faces)
    is_sampleable = np.isfinite(mesh.vertices).all() and np.isfinite(areas).all() and areas.sum() > 0
    if not is_sampleable:
        raise ValueError("Mesh geometry is empty, degenerate or non-finite")
    random = np.random.default_rng(seed)
    faces = random.choice(len(areas), size=int(count), p=areas / areas.sum())
    triangles = mesh.triangles[faces]
    # 平方根重参数化让三角形内的重心坐标服从均匀分布。
    a, b = np.sqrt(random.random((2, count)))
    b = b * b
    points = (1 - a[:, None]) * triangles[:, 0] + (a * (1 - b))[:, None] * triangles[:, 1] + (a * b)[:, None] * triangles[:, 2]
    return {"points": points, "normals": mesh.face_normals[faces]}


def _validate_raster_mesh(vertices: ArrayLike, faces: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    has_triangle_shapes = vertices.ndim == 2 and vertices.shape[1] == 3 and faces.ndim == 2 and faces.shape[1] == 3
    if not has_triangle_shapes:
        raise ValueError("Rasterization requires Nx3 vertices and Mx3 triangle indices")
    has_valid_indices = np.isfinite(vertices).all() and not (faces < 0).any() and not (faces >= len(vertices)).any()
    if not has_valid_indices:
        raise ValueError("Invalid rasterization mesh")
    return vertices, faces


def _diagnostic_pixel_map(width: int, height: int, max_side: int) -> tuple[np.ndarray, int, int]:
    """返回原始到缩小网格的像素中心映射及缩小后的宽高。"""
    scale = min(1.0, float(max_side) / max(width, height))
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    sx, sy = target_width / width, target_height / height
    pixel_map = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return pixel_map, target_width, target_height


def _draw_triangle_depth(depth: np.ndarray, xy: np.ndarray, z: np.ndarray) -> None:
    """把一个屏幕空间三角形按透视正确的深度原地写入 ``depth``（取最小值）。"""
    height, width = depth.shape
    left, top = np.maximum(np.ceil(xy.min(axis=0)), [0, 0]).astype(int)
    right, bottom = np.minimum(np.floor(xy.max(axis=0)), [width - 1, height - 1]).astype(int)
    if left > right or top > bottom:
        return
    a, b, c = xy
    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denominator) < MIN_TRIANGLE_AREA_TERM:
        return
    y, x = np.mgrid[top : bottom + 1, left : right + 1]
    wa = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
    wb = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
    wc = 1 - wa - wb
    is_inside = (wa >= -BARYCENTRIC_TOLERANCE) & (wb >= -BARYCENTRIC_TOLERANCE) & (wc >= -BARYCENTRIC_TOLERANCE)
    # 透视正确插值：在屏幕空间线性插值的是 1/z。
    inverse_depth = wa / z[0] + wb / z[1] + wc / z[2]
    values = np.full(x.shape, np.inf)
    is_visible = is_inside & (inverse_depth > 0)
    values[is_visible] = 1 / inverse_depth[is_visible]
    tile = depth[top : bottom + 1, left : right + 1]
    np.minimum(tile, values, out=tile)


# -------------------------------------------
# 相似变换配准
# -------------------------------------------
@dataclass(frozen=True)
class _IcpSettings:
    scale_bounds: tuple[float, float]
    iterations: int
    trim: float
    with_scale: bool = True


@dataclass(frozen=True)
class _PointPair:
    """同一分辨率下的资产采样点与观测点。"""

    source: np.ndarray
    target: np.ndarray


@dataclass(frozen=True)
class _AlignmentQuality:
    rmse: float
    fitness: float
    support: float

    def rank(self, target_size: float) -> float:
        """候选排序分数，越小越好。"""
        return self.rmse / target_size + RANK_FITNESS_WEIGHT * (1 - self.fitness) + RANK_SUPPORT_WEIGHT * (1 - self.support)


def estimate_similarity(source: ArrayLike, target: ArrayLike, with_scale: bool = True, weights: Optional[ArrayLike] = None) -> np.ndarray:
    """加权 Umeyama 估计把 ``source`` 对齐到 ``target`` 的 4x4 Sim(3)（或刚体）变换。

    Raises:
        ValueError: 对应点不足、非有限、共线/重合，或权重/尺度非法。
    """
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    has_paired_points = source.shape == target.shape and source.ndim == 2 and source.shape[1] == 3 and len(source) >= 3
    if not has_paired_points:
        raise ValueError("Similarity estimation needs at least three paired 3D points")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Similarity inputs must be finite")
    weights = _normalized_similarity_weights(weights, len(source))
    source_center, target_center = weights @ source, weights @ target
    source_zero, target_zero = source - source_center, target - target_center
    covariance = (target_zero * weights[:, None]).T @ source_zero
    u, singular, vt = np.linalg.svd(covariance)
    if np.count_nonzero(singular > max(singular[0], MIN_SINGULAR_VALUE) * SINGULAR_RANK_RATIO) < 2:
        raise ValueError("Collinear or coincident correspondences do not determine a 3D pose")
    sign = np.ones(3)
    sign[-1] = 1 if np.linalg.det(u @ vt) >= 0 else -1
    rotation = (u * sign) @ vt
    variance = np.sum(weights * np.sum(source_zero**2, axis=1))
    if variance < MIN_SOURCE_VARIANCE:
        raise ValueError("Source variance is zero")
    scale = float(np.dot(singular, sign) / variance) if with_scale else 1.0
    if scale <= 0:
        raise ValueError("Similarity scale must be positive")
    return _similarity_matrix(scale * rotation, source_center, target_center)


def register_points(source: ArrayLike, target: ArrayLike, config: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """把完整资产点 ``source`` 稳健地 Sim(3) 配准到观测点 ``target``，最后做刚体 ICP 收尾。

    分数只是几何诊断，不代表语义身份一致。

    Args:
        source: 资产表面采样点。
        target: 观测到的（可能残缺的）物体点。
        config: 可选项 ``coarse_points``、``fine_points``、``coarse_iterations``、``iterations``、
            ``refine_candidates``、``trim_fraction``、``scale_candidates``、``distance_ratio``、
            ``initial_transform``、``min_fitness``、``max_rmse_ratio``、``min_source_support``、``seed``。

    Returns:
        ``transform``、``scale``、``rotation_xyzw``、``translation_xyz``、``metrics`` 与 ``accepted``。

    Raises:
        ValueError: 点集不足或退化、尺寸为零，或 ``trim_fraction`` 越界。
    """
    options = dict(config or {})
    source = _validate_registration_points("source", source)
    target = _validate_registration_points("target", target)
    random = np.random.default_rng(options.get("seed", 0))
    fine_source = _random_subset(source, options.get("fine_points", DEFAULT_FINE_POINTS), random)
    fine_target = _random_subset(target, options.get("fine_points", DEFAULT_FINE_POINTS), random)
    coarse_source = _random_subset(fine_source, options.get("coarse_points", DEFAULT_COARSE_POINTS), random)
    coarse_target = _random_subset(fine_target, options.get("coarse_points", DEFAULT_COARSE_POINTS), random)

    source_size, target_size = point_extent(source), point_extent(target)
    if min(source_size, target_size) < MIN_REGISTRATION_SIZE:
        raise ValueError("Registration size is zero")
    base_scale = target_size / source_size
    trim = float(options.get("trim_fraction", DEFAULT_TRIM_FRACTION))
    if not MIN_TRIM_FRACTION <= trim <= 1:
        raise ValueError("trim_fraction must be in [0.5, 1]")
    scale_bounds = (base_scale * MIN_SCALE_FACTOR, base_scale * MAX_SCALE_FACTOR)
    radius = target_size * options.get("distance_ratio", DEFAULT_DISTANCE_RATIO)

    candidates = _initial_candidates(coarse_source, coarse_target, base_scale, options)
    coarse_settings = _IcpSettings(scale_bounds, options.get("coarse_iterations", DEFAULT_COARSE_ITERATIONS), trim)
    shortlisted = _rank_candidates(candidates, _PointPair(coarse_source, coarse_target), coarse_settings, radius, target_size)
    fine_settings = _IcpSettings(scale_bounds, options.get("iterations", DEFAULT_FINE_ITERATIONS), trim)
    shortlisted = shortlisted[: options.get("refine_candidates", DEFAULT_REFINE_CANDIDATES)]
    transform, quality = _refine_best_candidate(shortlisted, _PointPair(fine_source, fine_target), fine_settings, radius, target_size)
    return _registration_result(transform, quality, target_size, radius, options)


def _validate_registration_points(name: str, points: ArrayLike) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    has_enough_points = points.ndim == 2 and points.shape[1] == 3 and len(points) >= MIN_REGISTRATION_POINTS and np.isfinite(points).all()
    if not has_enough_points:
        raise ValueError(f"Registration {name} needs at least six finite 3D points")
    if np.linalg.matrix_rank(points - points.mean(axis=0), tol=RANK_TOLERANCE) < 2:
        raise ValueError(f"Registration {name} is geometrically degenerate")
    return points


def _normalized_similarity_weights(weights: Optional[ArrayLike], count: int) -> np.ndarray:
    weights = np.ones(count) if weights is None else np.asarray(weights, dtype=float)
    is_valid = weights.shape == (count,) and np.isfinite(weights).all() and not (weights < 0).any() and weights.sum() > 0
    if not is_valid:
        raise ValueError("Invalid similarity weights")
    return weights / weights.sum()


def _similarity_matrix(linear: np.ndarray, source_center: np.ndarray, target_center: np.ndarray) -> np.ndarray:
    """构造把 ``source_center`` 映射到 ``target_center`` 的齐次变换。"""
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    matrix[:3, 3] = target_center - matrix[:3, :3] @ source_center
    return matrix


def _random_subset(points: np.ndarray, count: int, random: np.random.Generator) -> np.ndarray:
    if len(points) <= count:
        return points
    return points[random.choice(len(points), count, replace=False)]


def _apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _principal_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回中位数中心和右手 PCA 主轴基。"""
    center = np.median(points, axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    basis = vt.T
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return center, basis


def _initial_candidates(source: np.ndarray, target: np.ndarray, base_scale: float, options: Mapping[str, Any]) -> list[np.ndarray]:
    """显式初值优先，其后是 24 个右手主轴对应乘以各尺度倍数的 PCA 初值。"""
    candidates = []
    if options.get("initial_transform") is not None:
        candidates.append(validate_transform(options["initial_transform"], allow_scale=True))
    source_center, source_basis = _principal_frame(source)
    target_center, target_basis = _principal_frame(target)
    for order, signs in product(permutations(range(3)), product((-1, 1), repeat=3)):
        axes = np.eye(3)[:, order] * np.asarray(signs)
        if np.linalg.det(axes) < 0:
            continue
        rotation = target_basis @ axes @ source_basis.T
        for multiplier in options.get("scale_candidates", DEFAULT_SCALE_CANDIDATES):
            candidates.append(_similarity_matrix(base_scale * multiplier * rotation, source_center, target_center))
    return candidates


def _rank_candidates(
    candidates: list[np.ndarray],
    points: _PointPair,
    settings: _IcpSettings,
    radius: float,
    target_size: float,
) -> list[np.ndarray]:
    """对每个初值做粗 ICP，按排序分数从好到差返回结果矩阵（同分保持原顺序）。"""
    ranked = []
    for initial in candidates:
        matrix = _run_icp(points.source, points.target, initial, settings)
        quality = _measure_alignment(points.source, points.target, matrix, radius, settings.trim)
        ranked.append((quality.rank(target_size), matrix))
    ranked.sort(key=lambda item: item[0])
    return [matrix for _, matrix in ranked]


def _refine_best_candidate(
    candidates: list[np.ndarray],
    points: _PointPair,
    settings: _IcpSettings,
    radius: float,
    target_size: float,
) -> tuple[np.ndarray, _AlignmentQuality]:
    """带尺度精 ICP 后再做固定尺度刚体 ICP，返回排序分数最好的变换和质量。"""
    rigid_settings = _IcpSettings(settings.scale_bounds, RIGID_POLISH_ITERATIONS, settings.trim, with_scale=False)
    refined = []
    for initial in candidates:
        matrix = _run_icp(points.source, points.target, initial, settings)
        matrix = _run_icp(points.source, points.target, matrix, rigid_settings)
        quality = _measure_alignment(points.source, points.target, matrix, radius, settings.trim)
        refined.append((quality.rank(target_size), matrix, quality))
    refined.sort(key=lambda item: item[0])
    _, transform, quality = refined[0]
    return transform, quality


def _registration_result(transform: np.ndarray, quality: _AlignmentQuality, target_size: float, radius: float, options: Mapping[str, Any]) -> dict[str, Any]:
    validate_transform(transform, allow_scale=True)
    scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
    rotation = transform[:3, :3] / scale
    metrics = {
        "rmse": quality.rmse,
        "rmse_ratio": quality.rmse / target_size,
        "fitness": quality.fitness,
        "observed_surface_coverage": quality.fitness,
        "source_support_fraction": quality.support,
        "target_size": target_size,
        "correspondence_distance": radius,
        "method": REGISTRATION_METHOD,
        "orientation_ambiguity_checked": False,
        "silhouette_checked": False,
    }
    has_coverage = quality.fitness >= options.get("min_fitness", DEFAULT_MIN_FITNESS)
    has_low_error = quality.rmse / target_size <= options.get("max_rmse_ratio", DEFAULT_MAX_RMSE_RATIO)
    has_support = quality.support >= options.get("min_source_support", DEFAULT_MIN_SOURCE_SUPPORT)
    return {
        "transform": transform,
        "scale": scale,
        "rotation_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(),
        "translation_xyz": transform[:3, 3].tolist(),
        "metrics": metrics,
        "accepted": bool(has_coverage and has_low_error and has_support),
    }


def _measure_alignment(source: np.ndarray, target: np.ndarray, matrix: np.ndarray, radius: float, trim: float) -> _AlignmentQuality:
    """截断 RMSE、观测覆盖率（fitness）和资产支撑率（support）。"""
    transformed = _apply_transform(source, matrix)
    distances = cKDTree(transformed).query(target, k=1)[0]
    cutoff = np.quantile(distances, trim)
    trimmed = distances[distances <= cutoff + QUALITY_TRIM_EPSILON]
    source_distances = cKDTree(target).query(transformed, k=1)[0]
    return _AlignmentQuality(
        rmse=float(np.sqrt(np.mean(trimmed**2))),
        fitness=float(np.mean(distances <= radius)),
        support=float(np.mean(source_distances <= radius)),
    )


def _run_icp(source: np.ndarray, target: np.ndarray, initial: np.ndarray, settings: _IcpSettings) -> np.ndarray:
    """截断鲁棒加权 ICP：目标点找最近的变换后源点；尺度越界时退回刚体增量。"""
    matrix = initial.copy()
    last_error = np.inf
    for _ in range(settings.iterations):
        transformed = _apply_transform(source, matrix)
        distances, matches = cKDTree(transformed).query(target, k=1)
        keep = distances <= max(np.quantile(distances, settings.trim), MIN_TRIM_DISTANCE)
        if keep.sum() < MIN_ICP_MATCHES:
            break
        kept_distances = distances[keep]
        robust_weights = 1 / (1 + (kept_distances / max(np.median(kept_distances), MIN_MEDIAN_DISTANCE)) ** 2)
        matched_source, matched_target = transformed[matches[keep]], target[keep]
        try:
            increment = estimate_similarity(matched_source, matched_target, with_scale=settings.with_scale, weights=robust_weights)
        except ValueError:
            break
        proposal = increment @ matrix
        scale = np.cbrt(np.linalg.det(proposal[:3, :3]))
        if not settings.scale_bounds[0] <= scale <= settings.scale_bounds[1]:
            proposal = estimate_similarity(matched_source, matched_target, with_scale=False, weights=robust_weights) @ matrix
        matrix = proposal
        error = float(np.mean(kept_distances**2))
        if _has_converged(last_error, error):
            break
        last_error = error
    return matrix


def _has_converged(last_error: float, error: float) -> bool:
    return bool(np.isfinite(last_error) and abs(last_error - error) < max(CONVERGENCE_ABSOLUTE, last_error * CONVERGENCE_RELATIVE))
