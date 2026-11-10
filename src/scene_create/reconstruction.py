"""生成物体与场景：物体补全、资产生成、配准与位姿校验、背景重建和场景组装。"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import binary_dilation

from scene_create.gaussian import (
    initialize_gaussians,
    inspect_gaussians,
    merge_gaussians,
    transform_gaussians,
)
from scene_create.geometry import (
    extent,
    load_triangle_mesh,
    map_mask,
    project_points,
    rasterize_depth,
    register_points,
    sample_mesh,
    voxel_fuse,
)
from scene_create.io import (
    atomic_json,
    load_mask,
    load_npz,
    read_cloud,
    save_image,
    write_cloud,
)

if TYPE_CHECKING:
    from scene_create.runtime import StageContext

# -------------------------------------------
# 配置常量
# -------------------------------------------
# fmt: off
MIN_CROP_SIDE_PIXELS = 8                         # 参考图裁剪的最小正方形边长（像素）
MIN_COMPLETION_IMAGE_SIZE = 64                   # 送入补全模型的最小图像边长（像素）
CROP_BACKGROUND_RGB = (255, 255, 255)            # 物体裁剪图之外区域填充的纯白背景

DEFAULT_QUALITY_MAX_SIDE = 256                   # 位姿校验栅格化的默认最长边（像素）
DEFAULT_OCCLUSION_TOLERANCE_RATIO = 0.05         # 渲染深度比观测深度更远时仍视为可见的相对容差
DEFAULT_MIN_MASK_IOU = 0.25                      # 位姿校验要求的最小平均可见轮廓 IoU
DEFAULT_MAX_RELATIVE_DEPTH_ERROR = 0.15          # 位姿校验允许的最大相对深度误差中位数
DEPTH_EPSILON = 1e-6                             # 深度相除时避免除零的下限

OVERLAY_IMAGE_WEIGHT = 0.6                       # 配准叠加图中原图的亮度权重
OVERLAY_OBSERVED_RGB = (0, 80, 0)                # 叠加图中观测物体轮廓的颜色增量（绿色）
OVERLAY_PREDICTED_RGB = (80, 0, 35)              # 叠加图中资产投影轮廓的颜色增量（红紫色）

FOREGROUND_DEPTH_TOLERANCE_RATIO = 0.03          # 物体点投影与观测深度一致时视为可见的相对容差
MIN_BACKGROUND_POINTS = 3                        # 构建背景 Gaussian 所需的最少点数
# fmt: on


# -------------------------------------------
# 物体补全
# -------------------------------------------
def crop_reference(view: Mapping, mask_path: Union[str, Path], destination: Path, config: Mapping) -> tuple[dict, list[Path]]:
    """以物体 mask 为中心裁剪参考视图，生成白底补全输入。

    Args:
        view: 输入视图记录，包含 `image_path`、`width`、`height`、`view_id`。
        mask_path: 原图尺寸的物体 mask。
        destination: 裁剪图及元数据的输出目录。
        config: `completion` 配置，包含 `padding` 和 `image_size`。

    Returns:
        补全请求所需的图像与 mask 路径，以及写出的产物路径。

    Raises:
        ValueError: mask 为空，或 `image_size` 小于下限。
    """
    mask = load_mask(mask_path, (view["width"], view["height"]))
    left, top, side = _square_crop_box(mask, config["padding"])
    box = (left, top, left + side, top + side)
    with Image.open(view["image_path"]) as image:
        crop = image.convert("RGB").crop(box)
    alpha = Image.fromarray((mask * 255).astype(np.uint8)).crop(box)
    white = Image.new("RGB", crop.size, CROP_BACKGROUND_RGB)
    white.paste(crop, mask=alpha)

    size = int(config["image_size"])
    if size < MIN_COMPLETION_IMAGE_SIZE:
        raise ValueError(f"Completion image_size must be at least {MIN_COMPLETION_IMAGE_SIZE}")
    white = white.resize((size, size), Image.Resampling.LANCZOS)
    alpha = alpha.resize((size, size), Image.Resampling.NEAREST)

    image_path = destination / "crop.png"
    mask_output = destination / "visible_mask.png"
    context_path = destination / "crop_context.png"
    metadata_path = destination / "crop_meta.json"
    save_image(image_path, white)
    save_image(mask_output, alpha)
    save_image(context_path, crop)
    metadata = {
        "view_id": view["view_id"],
        "bbox_xyxy": box,
        "size": size,
        "original_to_crop": _original_to_crop_matrix(left, top, size / side),
        "observed_pixels_only": True,
    }
    atomic_json(metadata_path, metadata)
    request = {"image_path": str(image_path), "mask_path": str(mask_output)}
    return request, [image_path, mask_output, context_path, metadata_path]


def _square_crop_box(mask: np.ndarray, padding: float) -> tuple[int, int, int]:
    """返回以 mask 包围盒为中心、按 padding 外扩的正方形左上角和边长。"""
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("Cannot crop an empty object mask")
    longest_side = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    side = max(MIN_CROP_SIDE_PIXELS, int(np.ceil(longest_side * (1 + 2 * padding))))
    left = int(np.floor((xs.min() + xs.max() + 1 - side) / 2))
    top = int(np.floor((ys.min() + ys.max() + 1 - side) / 2))
    return left, top, side


def _original_to_crop_matrix(left: int, top: int, scale: float) -> list[list[float]]:
    """原图像素中心到缩放后裁剪图像素中心的仿射矩阵。"""
    offset = (scale - 1) / 2
    return [
        [scale, 0, -left * scale + offset],
        [0, scale, -top * scale + offset],
        [0, 0, 1],
    ]


def complete_object(context: StageContext, item: dict) -> tuple[dict, list]:
    """裁剪物体参考视图并调用 FLUX.2 生成完整物体图像。

    Raises:
        ValueError: FLUX.2 没有返回可读取的补全图像。
    """
    directory = context.output / "objects" / item["object_id"] / "images"
    views = {view["view_id"]: view for view in context.results["ingest"]["views"]}
    request_object, artifacts = crop_reference(
        views[item["reference_view_id"]],
        item["reference_mask_path"],
        directory,
        context.config["completion"],
    )
    request_object.update(object_id=item["object_id"], category=item["category"], reference_images=[])
    completed = context.models.call("flux2", {"object": request_object}, directory)

    image_path = Path(completed["image_path"])
    if not image_path.is_file():
        raise ValueError("FLUX.2 returned no completed image")
    with Image.open(image_path) as image:
        image.verify()

    metadata = directory / "completion_meta.json"
    atomic_json(
        metadata,
        {
            **completed,
            "synthetic": True,
            "semantic_quality_verified": False,
            "source_view": item["reference_view_id"],
        },
    )
    update = {
        "status": "completed_image",
        "completed_image_path": str(image_path),
        "completion_metadata_path": str(metadata),
    }
    return update, artifacts + [image_path, metadata]


# -------------------------------------------
# 资产生成
# -------------------------------------------
def generate_asset(context: StageContext, item: dict) -> tuple[dict, list]:
    """调用 TRELLIS 从补全图像生成同坐标系的 mesh 与 Gaussian 资产。

    Raises:
        ValueError: TRELLIS 产物缺失，或 Gaussian 资产为空。
    """
    directory = context.output / "objects" / item["object_id"] / "assets"
    request_object = {
        "object_id": item["object_id"],
        "category": item["category"],
        "image_path": item["completed_image_path"],
        "mask_path": None,
        "reference_images": [],
    }
    asset = context.models.call("trellis", {"object": request_object}, directory)
    files = [asset[key] for key in ("mesh_path", "gaussian_path", "metadata_path")]
    if not all(Path(path).is_file() for path in files):
        raise ValueError("TRELLIS did not return mesh, Gaussian and metadata files")
    if inspect_gaussians(asset["gaussian_path"])["count"] <= 0:
        raise ValueError("TRELLIS returned an empty Gaussian asset")
    return {"status": "generated_asset", "asset": asset}, files


# -------------------------------------------
# 资产配准
# -------------------------------------------
def register_object(context: StageContext, item: dict) -> tuple[dict, list]:
    """把生成资产以 Sim(3) 配准到观测点云，并写出对齐后的 mesh 与 Gaussian。

    Raises:
        ValueError: 几何配准或重投影校验未通过；变换与诊断文件仍会写出。
    """
    config, output = context.config, context.output
    directory = output / "objects" / item["object_id"] / "registration"
    directory.mkdir(parents=True, exist_ok=True)
    seed = config["runtime"]["seed"]
    sampled = sample_mesh(item["asset"]["mesh_path"], count=config["registration"]["sample_count"], seed=seed)
    target = read_cloud(item["cloud_path"])
    registered = register_points(sampled["points"], target["points"], config={**config["registration"], "seed": seed})

    transform = np.asarray(registered["transform"], dtype=np.float64)
    reprojection, previews = validate_asset_pose(item, transform, output, config["registration"])
    registered["metrics"].update(reprojection)
    registered["accepted"] = registered["accepted"] and reprojection["reprojection_accepted"]
    metadata = _pose_metadata(registered, transform)

    transform_path = directory / "transform.json"
    atomic_json(transform_path, metadata)
    samples_path = directory / "mesh_samples.ply"
    write_cloud(samples_path, sampled["points"])
    if not registered["accepted"]:
        raise ValueError(f"Registration quality rejected: {registered['metrics']}; see {transform_path}")

    aligned_mesh, aligned_gaussian, aligned_samples = _write_aligned_assets(item["asset"], sampled["points"], transform, directory)
    update = {
        "status": "registered",
        "transform": metadata,
        "transform_path": str(transform_path),
        "aligned_mesh_path": str(aligned_mesh),
        "aligned_gaussian_path": str(aligned_gaussian),
    }
    return update, [transform_path, samples_path, aligned_mesh, aligned_gaussian, aligned_samples, *previews]


def _pose_metadata(registered: Mapping, transform: np.ndarray) -> dict:
    """配准结果中写入 `transform.json` 的位姿与质量字段。"""
    return {
        "T_scene_from_asset": transform,
        "translation_xyz": registered["translation_xyz"],
        "rotation_quaternion_xyzw": registered["rotation_xyzw"],
        "scale": registered["scale"],
        "metrics": registered["metrics"],
        "accepted": registered["accepted"],
        "source_frame": "asset",
        "target_frame": "first_input_camera_opencv",
        "scale_is_uniform": True,
    }


def _write_aligned_assets(asset: Mapping, sample_points: np.ndarray, transform: np.ndarray, directory: Path) -> tuple[Path, Path, Path]:
    """把世界变换烘焙进 mesh、Gaussian 和采样点，返回三者路径。"""
    aligned_mesh = directory / "aligned_mesh.glb"
    mesh = trimesh.load(asset["mesh_path"], force="scene", process=False)
    mesh.apply_transform(transform)
    mesh.export(str(aligned_mesh))

    aligned_gaussian = directory / "aligned_gaussian.ply"
    transform_gaussians(asset["gaussian_path"], aligned_gaussian, transform)

    aligned_samples = directory / "aligned_samples.ply"
    write_cloud(aligned_samples, sample_points @ transform[:3, :3].T + transform[:3, 3])
    return aligned_mesh, aligned_gaussian, aligned_samples


# -------------------------------------------
# 位姿校验
# -------------------------------------------
def validate_asset_pose(item: Mapping, transform: np.ndarray, output: Path, config: Mapping) -> tuple[dict, list[Path]]:
    """在每个观测视图中比较资产投影与观测 mask/深度，判断位姿是否可接受。

    Args:
        item: 物体记录，包含 `object_id`、`asset.mesh_path` 和 `observations`。
        transform: 资产到场景的 4x4 Sim(3) 变换。
        output: 流水线输出根目录，用于读取 `pointmaps/` 并写出叠加图。
        config: `registration` 配置；缺省项使用模块默认阈值。

    Returns:
        轮廓 IoU、深度误差和是否接受的汇总指标，以及各视图叠加图路径。
    """
    mesh = load_triangle_mesh(item["asset"]["mesh_path"], transform)
    metrics, artifacts = [], []
    for observation in item["observations"]:
        overlay_path = output / "objects" / item["object_id"] / "registration" / f"{observation['view_id']}_alignment.png"
        metrics.append(_check_view_alignment(observation, mesh, output, overlay_path, config))
        artifacts.append(overlay_path)
    return _summarize_pose_metrics(metrics, config), artifacts


def _check_view_alignment(observation: Mapping, mesh: trimesh.Trimesh, output: Path, overlay_path: Path, config: Mapping) -> dict:
    """计算单个视图的可见轮廓 IoU 和相对深度误差，并写出叠加诊断图。"""
    view_id = observation["view_id"]
    data = load_npz(output / "pointmaps" / f"{view_id}.npz")
    height, width = data["depth"].shape
    max_side = int(config.get("quality_max_side", DEFAULT_QUALITY_MAX_SIDE))
    rendered = rasterize_depth(mesh.vertices, mesh.faces, data["intrinsics"], data["world_from_camera"], width, height, max_side=max_side)

    shape = rendered["depth"].shape
    raster_size = (rendered["width"], rendered["height"])
    target = map_mask(load_mask(observation["geometry_mask_path"]), rendered["pixel_map"], shape)
    valid = map_mask(data["valid"], rendered["pixel_map"], shape)
    reference_depth = np.asarray(Image.fromarray(data["depth"].astype(np.float32), mode="F").resize(raster_size, Image.Resampling.NEAREST))

    tolerance_ratio = float(config.get("occlusion_tolerance_ratio", DEFAULT_OCCLUSION_TOLERANCE_RATIO))
    tolerance = tolerance_ratio * np.maximum(reference_depth, DEPTH_EPSILON)
    is_unoccluded = rendered["depth"] <= reference_depth + tolerance
    predicted = rendered["mask"] & valid & is_unoccluded
    expected = target & valid
    overlap = predicted & expected
    union = predicted | expected

    residual = np.abs(rendered["depth"][overlap] - reference_depth[overlap]) / np.maximum(reference_depth[overlap], DEPTH_EPSILON)
    depth_error = float(np.median(residual)) if len(residual) else None
    image = Image.fromarray(data["image"]).resize(raster_size, Image.Resampling.BILINEAR)
    _save_alignment_overlay(overlay_path, np.asarray(image), expected, predicted)
    return {
        "view_id": view_id,
        "visible_mask_iou": float(overlap.sum() / max(int(union.sum()), 1)),
        "median_relative_depth_error": depth_error,
        "observed_pixels": int(expected.sum()),
        "overlap_pixels": int(overlap.sum()),
    }


def _save_alignment_overlay(path: Path, image: np.ndarray, expected: np.ndarray, predicted: np.ndarray) -> None:
    """把观测轮廓（绿）与资产投影（红紫）叠加到变暗的原图上。"""
    overlay = image.astype(float) * OVERLAY_IMAGE_WEIGHT
    overlay[expected] += np.array(OVERLAY_OBSERVED_RGB)
    overlay[predicted] += np.array(OVERLAY_PREDICTED_RGB)
    save_image(path, np.clip(overlay, 0, 255).astype(np.uint8))


def _summarize_pose_metrics(metrics: Sequence[Mapping], config: Mapping) -> dict:
    """汇总有观测像素的视图，按 IoU 与深度误差阈值判定是否接受位姿。"""
    usable = [metric for metric in metrics if metric["observed_pixels"] > 0]
    mean_iou = float(np.mean([metric["visible_mask_iou"] for metric in usable])) if usable else 0.0
    depth_errors = [metric["median_relative_depth_error"] for metric in usable if metric["median_relative_depth_error"] is not None]
    median_depth_error = float(np.median(depth_errors)) if depth_errors else None

    min_mask_iou = float(config.get("min_mask_iou", DEFAULT_MIN_MASK_IOU))
    max_depth_error = float(config.get("max_relative_depth_error", DEFAULT_MAX_RELATIVE_DEPTH_ERROR))
    has_silhouette_match = bool(usable) and mean_iou >= min_mask_iou
    has_depth_match = median_depth_error is not None and median_depth_error <= max_depth_error
    return {
        "silhouette_checked": True,
        "mean_visible_mask_iou": mean_iou,
        "median_relative_depth_error": median_depth_error,
        "per_view": list(metrics),
        "reprojection_accepted": has_silhouette_match and has_depth_match,
        "preview_kind": "mask_alignment_overlay_not_photorealistic_render",
    }


# -------------------------------------------
# 背景重建
# -------------------------------------------
@dataclass
class BackgroundSamples:
    """移除已配准物体后，各视图收集到的背景点及训练视图。"""

    points: list[np.ndarray] = field(default_factory=list)
    colors: list[np.ndarray] = field(default_factory=list)
    weights: list[np.ndarray] = field(default_factory=list)
    training_views: list[dict] = field(default_factory=list)
    mask_paths: list[Path] = field(default_factory=list)


def reconstruct_background(context: StageContext) -> tuple[dict, list]:
    """从各视图移除已配准物体，融合背景点并初始化或训练 MaskGaussian 背景。

    Raises:
        ValueError: MaskGaussian 训练没有返回非空的 Gaussian PLY。
    """
    config, output = context.config, context.output
    registered, geometry = context.results["register"], context.results["depth"]
    directory = output / "background"
    directory.mkdir(parents=True, exist_ok=True)
    accepted = _registered_objects(registered)
    samples = _collect_background_samples(geometry["views"], accepted, config["background"], directory)
    xyz = np.concatenate(samples.points)
    rgb = np.concatenate(samples.colors)
    confidence = np.concatenate(samples.weights)
    artifacts = list(samples.mask_paths)
    result = {
        "removed_object_ids": [item["object_id"] for item in accepted],
        "single_view": len(geometry["views"]) == 1,
        "unobserved_background_completed": False,
    }
    if len(xyz) >= MIN_BACKGROUND_POINTS:
        xyz, rgb = _fuse_background_points(xyz, rgb, confidence, config)
        provenance = {"scale_status": geometry["scale_status"], "removed_object_ids": result["removed_object_ids"]}
        initial_gaussian, dataset_artifacts = _write_background_dataset(directory, xyz, rgb, samples.training_views, provenance)
        update, gaussian_artifacts = _build_background_gaussian(context, directory, initial_gaussian, result["single_view"])
        result.update(update)
        artifacts.extend(dataset_artifacts + gaussian_artifacts)
    else:
        result.update(status="empty", gaussian_path=None, trained=False, approximate=False)
    metadata = directory / "metadata.json"
    atomic_json(metadata, result)
    return result, artifacts + [metadata]


def _registered_objects(registered: Mapping) -> list[dict]:
    """已通过配准、作为独立资产放入场景的物体。"""
    return [item for item in registered["objects"] if item["status"] == "registered"]


def _collect_background_samples(views: Sequence[Mapping], accepted: Sequence[Mapping], settings: Mapping, directory: Path) -> BackgroundSamples:
    """逐视图剔除物体前景，写出背景有效 mask 并收集背景点。"""
    clouds = {item["object_id"]: read_cloud(item["cloud_path"])["points"] for item in accepted}
    dilation = int(settings["mask_dilation"])
    samples = BackgroundSamples()
    for view in views:
        data = load_npz(view["geometry_path"])
        foreground = _object_foreground(view["view_id"], data, accepted, clouds)
        if dilation > 0 and foreground.any():
            foreground = binary_dilation(foreground, iterations=dilation)
        valid = data["valid"] & ~foreground & np.isfinite(data["points_world"]).all(axis=-1)

        mask_path = directory / "masks" / f"{view['view_id']}.png"
        save_image(mask_path, (valid * 255).astype(np.uint8))
        samples.mask_paths.append(mask_path)
        samples.training_views.append(
            {
                "view_id": view["view_id"],
                "image_path": view["image_path"],
                "mask_path": str(mask_path),
                "geometry_path": view["geometry_path"],
            }
        )
        samples.points.append(data["points_world"][valid])
        samples.colors.append(data["image"][valid])
        samples.weights.append(np.maximum(data["conf"][valid], DEPTH_EPSILON))
    return samples


def _object_foreground(view_id: str, data: Mapping, accepted: Sequence[Mapping], clouds: Mapping) -> np.ndarray:
    """已配准物体在该视图中的前景：有观测时用其 mask，否则投影其观测点云。"""
    foreground = np.zeros(data["depth"].shape, dtype=bool)
    for item in accepted:
        observations = [observation for observation in item["observations"] if observation["view_id"] == view_id]
        if observations:
            foreground |= load_mask(observations[0]["geometry_mask_path"])
            continue
        rows, columns = _visible_projected_pixels(clouds[item["object_id"]], data)
        foreground[rows, columns] = True
    return foreground


def _visible_projected_pixels(points: np.ndarray, data: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """投影到图像内且与观测深度一致的点所在的像素行列。"""
    shape = data["depth"].shape
    uv, z = project_points(points, data["intrinsics"], data["world_from_camera"])
    is_finite = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0)
    u, v = np.zeros(len(z), dtype=int), np.zeros(len(z), dtype=int)
    u[is_finite], v[is_finite] = np.rint(uv[is_finite]).astype(int).T
    is_inside = is_finite & (u >= 0) & (v >= 0) & (u < shape[1]) & (v < shape[0])

    indices = np.flatnonzero(is_inside)
    depth_gap = np.abs(data["depth"][v[indices], u[indices]] - z[indices])
    is_visible = depth_gap <= FOREGROUND_DEPTH_TOLERANCE_RATIO * np.maximum(z[indices], DEPTH_EPSILON)
    indices = indices[is_visible]
    return v[indices], u[indices]


def _fuse_background_points(xyz: np.ndarray, rgb: np.ndarray, confidence: np.ndarray, config: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """按体素加权融合背景点，并以固定种子下采样到 `max_points`。"""
    settings = config["background"]
    voxel = settings["voxel_size"] or extent(xyz) * settings["voxel_ratio"]
    fused = voxel_fuse(xyz, rgb, confidence, voxel)
    xyz, rgb = fused["points"], fused["colors"]
    limit = settings["max_points"]
    if len(xyz) > limit:
        indices = np.random.default_rng(config["runtime"]["seed"]).choice(len(xyz), limit, replace=False)
        xyz, rgb = xyz[indices], rgb[indices]
    return xyz, rgb


def _write_background_dataset(directory: Path, xyz: np.ndarray, rgb: np.ndarray, training_views: Sequence[Mapping], provenance: Mapping) -> tuple[Path, list[Path]]:
    """写出初始点云、初始 Gaussian 和训练数据集清单，返回初始 Gaussian 及全部产物。

    Args:
        directory: 背景输出目录。
        xyz: 融合后的背景点坐标。
        rgb: 与 `xyz` 对应的 uint8 颜色。
        training_views: 每个视图的图像、背景 mask 与几何路径。
        provenance: 写入数据集的 `scale_status` 与 `removed_object_ids`。
    """
    initial_ply = directory / "initial_points.ply"
    initial_gaussian = directory / "initial_gaussian.ply"
    write_cloud(initial_ply, xyz, rgb)
    initialize_gaussians(xyz, rgb, initial_gaussian)
    dataset = directory / "dataset.json"
    atomic_json(
        dataset,
        {
            "schema_version": 1,
            "initial_ply": str(initial_ply),
            "views": list(training_views),
            "world_frame": "first_input_camera_opencv",
            "scale_status": provenance["scale_status"],
            "removed_object_ids": list(provenance["removed_object_ids"]),
        },
    )
    return initial_gaussian, [initial_ply, initial_gaussian, dataset]


def _build_background_gaussian(context: StageContext, directory: Path, initial_gaussian: Path, is_single_view: bool) -> tuple[dict, list]:
    """按背景模式选用初始 Gaussian 或训练 MaskGaussian，返回背景结果字段和新增产物。"""
    if context.config["background"]["mode"] == "init-only":
        update = {"status": "initialization_only", "gaussian_path": str(initial_gaussian), "trained": False, "approximate": True}
        return update, []
    dataset = directory / "dataset.json"
    trained = context.models.call("maskgaussian", {"dataset_path": str(dataset)}, directory)
    gaussian_path = trained.get("gaussian_path", trained.get("background_gaussian_path"))
    if not gaussian_path or not Path(gaussian_path).is_file():
        raise ValueError("MaskGaussian returned no Gaussian PLY")
    if inspect_gaussians(gaussian_path)["count"] <= 0:
        raise ValueError("MaskGaussian returned an empty background")
    update = {
        "status": "single_view_approximate" if is_single_view else "trained",
        "gaussian_path": gaussian_path,
        "trained": True,
        "approximate": is_single_view,
        "training": trained,
    }
    return update, [gaussian_path]


# -------------------------------------------
# 场景组装
# -------------------------------------------
def assemble_scene(context: StageContext) -> tuple[dict, list]:
    """合并已配准物体 mesh、背景与物体 Gaussian，写出场景清单和质量报告。"""
    output = context.output
    registered, background = context.results["register"], context.results["background"]
    geometry = context.results["depth"]
    directory = output / "scene"
    directory.mkdir(parents=True, exist_ok=True)
    accepted = _registered_objects(registered)
    objects = [_scene_object_record(item, directory) for item in registered["objects"]]

    mesh_path = _export_object_meshes(accepted, directory)
    gs_path, ranges = _merge_scene_gaussians(background, accepted, directory)
    artifacts = [path for path in (mesh_path, gs_path) if path is not None]

    failures = registered.get("failures", [])
    scene = {
        "schema_version": 1,
        "world_frame": geometry["world_frame"],
        "scale_status": geometry["scale_status"],
        "objects": objects,
        "background": {**background, "gaussian_path": _relative_path(background.get("gaussian_path"), directory)},
        "gaussian_scene": _relative_path(gs_path, directory),
        "gaussian_ranges": ranges,
        "default_representation": "gaussian",
        "cameras": _scene_cameras(geometry["views"]),
        "failures": failures,
        "quality": _scene_quality(len(geometry["views"]), len(accepted), len(objects)),
    }
    scene_path = directory / "scene.json"
    report_path = output / "reports" / "quality.json"
    atomic_json(scene_path, scene)
    atomic_json(report_path, _quality_report(scene, registered, background))
    result = {
        **{key: scene["quality"][key] for key in ("registered_objects", "total_objects")},
        "scene_path": str(scene_path),
        "gaussian_path": str(gs_path) if gs_path else None,
        "failures": failures,
        "approximate_background": background.get("approximate", False),
    }
    return result, artifacts + [scene_path, report_path]


def _relative_path(value: Optional[Union[str, Path]], directory: Path) -> Optional[str]:
    """把产物路径写成相对 `scene/` 目录的形式；空值保持为 `None`。"""
    if not value:
        return None
    # Path.relative_to 在 Python 3.9 中无法生成 `..`，而产物通常位于 scene/ 之外。
    return os.path.relpath(value, directory)


def _scene_object_record(item: Mapping, directory: Path) -> dict:
    """场景清单中的物体记录；未配准物体只保留观测并标记为留在背景中。"""
    is_registered = item["status"] == "registered"
    record = {
        "object_id": item["object_id"],
        "category": item["category"],
        "status": item["status"],
        "interactive": is_registered,
        "observed_cloud": _relative_path(item["cloud_path"], directory),
        "preserved_in_background": not is_registered,
    }
    if not is_registered:
        return record
    record.update(
        mesh=_relative_path(item["aligned_mesh_path"], directory),
        gaussian=_relative_path(item["aligned_gaussian_path"], directory),
        local_mesh=_relative_path(item["asset"]["mesh_path"], directory),
        local_gaussian=_relative_path(item["asset"]["gaussian_path"], directory),
        asset_to_world=item["transform"]["T_scene_from_asset"],
        render_transform=np.eye(4).tolist(),
        coordinates_baked=True,
        pose=item["transform"],
        generated_geometry=True,
        semantic_quality_verified=False,
    )
    return record


def _export_object_meshes(accepted: Sequence[Mapping], directory: Path) -> Optional[Path]:
    """把已对齐物体 mesh 合并导出为 `objects.glb`；没有物体时删除旧文件并返回 `None`。"""
    mesh_path = directory / "objects.glb"
    if not accepted:
        mesh_path.unlink(missing_ok=True)
        return None
    mesh_scene = trimesh.Scene()
    for item in accepted:
        mesh = trimesh.load(item["aligned_mesh_path"], force="scene", process=False)
        for index, node in enumerate(mesh.graph.nodes_geometry):
            transform, geometry_name = mesh.graph[node]
            name = f"{item['object_id']}_{index}"
            mesh_scene.add_geometry(mesh.geometry[geometry_name].copy(), geom_name=name, node_name=name, transform=transform)
    mesh_scene.export(str(mesh_path))
    return mesh_path


def _merge_scene_gaussians(background: Mapping, accepted: Sequence[Mapping], directory: Path) -> tuple[Optional[Path], list[dict]]:
    """按背景在前、物体在后的顺序合并 Gaussian，返回场景 PLY 和每个所有者的实例范围。

    没有任何 Gaussian 时删除旧的场景 PLY 并返回 `(None, [])`。
    """
    sources = [(item["object_id"], item["aligned_gaussian_path"]) for item in accepted]
    if background.get("gaussian_path"):
        sources.insert(0, ("background", background["gaussian_path"]))
    gs_path = directory / "scene_gaussian.ply"
    if not sources:
        gs_path.unlink(missing_ok=True)
        return None, []
    ranges = merge_gaussians([path for _, path in sources], gs_path)
    for (owner, _), item in zip(sources, ranges):
        item["object_id"] = owner
        if "path" in item:
            item["path"] = _relative_path(item["path"], directory)
    return gs_path, ranges


def _scene_quality(input_views: int, registered_objects: int, total_objects: int) -> dict:
    """场景质量摘要；没有真值，因此不声明任何准确率。"""
    return {
        "ground_truth_evaluated": False,
        "hidden_surfaces_are_generated_or_unknown": True,
        "input_views": input_views,
        "registered_objects": registered_objects,
        "total_objects": total_objects,
    }


def _scene_cameras(views: Sequence[Mapping]) -> list[dict]:
    """场景清单中记录的相机尺寸、内参与位姿。"""
    keys = ("view_id", "width", "height", "intrinsics", "world_from_camera")
    return [{key: view[key] for key in keys} for view in views]


def _quality_report(scene: Mapping, registered: Mapping, background: Mapping) -> dict:
    """`reports/quality.json`：可计算的几何指标和失败信息。"""
    objects = []
    for item in registered["objects"]:
        metrics = item.get("transform", {}).get("metrics")
        objects.append({"object_id": item["object_id"], "status": item["status"], "metrics": metrics})
    return {
        **scene["quality"],
        "failures": scene["failures"],
        "background_status": background["status"],
        "objects": objects,
    }
