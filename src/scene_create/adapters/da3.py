"""Depth Anything 3 多视图联合几何适配器：输出光轴深度、置信度、相机和像素映射。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from scene_create.io import atomic_npz
from scene_create.runtime import configure_device, require_local_model, resolve_dtype

# -------------------------------------------
# 配置常量
# -------------------------------------------
# fmt: off
PATCH_SIZE = 14                         # DA3 主干的 patch 边长，处理分辨率必须是其整数倍，单位：像素
MIN_PROCESS_RESOLUTION = 28             # 允许的最小处理分辨率（两个 patch），单位：像素
DEFAULT_PROCESS_RESOLUTION = 504        # 默认处理分辨率，单位：像素
DEFAULT_RESIZE_METHOD = "upper_bound_resize"
RESIZE_METHODS = ("upper_bound_resize", "upper_bound_crop", "lower_bound_resize", "lower_bound_crop")
DEFAULT_MAX_VIEWS = 16                  # 一次联合推理允许的最大视图数
DEFAULT_REF_VIEW_STRATEGY = "saddle_balanced"
SKY_PROBABILITY_THRESHOLD = 0.5         # 天空概率不低于该值的像素视为天空
REQUIRED_MODEL_FILES = ("config.json", "model.safetensors")
# fmt: on


@dataclass(frozen=True)
class PredictionArrays:
    """经过形状校验的 DA3 联合预测数组，首维为视图。"""

    depths: np.ndarray
    confidences: np.ndarray
    images: np.ndarray
    intrinsics: np.ndarray
    scene_from_camera: np.ndarray
    sky: Optional[np.ndarray]
    scale_status: str


# -------------------------------------------
# 像素映射与尺度判定
# -------------------------------------------
def preprocessing_maps(
    sizes: Sequence[tuple[int, int]],
    process_res: int = DEFAULT_PROCESS_RESOLUTION,
    method: str = DEFAULT_RESIZE_METHOD,
) -> tuple[list[np.ndarray], tuple[int, int]]:
    """复现 DA3 预处理，计算原图像素中心到处理网格像素中心的仿射映射。

    Args:
        sizes: 每个视图的原图 `(width, height)`。
        process_res: 处理分辨率。
        method: DA3 的 `process_res_method`。

    Returns:
        每个视图的 3x3 映射矩阵，以及批内统一的处理网格 `(height, width)`。

    Raises:
        ValueError: 预处理配置不受支持，或某个维度小于一个 patch。
    """
    if method not in RESIZE_METHODS or process_res < MIN_PROCESS_RESOLUTION:
        raise ValueError("Unsupported DA3 preprocessing configuration")
    shapes, mappings = [], []
    for width, height in sizes:
        mapping, shape = _single_view_map(width, height, process_res, method)
        mappings.append(mapping)
        shapes.append(shape)
    target_h = min(shape[0] for shape in shapes)
    target_w = min(shape[1] for shape in shapes)
    # 批内尺寸不一致时，上游会中心裁剪到最小公共网格。
    for matrix, (height, width) in zip(mappings, shapes):
        matrix[0, 2] -= (width - target_w) // 2
        matrix[1, 2] -= (height - target_h) // 2
    return mappings, (target_h, target_w)


def _single_view_map(width: int, height: int, process_res: int, method: str) -> tuple[np.ndarray, tuple[int, int]]:
    denominator = max(width, height) if method.startswith("upper") else min(width, height)
    scale = process_res / denominator
    w1, h1 = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    if method.endswith("resize"):
        w2 = max(1, int(np.floor(w1 / PATCH_SIZE + 0.5)) * PATCH_SIZE)
        h2 = max(1, int(np.floor(h1 / PATCH_SIZE + 0.5)) * PATCH_SIZE)
        sx, sy, dx, dy = w2 / width, h2 / height, 0, 0
    else:
        w2, h2 = w1 // PATCH_SIZE * PATCH_SIZE, h1 // PATCH_SIZE * PATCH_SIZE
        sx, sy = w1 / width, h1 / height
        dx, dy = (w1 - w2) // 2, (h1 - h2) // 2
    if min(w2, h2) < PATCH_SIZE:
        raise ValueError("Image aspect ratio creates a DA3 dimension smaller than one patch")
    mapping = np.array(
        [[sx, 0, (sx - 1) / 2 - dx], [0, sy, (sy - 1) / 2 - dy], [0, 0, 1]],
        dtype=float,
    )
    return mapping, (h2, w2)


def prediction_scale_status(model_name: str, is_metric: Any) -> str:
    """只依据模型实际返回的 `is_metric` 判定尺度，不接受配置标签。

    Returns:
        `"model_metric_estimate"` 或 `"relative"`。

    Raises:
        ValueError: 标志不是标量布尔值，或 Nested 模型未确认米制。
    """
    metric = False
    if is_metric is not None:
        flag = np.asarray(is_metric)
        if flag.size != 1 or flag.item() not in (0, 1):
            raise ValueError("DA3 is_metric must be a scalar boolean flag")
        metric = bool(flag.item())
    if "nested" in model_name.lower() and not metric:
        raise ValueError("Nested DA3 did not confirm metric scaling in its output")
    return "model_metric_estimate" if metric else "relative"


# -------------------------------------------
# 模型加载
# -------------------------------------------
def configured_model_type(base: type, torch: Any, dtype_name: str) -> type:
    """返回显式控制主干 autocast 精度的 DA3 子类，避免被上游默认精度覆盖。"""

    # 基类只能在延迟导入 depth_anything_3 后获得，因此子类在函数内定义。
    class ConfiguredDA3(base):
        # 参数与上游 DepthAnything3.forward 保持一致，不能合并或删减。
        def forward(
            self,
            image,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=None,
            infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy=DEFAULT_REF_VIEW_STRATEGY,
        ):
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=image.device.type,
                    dtype=getattr(torch, dtype_name),
                    enabled=dtype_name != "float32",
                ),
            ):
                return self.model(
                    image,
                    extrinsics,
                    intrinsics,
                    export_feat_layers,
                    infer_gs,
                    use_ray_pose,
                    ref_view_strategy,
                )

    return ConfiguredDA3


def load(config: Mapping[str, Any], root: Path) -> SimpleNamespace:
    """加载 any-view 或 Nested DA3 本地模型。

    Raises:
        FileNotFoundError: 本地模型目录缺失或不完整。
        ValueError: 选择了不提供相机解的独立 Metric/Mono 模型。
    """
    model_path = require_local_model(
        config.get("model_path"),
        REQUIRED_MODEL_FILES,
        allow_download=config.get("allow_download", False),
    )

    # 延迟加载：主环境可以不安装 torch 和 DA3。
    import torch
    from depth_anything_3.api import DepthAnything3

    device = configure_device(torch, config.get("device", "cuda"))
    is_bf16_supported = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    dtype_name = resolve_dtype(config.get("dtype", "auto"), device, bf16_supported=is_bf16_supported)
    torch.manual_seed(config.get("seed", 0))
    model = configured_model_type(DepthAnything3, torch, dtype_name).from_pretrained(
        model_path,
        local_files_only=not config.get("allow_download", False),
    )
    name = str(model.model_name).lower()
    if "metric" in name or "mono" in name:
        raise ValueError("Standalone DA3 metric/mono checkpoints do not supply the required joint camera solution; use an any-view or nested checkpoint")
    model = model.to(device=device)
    model.eval()
    return SimpleNamespace(
        torch=torch,
        network=model,
        name=name,
        device=device,
        dtype_name=dtype_name,
        checkpoint=model_path,
    )


# -------------------------------------------
# 推理
# -------------------------------------------
def run(model: SimpleNamespace, request: Mapping[str, Any]) -> dict:
    """对全部视图做一次联合推理，逐视图写出几何 NPZ。

    Returns:
        `views`（每项含 `view_id`、`geometry_path`、`scale_status`、`scale_source`）和 `provenance`。

    Raises:
        ValueError: 视图数超限，或模型输出与预期像素网格/相机格式不一致。
    """
    config, views = request["config"], request["views"]
    if not views or len(views) > int(config.get("max_views", DEFAULT_MAX_VIEWS)):
        raise ValueError("DA3 view count exceeds max_views; independent batches cannot be fused without inter-batch alignment")
    method = config.get("process_res_method", DEFAULT_RESIZE_METHOD)
    resolution = int(config.get("process_res", DEFAULT_PROCESS_RESOLUTION))
    mappings, shape = preprocessing_maps([(view["width"], view["height"]) for view in views], resolution, method)
    with model.torch.inference_mode():
        prediction = model.network.inference(
            image=[view["image_path"] for view in views],
            process_res=resolution,
            process_res_method=method,
            infer_gs=False,
            use_ray_pose=bool(config.get("use_ray_pose", False)),
            ref_view_strategy=config.get("ref_view_strategy", DEFAULT_REF_VIEW_STRATEGY),
            export_dir=None,
        )
    arrays = _validated_arrays(prediction, len(views), shape, model.name)
    scale_status = arrays.scale_status
    records = []
    for index, view in enumerate(views):
        path = _write_view_geometry(arrays, index, mappings[index], config, Path(request["output_dir"]) / f"{view['view_id']}.npz")
        records.append({"view_id": view["view_id"], "geometry_path": str(path), "scale_status": scale_status, "scale_source": model.name})
    return {"views": records, "provenance": _provenance(model, config, arrays, getattr(prediction, "scale_factor", None))}


def _provenance(model: SimpleNamespace, config: Mapping[str, Any], arrays: PredictionArrays, scale_factor: Any) -> dict:
    return {
        "model": model.name,
        "checkpoint": model.checkpoint,
        "device": model.device,
        "requested_dtype": config.get("dtype", "auto"),
        "effective_backbone_dtype": model.dtype_name,
        "geometry_head_dtype": "float32",
        "is_metric": arrays.scale_status == "model_metric_estimate",
        "reported_scale_factor": scale_factor,
        "sky_filtered": bool(config.get("exclude_sky", True) and arrays.sky is not None),
        "joint_inference": True,
        "depth_convention": "optical_axis",
        "mask_mapping_convention": "opencv_resize_half_pixel",
        "camera_convention": "opencv",
    }


def _validated_arrays(prediction: Any, view_count: int, shape: tuple[int, int], model_name: str) -> PredictionArrays:
    # 校验顺序与报错优先级保持和上游输出字段的依赖顺序一致。
    has_camera_and_confidence = prediction.intrinsics is not None and prediction.extrinsics is not None and prediction.conf is not None
    if not has_camera_and_confidence:
        raise ValueError("DA3 checkpoint did not produce intrinsics, extrinsics and confidence")
    depths = np.asarray(prediction.depth)
    if depths.shape != (view_count, *shape):
        raise ValueError(f"DA3 pixel mapping changed: predicted {depths.shape}, expected {(view_count, *shape)}; revalidate upstream preprocessing")
    scene_from_camera = _scene_from_camera(np.asarray(prediction.extrinsics), view_count)
    scale_status = prediction_scale_status(model_name, getattr(prediction, "is_metric", None))
    confidences = np.asarray(prediction.conf, dtype=np.float32)
    images = np.asarray(prediction.processed_images)
    is_same_grid = confidences.shape == depths.shape and images.shape == (*depths.shape, 3) and images.dtype == np.uint8
    if not is_same_grid:
        raise ValueError("DA3 depth, confidence and uint8 RGB must share the same pixel grid")
    sky = getattr(prediction, "sky", None)
    if sky is not None and np.asarray(sky).shape != depths.shape:
        raise ValueError("DA3 sky mask has an unexpected pixel grid")
    return PredictionArrays(
        depths=depths,
        confidences=confidences,
        images=images,
        intrinsics=prediction.intrinsics,
        scene_from_camera=scene_from_camera,
        sky=sky,
        scale_status=scale_status,
    )


def _scene_from_camera(extrinsics: np.ndarray, view_count: int) -> np.ndarray:
    # DA3 输出 world-to-camera；统一固定到第一视图相机系并求逆为 camera-to-scene。
    if extrinsics.shape == (view_count, 3, 4):
        matrices = np.broadcast_to(np.eye(4), (view_count, 4, 4)).copy()
        matrices[:, :3] = extrinsics
        extrinsics = matrices
    if extrinsics.shape != (view_count, 4, 4):
        raise ValueError("Unexpected DA3 extrinsics shape")
    return extrinsics[0] @ np.linalg.inv(extrinsics)


def _write_view_geometry(arrays: PredictionArrays, index: int, mapping: np.ndarray, config: Mapping[str, Any], path: Path) -> Path:
    confidence = arrays.confidences[index]
    depth = arrays.depths[index].astype(np.float32)
    valid = np.isfinite(depth) & (depth > float(config.get("min_depth", 0))) & (depth > 0)
    valid &= np.isfinite(confidence) & (confidence >= max(0, float(config.get("min_confidence", 0))))
    if config.get("exclude_sky", True) and arrays.sky is not None:
        sky_view = np.asarray(arrays.sky[index])
        valid &= np.isfinite(sky_view) & (sky_view < SKY_PROBABILITY_THRESHOLD)
    atomic_npz(
        path,
        depth=depth,
        conf=confidence,
        intrinsics=arrays.intrinsics[index],
        world_from_camera=arrays.scene_from_camera[index],
        image=arrays.images[index],
        original_to_processed=mapping,
        valid=valid,
    )
    return path
