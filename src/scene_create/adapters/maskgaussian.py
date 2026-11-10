"""MaskGaussian 背景适配器：用掩码监督训练背景 3D Gaussian，并用同一 CUDA 光栅器渲染场景预览。"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
from loguru import logger
from scipy.ndimage import binary_erosion

from scene_create.gaussian import read_gaussians, transform_gaussians
from scene_create.geometry import extent as point_extent
from scene_create.io import (
    atomic_json,
    file_digest,
    fingerprint,
    load_mask,
    load_npz,
    read_cloud,
    read_json,
    save_image,
    validate_id,
)
from scene_create.runtime import configure_device

# -------------------------------------------
# 配置常量
# -------------------------------------------
# fmt: off
DEFAULT_ITERATIONS = 1000                 # 背景训练默认迭代次数
DEFAULT_DEPTH_WEIGHT = 0.1                # 相对深度损失的默认权重
DEFAULT_DSSIM_WEIGHT = 0.2                # D-SSIM 损失权重，其余权重分给 L1
DEFAULT_MASK_WEIGHT = 0.0005              # 高斯存在概率的稀疏正则权重
DEFAULT_MAX_GAUSSIANS = 500_000           # 高斯数量上限；超过一半后不再密化
DEFAULT_LOG_INTERVAL = 50                 # 训练指标记录间隔，单位：迭代
DEFAULT_CHECKPOINT_INTERVAL = 250         # 检查点保存间隔，单位：迭代
DEFAULT_DENSIFY_FROM = 100                # 开始密化的迭代
DENSIFY_UNTIL_FRACTION = 0.6              # 未配置时，在总迭代数的该比例处停止密化
DENSIFY_INTERVAL = 100                    # 密化与剪枝的间隔，单位：迭代
MASK_PRUNE_INTERVAL = 500                 # 非密化阶段按 mask 剪枝的间隔，单位：迭代
SH_UPGRADE_INTERVAL = 1000                # SH 阶数提升一级的间隔，单位：迭代
DEFAULT_DENSIFY_GRAD_THRESHOLD = 0.0002   # 视空间梯度超过该值的高斯会被密化
DENSIFY_MIN_OPACITY = 0.005               # 密化时剪除不透明度低于该值的高斯
DEFAULT_EXPORT_PROBABILITY = 0.5          # 导出时保留的最低存在概率
EXPORT_MIN_OPACITY = 0.005                # 导出时剪除不透明度低于该值的高斯
DEPTH_ALPHA_THRESHOLD = 0.05              # 渲染累计不透明度超过该值的像素才参与深度监督
MIN_SCENE_EXTENT = 1e-4                   # 场景尺度下限，单位：场景单位
NEAR_EXTENT_RATIO = 1e-5                  # 默认近裁剪面相对场景尺度的比例
MIN_NEAR_PLANE = 1e-5                     # 近裁剪面下限，单位：场景单位
FAR_EXTENT_RATIO = 100                    # 默认远裁剪面相对场景尺度的比例
MIN_FAR_PLANE = 100                       # 远裁剪面下限，单位：场景单位
MAX_PINHOLE_SKEW = 1e-8                   # 光栅器要求零倾斜，超过该值即拒绝
DEFAULT_RENDER_MAX_SIDE = 768             # 场景预览图像的最长边，单位：像素
# fmt: on

# fmt: off
SSIM_WINDOW = 11                          # SSIM 高斯窗口边长，单位：像素
SSIM_SIGMA = 1.5                          # SSIM 高斯窗口标准差，单位：像素
SSIM_C1 = 0.01**2                         # SSIM 亮度项稳定常数（动态范围为 1）
SSIM_C2 = 0.03**2                         # SSIM 对比度项稳定常数（动态范围为 1）
# fmt: on

# MaskGaussian 上游训练的默认学习率与密化参数；同名配置项可以覆盖。
DEFAULT_OPTIMIZER_OPTIONS = {
    "percent_dense": 0.01,
    "position_lr_init": 0.00016,
    "position_lr_final": 0.0000016,
    "position_lr_delay_mult": 0.01,
    "feature_lr": 0.0025,
    "opacity_lr": 0.05,
    "mask_lr": 0.01,
    "scaling_lr": 0.005,
    "rotation_lr": 0.001,
}


# -------------------------------------------
# 数据模型
# -------------------------------------------
@dataclass(frozen=True)
class ViewCamera:
    """一个针孔相机视角：内参、camera-to-world 位姿和图像尺寸。"""

    intrinsic: np.ndarray
    world_from_camera: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class TrainingSettings:
    """解析并校验后的背景训练超参数。"""

    iterations: int
    sh_degree: int
    depth_weight: float
    dssim_weight: float
    mask_weight: float
    max_gaussians: int
    log_interval: int
    checkpoint_interval: int
    densify_from: int
    densify_until: int
    densify_grad_threshold: float
    export_probability: float
    extent: float
    near: float
    far: float


@dataclass(frozen=True)
class TrainingView:
    """一张训练视图在 CPU 上缓存的监督数据与对应相机。"""

    view_id: str
    camera: Any
    rgb: Any
    valid: Any
    ssim_valid: Any
    depth: Any
    conf: Any
    camera_from_world: Any


@dataclass(frozen=True)
class TrainingSession:
    """一次训练共享的模型、随机数、优化参数和检查点身份。"""

    gaussian: Any
    random: np.random.Generator
    options: SimpleNamespace
    settings: TrainingSettings
    checkpoint: Path
    signature: str


# -------------------------------------------
# 相机与损失
# -------------------------------------------
def projection_matrix(intrinsics: Any, width: int, height: int, near: float = 0.01, far: float = 100.0) -> np.ndarray:
    """构造与像素中心约定一致的 3DGS 投影矩阵（行主序，未转置）。

    Raises:
        ValueError: 内参、尺寸或裁剪面无效，或相机存在倾斜。
    """
    intrinsic = np.asarray(intrinsics, dtype=float)
    is_valid_input = intrinsic.shape == (3, 3) and min(width, height) > 0 and 0 < near < far and np.isfinite(intrinsic).all()
    if not is_valid_input:
        raise ValueError("Invalid camera projection parameters")
    is_zero_skew_pinhole = np.allclose(intrinsic[2], [0, 0, 1]) and abs(intrinsic[0, 1]) <= MAX_PINHOLE_SKEW and min(intrinsic[0, 0], intrinsic[1, 1]) > 0
    if not is_zero_skew_pinhole:
        raise ValueError("Gaussian rasterization requires a zero-skew pinhole camera")
    projection = np.zeros((4, 4), dtype=np.float32)
    projection[0, 0] = 2 * intrinsic[0, 0] / width
    projection[1, 1] = 2 * intrinsic[1, 1] / height
    projection[0, 2] = 2 * (intrinsic[0, 2] + 0.5) / width - 1
    projection[1, 2] = 2 * (intrinsic[1, 2] + 0.5) / height - 1
    projection[2, 2] = far / (far - near)
    projection[2, 3] = -far * near / (far - near)
    projection[3, 2] = 1
    return projection


def valid_window_mask(mask: Any, window: int = SSIM_WINDOW) -> np.ndarray:
    """返回整个 SSIM 窗口都落在有效区内的中心像素，避免前景和边界泄漏到损失中。

    Raises:
        ValueError: 窗口不是正奇数。
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("SSIM window must be odd and positive")
    return binary_erosion(
        np.asarray(mask, dtype=bool),
        structure=np.ones((window, window), bool),
        border_value=0,
    )


def masked_ssim(prediction: Any, reference: Any, valid_windows: Any, window: int = SSIM_WINDOW) -> Any:
    """只在完整有效窗口上计算 1 - SSIM 的均值；没有有效窗口时返回可反传的 0。"""
    # 延迟加载：主环境可以不安装 torch。
    import torch
    import torch.nn.functional as functional

    if not valid_windows.any():
        return prediction.sum() * 0
    coordinates = torch.arange(window, device=prediction.device, dtype=prediction.dtype) - window // 2
    kernel = torch.exp(-coordinates.square() / (2 * SSIM_SIGMA**2))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] @ kernel[None, :]).expand(3, 1, window, window).contiguous()
    padding = window // 2
    a, b = prediction[None], reference[None]
    mean_a = functional.conv2d(a, kernel, padding=padding, groups=3)
    mean_b = functional.conv2d(b, kernel, padding=padding, groups=3)
    variance_a = functional.conv2d(a * a, kernel, padding=padding, groups=3) - mean_a.square()
    variance_b = functional.conv2d(b * b, kernel, padding=padding, groups=3) - mean_b.square()
    covariance = functional.conv2d(a * b, kernel, padding=padding, groups=3) - mean_a * mean_b
    numerator = (2 * mean_a * mean_b + SSIM_C1) * (2 * covariance + SSIM_C2)
    denominator = (mean_a.square() + mean_b.square() + SSIM_C1) * (variance_a + variance_b + SSIM_C2)
    similarity = numerator / denominator
    return (1 - similarity[0][:, valid_windows]).mean()


def camera_for_view(model: SimpleNamespace, view: ViewCamera, near: float, far: float) -> Any:
    """把针孔相机转换为 MaskGaussian 光栅器使用的 `MiniCam`。"""
    world_view = model.torch.as_tensor(np.linalg.inv(view.world_from_camera), dtype=model.torch.float32, device=model.device).T.contiguous()
    projection = model.torch.as_tensor(projection_matrix(view.intrinsic, view.width, view.height, near, far), device=model.device).T.contiguous()
    return model.MiniCam(
        view.width,
        view.height,
        2 * math.atan(view.height / (2 * view.intrinsic[1, 1])),
        2 * math.atan(view.width / (2 * view.intrinsic[0, 0])),
        near,
        far,
        world_view,
        world_view @ projection,
    )


def _default_clip_range(extent: float) -> tuple[float, float]:
    return max(extent * NEAR_EXTENT_RATIO, MIN_NEAR_PLANE), max(extent * FAR_EXTENT_RATIO, MIN_FAR_PLANE)


def _render_rgb(model: SimpleNamespace, camera: Any, gaussians: Any, background: Any) -> np.ndarray:
    image = model.render(camera, gaussians, model.pipe, background)["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (image * 255).astype(np.uint8)


# -------------------------------------------
# 模型加载
# -------------------------------------------
def load(config: Mapping[str, Any], root: Path) -> SimpleNamespace:
    """导入 MaskGaussian 的 CUDA 光栅器与高斯模型类。

    Raises:
        ValueError: 未配置 CUDA 或 CUDA 不可用。
    """
    # 延迟加载：主环境可以不安装 torch 和 MaskGaussian 的 CUDA 扩展。
    import torch

    device = config.get("device", "cuda")
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("MaskGaussian requires CUDA and its compiled mask rasterizer")
    from gaussian_renderer import render
    from scene.cameras import MiniCam
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud

    return SimpleNamespace(
        torch=torch,
        device=configure_device(torch, device),
        render=render,
        MiniCam=MiniCam,
        GaussianModel=GaussianModel,
        BasicPointCloud=BasicPointCloud,
        pipe=SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False),
    )


def run(model: SimpleNamespace, request: Mapping[str, Any]) -> dict:
    """`mode == "render_scene"` 时渲染场景预览，否则训练背景 Gaussian。"""
    if request.get("mode") == "render_scene":
        return render_scene(model, request)
    return train(model, request)


# -------------------------------------------
# 背景训练
# -------------------------------------------
def train(model: SimpleNamespace, request: Mapping[str, Any]) -> dict:
    """从背景点云初始化并训练 MaskGaussian，支持同输入同配置的检查点续训。

    Returns:
        背景 Gaussian 路径、检查点、训练历史和近似性标记，同时写入 `training_metrics.json`。

    Raises:
        ValueError: 输入点过少、配置无效、检查点不匹配或没有有效训练像素。
        RuntimeError: 损失非有限、高斯被全部剪除或没有高斯通过导出阈值。
    """
    config, directory = request["config"], Path(request["output_dir"])
    model.torch.manual_seed(int(config.get("seed", 0)))
    random = np.random.default_rng(int(config.get("seed", 0)))
    dataset = read_json(request["dataset_path"])
    cloud = read_cloud(dataset["initial_ply"])
    if len(cloud["points"]) < 3:
        raise ValueError("MaskGaussian needs at least three initial points")
    settings = _training_settings(config, point_extent(cloud["points"], minimum=MIN_SCENE_EXTENT), len(dataset["views"]))
    options = _optimizer_options(config, settings.iterations)
    session = TrainingSession(
        gaussian=_coherent_model_type(model.GaussianModel)(settings.sh_degree),
        random=random,
        options=options,
        settings=settings,
        checkpoint=directory / "checkpoint.pt",
        signature=_dataset_signature(request, dataset, config),
    )

    first_iteration = _resume_or_initialize(model, session, cloud, bool(config.get("resume_checkpoint", True)))
    views = _load_training_views(model, dataset["views"], settings)
    if not views:
        raise ValueError("No valid background training pixels")

    started = time.monotonic()
    history = _optimize(model, session, views, first_iteration)
    output_path = _export_background(model, session, views, directory)
    result = {
        "gaussian_path": str(output_path),
        "checkpoint_path": str(session.checkpoint),
        "iterations": settings.iterations,
        "elapsed_seconds": time.monotonic() - started,
        "gaussians": len(session.gaussian.get_xyz),
        "history": history,
        "mask_mechanism": "upstream_gumbel_straight_through_probability",
        "background_masked_supervision": True,
        "depth_weight": settings.depth_weight,
        "single_view_approximate": len(views) == 1,
        "hidden_background_completed": False,
    }
    atomic_json(directory / "training_metrics.json", result)
    return result


def _training_settings(config: Mapping[str, Any], extent: float, view_count: int) -> TrainingSettings:
    default_near, default_far = _default_clip_range(extent)
    near = float(config.get("near", default_near))
    far = float(config.get("far", default_far))
    iterations = int(config.get("iterations", DEFAULT_ITERATIONS))
    degree = int(config.get("sh_degree", 0))
    if iterations < 1 or degree not in range(4):
        raise ValueError("Invalid training iteration count or SH degree")
    depth_weight = float(config.get("depth_weight", DEFAULT_DEPTH_WEIGHT))
    if view_count == 1 and depth_weight <= 0:
        raise ValueError("Single-view MaskGaussian requires a positive depth_weight; otherwise use explicit init-only mode")
    return TrainingSettings(
        iterations=iterations,
        sh_degree=degree,
        depth_weight=depth_weight,
        dssim_weight=float(config.get("dssim_weight", DEFAULT_DSSIM_WEIGHT)),
        mask_weight=float(config.get("mask_weight", DEFAULT_MASK_WEIGHT)),
        max_gaussians=int(config.get("max_gaussians", DEFAULT_MAX_GAUSSIANS)),
        log_interval=max(1, int(config.get("log_interval", DEFAULT_LOG_INTERVAL))),
        checkpoint_interval=max(1, int(config.get("checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL))),
        densify_from=int(config.get("densify_from", DEFAULT_DENSIFY_FROM)),
        densify_until=int(config.get("densify_until", max(0, int(iterations * DENSIFY_UNTIL_FRACTION)))),
        densify_grad_threshold=float(config.get("densify_grad_threshold", DEFAULT_DENSIFY_GRAD_THRESHOLD)),
        export_probability=float(config.get("export_probability", DEFAULT_EXPORT_PROBABILITY)),
        extent=extent,
        near=near,
        far=far,
    )


def _optimizer_options(config: Mapping[str, Any], iterations: int) -> SimpleNamespace:
    options = SimpleNamespace(**DEFAULT_OPTIMIZER_OPTIONS, position_lr_max_steps=iterations)
    for key in vars(options):
        if key in config:
            setattr(options, key, config[key])
    return options


def _coherent_model_type(base: type) -> type:
    """返回在一次迭代内复用同一采样 mask 的高斯模型子类。

    MaskGaussian 每次读取 `get_mask` 都会重新 Gumbel 采样；颜色、深度与 alpha 三次渲染
    必须看到同一个 mask，否则深度期望值没有意义。
    """

    # 基类只能在延迟导入 MaskGaussian 后获得，因此子类在函数内定义。
    class CoherentGaussianModel(base):
        cached_mask = None

        @property
        def get_mask(self):
            return super().get_mask if self.cached_mask is None else self.cached_mask

        def sample_iteration_mask(self):
            self.cached_mask = super().get_mask

    return CoherentGaussianModel


def _dataset_signature(request: Mapping[str, Any], dataset: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    view_digests = []
    for view in dataset["views"]:
        view_digests.append({key: file_digest(view[key]) for key in ("geometry_path", "image_path", "mask_path")})
    return fingerprint(
        {
            "dataset": file_digest(request["dataset_path"]),
            "initial": file_digest(dataset["initial_ply"]),
            "files": view_digests,
            "config": config,
        }
    )


def _resume_or_initialize(model: SimpleNamespace, session: TrainingSession, cloud: Mapping[str, np.ndarray], should_resume: bool) -> int:
    """恢复匹配的检查点并返回已完成的迭代数；否则从初始点云创建高斯并返回 0。"""
    if session.checkpoint.exists() and should_resume:
        return _restore_checkpoint(model, session)
    point_cloud = model.BasicPointCloud(points=cloud["points"], colors=cloud["colors"].astype(float) / 255, normals=np.zeros_like(cloud["points"]))
    session.gaussian.create_from_pcd(point_cloud, session.settings.extent)
    session.gaussian.training_setup(session.options)
    return 0


def _restore_checkpoint(model: SimpleNamespace, session: TrainingSession) -> int:
    torch = model.torch
    state = torch.load(str(session.checkpoint), map_location="cpu", weights_only=False)
    if state.get("signature") != session.signature:
        raise ValueError("Existing MaskGaussian checkpoint belongs to different inputs/configuration; move it or set resume_checkpoint: false")
    # GaussianModel.capture() 的第 1-7 项是可训练参数，第 8-10 项是密化统计张量。
    model_state = list(state["model"])
    for index in range(1, 11):
        if torch.is_tensor(model_state[index]):
            tensor = model_state[index].to(model.device)
            model_state[index] = torch.nn.Parameter(tensor) if index in range(1, 8) else tensor
    session.gaussian.restore(tuple(model_state), session.options)
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"], device=model.device)
    session.random.bit_generator.state = state["numpy_rng"]
    return int(state["iteration"])


def _save_checkpoint(model: SimpleNamespace, session: TrainingSession, iteration: int) -> None:
    temporary = session.checkpoint.with_suffix(".tmp")
    model.torch.save(
        {
            "signature": session.signature,
            "iteration": iteration,
            "model": session.gaussian.capture(),
            "torch_rng": model.torch.get_rng_state(),
            "cuda_rng": model.torch.cuda.get_rng_state(model.device),
            "numpy_rng": session.random.bit_generator.state,
        },
        str(temporary),
    )
    temporary.replace(session.checkpoint)


def _load_training_views(model: SimpleNamespace, records: Sequence[Mapping[str, Any]], settings: TrainingSettings) -> list[TrainingView]:
    torch = model.torch
    views = []
    for record in records:
        data = load_npz(record["geometry_path"])
        height, width = data["depth"].shape
        valid = load_mask(record["mask_path"], (width, height)) & data["valid"]
        if not valid.any():
            continue
        camera = camera_for_view(model, ViewCamera(data["intrinsics"], data["world_from_camera"], width, height), settings.near, settings.far)
        camera_from_world = np.linalg.inv(data["world_from_camera"])
        views.append(
            TrainingView(
                view_id=record["view_id"],
                camera=camera,
                rgb=torch.from_numpy(data["image"].copy()).permute(2, 0, 1).float() / 255,
                valid=torch.from_numpy(valid),
                ssim_valid=torch.from_numpy(valid_window_mask(valid)),
                depth=torch.from_numpy(data["depth"].copy()).float(),
                conf=torch.from_numpy(data["conf"].copy()).float(),
                camera_from_world=torch.as_tensor(camera_from_world, dtype=torch.float32, device=model.device),
            )
        )
    return views


def _optimize(model: SimpleNamespace, session: TrainingSession, views: Sequence[TrainingView], first_iteration: int) -> list[dict]:
    settings = session.settings
    black = model.torch.zeros(3, device=model.device)
    history = []
    for iteration in range(first_iteration + 1, settings.iterations + 1):
        view = views[int(session.random.integers(len(views)))]
        package, losses = _compute_losses(model, session, view, iteration, black)
        if not model.torch.isfinite(losses["loss"]):
            raise RuntimeError("Non-finite MaskGaussian training loss")
        losses["loss"].backward()
        _apply_gradients(model, session, package, iteration)
        is_last_iteration = iteration == settings.iterations
        if iteration % settings.log_interval == 0 or is_last_iteration:
            metrics = {
                "iteration": iteration,
                "l1": float(losses["l1"].detach()),
                "dssim": float(losses["dssim"].detach()),
                "relative_depth_loss": float(losses["depth"].detach()),
                "loss": float(losses["loss"].detach()),
                "gaussians": len(session.gaussian.get_xyz),
            }
            history.append(metrics)
            logger.info("Background iteration {}: loss={:.5f}, gaussians={}", iteration, metrics["loss"], metrics["gaussians"])
        if iteration % settings.checkpoint_interval == 0 or is_last_iteration:
            _save_checkpoint(model, session, iteration)
    return history


def _compute_losses(model: SimpleNamespace, session: TrainingSession, view: TrainingView, iteration: int, black: Any) -> tuple[dict, dict]:
    settings, gaussian = session.settings, session.gaussian
    reference, valid = view.rgb.to(model.device), view.valid.to(model.device)
    ssim_valid = view.ssim_valid.to(model.device)
    gaussian.update_learning_rate(iteration)
    if iteration % SH_UPGRADE_INTERVAL == 0:
        gaussian.oneupSHdegree()
    gaussian.sample_iteration_mask()
    package = model.render(view.camera, gaussian, model.pipe, black)
    rendered = package["render"]

    l1 = (rendered[:, valid] - reference[:, valid]).abs().mean()
    ssim_loss = masked_ssim(rendered, reference, ssim_valid) if settings.dssim_weight else rendered.sum() * 0
    loss = (1 - settings.dssim_weight) * l1 + settings.dssim_weight * ssim_loss + settings.mask_weight * package["mask"].mean().square()

    depth_loss = rendered.sum() * 0
    if settings.depth_weight > 0:
        supervised_depth_loss = _relative_depth_loss(model, gaussian, view, valid, black)
        if supervised_depth_loss is not None:
            depth_loss = supervised_depth_loss
            loss = loss + settings.depth_weight * depth_loss
    return package, {"loss": loss, "l1": l1, "dssim": ssim_loss, "depth": depth_loss}


def _relative_depth_loss(model: SimpleNamespace, gaussian: Any, view: TrainingView, valid: Any, black: Any) -> Optional[Any]:
    """置信度加权的相对深度误差；没有可监督像素时返回 `None`。

    原版光栅器不输出深度，这里用同一采样 mask 把相机深度和单位颜色各渲染一次，
    相除得到 alpha 归一化的期望深度。
    """
    torch = model.torch
    camera_from_world = view.camera_from_world
    depths = gaussian.get_xyz @ camera_from_world[2, :3] + camera_from_world[2, 3]
    depth_colors = depths.clamp_min(0)[:, None].expand(-1, 3).contiguous()
    depth_render = model.render(view.camera, gaussian, model.pipe, black, override_color=depth_colors)["render"][0]
    alpha = model.render(view.camera, gaussian, model.pipe, black, override_color=torch.ones_like(depth_colors))["render"][0]
    measured_depth = view.depth.to(model.device)
    depth_valid = valid & (alpha > DEPTH_ALPHA_THRESHOLD) & torch.isfinite(measured_depth) & (measured_depth > 0)
    if not depth_valid.any():
        return None
    predicted_depth = depth_render / alpha.clamp_min(1e-6)
    confidence = view.conf.to(model.device)[depth_valid].clamp_min(0)
    confidence = confidence / confidence.mean().clamp_min(1e-6)
    measured = measured_depth[depth_valid]
    relative_error = (predicted_depth[depth_valid] - measured).abs() / measured.clamp_min(1e-6)
    return (relative_error * confidence).mean()


def _apply_gradients(model: SimpleNamespace, session: TrainingSession, package: Mapping[str, Any], iteration: int) -> None:
    torch, gaussian, settings = model.torch, session.gaussian, session.settings
    with torch.no_grad():
        visible = package["visibility_filter"]
        gaussian.max_radii2D[visible] = torch.maximum(gaussian.max_radii2D[visible], package["radii"][visible])
        if package["viewspace_points"].grad is not None:
            gaussian.add_densification_stats(package["viewspace_points"], visible)
        gaussian.optimizer.step()
        gaussian.optimizer.zero_grad(set_to_none=True)
        gaussian.cached_mask = None
        is_densify_window = settings.densify_from <= iteration < settings.densify_until
        should_densify = is_densify_window and iteration % DENSIFY_INTERVAL == 0 and len(gaussian.get_xyz) < settings.max_gaussians // 2
        if should_densify:
            gaussian.densify_and_prune(settings.densify_grad_threshold, DENSIFY_MIN_OPACITY, settings.extent, None)
        elif iteration % MASK_PRUNE_INTERVAL == 0:
            gaussian.mask_prune()
        if len(gaussian.get_xyz) == 0:
            raise RuntimeError("MaskGaussian pruned all background primitives")


def _export_background(model: SimpleNamespace, session: TrainingSession, views: Sequence[TrainingView], directory: Path) -> Path:
    torch, gaussian = model.torch, session.gaussian
    with torch.no_grad():
        probability = torch.softmax(gaussian._mask_score, dim=-1)[:, 0]
        keep = (probability >= session.settings.export_probability) & (gaussian.get_opacity[:, 0] > EXPORT_MIN_OPACITY)
        if not keep.any():
            raise RuntimeError("No background Gaussians passed the export probability threshold")
        gaussian.cached_mask = None
        gaussian.prune_points(~keep)
        # 导出后的高斯视为确定存在，预览渲染不再随机采样 mask。
        gaussian.cached_mask = torch.ones_like(gaussian.get_opacity)
        raw_path = directory / "trained_with_masks.ply"
        output_path = directory / "background_gaussian.ply"
        gaussian.save_ply(str(raw_path))
        transform_gaussians(raw_path, output_path, np.eye(4))
        black = torch.zeros(3, device=model.device)
        for view in views:
            save_image(directory / "previews" / f"{view.view_id}.png", _render_rgb(model, view.camera, gaussian, black))
    return output_path


# -------------------------------------------
# 场景预览
# -------------------------------------------
def render_scene(model: SimpleNamespace, request: Mapping[str, Any]) -> dict:
    """在 `scene.json` 记录的每个相机视角渲染合并后的 Gaussian 场景。

    Returns:
        预览图片路径、高斯数量和渲染说明，同时写入 `previews.json`。

    Raises:
        ValueError: 场景没有 Gaussian 表示或 Gaussian 为空。
    """
    torch = model.torch
    scene_path = Path(request["scene_path"])
    scene = read_json(scene_path)
    if not scene.get("gaussian_scene"):
        raise ValueError("Scene has no Gaussian representation to render")
    vertices, degree = read_gaussians(scene_path.parent / scene["gaussian_scene"])
    if not len(vertices):
        raise ValueError("Cannot render an empty Gaussian scene")
    gaussians = _fixed_gaussians(model, vertices, degree)
    background = torch.zeros(3, device=model.device)
    minimum = gaussians.get_xyz.min(dim=0).values
    maximum = gaussians.get_xyz.max(dim=0).values
    extent = max(float(torch.linalg.vector_norm(maximum - minimum)), MIN_SCENE_EXTENT)
    near, far = _default_clip_range(extent)
    max_side = int(request["config"].get("render_max_side", DEFAULT_RENDER_MAX_SIDE))
    output_dir = Path(request["output_dir"])
    files = []
    with torch.no_grad():
        for view in scene["cameras"]:
            view_id = validate_id(view["view_id"])
            camera = camera_for_view(model, _preview_camera(view, max_side), near, far)
            path = output_dir / f"{view_id}.png"
            save_image(path, _render_rgb(model, camera, gaussians, background))
            files.append(str(path))
    result = {
        "images": files,
        "gaussian_count": len(vertices),
        "sh_degree": degree,
        "renderer": "MaskGaussian CUDA rasterizer; fixed all-present masks",
        "background_color": "black",
    }
    atomic_json(output_dir / "previews.json", result)
    return result


def _fixed_gaussians(model: SimpleNamespace, vertices: np.ndarray, degree: int) -> SimpleNamespace:
    """把 PLY 字段转换为光栅器读取的激活后属性，mask 固定为全部存在。"""
    torch, device = model.torch, model.device

    def tensor(names: Sequence[str]) -> Any:
        return torch.as_tensor(np.column_stack([vertices[name] for name in names]), dtype=torch.float32, device=device)

    gaussians = SimpleNamespace(
        get_xyz=tensor(["x", "y", "z"]),
        get_scaling=tensor([f"scale_{i}" for i in range(3)]).exp(),
        get_rotation=torch.nn.functional.normalize(tensor([f"rot_{i}" for i in range(4)]), dim=1),
        get_opacity=tensor(["opacity"]).sigmoid(),
        active_sh_degree=degree,
        max_sh_degree=degree,
    )
    gaussians.get_mask = torch.ones_like(gaussians.get_opacity)
    dc = tensor([f"f_dc_{i}" for i in range(3)])[:, None, :]
    count = (degree + 1) ** 2 - 1
    if count:
        rest = tensor([f"f_rest_{i}" for i in range(3 * count)]).reshape(-1, 3, count).transpose(1, 2)
    else:
        rest = torch.empty((len(vertices), 0, 3), device=device)
    gaussians.get_features = torch.cat([dc, rest], dim=1).contiguous()
    return gaussians


def _preview_camera(view: Mapping[str, Any], max_side: int) -> ViewCamera:
    scale = min(1.0, max_side / max(view["width"], view["height"]))
    width = max(1, round(view["width"] * scale))
    height = max(1, round(view["height"] * scale))
    sx, sy = width / view["width"], height / view["height"]
    # 缩放图像时按像素中心约定同步缩放内参。
    pixel_scale = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return ViewCamera(pixel_scale @ np.asarray(view["intrinsics"]), np.asarray(view["world_from_camera"]), width, height)
