"""TRELLIS 图像到 3D 适配器：只用本地组件，从同一潜变量采样导出 mesh 与 Gaussian 双资产。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Union

import numpy as np
from PIL import Image

from scene_create.io import atomic_json, read_json, save_image
from scene_create.runtime import configure_device

# -------------------------------------------
# 配置常量
# -------------------------------------------
# TRELLIS 原生坐标（Z 向上）到 GLB 资产坐标（Y 向上）的旋转，mesh 与 Gaussian 共用。
GLB_FROM_NATIVE = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)

# fmt: off
FOREGROUND_ALPHA_THRESHOLD = 204      # alpha 大于该值（约 80% 不透明）视为前景，取值 0-255
MIN_FOREGROUND_PIXELS = 16            # 可用前景的最少像素数，单位：像素
MIN_FOREGROUND_SPAN = 3               # 前景包围盒的最小宽或高，单位：像素
MIN_QUATERNION_NORM = 1e-8            # 小于该范数的四元数视为退化
OPACITY_CLIP_EPSILON = 1e-3           # 避免 FP16 opacity 饱和到 0/1 后导出 logit 无穷值
DEFAULT_SIMPLIFY_RATIO = 0.5          # 导出 GLB 时的网格简化比例
DEFAULT_TEXTURE_SIZE = 1024           # 导出 GLB 的纹理边长，单位：像素
DEFAULT_ATTENTION_BACKEND = "xformers"
DEFAULT_SPCONV_ALGO = "native"
DINO_MEAN = [0.485, 0.456, 0.406]     # DINOv2 输入归一化均值（ImageNet）
DINO_STD = [0.229, 0.224, 0.225]      # DINOv2 输入归一化标准差（ImageNet）
# fmt: on

# 实际使用的五个组件；未使用的 slat_decoder_rf 不加载。
REQUIRED_COMPONENTS = (
    "sparse_structure_flow_model",
    "sparse_structure_decoder",
    "slat_flow_model",
    "slat_decoder_mesh",
    "slat_decoder_gs",
)


# -------------------------------------------
# 本地组件与输入校验
# -------------------------------------------
def validate_foreground(alpha: Any, require_margin: bool = True) -> np.ndarray:
    """检查补全图前景是否可用于 TRELLIS。

    Args:
        alpha: HxW alpha 通道，取值 0-255。
        require_margin: 是否要求前景不接触图像边界。

    Returns:
        前景 bool mask。

    Raises:
        ValueError: 前景过少、接触边界或几何退化。
    """
    alpha = np.asarray(alpha)
    foreground = alpha > FOREGROUND_ALPHA_THRESHOLD
    if alpha.ndim != 2 or foreground.sum() < MIN_FOREGROUND_PIXELS:
        raise ValueError("Completed object has no usable foreground")
    touches_border = foreground[0].any() or foreground[-1].any() or foreground[:, 0].any() or foreground[:, -1].any()
    if require_margin and touches_border:
        raise ValueError("Completed foreground touches the image boundary; regenerate with more margin")
    ys, xs = np.where(foreground)
    if min(xs.max() - xs.min(), ys.max() - ys.min()) < MIN_FOREGROUND_SPAN:
        raise ValueError("Completed object foreground is geometrically degenerate")
    return foreground


def local_components(model_path: Union[str, Path, None]) -> tuple[dict, dict[str, str]]:
    """读取 `pipeline.json`，确认每个所需组件都有本地 `.json` 与 `.safetensors`。

    Returns:
        `pipeline.json` 的 `args`，以及组件名到本地文件主干路径的映射。

    Raises:
        ValueError: 未配置路径或清单缺少组件。
        FileNotFoundError: 清单或组件文件不存在；不会回退到远程下载。
    """
    if not model_path:
        raise ValueError("TRELLIS requires a local model_path")
    root = Path(model_path).expanduser().resolve()
    manifest = root / "pipeline.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"TRELLIS local pipeline manifest is missing: {manifest}")
    args = read_json(manifest).get("args")
    if not isinstance(args, dict) or not isinstance(args.get("models"), dict):
        raise ValueError("Invalid TRELLIS pipeline.json model mapping")
    components = {}
    for name in REQUIRED_COMPONENTS:
        reference = args["models"].get(name)
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"TRELLIS pipeline is missing component {name}")
        stem = (root / reference).resolve()
        for suffix in (".json", ".safetensors"):
            if not Path(str(stem) + suffix).is_file():
                raise FileNotFoundError(f"TRELLIS component {name} is incomplete: {stem}{suffix}; remote fallback is disabled")
        components[name] = str(stem)
    return args, components


def normalize_export_rotations(rotations: Any) -> np.ndarray:
    """把导出前的 wxyz 四元数归一化，不改变轴向。

    Raises:
        ValueError: 输入不是非空有限的 Nx4，或存在零范数四元数。
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    is_valid_shape = rotations.ndim == 2 and rotations.shape[1] == 4 and len(rotations) > 0
    if not is_valid_shape or not np.isfinite(rotations).all():
        raise ValueError("Gaussian rotations must be nonempty finite Nx4 wxyz values")
    norm = np.linalg.norm(rotations, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or (norm <= MIN_QUATERNION_NORM).any():
        raise ValueError("Gaussian rotation quaternions must have nonzero finite norm")
    return rotations / norm


def local_rembg_session(rembg: Any, model_path: Union[str, Path]) -> Any:
    """使用显式本地路径创建 U2Net 会话，不走 rembg 默认的自动下载会话。

    Raises:
        FileNotFoundError: 本地 ONNX 文件不存在。
    """
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Local U2Net ONNX model is missing: {path}")
    os.environ["U2NET_HOME"] = str(path.parent)
    os.environ["REMBG_HOME"] = str(path.parent)
    return rembg.new_session(
        "u2net_custom",
        model_path=str(path),
        providers=["CPUExecutionProvider"],
    )


# -------------------------------------------
# 模型加载
# -------------------------------------------
def load(config: Mapping[str, Any], root: Path) -> SimpleNamespace:
    """从本地组件、本地 DINOv2 源码与权重组装 TRELLIS 管线并放到 CUDA。

    Raises:
        ValueError: 组件清单无效或设备不是 CUDA。
        FileNotFoundError: 组件、DINOv2 源码或权重缺失。
    """
    args, component_paths = local_components(config.get("model_path"))
    if not str(config.get("device", "cuda")).startswith("cuda"):
        raise ValueError("TRELLIS requires CUDA with its compiled extensions")
    # TRELLIS 在导入时读取这两个环境变量，必须在导入前设置。
    os.environ["ATTN_BACKEND"] = config.get("attention_backend", DEFAULT_ATTENTION_BACKEND)
    os.environ["SPCONV_ALGO"] = config.get("spconv_algo", DEFAULT_SPCONV_ALGO)

    # 延迟加载：主环境可以不安装 torch、TRELLIS 及其 CUDA 扩展。
    import torch

    device = configure_device(torch, config.get("device", "cuda"))
    import rembg
    from trellis.pipelines import TrellisImageTo3DPipeline, samplers
    from trellis.utils import postprocessing_utils

    dino_repo, dino_weights = _local_dino_paths(config, root)

    # 基类只能在延迟导入 TRELLIS 后获得，因此子类在函数内定义。
    class LocalTrellisPipeline(TrellisImageTo3DPipeline):
        def _init_image_cond_model(self, name: str) -> None:
            self.models["image_cond_model"] = _load_local_dino(torch, dino_repo, dino_weights, name)
            self.image_cond_model_transform = _dino_transform()

    torch.manual_seed(config.get("seed", 0))
    sparse = args["sparse_structure_sampler"]
    slat = args["slat_sampler"]
    pipeline = LocalTrellisPipeline(
        models=_load_components(component_paths),
        sparse_structure_sampler=getattr(samplers, sparse["name"])(**sparse["args"]),
        slat_sampler=getattr(samplers, slat["name"])(**slat["args"]),
        slat_normalization=args["slat_normalization"],
        image_cond_model=args["image_cond_model"],
    )
    pipeline.sparse_structure_sampler_params = sparse["params"]
    pipeline.slat_sampler_params = slat["params"]
    pipeline.to(torch.device(device))
    return SimpleNamespace(
        torch=torch,
        pipeline=pipeline,
        rembg=rembg,
        rembg_model_path=Path(config.get("rembg_model_path") or root / "checkpoints" / "rembg" / "u2net.onnx"),
        rembg_session=None,
        postprocessing_utils=postprocessing_utils,
        component_paths=component_paths,
        dino_weights=str(dino_weights),
        device=device,
    )


def _local_dino_paths(config: Mapping[str, Any], root: Path) -> tuple[Path, str]:
    dino_repo = Path(config.get("dino_repo_path") or root / "third_parts" / "dinov2")
    dino_weights = config.get("dino_model_path")
    has_local_dino = bool(dino_weights) and Path(dino_weights).is_file() and (dino_repo / "hubconf.py").is_file()
    if not has_local_dino:
        raise FileNotFoundError("TRELLIS requires local dino_repo_path and dino_model_path; its upstream torch.hub auto-download is intentionally not used")
    return dino_repo, dino_weights


def _load_local_dino(torch: Any, dino_repo: Path, dino_weights: str, name: str) -> Any:
    model = torch.hub.load(str(dino_repo), name, source="local", pretrained=False)
    if str(dino_weights).endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(dino_weights)
    else:
        state = torch.load(dino_weights, map_location="cpu", weights_only=True)
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _dino_transform() -> Any:
    from torchvision import transforms

    return transforms.Compose([transforms.Normalize(mean=DINO_MEAN, std=DINO_STD)])


def _load_components(component_paths: Mapping[str, str]) -> dict[str, Any]:
    from safetensors.torch import load_file
    from trellis import models as trellis_models

    models = {}
    for name, stem in component_paths.items():
        component_config = read_json(stem + ".json")
        model_type = getattr(trellis_models, component_config["name"])
        model = model_type(**component_config["args"])
        model.load_state_dict(load_file(stem + ".safetensors"), strict=True)
        models[name] = model
    return models


# -------------------------------------------
# 推理与导出
# -------------------------------------------
def run(model: SimpleNamespace, request: Mapping[str, Any]) -> dict:
    """从补全图生成同一潜变量的 mesh 与 Gaussian，并以相同坐标系导出。

    Returns:
        `mesh_path`、`gaussian_path`、`metadata_path`、`gaussian_format`、`foreground_mask_path`。

    Raises:
        ValueError: 前景不可用，或模型返回空/非有限 mesh、非零 SH 阶 Gaussian。
    """
    config = request["config"]
    rgba = _foreground_image(model, request["object"]["image_path"])
    foreground = validate_foreground(np.asarray(rgba)[:, :, 3], bool(config.get("require_margin", True)))
    directory = Path(request["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    save_image(directory / "completed_rgba.png", rgba)
    save_image(directory / "completed_mask.png", (foreground * 255).astype(np.uint8))
    conditioned = model.pipeline.preprocess_image(rgba)
    save_image(directory / "trellis_input.png", conditioned)
    mesh, gaussian = _generate_assets(model, conditioned, config)
    _normalize_gaussian_rotations(model.torch, gaussian)
    _clamp_gaussian_opacities(model.torch, gaussian)
    glb = model.postprocessing_utils.to_glb(
        gaussian,
        mesh,
        simplify=float(config.get("simplify", DEFAULT_SIMPLIFY_RATIO)),
        texture_size=int(config.get("texture_size", DEFAULT_TEXTURE_SIZE)),
        fill_holes=bool(config.get("fill_holes", False)),
        verbose=True,
    )
    mesh_path, gaussian_path = directory / "mesh.glb", directory / "gaussian.ply"
    glb.export(str(mesh_path))
    gaussian.save_ply(str(gaussian_path), transform=GLB_FROM_NATIVE.tolist())
    metadata_path = directory / "asset_meta.json"
    metadata = _asset_metadata(model, config)
    atomic_json(metadata_path, metadata)
    return {
        "mesh_path": str(mesh_path),
        "gaussian_path": str(gaussian_path),
        "metadata_path": str(metadata_path),
        "gaussian_format": metadata["gaussian_format"],
        "foreground_mask_path": str(directory / "completed_mask.png"),
    }


def _foreground_image(model: SimpleNamespace, image_path: str) -> Image.Image:
    with Image.open(image_path) as image:
        image = image.copy()
    has_transparency = image.mode == "RGBA" and not np.all(np.asarray(image)[:, :, 3] == 255)
    if has_transparency:
        return image
    # U2Net 会话只在第一次需要抠图时创建，之后在同一已加载模型内复用。
    if model.rembg_session is None:
        model.rembg_session = local_rembg_session(model.rembg, model.rembg_model_path)
    return model.rembg.remove(image.convert("RGB"), session=model.rembg_session)


def _generate_assets(model: SimpleNamespace, conditioned: Image.Image, config: Mapping[str, Any]) -> tuple[Any, Any]:
    """从同一次潜变量采样得到 mesh 与 Gaussian，并校验两者可以导出。"""
    with model.torch.no_grad():
        output = model.pipeline.run(
            conditioned,
            seed=int(config.get("seed", 0)),
            formats=["mesh", "gaussian"],
            preprocess_image=False,
            sparse_structure_sampler_params=config.get("sparse_structure_sampler_params", {}),
            slat_sampler_params=config.get("slat_sampler_params", {}),
        )
    mesh, gaussian = output["mesh"][0], output["gaussian"][0]
    _validate_outputs(model.torch, mesh, gaussian)
    return mesh, gaussian


def _validate_outputs(torch: Any, mesh: Any, gaussian: Any) -> None:
    is_valid_mesh = len(mesh.vertices) > 0 and len(mesh.faces) > 0 and torch.isfinite(mesh.vertices).all()
    if not is_valid_mesh:
        raise ValueError("TRELLIS returned an empty or non-finite mesh")
    if getattr(gaussian, "sh_degree", 0) != 0:
        raise ValueError("Pinned TRELLIS save_ply only exports DC color; nonzero SH degree would lose information")


def _normalize_gaussian_rotations(torch: Any, gaussian: Any) -> None:
    # 上游 save_ply 直接写出未归一化的 rotation + bias，这里先归一化再写回。
    with torch.no_grad():
        raw_rotations = gaussian._rotation + gaussian.rots_bias[None, :]
        normalized = normalize_export_rotations(raw_rotations.detach().cpu().numpy())
        gaussian.from_rotation(torch.as_tensor(normalized, dtype=raw_rotations.dtype, device=raw_rotations.device))


def _clamp_gaussian_opacities(torch: Any, gaussian: Any) -> None:
    """将导出 opacity 限制在开区间内，保证 3DGS logit 编码有限。"""
    with torch.no_grad():
        opacities = gaussian.get_opacity
        if not bool(torch.isfinite(opacities).all()):
            raise ValueError("TRELLIS returned non-finite Gaussian opacity")
        gaussian.from_opacity(torch.clamp(opacities, min=OPACITY_CLIP_EPSILON, max=1 - OPACITY_CLIP_EPSILON))


def _asset_metadata(model: SimpleNamespace, config: Mapping[str, Any]) -> dict:
    return {
        "model": "TRELLIS-image",
        "checkpoint": config["model_path"],
        "seed": int(config.get("seed", 0)),
        "dino_weights": model.dino_weights,
        "component_checkpoints": model.component_paths,
        "device": model.device,
        "remote_component_fallback": False,
        "rotation_quaternion_order": "wxyz",
        "rotations_normalized": True,
        "opacity_clipped_for_finite_logit": True,
        "native_to_asset_rotation": GLB_FROM_NATIVE.tolist(),
        "same_latent_sample": True,
        "mesh_and_gaussian_same_frame": True,
        "asset_up": "Y",
        "asset_scale": "generative_normalized",
        "gaussian_format": "3dgs_sh0_logscale_logitopacity_wxyz",
        "synthetic": True,
        "semantic_quality_verified": False,
    }
