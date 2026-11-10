"""Diffusers 物体补全适配器：支持 FLUX.2 参考图生成与 SDXL Inpainting。"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image, ImageOps

from scene_create.io import atomic_json, save_image
from scene_create.runtime import configure_device, resolve_dtype

# -------------------------------------------
# 配置常量
# -------------------------------------------
# fmt: off
DEFAULT_IMAGE_SIDE = 768              # 生成图像的默认宽高，单位：像素
IMAGE_SIDE_MULTIPLE = 16              # FLUX.2 要求宽高为该值的整数倍，单位：像素
MIN_IMAGE_SIDE = 64                   # 生成图像允许的最小宽高，单位：像素
DEFAULT_INFERENCE_STEPS = 28          # 默认扩散采样步数
DEFAULT_GUIDANCE_SCALE = 4.0          # 默认文本引导强度
DEFAULT_DTYPE = "bfloat16"            # 未配置 dtype 时的推理精度
INPAINTING_PIPELINE = "AutoPipelineForInpainting"
SUPPORTED_PIPELINES = ("Flux2Pipeline", "Flux2KleinPipeline", INPAINTING_PIPELINE)
# fmt: on


# -------------------------------------------
# 提示词
# -------------------------------------------
def completion_prompt(category: str, extra: str = "") -> str:
    """生成保持物体身份、只补全缺失部分的编辑提示词。

    Raises:
        ValueError: 类别为空。
    """
    if not str(category).strip():
        raise ValueError("An object category is required for completion")
    return (
        f"Using the reference photograph, restore the same single {category} as a complete intact standalone object. "
        "Preserve its identity, visible shape, material, colors and viewpoint. Complete only missing, broken, cropped or occluded parts. "
        "Show the entire object centered with a clear margin on a plain white background. Do not include other objects, scenery, text or a cast shadow. " + str(extra)
    ).strip()


# -------------------------------------------
# 模型加载
# -------------------------------------------
def load(config: Mapping[str, Any], root: Path) -> SimpleNamespace:
    """加载配置指定的 Diffusers 图像补全管线。

    Raises:
        ValueError: 生成尺寸或管线类不受支持。
        ImportError: 当前 diffusers 版本不提供所选管线类。
        RuntimeError: 管线没有所需的图像编辑接口。
    """
    width = int(config.get("width", DEFAULT_IMAGE_SIDE))
    height = int(config.get("height", DEFAULT_IMAGE_SIDE))
    is_valid_size = min(width, height) >= MIN_IMAGE_SIDE and width % IMAGE_SIDE_MULTIPLE == 0 and height % IMAGE_SIDE_MULTIPLE == 0
    if not is_valid_size:
        raise ValueError("Completion image dimensions must be multiples of 16 and at least 64")
    class_name = config.get("pipeline_class", "Flux2Pipeline")
    if class_name not in SUPPORTED_PIPELINES:
        raise ValueError(f"Unsupported completion pipeline: {class_name}")

    # 延迟加载：主环境可以不安装 torch 和 diffusers。
    import diffusers
    import torch

    pipeline_type = _pipeline_type(diffusers, class_name)
    device = configure_device(torch, config.get("device", "cuda"))
    is_bf16_supported = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    dtype_name = resolve_dtype(config.get("dtype", DEFAULT_DTYPE), device, bf16_supported=is_bf16_supported)
    load_options = {
        "torch_dtype": getattr(torch, dtype_name),
        "local_files_only": not config.get("allow_download", False),
    }
    if config.get("variant"):
        load_options["variant"] = config["variant"]
    pipeline = pipeline_type.from_pretrained(config["model_path"], **load_options)
    _validate_pipeline(pipeline, class_name)
    if config.get("cpu_offload", True) and device.startswith("cuda"):
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device)
    return SimpleNamespace(
        torch=torch,
        pipeline=pipeline,
        class_name=class_name,
        implementation_name=type(pipeline).__name__,
        diffusers_version=diffusers.__version__,
        width=width,
        height=height,
    )


def _pipeline_type(diffusers: Any, class_name: str) -> Any:
    pipeline_type = getattr(diffusers, class_name, None)
    if pipeline_type is None:
        raise ImportError(f"{class_name} is unavailable in diffusers {diffusers.__version__}")
    return pipeline_type


def _validate_pipeline(pipeline: Any, class_name: str) -> None:
    parameters = inspect.signature(pipeline.__call__).parameters
    if "image" not in parameters:
        raise RuntimeError(f"{class_name} has no image editing interface")
    if class_name == INPAINTING_PIPELINE and "mask_image" not in parameters:
        raise RuntimeError(f"{class_name} has no inpainting mask interface")


# -------------------------------------------
# 推理
# -------------------------------------------
def run(model: SimpleNamespace, request: Mapping[str, Any]) -> dict:
    """以物体裁剪为参考图生成补全图，写出 `completed.png` 与 `generation.json`。

    Returns:
        `image_path` 和生成元数据 `metadata`。

    Raises:
        RuntimeError: 管线没有返回图像。
    """
    config, item = request["config"], request["object"]
    reference_paths = [item["image_path"], *item.get("reference_images", [])]
    prompt = config.get("prompt") or completion_prompt(item["category"], config.get("prompt_suffix", ""))
    seed = int(config.get("seed", 0))
    generator = model.torch.Generator(device="cpu").manual_seed(seed)
    with model.torch.inference_mode():
        if model.class_name == INPAINTING_PIPELINE:
            result, reference, visible_mask = _run_inpainting(model, config, item, prompt, generator)
        else:
            result = _run_reference_generation(model, config, reference_paths, prompt, generator)
            reference, visible_mask = None, None
    if not result.images:
        raise RuntimeError("Completion pipeline returned no image")
    completed_image = result.images[0].convert("RGB")
    if reference is not None and visible_mask is not None:
        completed_image = Image.composite(reference, completed_image, visible_mask)
    output_dir = Path(request["output_dir"])
    path = output_dir / "completed.png"
    save_image(path, completed_image)
    metadata = {
        "model": model.implementation_name,
        "pipeline": model.class_name,
        "diffusers_version": model.diffusers_version,
        "checkpoint": config["model_path"],
        "seed": seed,
        "prompt": prompt,
        "reference_images": reference_paths,
        "editing_mode": "masked_inpainting" if model.class_name == INPAINTING_PIPELINE else "reference_conditioned_generation",
        "pixel_locked_inpainting": model.class_name == INPAINTING_PIPELINE,
        "synthetic": True,
        "semantic_quality_verified": False,
    }
    atomic_json(output_dir / "generation.json", metadata)
    return {"image_path": str(path), "metadata": metadata}


def _run_reference_generation(
    model: SimpleNamespace,
    config: Mapping[str, Any],
    reference_paths: list[str],
    prompt: str,
    generator: Any,
) -> Any:
    references = []
    for path in reference_paths:
        with Image.open(path) as image:
            references.append(image.convert("RGB"))
    return model.pipeline(
        image=references,
        prompt=prompt,
        height=model.height,
        width=model.width,
        num_inference_steps=int(config.get("num_inference_steps", DEFAULT_INFERENCE_STEPS)),
        guidance_scale=float(config.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)),
        generator=generator,
    )


def _run_inpainting(
    model: SimpleNamespace,
    config: Mapping[str, Any],
    item: Mapping[str, Any],
    prompt: str,
    generator: Any,
) -> tuple[Any, Image.Image, Image.Image]:
    size = (model.width, model.height)
    with Image.open(item["image_path"]) as image:
        reference = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    with Image.open(item["mask_path"]) as image:
        visible_mask = image.convert("L").resize(size, Image.Resampling.NEAREST)
    inpainting_mask = ImageOps.invert(visible_mask)
    result = model.pipeline(
        image=reference,
        mask_image=inpainting_mask,
        prompt=prompt,
        negative_prompt=config.get("negative_prompt"),
        height=model.height,
        width=model.width,
        strength=float(config.get("strength", 0.99)),
        num_inference_steps=int(config.get("num_inference_steps", DEFAULT_INFERENCE_STEPS)),
        guidance_scale=float(config.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)),
        generator=generator,
    )
    return result, reference, visible_mask
