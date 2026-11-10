"""SAM3 文本类别实例分割适配器：按配置类别逐视图生成去重后的 0/255 实例 mask。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from scene_create.io import save_image
from scene_create.runtime import configure_device, resolve_dtype

# -------------------------------------------
# 配置常量
# -------------------------------------------
# fmt: off
DEFAULT_DEDUP_IOU = 0.85        # 同一视图内两个候选 mask 的 IoU 达到该值即视为重复
DEFAULT_MIN_MASK_PIXELS = 16    # 小于该像素数的候选 mask 直接丢弃，单位：像素
DEFAULT_SCORE_THRESHOLD = 0.5   # SAM3 处理器保留候选实例的最低置信度
DEFAULT_DTYPE = "float16"       # 未配置 dtype 时的推理精度
CHECKPOINT_NAME = "sam3.pt"     # model_path 指向目录时，目录内的权重文件名
# fmt: on


# -------------------------------------------
# 候选 mask 处理
# -------------------------------------------
def deduplicate_masks(
    candidates: Sequence[Mapping[str, Any]],
    threshold: float = DEFAULT_DEDUP_IOU,
    min_pixels: int = DEFAULT_MIN_MASK_PIXELS,
) -> list[dict]:
    """按置信度从高到低保留互不重复的候选 mask。

    Args:
        candidates: 含 `mask`、`score` 的候选实例。
        threshold: 与已保留 mask 的 IoU 达到该值即视为重复。
        min_pixels: 小于该前景像素数的候选被丢弃。

    Returns:
        保留的候选，`mask` 统一转换为 bool 数组。
    """
    kept = []
    for item in sorted(candidates, key=lambda value: value["score"], reverse=True):
        mask = np.asarray(item["mask"], dtype=bool)
        if mask.sum() < min_pixels:
            continue
        if not _overlaps_kept_mask(mask, kept, threshold):
            kept.append({**item, "mask": mask})
    return kept


def _overlaps_kept_mask(mask: np.ndarray, kept: Sequence[Mapping[str, Any]], threshold: float) -> bool:
    for existing in kept:
        intersection = np.logical_and(mask, existing["mask"]).sum()
        union = np.logical_or(mask, existing["mask"]).sum()
        if intersection / max(union, 1) >= threshold:
            return True
    return False


# -------------------------------------------
# 模型加载
# -------------------------------------------
def load_model(builder: Callable[..., Any], checkpoint: Optional[str], device: str, allow_download: bool = False) -> Any:
    """先在 CPU 上构建 SAM3，再显式放置到目标设备，避免上游默认使用 cuda:0。"""
    model = builder(
        checkpoint_path=checkpoint,
        load_from_HF=bool(allow_download),
        device="cpu",
        eval_mode=True,
    )
    return model.to(device).eval()


def load(config: Mapping[str, Any], root: Path) -> SimpleNamespace:
    """校验类别与本地权重后加载 SAM3 图像模型和处理器。

    Raises:
        ValueError: 类别提示为空或格式错误。
        FileNotFoundError: 本地权重缺失且禁止下载。
    """
    _validate_categories(config.get("categories", []))
    model_path = _resolve_checkpoint(config)

    # 延迟加载：主环境可以不安装 torch 和 SAM3。
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    device = configure_device(torch, config.get("device", "cuda"))
    is_bf16_supported = device.startswith("cuda") and torch.cuda.is_bf16_supported()
    dtype_name = resolve_dtype(config.get("dtype", DEFAULT_DTYPE), device, bf16_supported=is_bf16_supported)
    torch.manual_seed(config.get("seed", 0))
    model = load_model(build_sam3_image_model, model_path, device, config.get("allow_download", False))
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=float(config.get("score_threshold", DEFAULT_SCORE_THRESHOLD)),
    )
    return SimpleNamespace(
        torch=torch,
        processor=processor,
        device=device,
        dtype_name=dtype_name,
        checkpoint=model_path,
    )


def _validate_categories(categories: Any) -> None:
    if not isinstance(categories, list) or not categories:
        raise ValueError("SAM3 requires nonempty category prompts")
    for category in categories:
        if not isinstance(category, str) or not category.strip():
            raise ValueError("SAM3 requires nonempty category prompts")


def _resolve_checkpoint(config: Mapping[str, Any]) -> Optional[str]:
    model_path = config.get("model_path")
    if model_path and Path(model_path).is_dir():
        model_path = str(Path(model_path) / CHECKPOINT_NAME)
    if model_path and not Path(model_path).is_file():
        raise FileNotFoundError(model_path)
    if not model_path and not config.get("allow_download", False):
        raise FileNotFoundError("SAM3 requires a local model_path; downloads are disabled")
    return model_path


# -------------------------------------------
# 推理
# -------------------------------------------
def run(model: SimpleNamespace, request: Mapping[str, Any]) -> dict:
    """对每个视图按类别提示分割，去重后写出实例 mask。

    Returns:
        `instances` 列表（每项含 `view_id`、`local_id`、`mask_path`、`score`、`category`、`bbox_xyxy`）和 `provenance`。
    """
    config = request["config"]
    instances = []
    with model.torch.inference_mode(), _autocast_context(model):
        for view in request["views"]:
            candidates = _segment_view(model.processor, view, config["categories"])
            kept = deduplicate_masks(
                candidates,
                float(config.get("dedup_iou", DEFAULT_DEDUP_IOU)),
                int(config.get("min_mask_pixels", DEFAULT_MIN_MASK_PIXELS)),
            )
            instances.extend(_save_instances(kept, view, Path(request["output_dir"])))
    return {
        "instances": instances,
        "provenance": {
            "model": "sam3",
            "checkpoint": model.checkpoint,
            "device": model.device,
            "requested_dtype": config.get("dtype", DEFAULT_DTYPE),
            "effective_dtype": model.dtype_name,
            "prompt_source": "configured_categories",
            "exhaustive_interactability_guaranteed": False,
        },
    }


def _autocast_context(model: SimpleNamespace) -> Any:
    is_reduced_precision_cuda = model.device.startswith("cuda") and model.dtype_name != "float32"
    if not is_reduced_precision_cuda:
        return nullcontext()
    return model.torch.autocast(device_type="cuda", dtype=getattr(model.torch, model.dtype_name))


def _segment_view(processor: Any, view: Mapping[str, Any], categories: Sequence[str]) -> list[dict]:
    with Image.open(view["image_path"]) as image:
        state = processor.set_image(image.convert("RGB"))
    candidates = []
    for category in categories:
        processor.reset_all_prompts(state)
        prediction = processor.set_text_prompt(prompt=category, state=state)
        masks = prediction["masks"].detach().cpu().numpy()
        boxes = prediction["boxes"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        if masks.ndim == 4:
            masks = masks[:, 0]
        if masks.shape[1:] != (view["height"], view["width"]):
            raise ValueError("SAM3 returned masks with an unexpected image resolution")
        for mask, box, score in zip(masks, boxes, scores):
            candidates.append({"mask": mask, "bbox_xyxy": box.tolist(), "score": float(score), "category": category})
    return candidates


def _save_instances(kept: Sequence[dict], view: Mapping[str, Any], output_dir: Path) -> list[dict]:
    instances = []
    for index, item in enumerate(kept):
        local_id = f"instance_{index:03d}"
        target = output_dir / view["view_id"] / f"{local_id}.png"
        save_image(target, (item.pop("mask") * 255).astype(np.uint8))
        instances.append({**item, "view_id": view["view_id"], "local_id": local_id, "mask_path": str(target)})
    return instances
