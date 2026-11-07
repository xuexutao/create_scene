from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from scene_create.adapters.common import configure_device, resolve_dtype
from scene_create.io import save_image


def deduplicate_masks(candidates, threshold=0.85, min_pixels=16):
    kept = []
    for item in sorted(candidates, key=lambda value: value["score"], reverse=True):
        mask = np.asarray(item["mask"], dtype=bool)
        if mask.sum() < min_pixels:
            continue
        duplicate = False
        for existing in kept:
            intersection = np.logical_and(mask, existing["mask"]).sum()
            union = np.logical_or(mask, existing["mask"]).sum()
            if intersection / max(union, 1) >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append({**item, "mask": mask})
    return kept


def load_model(builder, checkpoint, device, allow_download=False):
    model = builder(
        checkpoint_path=checkpoint,
        load_from_HF=bool(allow_download),
        device="cpu",
        eval_mode=True,
    )
    return model.to(device).eval()


def run(request):
    config = request["config"]
    categories = config.get("categories", [])
    if (
        not isinstance(categories, list)
        or not categories
        or any(
            not isinstance(category, str) or not category.strip()
            for category in categories
        )
    ):
        raise ValueError("SAM3 requires nonempty category prompts")
    model_path = config.get("model_path")
    if model_path and Path(model_path).is_dir():
        model_path = str(Path(model_path) / "sam3.pt")
    if model_path and not Path(model_path).is_file():
        raise FileNotFoundError(model_path)
    if not model_path and not config.get("allow_download", False):
        raise FileNotFoundError(
            "SAM3 requires a local model_path; downloads are disabled"
        )

    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = configure_device(torch, config.get("device", "cuda"))
    dtype_name = resolve_dtype(
        config.get("dtype", "float16"),
        device,
        bf16_supported=device.startswith("cuda") and torch.cuda.is_bf16_supported(),
    )
    torch.manual_seed(config.get("seed", 0))
    model = load_model(
        build_sam3_image_model, model_path, device, config.get("allow_download", False)
    )
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=float(config.get("score_threshold", 0.5)),
    )
    context = (
        torch.autocast(device_type="cuda", dtype=getattr(torch, dtype_name))
        if device.startswith("cuda") and dtype_name != "float32"
        else nullcontext()
    )
    instances = []
    with torch.inference_mode(), context:
        for view in request["views"]:
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
                    raise ValueError(
                        "SAM3 returned masks with an unexpected image resolution"
                    )
                for mask, box, score in zip(masks, boxes, scores):
                    candidates.append(
                        {
                            "mask": mask,
                            "bbox_xyxy": box.tolist(),
                            "score": float(score),
                            "category": category,
                        }
                    )
            kept = deduplicate_masks(
                candidates,
                float(config.get("dedup_iou", 0.85)),
                int(config.get("min_mask_pixels", 16)),
            )
            for index, item in enumerate(kept):
                local_id = f"instance_{index:03d}"
                target = (
                    Path(request["output_dir"]) / view["view_id"] / f"{local_id}.png"
                )
                save_image(target, (item.pop("mask") * 255).astype(np.uint8))
                instances.append(
                    {
                        **item,
                        "view_id": view["view_id"],
                        "local_id": local_id,
                        "mask_path": str(target),
                    }
                )
    return {
        "instances": instances,
        "provenance": {
            "model": "sam3",
            "checkpoint": model_path,
            "device": device,
            "requested_dtype": config.get("dtype", "float16"),
            "effective_dtype": dtype_name,
            "prompt_source": "configured_categories",
            "exhaustive_interactability_guaranteed": False,
        },
    }
