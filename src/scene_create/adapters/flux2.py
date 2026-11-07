from __future__ import annotations

import inspect
from pathlib import Path

from PIL import Image

from scene_create.adapters.common import configure_device, resolve_dtype
from scene_create.io import atomic_json, save_image


def completion_prompt(category, extra=""):
    if not str(category).strip():
        raise ValueError("An object category is required for completion")
    return (
        f"Using the reference photograph, restore the same single {category} as a complete intact standalone object. "
        "Preserve its identity, visible shape, material, colors and viewpoint. Complete only missing, broken, cropped or occluded parts. "
        "Show the entire object centered with a clear margin on a plain white background. Do not include other objects, scenery, text or a cast shadow. "
        + str(extra)
    ).strip()


def run(request):
    import diffusers
    import torch

    config, item = request["config"], request["object"]
    class_name = config.get("pipeline_class", "Flux2Pipeline")
    if class_name not in ("Flux2Pipeline", "Flux2KleinPipeline"):
        raise ValueError("Only Flux2Pipeline or Flux2KleinPipeline are supported")
    pipeline_type = getattr(diffusers, class_name, None)
    if pipeline_type is None:
        raise ImportError(
            f"{class_name} is unavailable in diffusers {diffusers.__version__}; install a compatible FLUX.2 release in the worker environment"
        )
    if "image" not in inspect.signature(pipeline_type.__call__).parameters:
        raise RuntimeError(
            "Installed FLUX.2 backend has no verified reference-image editing interface"
        )
    device = configure_device(torch, config.get("device", "cuda"))
    dtype_name = resolve_dtype(
        config.get("dtype", "bfloat16"),
        device,
        bf16_supported=device.startswith("cuda") and torch.cuda.is_bf16_supported(),
    )
    width, height = int(config.get("width", 768)), int(config.get("height", 768))
    if min(width, height) < 64 or width % 16 or height % 16:
        raise ValueError(
            "FLUX.2 image dimensions must be multiples of 16 and at least 64"
        )
    model = pipeline_type.from_pretrained(
        config["model_path"],
        torch_dtype=getattr(torch, dtype_name),
        local_files_only=not config.get("allow_download", False),
    )
    if config.get("cpu_offload", True) and device.startswith("cuda"):
        model.enable_model_cpu_offload(device=device)
    else:
        model.to(device)
    references = []
    for path in [item["image_path"], *item.get("reference_images", [])]:
        with Image.open(path) as image:
            references.append(image.convert("RGB"))
    prompt = config.get("prompt") or completion_prompt(
        item["category"], config.get("prompt_suffix", "")
    )
    seed = int(config.get("seed", 0))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.inference_mode():
        result = model(
            image=references,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=int(config.get("num_inference_steps", 28)),
            guidance_scale=float(config.get("guidance_scale", 4.0)),
            generator=generator,
        )
    if not result.images:
        raise RuntimeError("FLUX.2 returned no image")
    path = Path(request["output_dir"]) / "completed.png"
    save_image(path, result.images[0].convert("RGB"))
    metadata = {
        "model": class_name,
        "diffusers_version": diffusers.__version__,
        "checkpoint": config["model_path"],
        "seed": seed,
        "prompt": prompt,
        "reference_images": [item["image_path"], *item.get("reference_images", [])],
        "editing_mode": "reference_conditioned_generation",
        "pixel_locked_inpainting": False,
        "synthetic": True,
        "semantic_quality_verified": False,
    }
    atomic_json(Path(request["output_dir"]) / "generation.json", metadata)
    return {"image_path": str(path), "metadata": metadata}
