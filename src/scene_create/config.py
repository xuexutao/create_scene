from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIRS = {
    "sam3": "sam3",
    "da3": "Depth-Anything-3",
    "flux2": "flux2",
    "trellis": "TRELLIS",
    "maskgaussian": "MaskGaussian",
}
STAGES = (
    "ingest",
    "segment",
    "depth",
    "objects",
    "complete",
    "assets",
    "register",
    "background",
    "assemble",
)

DEFAULTS = {
    "runtime": {
        "seed": 0,
        "worker_timeout": 3600,
        "retries": 0,
        "continue_on_object_error": True,
    },
    "models": {
        name: {
            "python": None,
            "repo_path": f"third_parts/{directory}",
            "model_path": None,
            "device": "cuda",
            "dtype": "float16",
            "allow_download": False,
        }
        for name, directory in MODEL_DIRS.items()
    },
    "geometry": {
        "min_confidence": 0.0,
        "min_points": 32,
        "outlier_neighbors": 16,
        "outlier_std_ratio": 2.5,
    },
    "fusion": {
        "voxel_size": None,
        "voxel_ratio": 0.005,
        "match_distance_ratio": 0.08,
        "min_overlap": 0.35,
        "max_color_distance": 0.4,
        "ambiguity_margin": 0.1,
    },
    "completion": {"padding": 0.25, "image_size": 768},
    "registration": {
        "sample_count": 12000,
        "min_fitness": 0.5,
        "max_rmse_ratio": 0.04,
        "min_mask_iou": 0.25,
        "max_relative_depth_error": 0.15,
        "quality_max_side": 256,
    },
    "background": {
        "mode": "train",
        "voxel_size": None,
        "voxel_ratio": 0.003,
        "max_points": 250000,
        "mask_dilation": 3,
    },
}
DEFAULTS["models"]["sam3"].update(
    {
        "categories": [
            "chair",
            "table",
            "sofa",
            "cabinet",
            "lamp",
            "bottle",
            "cup",
            "book",
            "vase",
            "plant",
            "monitor",
            "keyboard",
            "bag",
            "box",
        ],
        "score_threshold": 0.5,
    }
)
DEFAULTS["models"]["da3"].update(
    {"max_views": 16, "process_res": 504, "exclude_sky": True}
)
DEFAULTS["models"]["flux2"].update(
    {
        "pipeline_class": "Flux2Pipeline",
        "num_inference_steps": 28,
        "guidance_scale": 4.0,
        "cpu_offload": True,
    }
)
DEFAULTS["models"]["trellis"].update(
    {
        "texture_size": 1024,
        "dino_repo_path": "third_parts/dinov2",
        "dino_model_path": None,
        "rembg_model_path": "checkpoints/rembg/u2net.onnx",
        "attention_backend": "xformers",
    }
)
DEFAULTS["models"]["maskgaussian"].update(
    {"iterations": 1000, "sh_degree": 0, "depth_weight": 0.1}
)


def merge_dict(base: dict, update: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path=None, root=None) -> dict:
    root = Path(root or ROOT).resolve()
    settings = copy.deepcopy(DEFAULTS)
    if path:
        with Path(path).open(encoding="utf-8") as stream:
            supplied = yaml.safe_load(stream) or {}
        if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
            raise ValueError(
                "Configuration must be a mapping with known top-level sections"
            )
        settings = merge_dict(settings, supplied)
    if not set(settings["models"]).issubset(MODEL_DIRS):
        raise ValueError("Unknown model in configuration")
    for name, model in settings["models"].items():
        if not isinstance(model, dict):
            raise ValueError(f"models.{name} must be a mapping")
        for key in (
            "repo_path",
            "model_path",
            "dino_repo_path",
            "dino_model_path",
            "rembg_model_path",
        ):
            value = model.get(key)
            if value:
                expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
                if (
                    key == "model_path"
                    and model.get("allow_download")
                    and not expanded.is_absolute()
                    and not (root / expanded).exists()
                ):
                    continue
                model[key] = str(
                    expanded if expanded.is_absolute() else root / expanded
                )
        model["python"] = os.path.expandvars(model.get("python") or sys.executable)
        model.setdefault("seed", settings["runtime"]["seed"])
    if settings["runtime"]["worker_timeout"] <= 0 or settings["runtime"]["retries"] < 0:
        raise ValueError("Invalid timeout or retry count")
    if (
        settings["geometry"]["min_points"] < 3
        or settings["background"]["max_points"] < 3
    ):
        raise ValueError("Point limits must be at least three")
    if settings["background"]["mode"] not in ("train", "init-only"):
        raise ValueError("background.mode must be train or init-only")
    if not settings["models"]["sam3"]["categories"]:
        raise ValueError("SAM3 requires at least one category prompt")
    for section in ("fusion", "background"):
        if settings[section]["voxel_ratio"] <= 0 or (
            settings[section]["voxel_size"] is not None
            and settings[section]["voxel_size"] <= 0
        ):
            raise ValueError("Voxel sizes must be positive")
    settings["root"] = str(root)
    return settings
