from __future__ import annotations

from pathlib import Path

import numpy as np

from scene_create.adapters.common import (
    configure_device,
    require_local_model,
    resolve_dtype,
)
from scene_create.io import atomic_npz


def preprocessing_maps(sizes, process_res=504, method="upper_bound_resize"):
    if (
        method
        not in (
            "upper_bound_resize",
            "upper_bound_crop",
            "lower_bound_resize",
            "lower_bound_crop",
        )
        or process_res < 28
    ):
        raise ValueError("Unsupported DA3 preprocessing configuration")
    shapes, mappings = [], []
    for width, height in sizes:
        denominator = (
            max(width, height) if method.startswith("upper") else min(width, height)
        )
        scale = process_res / denominator
        w1, h1 = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
        if method.endswith("resize"):
            w2, h2 = (
                max(1, int(np.floor(w1 / 14 + 0.5)) * 14),
                max(1, int(np.floor(h1 / 14 + 0.5)) * 14),
            )
            sx, sy, dx, dy = w2 / width, h2 / height, 0, 0
        else:
            w2, h2 = w1 // 14 * 14, h1 // 14 * 14
            sx, sy = w1 / width, h1 / height
            dx, dy = (w1 - w2) // 2, (h1 - h2) // 2
        if min(w2, h2) < 14:
            raise ValueError(
                "Image aspect ratio creates a DA3 dimension smaller than one patch"
            )
        mappings.append(
            np.array(
                [[sx, 0, (sx - 1) / 2 - dx], [0, sy, (sy - 1) / 2 - dy], [0, 0, 1]],
                dtype=float,
            )
        )
        shapes.append((h2, w2))
    target_h, target_w = (
        min(shape[0] for shape in shapes),
        min(shape[1] for shape in shapes),
    )
    for matrix, (height, width) in zip(mappings, shapes):
        matrix[0, 2] -= (width - target_w) // 2
        matrix[1, 2] -= (height - target_h) // 2
    return mappings, (target_h, target_w)


def configured_model_type(base, torch, dtype_name):
    class ConfiguredDA3(base):
        def forward(
            self,
            image,
            extrinsics=None,
            intrinsics=None,
            export_feat_layers=None,
            infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy="saddle_balanced",
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


def prediction_scale_status(model_name, is_metric):
    if is_metric is None:
        metric = False
    else:
        flag = np.asarray(is_metric)
        if flag.size != 1 or flag.item() not in (0, 1):
            raise ValueError("DA3 is_metric must be a scalar boolean flag")
        metric = bool(flag.item())
    if "nested" in model_name.lower() and not metric:
        raise ValueError("Nested DA3 did not confirm metric scaling in its output")
    return "model_metric_estimate" if metric else "relative"


def run(request):
    config, views = request["config"], request["views"]
    if not views or len(views) > int(config.get("max_views", 16)):
        raise ValueError(
            "DA3 view count exceeds max_views; independent batches cannot be fused without inter-batch alignment"
        )
    model_path = require_local_model(
        config.get("model_path"),
        ("config.json", "model.safetensors"),
        allow_download=config.get("allow_download", False),
    )
    method = config.get("process_res_method", "upper_bound_resize")
    resolution = int(config.get("process_res", 504))
    mappings, shape = preprocessing_maps(
        [(view["width"], view["height"]) for view in views], resolution, method
    )

    import torch
    from depth_anything_3.api import DepthAnything3

    device = configure_device(torch, config.get("device", "cuda"))
    dtype_name = resolve_dtype(
        config.get("dtype", "auto"),
        device,
        bf16_supported=device.startswith("cuda") and torch.cuda.is_bf16_supported(),
    )
    torch.manual_seed(config.get("seed", 0))
    model = configured_model_type(DepthAnything3, torch, dtype_name).from_pretrained(
        model_path,
        local_files_only=not config.get("allow_download", False),
    )
    name = str(model.model_name).lower()
    if "metric" in name or "mono" in name:
        raise ValueError(
            "Standalone DA3 metric/mono checkpoints do not supply the required joint camera solution; use an any-view or nested checkpoint"
        )
    model = model.to(device=device)
    model.eval()
    with torch.inference_mode():
        prediction = model.inference(
            image=[view["image_path"] for view in views],
            process_res=resolution,
            process_res_method=method,
            infer_gs=False,
            use_ray_pose=bool(config.get("use_ray_pose", False)),
            ref_view_strategy=config.get("ref_view_strategy", "saddle_balanced"),
            export_dir=None,
        )
    if (
        prediction.intrinsics is None
        or prediction.extrinsics is None
        or prediction.conf is None
    ):
        raise ValueError(
            "DA3 checkpoint did not produce intrinsics, extrinsics and confidence"
        )
    depths = np.asarray(prediction.depth)
    if depths.shape != (len(views), *shape):
        raise ValueError(
            f"DA3 pixel mapping changed: predicted {depths.shape}, expected {(len(views), *shape)}; revalidate upstream preprocessing"
        )
    extrinsics = np.asarray(prediction.extrinsics)
    if extrinsics.shape == (len(views), 3, 4):
        matrices = np.broadcast_to(np.eye(4), (len(views), 4, 4)).copy()
        matrices[:, :3] = extrinsics
        extrinsics = matrices
    if extrinsics.shape != (len(views), 4, 4):
        raise ValueError("Unexpected DA3 extrinsics shape")
    world_from_camera = np.linalg.inv(extrinsics)
    scene_from_world = extrinsics[0]
    scale_status = prediction_scale_status(name, getattr(prediction, "is_metric", None))
    confidences = np.asarray(prediction.conf, dtype=np.float32)
    images = np.asarray(prediction.processed_images)
    if (
        confidences.shape != depths.shape
        or images.shape != (*depths.shape, 3)
        or images.dtype != np.uint8
    ):
        raise ValueError(
            "DA3 depth, confidence and uint8 RGB must share the same pixel grid"
        )
    sky = getattr(prediction, "sky", None)
    if sky is not None and np.asarray(sky).shape != depths.shape:
        raise ValueError("DA3 sky mask has an unexpected pixel grid")
    records = []
    for index, view in enumerate(views):
        confidence = confidences[index]
        depth = depths[index].astype(np.float32)
        image = images[index]
        valid = (
            np.isfinite(depth)
            & (depth > float(config.get("min_depth", 0)))
            & (depth > 0)
            & np.isfinite(confidence)
            & (confidence >= max(0, float(config.get("min_confidence", 0))))
        )
        if config.get("exclude_sky", True) and sky is not None:
            sky_view = np.asarray(sky[index])
            valid &= np.isfinite(sky_view) & (sky_view < 0.5)
        path = Path(request["output_dir"]) / f"{view['view_id']}.npz"
        atomic_npz(
            path,
            depth=depth,
            conf=confidence,
            intrinsics=prediction.intrinsics[index],
            world_from_camera=scene_from_world @ world_from_camera[index],
            image=image,
            original_to_processed=mappings[index],
            valid=valid,
        )
        records.append(
            {
                "view_id": view["view_id"],
                "geometry_path": str(path),
                "scale_status": scale_status,
                "scale_source": name,
            }
        )
    return {
        "views": records,
        "provenance": {
            "model": name,
            "checkpoint": model_path,
            "device": device,
            "requested_dtype": config.get("dtype", "auto"),
            "effective_backbone_dtype": dtype_name,
            "geometry_head_dtype": "float32",
            "is_metric": scale_status == "model_metric_estimate",
            "reported_scale_factor": getattr(prediction, "scale_factor", None),
            "sky_filtered": bool(config.get("exclude_sky", True) and sky is not None),
            "joint_inference": True,
            "depth_convention": "optical_axis",
            "mask_mapping_convention": "opencv_resize_half_pixel",
            "camera_convention": "opencv",
        },
    }
