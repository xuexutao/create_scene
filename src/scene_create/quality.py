from __future__ import annotations

import numpy as np
from PIL import Image

from scene_create.geometry.raster import load_mesh_arrays, rasterize_depth
from scene_create.io import load_mask, load_npz, save_image
from scene_create.observations import map_mask


def validate_asset_pose(item, transform, output, config):
    vertices, faces = load_mesh_arrays(item["asset"]["mesh_path"], transform)
    metrics, artifacts = [], []
    for observation in item["observations"]:
        view_id = observation["view_id"]
        data = load_npz(output / "pointmaps" / f"{view_id}.npz")
        height, width = data["depth"].shape
        rendered = rasterize_depth(
            vertices,
            faces,
            data["intrinsics"],
            data["world_from_camera"],
            width,
            height,
            max_side=int(config.get("quality_max_side", 256)),
        )
        shape = rendered["depth"].shape
        target = map_mask(
            load_mask(observation["geometry_mask_path"]), rendered["pixel_map"], shape
        )
        valid = map_mask(data["valid"], rendered["pixel_map"], shape)
        image = Image.fromarray(data["image"]).resize(
            (rendered["width"], rendered["height"]), Image.Resampling.BILINEAR
        )
        reference_depth = np.asarray(
            Image.fromarray(data["depth"].astype(np.float32), mode="F").resize(
                (rendered["width"], rendered["height"]), Image.Resampling.NEAREST
            )
        )
        tolerance = float(config.get("occlusion_tolerance_ratio", 0.05)) * np.maximum(
            reference_depth, 1e-6
        )
        predicted = (
            rendered["mask"]
            & valid
            & (rendered["depth"] <= reference_depth + tolerance)
        )
        expected = target & valid
        overlap = predicted & expected
        union = predicted | expected
        iou = float(overlap.sum() / max(int(union.sum()), 1))
        residual = np.abs(
            rendered["depth"][overlap] - reference_depth[overlap]
        ) / np.maximum(reference_depth[overlap], 1e-6)
        depth_error = float(np.median(residual)) if len(residual) else None
        metrics.append(
            {
                "view_id": view_id,
                "visible_mask_iou": iou,
                "median_relative_depth_error": depth_error,
                "observed_pixels": int(expected.sum()),
                "overlap_pixels": int(overlap.sum()),
            }
        )
        overlay = np.asarray(image).astype(float) * 0.6
        overlay[expected] += np.array([0, 80, 0])
        overlay[predicted] += np.array([80, 0, 35])
        path = (
            output
            / "objects"
            / item["object_id"]
            / "registration"
            / f"{view_id}_alignment.png"
        )
        save_image(path, np.clip(overlay, 0, 255).astype(np.uint8))
        artifacts.append(path)
    usable = [metric for metric in metrics if metric["observed_pixels"] > 0]
    mean_iou = (
        float(np.mean([metric["visible_mask_iou"] for metric in usable]))
        if usable
        else 0.0
    )
    depth_errors = [
        metric["median_relative_depth_error"]
        for metric in usable
        if metric["median_relative_depth_error"] is not None
    ]
    median_depth_error = float(np.median(depth_errors)) if depth_errors else None
    accepted = (
        bool(usable)
        and mean_iou >= float(config.get("min_mask_iou", 0.25))
        and median_depth_error is not None
        and median_depth_error <= float(config.get("max_relative_depth_error", 0.15))
    )
    return {
        "silhouette_checked": True,
        "mean_visible_mask_iou": mean_iou,
        "median_relative_depth_error": median_depth_error,
        "per_view": metrics,
        "reprojection_accepted": accepted,
        "preview_kind": "mask_alignment_overlay_not_photorealistic_render",
    }, artifacts
