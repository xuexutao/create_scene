from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from scene_create.assets.gaussian import initialize_gaussians, inspect_gaussians
from scene_create.geometry.camera import project_points
from scene_create.geometry.cloud import voxel_fuse
from scene_create.io import (
    atomic_json,
    load_mask,
    load_npz,
    read_cloud,
    save_image,
    write_cloud,
)
from scene_create.objects import extent


def reconstruct_background(registered, geometry, config, runner, destination, output):
    directory = output / "background"
    directory.mkdir(parents=True, exist_ok=True)
    accepted = [
        item for item in registered["objects"] if item["status"] == "registered"
    ]
    points, colors, weights, training_views, artifacts = [], [], [], [], []
    clouds = {
        item["object_id"]: read_cloud(item["cloud_path"])["points"] for item in accepted
    }
    for view in geometry["views"]:
        data = load_npz(view["geometry_path"])
        foreground = np.zeros(data["depth"].shape, dtype=bool)
        for item in accepted:
            observations = [
                observation
                for observation in item["observations"]
                if observation["view_id"] == view["view_id"]
            ]
            if observations:
                foreground |= load_mask(observations[0]["geometry_mask_path"])
            else:
                observed = clouds[item["object_id"]]
                uv, z = project_points(
                    observed, data["intrinsics"], data["world_from_camera"]
                )
                finite = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0)
                u, v = np.zeros(len(z), dtype=int), np.zeros(len(z), dtype=int)
                u[finite], v[finite] = np.rint(uv[finite]).astype(int).T
                inside = (
                    finite
                    & (u >= 0)
                    & (v >= 0)
                    & (u < foreground.shape[1])
                    & (v < foreground.shape[0])
                )
                indices = np.flatnonzero(inside)
                visible = np.abs(
                    data["depth"][v[indices], u[indices]] - z[indices]
                ) <= 0.03 * np.maximum(z[indices], 1e-6)
                indices = indices[visible]
                foreground[v[indices], u[indices]] = True
        dilation = int(config["background"]["mask_dilation"])
        if dilation > 0 and foreground.any():
            foreground = binary_dilation(foreground, iterations=dilation)
        valid = (
            data["valid"] & ~foreground & np.isfinite(data["points_world"]).all(axis=-1)
        )
        mask_path = directory / "masks" / f"{view['view_id']}.png"
        save_image(mask_path, (valid * 255).astype(np.uint8))
        artifacts.append(mask_path)
        training_views.append(
            {
                "view_id": view["view_id"],
                "image_path": view["image_path"],
                "mask_path": str(mask_path),
                "geometry_path": view["geometry_path"],
            }
        )
        points.append(data["points_world"][valid])
        colors.append(data["image"][valid])
        weights.append(np.maximum(data["conf"][valid], 1e-6))
    xyz, rgb, confidence = (
        np.concatenate(points),
        np.concatenate(colors),
        np.concatenate(weights),
    )
    result = {
        "removed_object_ids": [item["object_id"] for item in accepted],
        "single_view": len(geometry["views"]) == 1,
        "unobserved_background_completed": False,
    }
    if len(xyz) < 3:
        result.update(
            status="empty", gaussian_path=None, trained=False, approximate=False
        )
        metadata = directory / "metadata.json"
        atomic_json(metadata, result)
        return result, artifacts + [metadata]
    voxel = (
        config["background"]["voxel_size"]
        or extent(xyz) * config["background"]["voxel_ratio"]
    )
    fused = voxel_fuse(xyz, rgb, confidence, voxel)
    xyz, rgb = fused["points"], fused["colors"]
    limit = config["background"]["max_points"]
    if len(xyz) > limit:
        indices = np.random.default_rng(config["runtime"]["seed"]).choice(
            len(xyz), limit, replace=False
        )
        xyz, rgb = xyz[indices], rgb[indices]
    initial_ply = directory / "initial_points.ply"
    initial_gaussian = directory / "initial_gaussian.ply"
    write_cloud(initial_ply, xyz, rgb)
    initialize_gaussians(xyz, rgb, initial_gaussian)
    dataset = directory / "dataset.json"
    atomic_json(
        dataset,
        {
            "schema_version": 1,
            "initial_ply": str(initial_ply),
            "views": training_views,
            "world_frame": "first_input_camera_opencv",
            "scale_status": geometry["scale_status"],
            "removed_object_ids": result["removed_object_ids"],
        },
    )
    artifacts.extend([initial_ply, initial_gaussian, dataset])
    if config["background"]["mode"] == "init-only":
        result.update(
            status="initialization_only",
            gaussian_path=str(initial_gaussian),
            trained=False,
            approximate=True,
        )
    else:
        trained = runner.call("maskgaussian", {"dataset_path": str(dataset)}, directory)
        gaussian_path = trained.get(
            "gaussian_path", trained.get("background_gaussian_path")
        )
        if not gaussian_path or not Path(gaussian_path).is_file():
            raise ValueError("MaskGaussian returned no Gaussian PLY")
        if inspect_gaussians(gaussian_path)["count"] <= 0:
            raise ValueError("MaskGaussian returned an empty background")
        result.update(
            status="single_view_approximate" if result["single_view"] else "trained",
            gaussian_path=gaussian_path,
            trained=True,
            approximate=result["single_view"],
            training=trained,
        )
        artifacts.append(gaussian_path)
    metadata = directory / "metadata.json"
    atomic_json(metadata, result)
    return result, artifacts + [metadata]
