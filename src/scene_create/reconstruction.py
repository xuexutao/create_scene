from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from scene_create.assets.gaussian import inspect_gaussians, transform_gaussians
from scene_create.geometry.registration import register_points, sample_mesh
from scene_create.io import (
    atomic_json,
    file_digest,
    load_mask,
    read_cloud,
    read_json,
    save_image,
    write_cloud,
)


def crop_reference(view, mask_path, destination, config):
    mask = load_mask(mask_path, (view["width"], view["height"]))
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("Cannot crop an empty object mask")
    side = max(
        8,
        int(
            np.ceil(
                max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
                * (1 + 2 * config["padding"])
            )
        ),
    )
    left = int(np.floor((xs.min() + xs.max() + 1 - side) / 2))
    top = int(np.floor((ys.min() + ys.max() + 1 - side) / 2))
    box = (left, top, left + side, top + side)
    with Image.open(view["image_path"]) as image:
        crop = image.convert("RGB").crop(box)
    alpha = Image.fromarray((mask * 255).astype(np.uint8)).crop(box)
    white = Image.new("RGB", crop.size, (255, 255, 255))
    white.paste(crop, mask=alpha)
    size = int(config["image_size"])
    if size < 64:
        raise ValueError("Completion image_size must be at least 64")
    white = white.resize((size, size), Image.Resampling.LANCZOS)
    alpha = alpha.resize((size, size), Image.Resampling.NEAREST)
    image_path, mask_output = destination / "crop.png", destination / "visible_mask.png"
    save_image(image_path, white)
    save_image(mask_output, alpha)
    save_image(destination / "crop_context.png", crop)
    scale = size / side
    metadata = {
        "view_id": view["view_id"],
        "bbox_xyxy": box,
        "size": size,
        "original_to_crop": [
            [scale, 0, -left * scale + (scale - 1) / 2],
            [0, scale, -top * scale + (scale - 1) / 2],
            [0, 0, 1],
        ],
        "observed_pixels_only": True,
    }
    atomic_json(destination / "crop_meta.json", metadata)
    return {"image_path": str(image_path), "mask_path": str(mask_output)}, [
        image_path,
        mask_output,
        destination / "crop_context.png",
        destination / "crop_meta.json",
    ]


def process_objects(previous, stage, config, handler, stage_cache=None):
    result = copy.deepcopy(previous)
    result.setdefault("failures", [])
    artifacts = []
    required = {
        "complete": "observed",
        "assets": "completed_image",
        "register": "generated_asset",
    }[stage]
    for item in result["objects"]:
        if item["status"] != required:
            if stage == "complete" and item["status"] == "insufficient_geometry":
                result["failures"].append(
                    {
                        "object_id": item["object_id"],
                        "stage": stage,
                        "error": "Insufficient observed points",
                    }
                )
            continue
        log = logger.bind(stage=stage, object_id=item["object_id"])
        cache_path = (
            Path(stage_cache) / f"{item['object_id']}.json" if stage_cache else None
        )
        try:
            cached = (
                read_json(cache_path) if cache_path and cache_path.is_file() else None
            )
            if cached and all(
                Path(value["path"]).is_file()
                and file_digest(value["path"]) == value["sha256"]
                for value in cached["artifacts"]
            ):
                update, created = (
                    cached["update"],
                    [value["path"] for value in cached["artifacts"]],
                )
                log.info("Reusing verified object result")
            else:
                update, created = handler(item)
                if cache_path:
                    atomic_json(
                        cache_path,
                        {
                            "update": update,
                            "artifacts": [
                                {"path": str(path), "sha256": file_digest(path)}
                                for path in created
                            ],
                        },
                    )
            item.update(update)
            artifacts.extend(created)
            if cache_path:
                artifacts.append(cache_path)
            log.info("Object {}: {}", item["object_id"], item["status"])
        except Exception as error:
            item["status"] = f"{stage}_failed"
            item["error"] = str(error)
            result["failures"].append(
                {"object_id": item["object_id"], "stage": stage, "error": str(error)}
            )
            log.exception("Object processing failed")
            if not config["runtime"]["continue_on_object_error"]:
                raise
    return result, artifacts


def complete_objects(previous, ingested, config, runner, output, stage_cache=None):
    views = {view["view_id"]: view for view in ingested["views"]}

    def handler(item):
        directory = output / "objects" / item["object_id"] / "images"
        request_object, artifacts = crop_reference(
            views[item["reference_view_id"]],
            item["reference_mask_path"],
            directory,
            config["completion"],
        )
        request_object.update(
            object_id=item["object_id"], category=item["category"], reference_images=[]
        )
        completed = runner.call("flux2", {"object": request_object}, directory)
        image_path = Path(completed["image_path"])
        if not image_path.is_file():
            raise ValueError("FLUX.2 returned no completed image")
        with Image.open(image_path) as image:
            image.verify()
        metadata = directory / "completion_meta.json"
        atomic_json(
            metadata,
            {
                **completed,
                "synthetic": True,
                "semantic_quality_verified": False,
                "source_view": item["reference_view_id"],
            },
        )
        return {
            "status": "completed_image",
            "completed_image_path": str(image_path),
            "completion_metadata_path": str(metadata),
        }, artifacts + [image_path, metadata]

    return process_objects(previous, "complete", config, handler, stage_cache)


def generate_assets(previous, config, runner, output, stage_cache=None):
    def handler(item):
        directory = output / "objects" / item["object_id"] / "assets"
        asset = runner.call(
            "trellis",
            {
                "object": {
                    "object_id": item["object_id"],
                    "category": item["category"],
                    "image_path": item["completed_image_path"],
                    "mask_path": None,
                    "reference_images": [],
                }
            },
            directory,
        )
        for key in ("mesh_path", "gaussian_path", "metadata_path"):
            if not Path(asset[key]).is_file():
                raise ValueError(f"TRELLIS output missing: {key}")
        gaussian = inspect_gaussians(asset["gaussian_path"])
        if gaussian["count"] <= 0:
            raise ValueError("TRELLIS returned an empty Gaussian asset")
        return {"status": "generated_asset", "asset": asset}, [
            asset[key] for key in ("mesh_path", "gaussian_path", "metadata_path")
        ]

    return process_objects(previous, "assets", config, handler, stage_cache)


def register_assets(previous, config, output, stage_cache=None):
    def handler(item):
        import trimesh

        directory = output / "objects" / item["object_id"] / "registration"
        directory.mkdir(parents=True, exist_ok=True)
        sampled = sample_mesh(
            item["asset"]["mesh_path"],
            count=config["registration"]["sample_count"],
            seed=config["runtime"]["seed"],
        )
        target = read_cloud(item["cloud_path"])
        registered = register_points(
            sampled["points"],
            target["points"],
            config={**config["registration"], "seed": config["runtime"]["seed"]},
        )
        transform = np.asarray(registered["transform"], dtype=np.float64)
        from scene_create.quality import validate_asset_pose

        reprojection, previews = validate_asset_pose(
            item, transform, output, config["registration"]
        )
        registered["metrics"].update(reprojection)
        registered["accepted"] = (
            registered["accepted"] and reprojection["reprojection_accepted"]
        )
        metadata = {
            "T_scene_from_asset": transform,
            "translation_xyz": registered["translation_xyz"],
            "rotation_quaternion_xyzw": registered["rotation_xyzw"],
            "scale": registered["scale"],
            "metrics": registered["metrics"],
            "accepted": registered["accepted"],
            "source_frame": "asset",
            "target_frame": "first_input_camera_opencv",
            "scale_is_uniform": True,
        }
        transform_path = directory / "transform.json"
        atomic_json(transform_path, metadata)
        samples_path = directory / "mesh_samples.ply"
        write_cloud(samples_path, sampled["points"])
        if not registered["accepted"]:
            raise ValueError(
                f"Registration quality rejected: {registered['metrics']}; see {transform_path}"
            )
        aligned_mesh = directory / "aligned_mesh.glb"
        mesh = trimesh.load(item["asset"]["mesh_path"], force="scene", process=False)
        mesh.apply_transform(transform)
        mesh.export(str(aligned_mesh))
        aligned_gaussian = directory / "aligned_gaussian.ply"
        transform_gaussians(item["asset"]["gaussian_path"], aligned_gaussian, transform)
        aligned_samples = directory / "aligned_samples.ply"
        write_cloud(
            aligned_samples, sampled["points"] @ transform[:3, :3].T + transform[:3, 3]
        )
        return {
            "status": "registered",
            "transform": metadata,
            "transform_path": str(transform_path),
            "aligned_mesh_path": str(aligned_mesh),
            "aligned_gaussian_path": str(aligned_gaussian),
        }, [
            transform_path,
            samples_path,
            aligned_mesh,
            aligned_gaussian,
            aligned_samples,
            *previews,
        ]

    return process_objects(previous, "register", config, handler, stage_cache)
