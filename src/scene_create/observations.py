from __future__ import annotations

import numpy as np

from scene_create.geometry.camera import depth_to_points, validate_transform
from scene_create.io import (
    atomic_json,
    atomic_npz,
    load_mask,
    load_npz,
    save_image,
    validate_id,
)


def map_mask(mask, original_to_processed, shape):
    matrix = np.asarray(original_to_processed, dtype=float)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or abs(np.linalg.det(matrix)) < 1e-12
    ):
        raise ValueError("Invalid original_to_processed mapping")
    y, x = np.indices(shape)
    homogeneous = np.stack([x.ravel(), y.ravel(), np.ones(x.size)])
    original = np.linalg.inv(matrix) @ homogeneous
    finite = np.isfinite(original).all(axis=0) & (np.abs(original[2]) > 1e-12)
    original[:2, finite] /= original[2:3, finite]
    u, v = np.zeros(x.size, dtype=np.int64), np.zeros(x.size, dtype=np.int64)
    u[finite], v[finite] = np.rint(original[:2, finite]).astype(np.int64)
    inside = finite & (u >= 0) & (v >= 0) & (u < mask.shape[1]) & (v < mask.shape[0])
    result = np.zeros(x.size, dtype=bool)
    result[inside] = mask[v[inside], u[inside]]
    return result.reshape(shape)


def segmentation(ingested, config, runner, destination):
    if ingested["provenance"] != "external_observations":
        result = runner.call("sam3", {"views": ingested["views"]}, destination)
    else:
        instances = []
        for view in ingested["views"]:
            for item in view["imported_masks"]:
                mask = load_mask(item["mask_path"], (view["width"], view["height"]))
                if not mask.any():
                    raise ValueError(f"Empty imported mask: {item['mask_path']}")
                target = destination / view["view_id"] / f"{item['local_id']}.png"
                save_image(target, (mask * 255).astype(np.uint8))
                ys, xs = np.where(mask)
                instances.append(
                    {
                        **item,
                        "view_id": view["view_id"],
                        "mask_path": str(target),
                        "score": float(item.get("score", 1.0)),
                        "bbox_xyxy": [
                            int(xs.min()),
                            int(ys.min()),
                            int(xs.max() + 1),
                            int(ys.max() + 1),
                        ],
                    }
                )
        result = {"instances": instances, "provenance": "external_observations"}
    views = {view["view_id"]: view for view in ingested["views"]}
    seen, artifacts = set(), []
    for item in result["instances"]:
        key = (item["view_id"], validate_id(item["local_id"]))
        if key in seen or key[0] not in views:
            raise ValueError(f"Duplicate or unknown segmentation instance: {key}")
        seen.add(key)
        score = float(item["score"])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Instance confidence must be in [0, 1]")
        view = views[key[0]]
        if not load_mask(item["mask_path"], (view["width"], view["height"])).any():
            raise ValueError("Segmentation contains an empty mask")
        artifacts.append(item["mask_path"])
    return result, artifacts


def validate_geometry(data, original_size):
    required = {
        "depth",
        "conf",
        "intrinsics",
        "world_from_camera",
        "image",
        "original_to_processed",
    }
    missing = required - set(data)
    if missing:
        raise ValueError(f"Geometry NPZ is missing fields: {sorted(missing)}")
    depth = np.asarray(data["depth"], dtype=np.float32)
    confidence = np.asarray(data["conf"], dtype=np.float32)
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
    image = np.asarray(data["image"])
    if (
        depth.ndim != 2
        or depth.shape != confidence.shape
        or image.shape != (*depth.shape, 3)
    ):
        raise ValueError("Depth, confidence and processed RGB must share a pixel grid")
    if image.dtype != np.uint8:
        raise ValueError("Processed RGB must be uint8")
    if (
        intrinsics.shape != (3, 3)
        or not np.isfinite(intrinsics).all()
        or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0
        or not np.allclose(intrinsics[2], [0, 0, 1])
    ):
        raise ValueError("Invalid camera intrinsics")
    matrix = np.asarray(data["world_from_camera"], dtype=np.float64)
    validate_transform(matrix)
    mapping = np.asarray(data["original_to_processed"], dtype=np.float64)
    if (
        mapping.shape != (3, 3)
        or not np.isfinite(mapping).all()
        or abs(np.linalg.det(mapping)) < 1e-12
    ):
        raise ValueError("Invalid original-to-processed pixel mapping")
    valid = np.asarray(data.get("valid", np.ones(depth.shape, dtype=bool)), dtype=bool)
    if valid.shape != depth.shape:
        raise ValueError("Invalid depth validity mask")
    valid = (
        valid
        & np.isfinite(depth)
        & (depth > 0)
        & np.isfinite(confidence)
        & (confidence >= 0)
    )
    if not valid.any():
        raise ValueError("Geometry has no valid depth pixels")
    return dict(
        depth=depth,
        conf=confidence,
        intrinsics=intrinsics,
        world_from_camera=matrix,
        image=image,
        original_to_processed=mapping,
        valid=valid,
    )


def geometry(ingested, config, runner, destination, output):
    if ingested["provenance"] == "external_observations":
        predicted = {
            "views": [
                {
                    "view_id": view["view_id"],
                    "geometry_path": view["imported_geometry"],
                    "scale_status": view["scale_status"],
                    "scale_source": view["scale_source"],
                }
                for view in ingested["views"]
            ],
            "provenance": "external_observations",
        }
    else:
        predicted = runner.call("da3", {"views": ingested["views"]}, destination)
    by_id = {item["view_id"]: item for item in predicted["views"]}
    if len(by_id) != len(predicted["views"]) or set(by_id) != {
        view["view_id"] for view in ingested["views"]
    }:
        raise ValueError("DA3 must return geometry for each input view exactly once")
    processed = []
    for view in ingested["views"]:
        record = by_id[view["view_id"]]
        data = validate_geometry(
            load_npz(record["geometry_path"]), (view["width"], view["height"])
        )
        processed.append((view, record, data))
    scene_from_world = np.linalg.inv(processed[0][2]["world_from_camera"])
    statuses = {record.get("scale_status", "relative") for _, record, _ in processed}
    if len(statuses) != 1:
        raise ValueError("Mixed scale conventions across views are not allowed")
    records, artifacts = [], []
    for view, prediction, data in processed:
        data["world_from_camera"] = scene_from_world @ data["world_from_camera"]
        data["valid"] &= data["conf"] >= config["geometry"]["min_confidence"]
        safe_depth = np.where(data["valid"], data["depth"], 0)
        points = depth_to_points(
            safe_depth, data["intrinsics"], data["world_from_camera"]
        )
        points[~data["valid"]] = np.nan
        camera_points = depth_to_points(safe_depth, data["intrinsics"], np.eye(4))
        camera_points[~data["valid"]] = np.nan
        point_path = output / "pointmaps" / f"{view['view_id']}.npz"
        depth_path = output / "depth" / f"{view['view_id']}.npz"
        image_path = destination / f"{view['view_id']}_processed.png"
        atomic_npz(point_path, **data, points_world=points, points_camera=camera_points)
        atomic_npz(
            depth_path, depth=data["depth"], conf=data["conf"], valid=data["valid"]
        )
        save_image(image_path, data["image"])
        records.append(
            {
                "view_id": view["view_id"],
                "geometry_path": str(point_path),
                "image_path": str(image_path),
                "width": data["depth"].shape[1],
                "height": data["depth"].shape[0],
                "scale_status": prediction.get("scale_status", "relative"),
                "scale_source": prediction.get("scale_source", "unspecified"),
                "intrinsics": data["intrinsics"].tolist(),
                "world_from_camera": data["world_from_camera"].tolist(),
            }
        )
        artifacts.extend([point_path, depth_path, image_path])
    result = {
        "views": records,
        "provenance": predicted.get("provenance"),
        "world_frame": "first_input_camera_opencv",
        "scale_status": next(iter(statuses)),
    }
    cameras = output / "metadata" / "cameras.json"
    atomic_json(cameras, result)
    return result, artifacts + [cameras]
