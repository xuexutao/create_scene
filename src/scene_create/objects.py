from __future__ import annotations

from collections import defaultdict

import numpy as np

from scene_create.geometry.camera import project_points
from scene_create.geometry.cloud import clean_cloud, cloud_overlap, voxel_fuse
from scene_create.io import (
    atomic_json,
    atomic_npz,
    load_mask,
    load_npz,
    safe_name,
    save_image,
    write_cloud,
)
from scene_create.observations import map_mask


def extent(points):
    if len(points) < 2:
        return 1e-6
    return max(
        float(
            np.linalg.norm(
                np.quantile(points, 0.95, axis=0) - np.quantile(points, 0.05, axis=0)
            )
        ),
        1e-6,
    )


def projected_support(points, instance):
    if "camera_data" not in instance:
        return 1.0
    data = instance["camera_data"]
    uv, z = project_points(points, data["intrinsics"], data["world_from_camera"])
    finite = np.isfinite(uv).all(axis=1) & (z > 0)
    pixels = np.zeros((len(points), 2), dtype=int)
    pixels[finite] = np.rint(uv[finite]).astype(int)
    u, v = pixels.T
    mask = instance["processed_mask"]
    inside = finite & (u >= 0) & (v >= 0) & (u < mask.shape[1]) & (v < mask.shape[0])
    indices = np.flatnonzero(inside)
    if not len(indices):
        return 0.0
    observed = data["depth"][v[indices], u[indices]]
    visible = data["valid"][v[indices], u[indices]] & (
        z[indices] <= observed + 0.05 * np.maximum(observed, 1e-6)
    )
    indices = indices[visible]
    if len(indices) < 3:
        return 0.0
    return float(mask[v[indices], u[indices]].mean())


def associate_instances(instances, fusion):
    parents = list(range(len(instances)))
    members = {index: {index} for index in parents}
    warnings = []

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def merge(a, b):
        a, b = root(a), root(b)
        if a == b:
            return
        left, right = members[a], members[b]
        if {instances[i]["view_id"] for i in left} & {
            instances[i]["view_id"] for i in right
        }:
            return
        names = {instances[i].get("object_id") for i in left | right} - {None}
        if len(names) > 1:
            return
        parents[b] = a
        members[a] |= members.pop(b)

    explicit = defaultdict(list)
    for index, instance in enumerate(instances):
        if instance.get("object_id"):
            explicit[instance["object_id"]].append(index)
    for name, indices in explicit.items():
        if len({instances[i]["view_id"] for i in indices}) != len(indices):
            raise ValueError(
                f"The same explicit object ID occurs twice in one view: {name}"
            )
        if len({instances[i]["category"] for i in indices}) != 1:
            raise ValueError(f"Conflicting categories for explicit object ID {name}")
        for index in indices[1:]:
            merge(indices[0], index)
    edges = []
    competing = defaultdict(list)
    for i, a in enumerate(instances):
        for j in range(i + 1, len(instances)):
            b = instances[j]
            if (
                a["view_id"] == b["view_id"]
                or a["category"].lower() != b["category"].lower()
            ):
                continue
            if (
                a.get("object_id")
                and b.get("object_id")
                and a["object_id"] != b["object_id"]
            ):
                continue
            if min(len(a["points"]), len(b["points"])) < 3:
                continue
            color_distance = np.linalg.norm(
                np.mean(a["colors"], axis=0) - np.mean(b["colors"], axis=0)
            ) / (255 * np.sqrt(3))
            if color_distance > fusion["max_color_distance"]:
                continue
            size = max(extent(a["points"]), extent(b["points"]))
            if (
                np.linalg.norm(
                    np.median(a["points"], axis=0) - np.median(b["points"], axis=0)
                )
                > 1.5 * size
            ):
                continue
            points_a = a["points"][:: max(1, len(a["points"]) // 2000)]
            points_b = b["points"][:: max(1, len(b["points"]) // 2000)]
            score = float(
                cloud_overlap(points_a, points_b, size * fusion["match_distance_ratio"])
            )
            if score >= fusion["min_overlap"]:
                support = min(
                    projected_support(points_a, b), projected_support(points_b, a)
                )
                if support < fusion.get("min_projected_support", 0.5):
                    continue
                score *= 1 - 0.2 * float(color_distance)
                edges.append((score, i, j))
                competing[(i, b["view_id"])].append(score)
                competing[(j, a["view_id"])].append(score)
    ambiguous = set()
    for key, scores in competing.items():
        ordered = sorted(scores, reverse=True)
        if len(ordered) > 1 and ordered[0] - ordered[1] < fusion["ambiguity_margin"]:
            ambiguous.add(key)
    for score, i, j in sorted(edges, reverse=True):
        if (i, instances[j]["view_id"]) in ambiguous or (
            j,
            instances[i]["view_id"],
        ) in ambiguous:
            continue
        merge(i, j)
    if ambiguous:
        warnings.append(
            f"{len(ambiguous)} ambiguous cross-view assignments were kept separate; provide explicit object_id values to review"
        )
    return [sorted(group) for group in members.values()], warnings


def build_objects(segmented, geometry, ingested, config, destination, output):
    views = {item["view_id"]: item for item in ingested["views"]}
    geometry_views = {item["view_id"]: item for item in geometry["views"]}
    by_view = defaultdict(list)
    for instance in segmented["instances"]:
        by_view[instance["view_id"]].append(instance)
    instances, artifacts = [], []
    for view_id, entries in by_view.items():
        data = load_npz(geometry_views[view_id]["geometry_path"])
        masks = [
            map_mask(
                load_mask(item["mask_path"]),
                data["original_to_processed"],
                data["depth"].shape,
            )
            for item in entries
        ]
        owner = np.full(data["depth"].shape, -1, dtype=np.int32)
        for index in sorted(
            range(len(entries)), key=lambda i: entries[i]["score"], reverse=True
        ):
            owner[masks[index] & (owner < 0)] = index
        for index, item in enumerate(entries):
            keep = (
                masks[index]
                & (owner == index)
                & data["valid"]
                & np.isfinite(data["points_world"]).all(axis=-1)
            )
            pixel_indices = np.flatnonzero(keep)
            points = data["points_world"][keep]
            colors = data["image"][keep]
            weights = np.maximum(data["conf"][keep], 1e-6)
            raw_path = destination / view_id / f"{item['local_id']}_raw.ply"
            write_cloud(raw_path, points, colors)
            artifacts.append(raw_path)
            if len(points) >= 3:
                cleaned = clean_cloud(
                    points,
                    colors,
                    weights,
                    neighbors=config["geometry"]["outlier_neighbors"],
                    std_ratio=config["geometry"]["outlier_std_ratio"],
                )
                points, colors, weights = (
                    cleaned["points"],
                    cleaned["colors"],
                    cleaned["weights"],
                )
                pixel_indices = pixel_indices[cleaned["indices"]]
            instances.append(
                {
                    **item,
                    "points": points,
                    "colors": colors,
                    "weights": weights,
                    "pixel_indices": pixel_indices,
                    "processed_mask": masks[index],
                    "pixel_count": int(load_mask(item["mask_path"]).sum()),
                    "raw_cloud_path": str(raw_path),
                    "camera_data": data,
                }
            )
    groups, warnings = associate_instances(instances, config["fusion"])
    groups.sort(
        key=lambda indices: (
            instances[indices[0]]["category"],
            instances[indices[0]]["view_id"],
            instances[indices[0]]["local_id"],
        )
    )
    counts, used, objects, mask_records = (
        defaultdict(int),
        {i["object_id"] for i in instances if i.get("object_id")},
        [],
        [],
    )
    for group in groups:
        observations = [instances[index] for index in group]
        category = observations[0]["category"]
        explicit_names = {item.get("object_id") for item in observations} - {None}
        if explicit_names:
            object_id = next(iter(explicit_names))
        else:
            prefix = safe_name(category)
            while True:
                counts[prefix] += 1
                object_id = f"{prefix}_{counts[prefix]:03d}"
                if object_id not in used:
                    break
            used.add(object_id)
        base = output / "objects" / object_id
        references, all_points, all_colors, all_weights, all_views, all_pixels = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        reference = max(
            observations, key=lambda item: item["score"] * np.sqrt(item["pixel_count"])
        )
        for view_index, item in enumerate(observations):
            view_id = item["view_id"]
            mask_path = output / "masks" / view_id / f"{object_id}.png"
            save_image(mask_path, (load_mask(item["mask_path"]) * 255).astype(np.uint8))
            processed_mask = base / "clouds" / "views" / f"{view_id}_mask.png"
            save_image(processed_mask, (item["processed_mask"] * 255).astype(np.uint8))
            cloud_path = base / "clouds" / "views" / f"{view_id}.ply"
            write_cloud(cloud_path, item["points"], item["colors"])
            record = {
                "view_id": view_id,
                "local_id": item["local_id"],
                "mask_path": str(mask_path),
                "geometry_mask_path": str(processed_mask),
                "cloud_path": str(cloud_path),
                "score": item["score"],
                "pixel_count": item["pixel_count"],
                "point_count": len(item["points"]),
            }
            references.append(record)
            mask_records.append(
                {
                    **record,
                    "object_id": object_id,
                    "category": category,
                    "width": views[view_id]["width"],
                    "height": views[view_id]["height"],
                }
            )
            artifacts.extend([mask_path, processed_mask, cloud_path])
            all_points.append(item["points"])
            all_colors.append(item["colors"])
            all_weights.append(item["weights"])
            all_views.append(np.full(len(item["points"]), view_index, dtype=np.int32))
            all_pixels.append(item["pixel_indices"])
        points, colors, weights = (
            np.concatenate(all_points),
            np.concatenate(all_colors),
            np.concatenate(all_weights),
        )
        voxel = (
            config["fusion"]["voxel_size"]
            or extent(points) * config["fusion"]["voxel_ratio"]
        )
        fused = (
            voxel_fuse(points, colors, weights, voxel)
            if len(points)
            else {
                "points": points,
                "colors": colors,
                "weights": weights,
                "counts": np.empty(0, dtype=np.int32),
            }
        )
        cloud_path = base / "clouds" / "fused.ply"
        provenance_path = base / "clouds" / "observations.npz"
        write_cloud(cloud_path, fused["points"], fused["colors"])
        atomic_npz(
            provenance_path,
            points=points,
            colors=colors,
            weights=weights,
            view_indices=np.concatenate(all_views),
            pixel_indices=np.concatenate(all_pixels),
            fused_weights=fused["weights"],
            fused_counts=fused["counts"],
        )
        representative_mask = output / "masks" / f"{object_id}.png"
        save_image(
            representative_mask,
            (load_mask(reference["mask_path"]) * 255).astype(np.uint8),
        )
        artifacts.extend([cloud_path, provenance_path, representative_mask])
        objects.append(
            {
                "object_id": object_id,
                "category": category,
                "observations": references,
                "reference_view_id": reference["view_id"],
                "reference_mask_path": str(representative_mask),
                "cloud_path": str(cloud_path),
                "observations_path": str(provenance_path),
                "point_count": len(fused["points"]),
                "voxel_size": voxel,
                "status": "observed"
                if len(fused["points"]) >= config["geometry"]["min_points"]
                else "insufficient_geometry",
            }
        )
    metadata = output / "masks" / "metadata.json"
    object_metadata = output / "metadata" / "objects.json"
    result = {"objects": objects, "warnings": warnings}
    atomic_json(
        metadata,
        {
            "instances": mask_records,
            "representatives": {
                item["object_id"]: item["reference_view_id"] for item in objects
            },
        },
    )
    atomic_json(object_metadata, result)
    return result, artifacts + [metadata, object_metadata]
