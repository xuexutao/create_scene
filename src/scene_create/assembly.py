from __future__ import annotations

import os

import numpy as np
import trimesh

from scene_create.assets.gaussian import merge_gaussians
from scene_create.io import atomic_json


def assemble_scene(registered, background, geometry, output):
    directory = output / "scene"
    directory.mkdir(parents=True, exist_ok=True)

    def relative(value):
        return os.path.relpath(value, directory) if value else None

    accepted = [
        item for item in registered["objects"] if item["status"] == "registered"
    ]
    mesh_scene, gs_paths, gs_owners, objects, artifacts = (
        trimesh.Scene(),
        [],
        [],
        [],
        [],
    )
    if background.get("gaussian_path"):
        gs_paths.append(background["gaussian_path"])
        gs_owners.append("background")
    for item in registered["objects"]:
        record = {
            "object_id": item["object_id"],
            "category": item["category"],
            "status": item["status"],
            "interactive": item["status"] == "registered",
            "observed_cloud": relative(item["cloud_path"]),
            "preserved_in_background": item["status"] != "registered",
        }
        if item["status"] == "registered":
            mesh = trimesh.load(item["aligned_mesh_path"], force="scene", process=False)
            for index, node in enumerate(mesh.graph.nodes_geometry):
                transform, geometry_name = mesh.graph[node]
                name = f"{item['object_id']}_{index}"
                mesh_scene.add_geometry(
                    mesh.geometry[geometry_name].copy(),
                    geom_name=name,
                    node_name=name,
                    transform=transform,
                )
            gs_paths.append(item["aligned_gaussian_path"])
            gs_owners.append(item["object_id"])
            record.update(
                mesh=relative(item["aligned_mesh_path"]),
                gaussian=relative(item["aligned_gaussian_path"]),
                local_mesh=relative(item["asset"]["mesh_path"]),
                local_gaussian=relative(item["asset"]["gaussian_path"]),
                asset_to_world=item["transform"]["T_scene_from_asset"],
                render_transform=np.eye(4).tolist(),
                coordinates_baked=True,
                pose=item["transform"],
                generated_geometry=True,
                semantic_quality_verified=False,
            )
        objects.append(record)
    if accepted:
        mesh_path = directory / "objects.glb"
        mesh_scene.export(str(mesh_path))
        artifacts.append(mesh_path)
    else:
        (directory / "objects.glb").unlink(missing_ok=True)
    ranges = []
    gs_path = None
    if gs_paths:
        gs_path = directory / "scene_gaussian.ply"
        ranges = merge_gaussians(gs_paths, gs_path)
        for owner, item in zip(gs_owners, ranges):
            item["object_id"] = owner
            if "path" in item:
                item["path"] = relative(item["path"])
        artifacts.append(gs_path)
    else:
        (directory / "scene_gaussian.ply").unlink(missing_ok=True)
    failures = registered.get("failures", [])
    scene = {
        "schema_version": 1,
        "world_frame": geometry["world_frame"],
        "scale_status": geometry["scale_status"],
        "objects": objects,
        "background": {
            **background,
            "gaussian_path": relative(background.get("gaussian_path")),
        },
        "gaussian_scene": relative(gs_path),
        "gaussian_ranges": ranges,
        "default_representation": "gaussian",
        "cameras": [
            {
                key: view[key]
                for key in (
                    "view_id",
                    "width",
                    "height",
                    "intrinsics",
                    "world_from_camera",
                )
            }
            for view in geometry["views"]
        ],
        "failures": failures,
        "quality": {
            "ground_truth_evaluated": False,
            "hidden_surfaces_are_generated_or_unknown": True,
            "input_views": len(geometry["views"]),
            "registered_objects": len(accepted),
            "total_objects": len(objects),
        },
    }
    scene_path = directory / "scene.json"
    report_path = output / "reports" / "quality.json"
    atomic_json(scene_path, scene)
    atomic_json(
        report_path,
        {
            **scene["quality"],
            "failures": failures,
            "background_status": background["status"],
            "objects": [
                {
                    "object_id": item["object_id"],
                    "status": item["status"],
                    "metrics": item.get("transform", {}).get("metrics"),
                }
                for item in registered["objects"]
            ],
        },
    )
    return {
        "scene_path": str(scene_path),
        "gaussian_path": str(gs_path) if gs_path else None,
        "registered_objects": len(accepted),
        "total_objects": len(objects),
        "failures": failures,
        "approximate_background": background.get("approximate", False),
    }, artifacts + [scene_path, report_path]
