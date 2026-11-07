import numpy as np
import pytest
import trimesh
from PIL import Image

from scene_create.config import load_config
from scene_create.geometry.raster import rasterize_depth
from scene_create.io import atomic_json, atomic_npz
from scene_create.quality import validate_asset_pose
from scene_create.reconstruction import process_objects


def test_depth_rasterizer_respects_front_surface():
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    vertices = mesh.vertices + [0, 0, 3]
    intrinsic = np.array([[40, 0, 31.5], [0, 40, 31.5], [0, 0, 1]])
    result = rasterize_depth(vertices, mesh.faces, intrinsic, np.eye(4), 64, 64)
    assert result["mask"][32, 32]
    assert result["depth"][32, 32] == pytest.approx(2.5)
    assert not result["mask"][0, 0]


def test_pose_gate_accepts_exact_mesh_and_rejects_translation(tmp_path):
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    path = tmp_path / "asset.glb"
    mesh.export(path)
    intrinsic = np.array([[40, 0, 31.5], [0, 40, 31.5], [0, 0, 1]])
    transform = np.eye(4)
    transform[2, 3] = 3
    raster = rasterize_depth(
        mesh.vertices + [0, 0, 3], mesh.faces, intrinsic, np.eye(4), 64, 64
    )
    observed_depth = np.where(raster["mask"], raster["depth"], 5)
    atomic_npz(
        tmp_path / "pointmaps" / "view_000.npz",
        depth=observed_depth,
        image=np.full((64, 64, 3), 160, np.uint8),
        valid=np.ones((64, 64), bool),
        intrinsics=intrinsic,
        world_from_camera=np.eye(4),
    )
    mask_path = tmp_path / "mask.png"
    Image.fromarray((raster["mask"] * 255).astype(np.uint8)).save(mask_path)
    item = {
        "object_id": "box_001",
        "asset": {"mesh_path": str(path)},
        "observations": [{"view_id": "view_000", "geometry_mask_path": str(mask_path)}],
    }
    result, previews = validate_asset_pose(item, transform, tmp_path, {})
    assert result["reprojection_accepted"]
    assert result["mean_visible_mask_iou"] == pytest.approx(1)
    assert all(path.exists() for path in previews)
    transform[0, 3] = 1.5
    result, _ = validate_asset_pose(item, transform, tmp_path, {})
    assert not result["reprojection_accepted"]


def test_failed_object_retry_reuses_only_successful_results(tmp_path):
    previous = {
        "objects": [
            {"object_id": "cup_001", "status": "observed"},
            {"object_id": "cup_002", "status": "observed"},
        ]
    }
    attempts = {"cup_001": 0, "cup_002": 0}

    def handler(item):
        name = item["object_id"]
        attempts[name] += 1
        if name == "cup_002" and attempts[name] == 1:
            raise ValueError("transient test failure")
        path = tmp_path / f"{name}_result.json"
        atomic_json(path, {"ok": True})
        return {"status": "completed_image"}, [path]

    config = load_config()
    first, _ = process_objects(
        previous, "complete", config, handler, tmp_path / "cache"
    )
    assert len(first["failures"]) == 1
    second, _ = process_objects(
        previous, "complete", config, handler, tmp_path / "cache"
    )
    assert not second["failures"]
    assert attempts == {"cup_001": 1, "cup_002": 2}
