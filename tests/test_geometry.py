import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from scene_create.geometry.camera import (
    depth_to_points,
    project_points,
    validate_transform,
)
from scene_create.geometry.cloud import clean_cloud, voxel_fuse
from scene_create.geometry.registration import (
    estimate_similarity,
    register_points,
    sample_mesh,
)


def test_camera_roundtrip():
    depth = np.full((11, 17), 2.5)
    intrinsic = np.array([[20, 0, 7], [0, 21, 4], [0, 0, 1]], dtype=float)
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", [0.2, 0.1, -0.4]).as_matrix()
    matrix[:3, 3] = [2, -3, 1]
    world = depth_to_points(depth, intrinsic, matrix)
    pixels, z = project_points(world.reshape(-1, 3), intrinsic, matrix)
    y, x = np.indices(depth.shape)
    np.testing.assert_allclose(
        pixels, np.column_stack([x.ravel(), y.ravel()]), atol=1e-10
    )
    np.testing.assert_allclose(z, depth.ravel())


def test_uniform_scale_only():
    matrix = np.eye(4)
    matrix[:3, :3] *= 2
    validate_transform(matrix, allow_scale=True)
    with pytest.raises(ValueError):
        validate_transform(matrix)
    matrix[0, 0] = 3
    with pytest.raises(ValueError):
        validate_transform(matrix, allow_scale=True)


def test_weighted_voxel_fusion():
    result = voxel_fuse(
        np.array([[0, 0, 0], [0.2, 0, 0], [2, 0, 0]]),
        np.array([[0, 0, 0], [100, 200, 40], [7, 8, 9]], dtype=np.uint8),
        [1, 3, 1],
        1,
    )
    np.testing.assert_allclose(result["points"], [[0.15, 0, 0], [2, 0, 0]])
    np.testing.assert_allclose(result["weights"], [4, 1])
    np.testing.assert_array_equal(result["colors"], [[75, 150, 30], [7, 8, 9]])


def test_clean_indices_preserve_correspondence():
    points = np.array([[0, 0, 0], [1, 1, 1], [np.nan, 1, 2]])
    result = clean_cloud(points, np.zeros((3, 3)), [1, 1, 1])
    np.testing.assert_array_equal(result["indices"], [0, 1])


def test_known_similarity():
    source = np.random.default_rng(2).normal(size=(50, 3))
    rotation = Rotation.from_euler("zyx", [0.3, -0.8, 0.1]).as_matrix()
    target = source @ (rotation * 2.3).T + [2, 3, -4]
    transform = estimate_similarity(source, target)
    np.testing.assert_allclose(
        source @ transform[:3, :3].T + transform[:3, 3], target, atol=1e-10
    )
    with pytest.raises(ValueError):
        estimate_similarity(np.zeros((5, 3)), np.zeros((5, 3)))


def test_partial_registration_with_initial_pose():
    random = np.random.default_rng(5)
    source = random.normal(size=(400, 3)) * [1.0, 0.4, 0.2]
    true = np.eye(4)
    true[:3, :3] = 1.7 * Rotation.from_euler("xyz", [0.2, -0.1, 0.4]).as_matrix()
    true[:3, 3] = [1.2, -0.7, 2.5]
    target = source[source[:, 0] > -0.3] @ true[:3, :3].T + true[:3, 3]
    result = register_points(
        source,
        target,
        {
            "initial_transform": true.tolist(),
            "coarse_iterations": 2,
            "iterations": 15,
            "refine_candidates": 2,
            "scale_candidates": [1.0],
        },
    )
    assert result["accepted"]
    assert result["metrics"]["rmse_ratio"] < 1e-5
    np.testing.assert_allclose(result["transform"], true, atol=1e-4)


def test_mesh_sampling_surface_area_and_seed(tmp_path):
    mesh = trimesh.creation.box(extents=[1, 2, 3])
    path = tmp_path / "mesh.glb"
    mesh.export(path)
    a = sample_mesh(path, count=10000, seed=5)
    b = sample_mesh(path, count=10000, seed=5)
    np.testing.assert_array_equal(a["points"], b["points"])
    np.testing.assert_allclose(np.linalg.norm(a["normals"], axis=1), 1)
    x_faces = np.isclose(np.abs(a["points"][:, 0]), 0.5)
    assert abs(x_faces.mean() - 12 / 22) < 0.03
