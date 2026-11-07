import numpy as np
import pytest
from plyfile import PlyData
from scipy.spatial.transform import Rotation

from scene_create.assets.gaussian import (
    initialize_gaussians,
    inspect_gaussians,
    merge_gaussians,
    rotate_sh,
    sh_basis,
    transform_gaussians,
)
from scene_create.io import write_cloud


def test_gaussian_sim3(tmp_path):
    source, target = tmp_path / "input.ply", tmp_path / "output.ply"
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    initialize_gaussians(xyz, np.full((3, 3), 150), source)
    matrix = np.eye(4)
    rotation = Rotation.from_euler("xyz", [0.2, 0.7, -0.4]).as_matrix()
    matrix[:3, :3], matrix[:3, 3] = 2 * rotation, [1, 2, 3]
    transform_gaussians(source, target, matrix)
    a, b = PlyData.read(source)["vertex"].data, PlyData.read(target)["vertex"].data
    actual = np.column_stack([b[name] for name in ("x", "y", "z")])
    np.testing.assert_allclose(
        actual, xyz @ matrix[:3, :3].T + matrix[:3, 3], atol=1e-6
    )
    np.testing.assert_allclose(b["scale_0"] - a["scale_0"], np.log(2), atol=1e-6)
    quaternion = np.array([b[f"rot_{i}"][0] for i in (1, 2, 3, 0)])
    np.testing.assert_allclose(
        Rotation.from_quat(quaternion).as_matrix(), rotation, atol=1e-6
    )
    assert inspect_gaussians(target)["sh_degree"] == 0


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_sh_rotation_direction_consistency(degree):
    random = np.random.default_rng(8)
    coeffs = random.normal(size=(4, 3, (degree + 1) ** 2))
    rotation = Rotation.from_euler("xyz", [0.6, -0.3, 1.0]).as_matrix()
    directions = random.normal(size=(19, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    expected = np.einsum("mk,nck->ncm", sh_basis(directions @ rotation, degree), coeffs)
    actual = np.einsum(
        "mk,nck->ncm", sh_basis(directions, degree), rotate_sh(coeffs, rotation)
    )
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_merge_degrees_and_reject_plain_points(tmp_path):
    paths = [tmp_path / "a.ply", tmp_path / "b.ply"]
    for degree, path in zip([0, 2], paths):
        initialize_gaussians(
            [[0, 0, 0], [1, 0, 0]], [[255, 0, 0], [0, 255, 0]], path, sh_degree=degree
        )
    ranges = merge_gaussians(paths, tmp_path / "merged.ply")
    assert [(item["start"], item["count"]) for item in ranges] == [(0, 2), (2, 2)]
    assert inspect_gaussians(tmp_path / "merged.ply")["sh_degree"] == 2
    write_cloud(tmp_path / "xyz.ply", [[0, 0, 0]])
    with pytest.raises(ValueError, match="Not a standard"):
        inspect_gaussians(tmp_path / "xyz.ply")
