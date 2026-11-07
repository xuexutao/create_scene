import numpy as np
import pytest

from scene_create.adapters.maskgaussian import projection_matrix, valid_window_mask


def test_offcenter_projection_matches_pinhole():
    intrinsic = np.array([[400, 0, 120], [0, 420, 90], [0, 0, 1]], dtype=float)
    projection = projection_matrix(intrinsic, 640, 480)
    points = np.array([[0, 0, 2, 1], [0.2, -0.1, 3, 1]])
    homogeneous = points @ projection.T
    ndc = homogeneous[:, :2] / homogeneous[:, 3:4]
    pixels = ((ndc + 1) * np.array([640, 480]) - 1) / 2
    expected = points[:, :3] @ intrinsic.T
    expected = expected[:, :2] / expected[:, 2:3]
    np.testing.assert_allclose(pixels, expected, atol=1e-5)


def test_ssim_windows_exclude_foreground_and_border():
    mask = np.ones((25, 25), bool)
    mask[12, 12] = False
    windows = valid_window_mask(mask, 11)
    assert not windows[:5].any()
    assert not windows[7:18, 7:18].any()
    assert windows[5, 5]
    with pytest.raises(ValueError):
        valid_window_mask(mask, 4)
