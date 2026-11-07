import numpy as np
import pytest

from scene_create.adapters.da3 import preprocessing_maps
from scene_create.adapters.sam3 import deduplicate_masks
from scene_create.observations import map_mask, validate_geometry


def test_resize_mapping_uses_actual_dimensions():
    maps, shape = preprocessing_maps([(640, 480)])
    assert shape == (378, 504)
    np.testing.assert_allclose(
        maps[0],
        [
            [504 / 640, 0, (504 / 640 - 1) / 2],
            [0, 378 / 480, (378 / 480 - 1) / 2],
            [0, 0, 1],
        ],
    )


def test_batch_center_crop_mapping():
    maps, shape = preprocessing_maps([(640, 480), (480, 640)])
    assert shape == (378, 378)
    assert maps[0][0, 2] < -63
    assert maps[1][1, 2] < -63
    full = np.ones((480, 640), bool)
    assert map_mask(full, maps[0], shape).all()


def test_mask_affine_does_not_stretch_crop():
    mask = np.zeros((20, 30), bool)
    mask[4:8, 12:16] = True
    actual = map_mask(mask, np.array([[1, 0, -10], [0, 1, -2], [0, 0, 1]]), (10, 10))
    expected = np.zeros((10, 10), bool)
    expected[2:6, 2:6] = True
    np.testing.assert_array_equal(actual, expected)


def test_sam_dedup_preserves_distinct_instances():
    a = np.zeros((20, 20), bool)
    a[2:8, 2:8] = True
    b = np.zeros((20, 20), bool)
    b[10:18, 10:18] = True
    candidates = [
        {"mask": a, "score": 0.9},
        {"mask": a.copy(), "score": 0.8},
        {"mask": b, "score": 0.7},
    ]
    assert len(deduplicate_masks(candidates)) == 2


def test_geometry_rejects_ambiguous_mapping():
    data = {
        "depth": np.ones((4, 5)),
        "conf": np.ones((4, 5)),
        "image": np.zeros((4, 5, 3), np.uint8),
        "intrinsics": np.eye(3),
        "world_from_camera": np.eye(4),
        "original_to_processed": np.zeros((3, 3)),
    }
    with pytest.raises(ValueError, match="mapping"):
        validate_geometry(data, (5, 4))
