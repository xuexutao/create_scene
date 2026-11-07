from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scene_create.config import load_config
from scene_create.inputs import resolve_input
from scene_create.io import (
    atomic_json,
    atomic_npz,
    load_npz,
    read_cloud,
    read_json,
    safe_name,
    validate_id,
    write_cloud,
)
from scene_create.runtime import StageStore, output_lock


def test_atomic_arrays_and_clouds(tmp_path):
    points = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    colors = np.array([[255, 10, 0], [3, 100, 200]], dtype=np.uint8)
    atomic_npz(tmp_path / "a.npz", points=points)
    np.testing.assert_allclose(load_npz(tmp_path / "a.npz")["points"], points)
    write_cloud(tmp_path / "a.ply", points, colors)
    actual = read_cloud(tmp_path / "a.ply")
    np.testing.assert_allclose(actual["points"], points)
    np.testing.assert_array_equal(actual["colors"], colors)


def test_invalid_cloud_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_cloud(tmp_path / "bad.ply", [[float("nan"), 0, 0]])


@pytest.mark.parametrize("value", ["../escape", "a/b", "..", "", "a.b"])
def test_unsafe_identifiers(value):
    with pytest.raises(ValueError):
        validate_id(value)
    assert "/" not in safe_name(value)


def test_cache_integrity_and_config_invalidation(tmp_path):
    store = StageStore(tmp_path, "run1")
    calls = []

    def operation(directory):
        calls.append(1)
        output = directory / "asset.json"
        atomic_json(output, {"count": len(calls)})
        return {"value": len(calls)}, [output]

    assert store.run("stage", "a" * 64, operation)["value"] == 1
    assert store.run("stage", "a" * 64, operation)["value"] == 1
    asset = tmp_path / "intermediate" / "stage" / ("a" * 16) / "asset.json"
    atomic_json(asset, {"corrupted": True})
    assert store.run("stage", "a" * 64, operation)["value"] == 2
    assert store.run("stage", "b" * 64, operation)["value"] == 3
    resumed = StageStore(tmp_path, "run2", resume=True)
    assert resumed.run("stage", "b" * 64, operation)["value"] == 3
    with pytest.raises(ValueError):
        StageStore(tmp_path, "run3")


def test_failed_stage_is_not_cached(tmp_path):
    store = StageStore(tmp_path, "run1")

    def failure(directory):
        raise RuntimeError("intentional test error")

    with pytest.raises(RuntimeError):
        store.run("stage", "a" * 64, failure)
    manifest = read_json(tmp_path / "run_manifest.json")
    assert manifest["stages"]["stage"]["status"] == "failed"
    assert manifest["status"] == "failed"
    result = store.run("stage", "a" * 64, lambda directory: ({"ok": True}, []))
    assert result["ok"]


def test_partial_stage_is_retried(tmp_path):
    store = StageStore(tmp_path, "run1")
    calls = []

    def partial(directory):
        calls.append(1)
        return {"failures": [{"object_id": "cup_001"}]}, []

    store.run("stage", "c" * 64, partial)
    store.run("stage", "c" * 64, partial)
    assert len(calls) == 2
    assert store.data["stages"]["stage"]["status"] == "partial"


def test_lock_rejects_concurrent_writer(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with output_lock(tmp_path):
                pass


def test_configuration_and_inputs(tmp_path):
    config = load_config(root=tmp_path)
    assert config["models"]["sam3"]["allow_download"] is False
    assert Path(config["models"]["da3"]["repo_path"]).is_absolute()
    image = tmp_path / "test.png"
    Image.new("RGB", (32, 24), "white").save(image)
    inputs = resolve_input({"image": image})
    assert inputs["views"][0]["view_id"] == "view_000"
    assert inputs["provenance"] == "input_images"
    with pytest.raises(ValueError, match="No input images"):
        empty = tmp_path / "empty"
        empty.mkdir()
        resolve_input({"images": empty})


def test_config_rejects_typos(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("modles: {}\n")
    with pytest.raises(ValueError):
        load_config(path)
