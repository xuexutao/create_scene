from unittest.mock import Mock

import numpy as np
import pytest

from scene_create.adapters.flux2 import completion_prompt
from scene_create.adapters.trellis import (
    GLB_FROM_NATIVE,
    local_components,
    local_rembg_session,
    normalize_export_rotations,
    validate_foreground,
)
from scene_create.io import atomic_json


def test_trellis_export_frames_match():
    native = np.array([[0.2, 0.3, 0.4], [0, 1, 0]])
    glb_row_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    np.testing.assert_allclose(native @ glb_row_matrix, native @ GLB_FROM_NATIVE.T)
    assert np.linalg.det(GLB_FROM_NATIVE) == 1


def test_foreground_quality_gate():
    alpha = np.zeros((32, 32), np.uint8)
    alpha[8:24, 8:24] = 255
    assert validate_foreground(alpha).sum() == 256
    with pytest.raises(ValueError):
        validate_foreground(np.full((32, 32), 255))
    with pytest.raises(ValueError):
        validate_foreground(np.zeros((32, 32)))


def test_completion_prompt_identity_and_class():
    prompt = completion_prompt("chair")
    assert "chair" in prompt and "Preserve its identity" in prompt
    with pytest.raises(ValueError):
        completion_prompt("")


def local_pipeline_fixture(tmp_path):
    names = [
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "slat_flow_model",
        "slat_decoder_mesh",
        "slat_decoder_gs",
    ]
    models = {}
    for name in names:
        models[name] = "ckpts/" + name
        atomic_json(
            tmp_path / (models[name] + ".json"), {"name": "fixture_only", "args": {}}
        )
        (tmp_path / (models[name] + ".safetensors")).touch()
    models["slat_decoder_rf"] = "remote-owner/unused-component"
    atomic_json(tmp_path / "pipeline.json", {"args": {"models": models}})
    return names


def test_only_required_local_trellis_components_are_selected(tmp_path):
    names = local_pipeline_fixture(tmp_path)
    _, components = local_components(tmp_path)
    assert set(components) == set(names)
    assert "slat_decoder_rf" not in components


def test_missing_component_refuses_remote_fallback(tmp_path):
    names = local_pipeline_fixture(tmp_path)
    (tmp_path / "ckpts" / (names[0] + ".safetensors")).unlink()
    with pytest.raises(FileNotFoundError, match="remote fallback is disabled"):
        local_components(tmp_path)


def test_remote_only_pipeline_cannot_be_loaded(tmp_path):
    with pytest.raises(FileNotFoundError, match="local pipeline"):
        local_components(tmp_path / "not_downloaded")


def test_u2net_uses_explicit_local_custom_session(tmp_path, monkeypatch):
    monkeypatch.setenv("U2NET_HOME", "untouched_by_test_cleanup")
    monkeypatch.setenv("REMBG_HOME", "untouched_by_test_cleanup")
    model = tmp_path / "custom_u2net.onnx"
    model.touch()
    rembg = Mock()
    assert local_rembg_session(rembg, model) is rembg.new_session.return_value
    rembg.new_session.assert_called_once_with(
        "u2net_custom",
        model_path=str(model.resolve()),
        providers=["CPUExecutionProvider"],
    )
    with pytest.raises(FileNotFoundError):
        local_rembg_session(rembg, tmp_path / "missing.onnx")
    assert rembg.new_session.call_count == 1


def test_export_rotations_are_normalized_without_axis_change():
    rotations = np.array([[2, 0, 0, 0], [0, 1, 2, 3]], dtype=float)
    normalized = normalize_export_rotations(rotations)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1)
    np.testing.assert_allclose(normalized[0], [1, 0, 0, 0])
    np.testing.assert_allclose(normalized[1], rotations[1] / np.sqrt(14))


@pytest.mark.parametrize(
    "rotations",
    [np.zeros((1, 4)), [[1, 0, float("nan"), 0]], np.zeros((0, 4)), np.ones((2, 3))],
)
def test_invalid_export_rotations_are_rejected(rotations):
    with pytest.raises(ValueError):
        normalize_export_rotations(rotations)
