from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scene_create.adapters.common import (
    configure_device,
    require_local_model,
    resolve_dtype,
)
from scene_create.adapters.da3 import (
    configured_model_type,
    prediction_scale_status,
    run as run_da3,
)
from scene_create.adapters.sam3 import load_model, run as run_sam3


def torch_device_stub(requested):
    kind, _, index = requested.partition(":")
    return SimpleNamespace(type=kind, index=int(index) if index else None)


@pytest.mark.parametrize(
    "requested,expected", [("cuda", 2), ("cuda:0", 0), ("cuda:3", 3)]
)
def test_device_selection_passes_integer_index(requested, expected):
    torch = Mock()
    torch.device.side_effect = torch_device_stub
    torch.cuda.is_available.return_value = True
    torch.cuda.current_device.return_value = 2
    assert configure_device(torch, requested) == f"cuda:{expected}"
    torch.cuda.set_device.assert_called_once_with(expected)


def test_cpu_device_does_not_touch_cuda():
    torch = Mock()
    torch.device.side_effect = torch_device_stub
    assert configure_device(torch, "cpu") == "cpu"
    torch.cuda.set_device.assert_not_called()


@pytest.mark.parametrize("requested", ["cpu", "cuda:1"])
def test_float32_precision_is_not_downgraded(requested):
    assert resolve_dtype("float32", requested, bf16_supported=True) == "float32"


def test_automatic_precision_and_unsupported_requests():
    assert resolve_dtype("auto", "cpu") == "float32"
    assert resolve_dtype("auto", "cuda:0", False) == "float16"
    assert resolve_dtype("auto", "cuda:1", True) == "bfloat16"
    with pytest.raises(ValueError, match="BF16"):
        resolve_dtype("bfloat16", "cuda:0", False)
    with pytest.raises(ValueError, match="CPU"):
        resolve_dtype("float16", "cpu")
    with pytest.raises(ValueError):
        resolve_dtype("int8", "cuda:0")


def test_sam_builder_explicitly_places_model_on_indexed_gpu():
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    builder = Mock(return_value=model)
    assert load_model(builder, "/weights/sam3.pt", "cuda:3") is model
    builder.assert_called_once_with(
        checkpoint_path="/weights/sam3.pt",
        load_from_HF=False,
        device="cpu",
        eval_mode=True,
    )
    model.to.assert_called_once_with("cuda:3")


def test_missing_weights_and_view_limit_fail_without_loading_models(tmp_path):
    with pytest.raises(FileNotFoundError, match="model_path"):
        run_sam3({"config": {"categories": ["chair"], "allow_download": False}})
    with pytest.raises(ValueError, match="max_views"):
        run_da3({"config": {"max_views": 1}, "views": [{}, {}]})
    model_path = tmp_path / "incomplete"
    model_path.mkdir()
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        require_local_model(model_path, ("config.json", "model.safetensors"))
    with pytest.raises(FileNotFoundError):
        require_local_model(tmp_path / "missing", (), allow_download=True)
    assert require_local_model("owner/model", (), allow_download=True) == "owner/model"


@pytest.mark.parametrize("flag", [False, 0, np.array(0), None])
def test_relative_scale_does_not_claim_meters(flag):
    assert prediction_scale_status("da3-large", flag) == "relative"
    with pytest.raises(ValueError, match="metric scaling"):
        prediction_scale_status("da3nested-giant-large", flag)


@pytest.mark.parametrize("flag", [True, 1, np.array(1)])
def test_metric_scale_requires_actual_model_flag(flag):
    assert (
        prediction_scale_status("da3nested-giant-large", flag)
        == "model_metric_estimate"
    )


@pytest.mark.parametrize("flag", [2, float("nan"), "true", [0, 1]])
def test_malformed_metric_flags_are_rejected(flag):
    with pytest.raises(ValueError):
        prediction_scale_status("da3-large", flag)


@pytest.mark.parametrize(
    "dtype_name,enabled", [("float32", False), ("float16", True), ("bfloat16", True)]
)
def test_da3_forward_precision_boundary(dtype_name, enabled):
    torch = SimpleNamespace(
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        inference_mode=Mock(side_effect=nullcontext),
        autocast=Mock(side_effect=lambda **kwargs: nullcontext()),
    )
    wrapper = configured_model_type(object, torch, dtype_name)()
    wrapper.model = Mock(return_value={"boundary_test_only": True})
    image = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    wrapper.forward(image, use_ray_pose=True)
    torch.autocast.assert_called_once_with(
        device_type="cuda", dtype=dtype_name, enabled=enabled
    )
    wrapper.model.assert_called_once_with(
        image, None, None, None, False, True, "saddle_balanced"
    )
