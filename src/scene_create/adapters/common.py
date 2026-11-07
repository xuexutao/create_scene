from __future__ import annotations

from pathlib import Path


def configure_device(torch, requested):
    target = torch.device(requested)
    if target.type == "cpu":
        return "cpu"
    if target.type != "cuda":
        raise ValueError("Model workers support only cpu or cuda[:index] devices")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this model worker")
    index = target.index if target.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(index)
    return f"cuda:{index}"


def resolve_dtype(requested, device, bf16_supported=False):
    if requested not in ("auto", "float16", "bfloat16", "float32"):
        raise ValueError(f"Unsupported inference dtype: {requested}")
    if device == "cpu":
        if requested not in ("auto", "float32"):
            raise ValueError("CPU inference requires dtype: float32 or auto")
        return "float32"
    if requested == "auto":
        return "bfloat16" if bf16_supported else "float16"
    if requested == "bfloat16" and not bf16_supported:
        raise ValueError("The selected CUDA device does not support BF16")
    return requested


def require_local_model(value, files, allow_download=False):
    if not value:
        raise ValueError("A model_path must be configured")
    path = Path(value).expanduser()
    if path.is_dir():
        missing = [name for name in files if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete local model {path}: missing {missing}")
        return str(path.resolve())
    if path.exists():
        raise ValueError(f"Expected a complete model directory: {path}")
    if not allow_download or path.is_absolute() or str(value).startswith("."):
        raise FileNotFoundError(f"Local model directory does not exist: {path}")
    return str(value)
