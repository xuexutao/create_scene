from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import sys
from pathlib import Path

from loguru import logger

from scene_create.io import atomic_json, read_json
from scene_create.logging_utils import (
    LoguruArgumentParser,
    capture_external_output,
    configure_logging,
)

MODULES = {
    "sam3": ["torch", "sam3"],
    "da3": ["torch", "depth_anything_3"],
    "flux2": ["torch", "diffusers", "transformers"],
    "trellis": ["torch", "trellis", "spconv", "nvdiffrast", "rembg"],
    "maskgaussian": [
        "torch",
        "mask_diff_gaussian_rasterization",
        "diff_gaussian_rasterization",
        "simple_knn",
    ],
}


def prepare(request):
    name = request["operation"]
    if name not in MODULES:
        raise ValueError(f"Unknown model operation: {name}")
    config = request["config"]
    repository = Path(config["repo_path"])
    source = repository / "src" if name in ("da3", "flux2") else repository
    sys.path.insert(0, str(source))
    root = Path(request["root"])
    os.environ.setdefault("HF_HOME", str(root / "checkpoints" / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(root / "checkpoints" / "torch"))
    if not config.get("allow_download", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return name


def probe(request):
    name = prepare(request)
    available = {}
    for module in MODULES[name]:
        try:
            available[module] = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError, AttributeError):
            available[module] = False
    result = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "modules": available,
        "cuda_available": False,
    }
    if available.get("torch"):
        try:
            import torch

            result.update(
                torch=str(torch.__version__),
                cuda_available=torch.cuda.is_available(),
                torch_cuda=torch.version.cuda,
            )
            if torch.cuda.is_available() and request["config"].get(
                "device", "cuda"
            ).startswith("cuda"):
                from scene_create.adapters.common import configure_device

                device = configure_device(
                    torch, request["config"].get("device", "cuda")
                )
                index = int(device.split(":")[1])
                result.update(
                    device=device,
                    gpu=torch.cuda.get_device_name(index),
                    memory_gib=torch.cuda.get_device_properties(index).total_memory
                    / 1024**3,
                    capability=list(torch.cuda.get_device_capability(index)),
                )
        except Exception as error:
            result["torch_error"] = str(error)
    return result


def main(argv=None):
    configure_logging(stage="worker", serialized_console=True)
    parser = LoguruArgumentParser(description="Isolated scene-create model worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--probe", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        request = read_json(arguments.request)
        context = request.get("logging_context", {})
        configure_logging(
            run_id=context.get("run_id", "-"),
            stage=request.get("operation", "worker"),
            view_id=context.get("view_id", "-"),
            object_id=context.get("object_id", "-"),
            worker_id=context.get("worker_id", "-"),
            serialized_console=True,
        )
        logger.info(
            "Starting {}", "environment probe" if arguments.probe else "model worker"
        )
        with capture_external_output():
            if arguments.probe:
                result = probe(request)
            else:
                name = prepare(request)
                model = request["config"].get("model_path")
                if (
                    name != "maskgaussian"
                    and not request["config"].get("allow_download")
                    and (not model or not Path(model).exists())
                ):
                    raise FileNotFoundError(
                        f"models.{name}.model_path must reference local model weights; automatic downloading is disabled"
                    )
                result = importlib.import_module(f"scene_create.adapters.{name}").run(
                    request
                )
        atomic_json(arguments.response, {"status": "success", "result": result})
        logger.info(
            "{} completed", "Environment probe" if arguments.probe else "Model worker"
        )
        return 0
    except Exception as error:
        logger.exception("Worker failed: {}", error)
        atomic_json(
            arguments.response,
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
