"""进程内执行：输出目录锁、阶段缓存、按物体循环、模型延迟加载与环境预检。"""

from __future__ import annotations

import copy
import fcntl
import gc
import importlib
import importlib.util
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Optional, Union

from loguru import logger

from scene_create.io import atomic_json, file_digest, read_json

# -------------------------------------------
# 配置常量
# -------------------------------------------
# fmt: off
FINGERPRINT_DIRECTORY_CHARS = 16   # 阶段中间目录名使用的指纹前缀长度
RETRY_BACKOFF_MAX_SECONDS = 10     # 模型重试时指数退避的最长等待时间
GIT_REVISION_TIMEOUT_SECONDS = 15  # doctor 读取第三方仓库 commit 的超时时间
TRELLIS_MIN_GPU_MEMORY_GIB = 16    # TRELLIS 文档要求的最小显存
BF16_MIN_COMPUTE_MAJOR = 8         # 原生 BF16 所需的最低 CUDA 计算能力主版本
SAM3_MIN_PYTHON = (3, 12)          # 锁定版本 SAM3 要求的最低 Python 版本
BYTES_PER_GIB = 1024**3            # 显存字节数换算为 GiB 的除数
# fmt: on

SOURCE_REVISION_MISMATCH = "Source commit differs from third_parts/manifest.json; adapter compatibility must be revalidated"
DOWNLOAD_WARNING = "Weights will require network access, sufficient disk space and any necessary license approval"
DOCTOR_NOTES = (
    "All five models run in this Python interpreter; module discovery is not a real-model inference test.",
    "Weights, licenses and CUDA extensions must be provisioned in this environment.",
)


# -------------------------------------------
# 模型登记
# -------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """一个模型在 ``third_parts/`` 中的位置、预检模块与是否需要权重。"""

    directory: str
    modules: tuple[str, ...]
    source: str = ""
    needs_weights: bool = True


MODELS = {
    "sam3": ModelSpec("sam3", ("torch", "sam3")),
    "da3": ModelSpec("Depth-Anything-3", ("torch", "depth_anything_3"), "src"),
    "flux2": ModelSpec("flux2", ("torch", "diffusers", "transformers"), "src"),
    "trellis": ModelSpec("TRELLIS", ("torch", "trellis", "spconv", "nvdiffrast", "rembg")),
    "maskgaussian": ModelSpec(
        "MaskGaussian",
        ("torch", "mask_diff_gaussian_rasterization", "diff_gaussian_rasterization", "simple_knn"),
        needs_weights=False,
    ),
}


# -------------------------------------------
# 输出目录与阶段缓存
# -------------------------------------------
@contextmanager
def output_lock(output: Union[str, Path]) -> Iterator[None]:
    """独占输出目录，防止两个进程同时写入。

    Raises:
        RuntimeError: 另一个进程已持有该目录的锁。
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is writing to {output}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def verified(artifacts: Iterable[Mapping[str, str]]) -> bool:
    """判断记录的产物是否都存在且 SHA-256 未变。"""
    for item in artifacts:
        path = Path(item["path"])
        if not path.is_file() or file_digest(path) != item["sha256"]:
            return False
    return True


def artifact_records(paths: Iterable[Union[str, Path]]) -> list[dict[str, str]]:
    """为去重、排序后的产物路径生成 ``{path, sha256}`` 记录。"""
    unique_paths = sorted({str(Path(path).resolve()) for path in paths})
    return [{"path": path, "sha256": file_digest(path)} for path in unique_paths]


class StageStore:
    """``run_manifest.json`` 的读写：阶段指纹、已校验产物和错误。"""

    def __init__(self, output: Path, run_id: str) -> None:
        self.output = Path(output)
        self.path = self.output / "run_manifest.json"
        if self.path.exists():
            self.data = read_json(self.path)
        else:
            self.data = {"schema_version": 1, "run_id": run_id, "stages": {}}
        self.data["last_run_id"] = run_id
        self.data["status"] = "running"
        self.save()

    def save(self) -> None:
        """原子写回运行清单。"""
        atomic_json(self.path, self.data)

    def lookup(self, name: str, digest: str) -> Optional[dict]:
        """仅当指纹和产物文件都仍然一致时返回已保存的结果，否则返回 None。"""
        previous = self.data["stages"].get(name, {})
        is_completed = previous.get("status") == "completed"
        is_current = previous.get("fingerprint") == digest
        if is_completed and is_current and verified(previous.get("artifacts", [])):
            logger.info("Using verified cached result")
            return previous["result"]
        return None

    def begin(self, name: str, digest: str, dependents: Sequence[str] = ()) -> Path:
        """登记阶段开始，必要时把下游阶段标记为 stale，返回本次中间目录。"""
        previous = self.data["stages"].get(name, {})
        is_changed = previous.get("fingerprint") != digest
        was_unfinished = previous.get("status") not in (None, "completed")
        if is_changed or was_unfinished:
            for dependent in dependents:
                if dependent in self.data["stages"]:
                    self.data["stages"][dependent]["status"] = "stale"
        directory = self.output / "intermediate" / name / digest[:FINGERPRINT_DIRECTORY_CHARS]
        directory.mkdir(parents=True, exist_ok=True)
        self.data["stages"][name] = {"status": "running", "fingerprint": digest, "started_at": time.time(), "directory": str(directory)}
        self.save()
        logger.info("Starting stage")
        return directory

    def complete(self, name: str, result: dict, artifacts: Iterable[Union[str, Path]]) -> dict:
        """保存阶段结果与产物校验和，返回经 JSON 往返后的结果。"""
        entry = self.data["stages"][name]
        result_path = Path(entry["directory"]) / "result.json"
        atomic_json(result_path, result)
        entry.update(
            status="partial" if result.get("failures") else "completed",
            result=read_json(result_path),
            artifacts=artifact_records([*artifacts, result_path]),
            elapsed_seconds=time.time() - entry["started_at"],
        )
        self.save()
        logger.info("Completed in {:.2f}s", entry["elapsed_seconds"])
        return entry["result"]

    def fail(self, name: str, error: BaseException) -> None:
        """记录阶段失败及异常信息。"""
        entry = self.data["stages"][name]
        entry.update(status="failed", error_type=type(error).__name__, error=str(error), elapsed_seconds=time.time() - entry["started_at"])
        self.data["status"] = "failed"
        self.save()
        logger.opt(exception=error).error("Stage failed: {}", error)

    def finish(self, status: str, targets: Iterable[str]) -> None:
        """记录整次运行的最终状态与请求的目标阶段。"""
        self.data.update(status=status, targets=list(targets), finished_at=time.time())
        self.save()


# -------------------------------------------
# 阶段上下文与按物体执行
# -------------------------------------------
@dataclass
class StageContext:
    """一个阶段执行时可见的配置、目录、输入、上游结果与模型运行时。"""

    stage: str
    config: dict
    output: Path
    directory: Path
    inputs: dict
    results: dict
    models: ModelRuntime

    @property
    def imported(self) -> bool:
        """输入是否为外部导入的观测。"""
        return self.inputs["provenance"] == "external_observations"


@dataclass(frozen=True)
class ObjectStep:
    """对 ``source`` 阶段结果中的每个物体执行 ``handler(context, item)``。"""

    source: str
    accepts: str
    handler: Callable[[StageContext, dict], tuple[dict, list]]
    rejects: dict[str, str] = field(default_factory=dict)


def run_objects(context: StageContext, step: ObjectStep) -> tuple[dict, list]:
    """Object-level loop with per-object verified reuse and failure isolation.

    Returns:
        上游结果的深拷贝（物体状态已更新、失败列入 ``failures``）与产物路径列表。

    Raises:
        Exception: ``runtime.continue_on_object_error`` 为假时，原样抛出首个物体失败。
    """
    result = copy.deepcopy(context.results[step.source])
    result.setdefault("failures", [])
    artifacts = []
    for item in result["objects"]:
        if item["status"] in step.rejects:
            result["failures"].append(_object_failure(item, context.stage, step.rejects[item["status"]]))
            continue
        if item["status"] != step.accepts:
            continue
        with logger.contextualize(object_id=item["object_id"]):
            # 物体失败隔离边界：单个物体的任意异常都记录为失败，其余物体继续处理。
            try:
                artifacts.extend(_process_object(context, step, item))
            except Exception as error:
                item["status"] = f"{context.stage}_failed"
                item["error"] = str(error)
                result["failures"].append(_object_failure(item, context.stage, str(error)))
                logger.exception("Object processing failed")
                if not context.config["runtime"]["continue_on_object_error"]:
                    raise
    return result, artifacts


def _process_object(context: StageContext, step: ObjectStep, item: dict) -> list:
    """复用已校验的单物体结果，否则执行 handler 并缓存；原地更新 ``item``，返回产物路径。"""
    cache_path = context.directory / f"{item['object_id']}.json"
    cached = read_json(cache_path) if cache_path.is_file() else None
    if cached and verified(cached["artifacts"]):
        update = cached["update"]
        created = [value["path"] for value in cached["artifacts"]]
        logger.info("Reusing verified object result")
    else:
        update, created = step.handler(context, item)
        atomic_json(cache_path, {"update": update, "artifacts": artifact_records(created)})
    item.update(update)
    logger.info("Object {}: {}", item["object_id"], item["status"])
    return [*created, cache_path]


def _object_failure(item: Mapping[str, Any], stage: str, error: str) -> dict[str, str]:
    return {"object_id": item["object_id"], "stage": stage, "error": error}


# -------------------------------------------
# 设备、精度与本地权重
# -------------------------------------------
def configure_device(torch: ModuleType, requested: str) -> str:
    """把 ``cpu``/``cuda``/``cuda:N`` 解析为确定的设备并设为当前 CUDA 设备。

    Raises:
        ValueError: 不是 cpu 或 cuda 设备。
        RuntimeError: 请求 CUDA 但当前环境不可用。
    """
    target = torch.device(requested)
    if target.type == "cpu":
        return "cpu"
    if target.type != "cuda":
        raise ValueError("Models support only cpu or cuda[:index] devices")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    index = target.index
    if index is None:
        index = torch.cuda.current_device()
    torch.cuda.set_device(index)
    return f"cuda:{index}"


def resolve_dtype(requested: str, device: str, bf16_supported: bool = False) -> str:
    """根据设备能力确定实际推理精度。

    Raises:
        ValueError: 精度名称未知，或设备不支持所请求的精度。
    """
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


def require_local_model(value: Optional[Union[str, Path]], files: Iterable[str], allow_download: bool = False) -> str:
    """确认本地模型目录完整；仅在允许下载时接受远程仓库名。

    Raises:
        ValueError: 未配置路径，或路径不是目录。
        FileNotFoundError: 本地目录缺失或不完整。
    """
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
    looks_local = path.is_absolute() or str(value).startswith(".")
    if not allow_download or looks_local:
        raise FileNotFoundError(f"Local model directory does not exist: {path}")
    return str(value)


def model_source(name: str, config: Mapping[str, Any]) -> Path:
    """把模型的第三方源码目录加入 ``sys.path``（只加一次）并返回该目录。"""
    source = Path(config["repo_path"]) / MODELS[name].source
    # third_parts 中的上游源码不可 pip 安装，只能在延迟加载时加入导入路径。
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


def configure_model_environment(config: Mapping[str, Any]) -> None:
    """设置缓存目录和离线开关；Hugging Face 只在首次导入时读取它们。"""
    root = Path(config["root"])
    os.environ.setdefault("HF_HOME", str(root / "checkpoints" / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(root / "checkpoints" / "torch"))
    if not any(model.get("allow_download") for model in config["models"].values()):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


# -------------------------------------------
# 进程内模型运行时
# -------------------------------------------
class ModelRuntime:
    """在当前解释器中按需导入并加载模型适配器，同一时刻只保留一个模型。

    适配器模块提供 ``load(config, root) -> handle`` 和 ``run(handle, request) -> dict``。
    同一模型被重复调用（例如逐个物体）时句柄保持常驻；加载其他模型之前或所属阶段结束时释放。
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.active: Optional[tuple[str, ModuleType, Any]] = None
        configure_model_environment(config)

    def __enter__(self) -> ModelRuntime:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    def call(self, name: str, payload: Mapping[str, Any], destination: Union[str, Path]) -> dict:
        """调用模型；失败时释放并重新加载，最多重试 ``runtime.retries`` 次。

        Raises:
            Exception: 最后一次尝试中适配器抛出的异常。
        """
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        request = {"root": self.config["root"], "config": self.config["models"][name], "output_dir": str(destination.resolve()), **payload}
        attempts = 1 + self.config["runtime"]["retries"]
        for attempt in range(attempts):
            # 第三方模型可能抛出任意异常；释放后重新加载重试，最后一次原样抛出。
            try:
                adapter, handle = self.load(name)
                return adapter.run(handle, request)
            except Exception as error:
                if attempt + 1 == attempts:
                    raise
                logger.opt(exception=error).warning("{} failed; retry {}/{}", name, attempt + 1, attempts - 1)
                self.release()
                time.sleep(min(2**attempt, RETRY_BACKOFF_MAX_SECONDS))
        raise RuntimeError("Model call did not return a result")

    def load(self, name: str) -> tuple[ModuleType, Any]:
        """返回已加载的适配器与句柄；需要时先释放当前模型再加载。

        Raises:
            FileNotFoundError: 需要权重但本地权重不存在且不允许下载。
        """
        if self.active and self.active[0] == name:
            return self.active[1], self.active[2]
        self.release()
        config = self.config["models"][name]
        _require_configured_weights(name, config)
        model_source(name, config)
        adapter = importlib.import_module(f"scene_create.adapters.{name}")
        started = time.monotonic()
        logger.info("Loading {} in-process", name)
        handle = adapter.load(config, Path(self.config["root"]))
        logger.info("{} loaded in {:.1f}s", name, time.monotonic() - started)
        self.active = (name, adapter, handle)
        return adapter, handle

    def release(self) -> None:
        """释放当前模型并清空 CUDA 缓存。"""
        if self.active is None:
            return
        logger.info("Releasing {}", self.active[0])
        self.active = None
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _require_configured_weights(name: str, config: Mapping[str, Any]) -> None:
    """在导入任何重量级依赖之前确认本地权重存在。"""
    weights = config.get("model_path")
    has_local_weights = bool(weights) and Path(weights).exists()
    if MODELS[name].needs_weights and not config.get("allow_download") and not has_local_weights:
        raise FileNotFoundError(f"models.{name}.model_path must reference local model weights; automatic downloading is disabled")


# -------------------------------------------
# 环境预检
# -------------------------------------------
def probe_environment(name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """检查当前解释器中的模块可发现性和设备信息。"""
    model_source(name, config)
    available = {}
    for module in MODELS[name].modules:
        try:
            available[module] = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError, AttributeError):
            available[module] = False
    result: dict[str, Any] = {"python": platform.python_version(), "executable": sys.executable, "modules": available, "cuda_available": False}
    if not available.get("torch"):
        return result
    # torch 或驱动损坏时可能抛出任意异常；预检只记录原因，不中断其他模型的检查。
    try:
        result.update(_probe_torch(config["device"]))
    except Exception as error:
        result["torch_error"] = str(error)
    return result


def _probe_torch(requested_device: str) -> dict[str, Any]:
    """读取 torch、CUDA 版本以及所选 GPU 的型号、显存和计算能力。"""
    # torch 只在预检确认可发现后才导入，避免 CPU 主环境的启动开销。
    import torch

    facts: dict[str, Any] = {"torch": str(torch.__version__), "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    if not torch.cuda.is_available() or not requested_device.startswith("cuda"):
        return facts
    device = configure_device(torch, requested_device)
    index = int(device.split(":")[1])
    facts.update(
        device=device,
        gpu=torch.cuda.get_device_name(index),
        memory_gib=torch.cuda.get_device_properties(index).total_memory / BYTES_PER_GIB,
        capability=list(torch.cuda.get_device_capability(index)),
    )
    return facts


def diagnose(config: Mapping[str, Any]) -> dict[str, Any]:
    """不加载模型，检查五个模型的源码、权重和依赖模块。"""
    manifest_path = Path(config["root"]) / "third_parts" / "manifest.json"
    pins = read_json(manifest_path).get("repositories", {}) if manifest_path.is_file() else {}
    report = {"schema_version": 2, "ready": True, "python": sys.executable, "models": {}, "notes": list(DOCTOR_NOTES)}
    configure_model_environment(config)
    for name, model in config["models"].items():
        issues: list[str] = []
        warnings: list[str] = []
        item: dict[str, Any] = {"repository": model["repo_path"], "model_path": model.get("model_path")}
        commit = _check_source_revision(Path(model["repo_path"]), pins.get(name, {}).get("commit"), issues, warnings)
        if commit:
            item["commit"] = commit
        _check_weights(name, model, issues, warnings)
        item["environment"] = probe_environment(name, model)
        _check_environment(name, model, item["environment"], issues, warnings)
        item.update(ready=not issues, issues=issues, warnings=warnings)
        report["models"][name] = item
        report["ready"] = report["ready"] and item["ready"]
    return report


def _check_source_revision(repo: Path, expected: Optional[str], issues: list[str], warnings: list[str]) -> Optional[str]:
    """读取第三方仓库 commit 并与锁定版本比较；结论追加到 ``issues``/``warnings``。"""
    if not repo.is_dir():
        issues.append("Third-party repository is missing")
        return None
    command = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=GIT_REVISION_TIMEOUT_SECONDS)
    if command.returncode != 0:
        warnings.append("Cannot establish the source revision")
        return None
    commit = command.stdout.strip()
    if expected and commit != expected:
        issues.append(SOURCE_REVISION_MISMATCH)
    return commit


def _check_weights(name: str, model: Mapping[str, Any], issues: list[str], warnings: list[str]) -> None:
    """检查主权重以及 TRELLIS 的 DINOv2/U2Net 辅助资源。"""
    if MODELS[name].needs_weights:
        weights = model.get("model_path")
        if not weights:
            issues.append(f"Configure models.{name}.model_path")
        elif not Path(weights).exists():
            if model.get("allow_download"):
                warnings.append(DOWNLOAD_WARNING)
            else:
                issues.append("Local weights do not exist; downloads are disabled")
    if name != "trellis":
        return
    for key in ("dino_model_path", "rembg_model_path"):
        if not model.get(key) or not Path(model[key]).is_file():
            issues.append(f"Configure a local models.trellis.{key}")
    if not (Path(model.get("dino_repo_path", "")) / "hubconf.py").is_file():
        issues.append("The local DINOv2 source repository is missing")


def _check_environment(name: str, model: Mapping[str, Any], probe: Mapping[str, Any], issues: list[str], warnings: list[str]) -> None:
    """根据模块发现与设备信息判断当前解释器能否运行该模型。"""
    missing = [module for module, present in probe["modules"].items() if not present]
    if missing:
        issues.append("Missing modules: " + ", ".join(missing))
    if probe.get("torch_error"):
        issues.append("Torch/device probe failed: " + probe["torch_error"])
    if model["device"].startswith("cuda") and not probe["cuda_available"]:
        issues.append("CUDA is unavailable in this Python environment")
    if name == "sam3" and sys.version_info < SAM3_MIN_PYTHON:
        issues.append("Pinned SAM3 requires Python >= 3.12")
    if name == "trellis" and probe.get("memory_gib", TRELLIS_MIN_GPU_MEMORY_GIB) < TRELLIS_MIN_GPU_MEMORY_GIB:
        warnings.append("Available GPU memory is below TRELLIS's documented 16GB prerequisite")
    has_native_bf16 = probe.get("capability", [0])[0] >= BF16_MIN_COMPUTE_MAJOR
    if model.get("dtype") == "bfloat16" and not has_native_bf16:
        issues.append("Configured bfloat16 requires a compatible GPU; this GPU does not support native BF16")
