"""配置、统一的阶段注册表与单进程场景 Pipeline。"""

from __future__ import annotations

import copy
import os
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from loguru import logger

from scene_create import observations, reconstruction
from scene_create.io import atomic_json, file_digest, fingerprint
from scene_create.logging_utils import configure_logging
from scene_create.runtime import MODELS, ModelRuntime, ObjectStep, StageContext, StageStore, output_lock, run_objects

# -------------------------------------------
# 配置常量
# -------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

# fmt: off
MIN_POINT_LIMIT = 3  # 物体与背景点数下限的最小允许值，少于三点无法构成曲面
# fmt: on

DEFAULTS = {
    "runtime": {"seed": 0, "retries": 0, "continue_on_object_error": True},
    "models": {
        name: {
            "repo_path": f"third_parts/{spec.directory}",
            "model_path": None,
            "device": "cuda",
            "dtype": "float16",
            "allow_download": False,
        }
        for name, spec in MODELS.items()
    },
    "geometry": {
        "min_confidence": 0.0,
        "min_points": 32,
        "outlier_neighbors": 16,
        "outlier_std_ratio": 2.5,
    },
    "fusion": {
        "voxel_size": None,
        "voxel_ratio": 0.005,
        "match_distance_ratio": 0.08,
        "min_overlap": 0.35,
        "max_color_distance": 0.4,
        "ambiguity_margin": 0.1,
    },
    "completion": {"padding": 0.25, "image_size": 768},
    "registration": {
        "sample_count": 12000,
        "min_fitness": 0.5,
        "max_rmse_ratio": 0.04,
        "min_mask_iou": 0.25,
        "max_relative_depth_error": 0.15,
        "quality_max_side": 256,
    },
    "background": {
        "mode": "train",
        "voxel_size": None,
        "voxel_ratio": 0.003,
        "max_points": 250000,
        "mask_dilation": 3,
    },
}
DEFAULTS["models"]["sam3"].update(
    {
        "categories": ["chair", "table", "sofa", "cabinet", "lamp", "bottle", "cup", "book", "vase", "plant", "monitor", "keyboard", "bag", "box"],
        "score_threshold": 0.5,
    }
)
DEFAULTS["models"]["da3"].update({"max_views": 16, "process_res": 504, "exclude_sky": True})
DEFAULTS["models"]["flux2"].update(
    {
        "pipeline_class": "AutoPipelineForInpainting",
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "strength": 0.99,
        "cpu_offload": True,
    }
)
DEFAULTS["models"]["trellis"].update(
    {
        "texture_size": 1024,
        "dino_repo_path": "third_parts/dinov2",
        "dino_model_path": None,
        "rembg_model_path": "checkpoints/rembg/u2net.onnx",
        "attention_backend": "xformers",
    }
)
DEFAULTS["models"]["maskgaussian"].update({"iterations": 1000, "sh_degree": 0, "depth_weight": 0.1})

PATH_KEYS = ("repo_path", "model_path", "dino_repo_path", "dino_model_path", "rembg_model_path")
REMOVED_KEYS = {
    ("runtime", "worker_timeout"): "models run in-process and have no worker timeout",
    **{("models", name, "python"): "all models run in the interpreter executing main.py" for name in MODELS},
}


# -------------------------------------------
# 配置加载
# -------------------------------------------
def merge_dict(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict:
    """递归合并字典，返回新字典，不修改输入。"""
    result = copy.deepcopy(dict(base))
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Optional[Union[str, Path]] = None, root: Optional[Union[str, Path]] = None) -> dict:
    """读取 YAML 配置，与默认值合并、解析路径并校验。

    Args:
        path: YAML 配置文件；为 ``None`` 时只使用默认值。
        root: 解析相对路径的项目根目录；默认为本仓库根目录。

    Raises:
        ValueError: 配置结构、取值非法，或使用了已移除的配置项。
    """
    root = Path(root or ROOT).resolve()
    settings = copy.deepcopy(DEFAULTS)
    if path:
        settings = merge_dict(settings, _read_supplied_config(Path(path)))
    if not set(settings["models"]).issubset(MODELS):
        raise ValueError("Unknown model in configuration")
    for name, model in settings["models"].items():
        if not isinstance(model, dict):
            raise ValueError(f"models.{name} must be a mapping")
        _resolve_model_paths(model, root)
        model.setdefault("seed", settings["runtime"]["seed"])
    _validate_settings(settings)
    settings["root"] = str(root)
    return settings


def _read_supplied_config(path: Path) -> dict:
    """读取用户 YAML，拒绝未知顶层节和已移除的 worker 配置项。"""
    with path.open(encoding="utf-8") as stream:
        supplied = yaml.safe_load(stream) or {}
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("Configuration must be a mapping with known top-level sections")
    for keys, reason in REMOVED_KEYS.items():
        section = supplied
        for key in keys[:-1]:
            section = section.get(key) if isinstance(section, dict) else None
        if isinstance(section, dict) and keys[-1] in section:
            raise ValueError(f"{'.'.join(keys)} is no longer supported: {reason}")
    return supplied


def _resolve_model_paths(model: dict, root: Path) -> None:
    """把模型配置中的路径展开为绝对路径（原地修改 ``model``）。

    允许下载且本地不存在的相对 ``model_path`` 视为远程仓库名，保持原样。
    """
    for key in PATH_KEYS:
        value = model.get(key)
        if not value:
            continue
        # 先展开 ~ 再替换环境变量，与重构前顺序一致；pathlib 没有 expandvars 的等价接口。
        expanded = Path(os.path.expandvars(str(Path(str(value)).expanduser())))
        is_remote_name = key == "model_path" and model.get("allow_download") and not expanded.is_absolute() and not (root / expanded).exists()
        if is_remote_name:
            continue
        model[key] = str(expanded if expanded.is_absolute() else root / expanded)


def _validate_settings(settings: Mapping[str, Any]) -> None:
    """校验与具体模型无关的数值和枚举配置。"""
    if settings["runtime"]["retries"] < 0:
        raise ValueError("runtime.retries must be nonnegative")
    has_small_point_limit = settings["geometry"]["min_points"] < MIN_POINT_LIMIT or settings["background"]["max_points"] < MIN_POINT_LIMIT
    if has_small_point_limit:
        raise ValueError("Point limits must be at least three")
    if settings["background"]["mode"] not in ("train", "init-only"):
        raise ValueError("background.mode must be train or init-only")
    if not settings["models"]["sam3"]["categories"]:
        raise ValueError("SAM3 requires at least one category prompt")
    for section in ("fusion", "background"):
        voxel_size = settings[section]["voxel_size"]
        has_invalid_size = voxel_size is not None and voxel_size <= 0
        if settings[section]["voxel_ratio"] <= 0 or has_invalid_size:
            raise ValueError("Voxel sizes must be positive")


# -------------------------------------------
# 阶段注册表
# -------------------------------------------
def always(config: Mapping[str, Any], imported: bool) -> bool:
    """模型配置始终参与指纹。"""
    return True


def unless_imported(config: Mapping[str, Any], imported: bool) -> bool:
    """只有不是外部导入观测时模型才会运行。"""
    return not imported


def trains_background(config: Mapping[str, Any], imported: bool) -> bool:
    """只有训练模式才会运行 MaskGaussian。"""
    return config["background"]["mode"] == "train"


@dataclass(frozen=True)
class StageSpec:
    """一个流水线阶段：依赖、参与缓存指纹的配置节和执行器。

    ``run(context)`` 与 ``objects`` 二者恰有其一负责执行该阶段。``model`` 是它可能调用的
    适配器；``model_enabled`` 决定该模型的配置和权重是否计入缓存指纹。
    """

    name: str
    requires: tuple[str, ...] = ()
    run: Optional[Callable[[StageContext], tuple[dict, list]]] = None
    objects: Optional[ObjectStep] = None
    settings: tuple[str, ...] = ()
    model: Optional[str] = None
    model_enabled: Callable[[Mapping[str, Any], bool], bool] = always

    def execute(self, context: StageContext) -> tuple[dict, list]:
        """执行本阶段，返回结果与产物路径。"""
        if self.objects is not None:
            return run_objects(context, self.objects)
        return self.run(context)


STAGE_SPECS = (
    StageSpec("ingest", run=observations.ingest),
    StageSpec("segment", ("ingest",), observations.segment, model="sam3", model_enabled=unless_imported),
    StageSpec("depth", ("ingest",), observations.estimate_depth, settings=("geometry",), model="da3", model_enabled=unless_imported),
    StageSpec("objects", ("ingest", "segment", "depth"), observations.build_objects, settings=("geometry", "fusion")),
    StageSpec(
        "complete",
        ("ingest", "objects"),
        objects=ObjectStep("objects", "observed", reconstruction.complete_object, {"insufficient_geometry": "Insufficient observed points"}),
        settings=("completion",),
        model="flux2",
    ),
    StageSpec("assets", ("complete",), objects=ObjectStep("complete", "completed_image", reconstruction.generate_asset), model="trellis"),
    StageSpec("register", ("assets",), objects=ObjectStep("assets", "generated_asset", reconstruction.register_object), settings=("registration",)),
    StageSpec(
        "background",
        ("register", "depth"),
        reconstruction.reconstruct_background,
        settings=("background",),
        model="maskgaussian",
        model_enabled=trains_background,
    ),
    StageSpec("assemble", ("register", "background", "depth"), reconstruction.assemble_scene),
)
STAGES = {spec.name: spec for spec in STAGE_SPECS}
STAGE_NAMES = tuple(STAGES)


def _validate_registry() -> None:
    """导入时检查注册表：执行器唯一、依赖在前、模型已登记。"""
    seen: set[str] = set()
    for spec in STAGE_SPECS:
        if (spec.run is None) == (spec.objects is None):
            raise ValueError(f"Stage {spec.name} needs exactly one executor")
        if not seen.issuperset(spec.requires):
            raise ValueError(f"Stage {spec.name} must follow its dependencies")
        if spec.objects and spec.objects.source not in spec.requires:
            raise ValueError(f"Stage {spec.name} reads objects from a non-dependency")
        if spec.model and spec.model not in MODELS:
            raise ValueError(f"Stage {spec.name} uses an unknown model")
        seen.add(spec.name)


_validate_registry()


def stage_plan(targets: Optional[Iterable[str]] = None) -> list[str]:
    """Requested stages plus their transitive dependencies, in registry order.

    Raises:
        ValueError: 包含未登记的阶段名。
    """
    pending = list(targets or STAGE_NAMES)
    unknown = sorted(set(pending) - set(STAGES))
    if unknown:
        raise ValueError(f"Unknown pipeline stage: {', '.join(unknown)}")
    needed: set[str] = set()
    while pending:
        name = pending.pop()
        if name not in needed:
            needed.add(name)
            pending.extend(STAGES[name].requires)
    return [name for name in STAGE_NAMES if name in needed]


def dependents(name: str) -> list[str]:
    """直接或间接依赖 ``name`` 的下游阶段，按注册表顺序。"""
    affected = {name}
    for spec in STAGE_SPECS:
        if affected.intersection(spec.requires):
            affected.add(spec.name)
    return [stage for stage in STAGE_NAMES if stage in affected - {name}]


# -------------------------------------------
# 缓存指纹
# -------------------------------------------
def weight_inventory(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """列出模型权重文件的路径、大小与修改时间，用作廉价的权重指纹。"""
    inventory = []
    for key in ("model_path", "dino_model_path", "rembg_model_path"):
        weights = model.get(key)
        if not weights or not Path(weights).exists():
            continue
        path = Path(weights)
        if path.is_file():
            files = [path]
        else:
            files = sorted(item for item in path.rglob("*") if item.is_file())
        for item in files:
            status = item.stat()
            inventory.append({"path": str(item), "size": status.st_size, "mtime_ns": status.st_mtime_ns})
    return inventory


def source_fingerprint(directory: Union[str, Path]) -> Optional[str]:
    """目录下全部 ``*.py`` 内容的指纹；目录不存在时为 ``None``。"""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    digests = {str(path.relative_to(directory)): file_digest(path) for path in sorted(directory.rglob("*.py"))}
    return fingerprint(digests)


# -------------------------------------------
# 场景 Pipeline
# -------------------------------------------
@dataclass
class _PipelineRun:
    """一次运行中各阶段共享的状态。"""

    store: StageStore
    models: ModelRuntime
    inputs: dict
    code: Optional[str]
    results: dict[str, dict] = field(default_factory=dict)
    identities: dict[str, str] = field(default_factory=dict)

    @property
    def imported(self) -> bool:
        """输入是否为外部导入的观测。"""
        return self.inputs["provenance"] == "external_observations"


class ScenePipeline:
    """在当前解释器中按注册表顺序执行阶段，并复用已校验的结果。"""

    def __init__(self, config: dict, output: Union[str, Path]) -> None:
        self.config = config
        self.output = Path(output).expanduser().resolve()

    def stage_settings(self, spec: StageSpec, imported: bool) -> dict[str, Any]:
        """返回可能改变该阶段结果的配置，用于计算阶段指纹。"""
        subset: dict[str, Any] = {key: self.config[key] for key in spec.settings}
        subset["runtime"] = {key: self.config["runtime"][key] for key in ("seed", "continue_on_object_error")}
        if not spec.model or not spec.model_enabled(self.config, imported):
            return subset
        model = self.config["models"][spec.model]
        subset.update(model=model, weight_inventory=weight_inventory(model), source_fingerprint=source_fingerprint(model["repo_path"]))
        manifest = Path(self.config["root"]) / "third_parts" / "manifest.json"
        if manifest.is_file():
            subset["source_manifest"] = file_digest(manifest)
        return subset

    def check_output(self, inputs: Mapping[str, Any]) -> None:
        """拒绝不属于本流水线的非空目录，以及与输入图片相同的目录。

        Raises:
            ValueError: 输出目录不可用。
        """
        is_foreign = self.output.exists() and not (self.output / "run_manifest.json").exists() and any(self.output.iterdir())
        if is_foreign:
            raise ValueError("Output directory is nonempty and is not owned by this pipeline; choose a new directory")
        for view in inputs["views"]:
            if self.output == Path(view["source_path"]).parent:
                raise ValueError("Output directory cannot be the input image directory")

    def run(
        self,
        source: Mapping[str, Path],
        stages: Optional[Sequence[str]] = None,
        stage_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> dict[str, Any]:
        """Run ``stages`` (default: all) and their dependencies.

        Stages whose fingerprint and artifacts already verify in this output's
        ``run_manifest.json`` are reused; everything else runs in-process.

        Args:
            source: 唯一的输入来源，键为 ``image``/``images``/``manifest``/``observations``。
            stages: 目标阶段；为 ``None`` 时运行全部阶段。
            stage_callback: 阶段进度回调，参数为阶段名、已完成数和总阶段数。

        Returns:
            运行状态、目标阶段、实际计划、输出位置与各目标阶段的结果。

        Raises:
            ValueError: 输入、阶段名或输出目录非法。
            Exception: 阶段执行失败时原样抛出，失败已写入运行清单。
        """
        targets = list(dict.fromkeys(stages or STAGE_NAMES))
        plan = stage_plan(targets)
        inputs = observations.resolve_input(source)
        self.check_output(inputs)
        run_id = time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        with output_lock(self.output), ModelRuntime(self.config) as models:
            configure_logging(self.output, run_id, stage="pipeline")
            logger.info("Stages: {}", ", ".join(plan))
            state = _PipelineRun(StageStore(self.output, run_id), models, inputs, source_fingerprint(Path(__file__).parent))
            atomic_json(self.output / "metadata" / "config.resolved.json", self.config)
            for index, name in enumerate(plan):
                if stage_callback is not None:
                    stage_callback(name, index, len(plan))
                with logger.contextualize(stage=name):
                    self._run_stage(state, STAGES[name])
                if stage_callback is not None:
                    stage_callback(name, index + 1, len(plan))
            status = self.status(state.results)
            state.store.finish(status, targets)
            logger.info("Pipeline {}", status)
            return {
                "status": status,
                "stages": targets,
                "plan": plan,
                "output": str(self.output),
                "manifest": str(state.store.path),
                "results": {name: state.results[name] for name in targets},
            }

    def _run_stage(self, state: _PipelineRun, spec: StageSpec) -> None:
        """复用或执行一个阶段，把结果和身份指纹写回 ``state``。"""
        digest = fingerprint(
            {
                "stage": spec.name,
                "code": state.code,
                "settings": self.stage_settings(spec, state.imported),
                "inputs": state.inputs if spec.name == "ingest" else None,
                "dependencies": {item: state.identities[item] for item in spec.requires},
            }
        )
        result = state.store.lookup(spec.name, digest)
        if result is None:
            result = self._execute_stage(state, spec, digest)
        state.results[spec.name] = result
        record = state.store.data["stages"][spec.name]
        state.identities[spec.name] = fingerprint({"fingerprint": digest, "result": result, "artifacts": record["artifacts"]})

    def _execute_stage(self, state: _PipelineRun, spec: StageSpec, digest: str) -> dict:
        """在新的中间目录中执行阶段；无论成败都释放本阶段加载的模型。"""
        context = StageContext(
            stage=spec.name,
            config=self.config,
            output=self.output,
            directory=state.store.begin(spec.name, digest, dependents(spec.name)),
            inputs=state.inputs,
            results={item: state.results[item] for item in spec.requires},
            models=state.models,
        )
        # 阶段执行边界：任意失败都写入运行清单后原样抛出。
        try:
            result, artifacts = spec.execute(context)
        except Exception as error:
            state.store.fail(spec.name, error)
            raise
        finally:
            state.models.release()
        return state.store.complete(spec.name, result, artifacts)

    @staticmethod
    def status(results: Mapping[str, Mapping[str, Any]]) -> str:
        """汇总运行状态：``partial``、``approximate`` 或 ``completed``。"""
        has_failures = any(result.get("failures") for result in results.values())
        objects = results.get("objects", {}).get("objects", [])
        has_insufficient = any(item.get("status") == "insufficient_geometry" for item in objects)
        if has_failures or has_insufficient:
            return "partial"
        if results.get("background", {}).get("approximate", False):
            return "approximate"
        return "completed"
