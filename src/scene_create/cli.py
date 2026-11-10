"""命令行入口：默认运行场景重建流水线，另提供 doctor 与 preview 子命令。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, TextIO

from loguru import logger

from scene_create import runtime
from scene_create.io import atomic_json, json_value
from scene_create.logging_utils import LoguruArgumentParser, capture_external_output, configure_logging
from scene_create.pipeline import ROOT, STAGE_NAMES, ScenePipeline, load_config

# -------------------------------------------
# 配置常量
# -------------------------------------------
# fmt: off
EXIT_SUCCESS = 0         # 请求的阶段全部完成
EXIT_FAILURE = 2         # 输入、环境或执行错误；doctor 表示未就绪
EXIT_PARTIAL = 3         # 部分物体失败或背景只是近似结果
EXIT_INTERRUPTED = 130   # 用户中断（SIGINT 的惯用退出码）
# fmt: on

COMMANDS = ("doctor", "preview")
SOURCE_KEYS = ("image", "images", "manifest", "observations")


# -------------------------------------------
# 进度展示
# -------------------------------------------
class _StageProgress:
    """向交互终端输出按阶段计数的总进度。"""

    def __init__(self, stream: Optional[TextIO] = None, width: int = 30) -> None:
        self.stream = stream or sys.__stderr__
        self.width = width
        self.enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        self.current_stage: Optional[str] = None

    def __call__(self, stage: str, completed: int, total: int) -> None:
        if not self.enabled:
            return
        is_new_stage = stage != self.current_stage
        is_finished = completed == total
        if not is_new_stage and not is_finished:
            return
        self.current_stage = stage
        filled = self.width * completed // max(total, 1)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = 100 * completed // max(total, 1)
        label = "completed" if is_finished else stage
        self.stream.write(f"Pipeline [{bar}] {completed}/{total} {percent:3d}% | {label}\n")
        self.stream.flush()


# -------------------------------------------
# 命令行参数
# -------------------------------------------
def make_parser(command: Optional[str] = None) -> argparse.ArgumentParser:
    """返回子命令 ``command`` 的参数解析器；``None`` 表示默认的重建命令。"""
    if command == "doctor":
        return _make_doctor_parser()
    if command == "preview":
        return _make_preview_parser()
    return _make_reconstruct_parser()


def _make_doctor_parser() -> argparse.ArgumentParser:
    parser = LoguruArgumentParser(prog="scene-create doctor", description="Check sources, weights and modules for all five models in this interpreter without running inference")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--json", type=Path, dest="json_path")
    return parser


def _make_preview_parser() -> argparse.ArgumentParser:
    parser = LoguruArgumentParser(prog="scene-create preview", description="Render an assembled Gaussian scene with the MaskGaussian CUDA rasterizer")
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _make_reconstruct_parser() -> argparse.ArgumentParser:
    parser = LoguruArgumentParser(
        prog="scene-create",
        description="Object-centric scene reconstruction with SAM3, DA3, FLUX.2, TRELLIS and MaskGaussian. Runs every stage by default and reuses verified results already in --output.",
        epilog="Other commands: 'doctor' (environment check) and 'preview --scene scene.json' (Gaussian renders).",
    )
    stage_help = f"Target stages; their dependencies run or are reused automatically. One or more of: {', '.join(STAGE_NAMES)}. Default: all"
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--images", type=Path, help="Directory of overlapping views of the same static scene")
    source.add_argument("--manifest", type=Path, help="JSON input view manifest")
    source.add_argument("--observations", type=Path, help="Import externally computed masks and geometry; provenance is marked external")
    parser.add_argument("--stage", nargs="+", choices=STAGE_NAMES, dest="stages", metavar="STAGE", help=stage_help)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--strict", action="store_true", help="Stop immediately on any failed object")
    parser.add_argument("--background-mode", choices=["train", "init-only"])
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--seed", type=int)
    return parser


def apply_overrides(config: dict, arguments: argparse.Namespace) -> dict:
    """把命令行覆盖项写入 ``config``（原地修改）并返回它。"""
    if arguments.strict:
        config["runtime"]["continue_on_object_error"] = False
    if arguments.background_mode:
        config["background"]["mode"] = arguments.background_mode
    if arguments.categories:
        config["models"]["sam3"]["categories"] = arguments.categories
    if arguments.seed is not None:
        config["runtime"]["seed"] = arguments.seed
        for model in config["models"].values():
            model["seed"] = arguments.seed
    return config


# -------------------------------------------
# 结果输出
# -------------------------------------------
def emit_result(result: Any) -> None:
    """把机器可读的 CLI 结果写到标准输出；它不是诊断日志。"""
    payload = json.dumps(result, ensure_ascii=False, indent=2, default=json_value, allow_nan=False)
    sys.stdout.write(payload + "\n")
    sys.stdout.flush()


# -------------------------------------------
# 子命令
# -------------------------------------------
def doctor(config: dict, arguments: argparse.Namespace) -> int:
    """输出环境预检报告；未就绪时返回 2。"""
    with capture_external_output():
        result = runtime.diagnose(config)
    if arguments.json_path:
        atomic_json(arguments.json_path, result)
    emit_result(result)
    return EXIT_SUCCESS if result["ready"] else EXIT_FAILURE


def preview(config: dict, arguments: argparse.Namespace) -> int:
    """用 MaskGaussian 光栅器渲染已组装场景的各相机视角。

    Raises:
        FileNotFoundError: ``--scene`` 指向的文件不存在。
    """
    scene_path = arguments.scene.expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    destination = (arguments.output or scene_path.parent / "previews").resolve()
    request = {"mode": "render_scene", "scene_path": str(scene_path)}
    with runtime.output_lock(destination), capture_external_output():
        configure_logging(destination, "preview", stage="preview")
        with runtime.ModelRuntime(config) as models:
            result = models.call("maskgaussian", request, destination)
    emit_result(result)
    return EXIT_SUCCESS


def reconstruct(config: dict, arguments: argparse.Namespace) -> int:
    """运行重建流水线；部分失败或近似结果返回 3。"""
    source = {key: getattr(arguments, key) for key in SOURCE_KEYS if getattr(arguments, key) is not None}
    pipeline = ScenePipeline(apply_overrides(config, arguments), arguments.output)
    progress = _StageProgress()
    with capture_external_output():
        result = pipeline.run(source, arguments.stages, stage_callback=progress)
    emit_result(result)
    return EXIT_SUCCESS if result["status"] == "completed" else EXIT_PARTIAL


# -------------------------------------------
# 命令行入口
# -------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """解析参数、加载配置并执行子命令，返回进程退出码。"""
    configure_logging(stage="cli")
    try:
        arguments_list = list(sys.argv[1:] if argv is None else argv)
        command = arguments_list.pop(0) if arguments_list and arguments_list[0] in COMMANDS else None
        arguments = make_parser(command).parse_args(arguments_list)
        config = load_config(arguments.config)
        handler = {"doctor": doctor, "preview": preview}.get(command, reconstruct)
        return handler(config, arguments)
    except KeyboardInterrupt:
        logger.warning("Interrupted; rerun the same command to reuse verified completed stages.")
        return EXIT_INTERRUPTED
    except Exception as error:
        logger.exception("Command failed: {}", error)
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
