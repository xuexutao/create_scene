from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

from scene_create.logging_utils import (
    LoguruArgumentParser,
    capture_external_output,
    configure_logging,
)


def make_parser():
    from scene_create.config import ROOT, STAGES

    parser = LoguruArgumentParser(
        prog="scene-create",
        description="Object-centric reconstruction with SAM3, DA3, FLUX.2, TRELLIS and MaskGaussian",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    doctor = subcommands.add_parser(
        "doctor",
        help="Check model environments, repositories and weights without running inference",
    )
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--json", type=Path, dest="json_path")
    preview = subcommands.add_parser(
        "preview",
        help="Render an assembled Gaussian scene with the MaskGaussian CUDA worker",
    )
    preview.add_argument("--scene", required=True, type=Path)
    preview.add_argument("--config", type=Path)
    preview.add_argument("--output", type=Path)
    run = subcommands.add_parser(
        "run", help="Reconstruct a scene or execute through one stage"
    )
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument(
        "--images",
        type=Path,
        help="Directory of overlapping views of the same static scene",
    )
    source.add_argument("--manifest", type=Path, help="JSON input view manifest")
    source.add_argument(
        "--observations",
        type=Path,
        help="Import externally computed masks and geometry; provenance is marked external",
    )
    run.add_argument("--config", type=Path)
    run.add_argument("--output", type=Path, default=ROOT / "output")
    run.add_argument("--through", choices=STAGES, default="assemble")
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--strict", action="store_true", help="Stop immediately on any failed object"
    )
    run.add_argument("--background-mode", choices=["train", "init-only"])
    run.add_argument("--categories", nargs="+")
    run.add_argument("--seed", type=int)
    return parser


def emit_result(result):
    """Write the machine-readable CLI result, not a diagnostic log."""
    from scene_create.io import json_value

    payload = json.dumps(
        result, ensure_ascii=False, indent=2, default=json_value, allow_nan=False
    )
    sys.stdout.write(payload + "\n")
    sys.stdout.flush()


def main(argv=None):
    configure_logging(stage="cli")
    try:
        arguments = make_parser().parse_args(argv)
        from scene_create.config import load_config
        from scene_create.io import atomic_json

        config = load_config(arguments.config)
        if arguments.command == "doctor":
            from scene_create.doctor import diagnose

            with capture_external_output():
                result = diagnose(config)
            if arguments.json_path:
                atomic_json(arguments.json_path, result)
            emit_result(result)
            return 0 if result["ready"] else 2
        if arguments.command == "preview":
            from scene_create.runtime import WorkerRunner, output_lock

            scene_path = arguments.scene.expanduser().resolve()
            if not scene_path.is_file():
                raise FileNotFoundError(scene_path)
            destination = (arguments.output or scene_path.parent / "previews").resolve()
            with output_lock(destination):
                configure_logging(destination, "preview")
                result = WorkerRunner(config, destination, run_id="preview").call(
                    "maskgaussian",
                    {"mode": "render_scene", "scene_path": str(scene_path)},
                    destination,
                )
            emit_result(result)
            return 0
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
        from scene_create.pipeline import ScenePipeline

        source = {
            key: getattr(arguments, key)
            for key in ("image", "images", "manifest", "observations")
            if getattr(arguments, key) is not None
        }
        with capture_external_output():
            result = ScenePipeline(config, arguments.output).run(
                source, arguments.through, arguments.resume
            )
        emit_result(result)
        return 0 if result["status"] == "completed" else 3
    except KeyboardInterrupt:
        logger.warning("Interrupted; completed stages can be resumed with --resume.")
        return 130
    except Exception as error:
        logger.exception("Command failed: {}", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
