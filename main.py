from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from loguru import logger
    from scene_create.logging_utils import configure_logging

    configure_logging(stage="bootstrap")
    try:
        from scene_create.cli import main
    except ModuleNotFoundError as error:
        logger.exception(
            "Missing dependency: {}. Use the project .venv or install this project's dependencies.",
            error.name,
        )
        raise SystemExit(2)
    raise SystemExit(main())
