"""scene_create 入口：``python main.py --image photo.png`` 运行全部阶段。

``--stage`` 指定目标阶段（依赖自动补齐），``python main.py doctor`` 检查环境，
``python main.py preview`` 渲染已组装的 Gaussian 场景。五个模型都在本解释器中运行。
"""

import sys
from pathlib import Path

from loguru import logger

# 允许不安装本项目即可从仓库根目录直接运行。
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from scene_create.logging_utils import configure_logging  # noqa: E402

# -------------------------------------------
# 命令行入口
# -------------------------------------------
if __name__ == "__main__":
    configure_logging(stage="bootstrap")
    try:
        from scene_create.cli import main
    except ModuleNotFoundError as error:
        logger.exception("Missing dependency: {}. Use the project .venv or install this project's dependencies.", error.name)
        raise SystemExit(2) from error
    raise SystemExit(main())
