conda activate "$HOME/scene-envs/main"
cd /apdcephfs/private_xt/dev_workspace/scene_create

python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

from loguru import logger

base = (Path.cwd() / "third_parts").resolve()
manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
env = dict(os.environ, GIT_TERMINAL_PROMPT="0")


def run(*args):
    result = subprocess.run(
        args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode:
        logger.error("{}", result.stdout)
        raise RuntimeError(f"命令失败：{' '.join(args)}")
    return result.stdout.strip()


def fetch_repository(name, entry):
    target = (base / entry["directory"]).resolve()
    if target == base or not target.is_relative_to(base):
        raise ValueError(f"仓库目录超出 third_parts：{target}")

    url = entry["url"]
    commit = entry["commit"]
    if not url.startswith("https://"):
        raise ValueError(f"请使用 HTTPS 仓库地址：{url}")

    existing = (target / ".git").exists()
    if existing:
        dirty = run(
            "git", "-C", str(target),
            "status", "--porcelain", "--untracked-files=all",
        )
        if dirty:
            raise RuntimeError(f"{target} 有本地改动，请先处理，脚本不会覆盖")

        current = run("git", "-C", str(target), "rev-parse", "HEAD")
        if current == commit:
            logger.info("{} 已处于指定版本，跳过", name)
            return
    else:
        if target.exists() and any(target.iterdir()):
            raise RuntimeError(f"{target} 非空且不是独立 Git 仓库，不覆盖")
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("克隆 {} → {}", name, target)
        run(
            "git", "clone", "--depth", "1",
            "--filter=blob:none", "--no-checkout",
            url, str(target),
        )

    logger.info("{} 切换到 {}", name, commit[:12])
    run("git", "-C", str(target), "fetch", "--no-tags", "--depth", "1", url, commit)
    run("git", "-C", str(target), "checkout", "--detach", commit)


try:
    for section in ("repositories", "required_submodules"):
        for name, entry in manifest.get(section, {}).items():
            fetch_repository(name, entry)

    for entry in manifest.get("required_submodules", {}).values():
        target = (base / entry["directory"]).resolve()
        run(
            "git", "-C", str(target),
            "submodule", "update", "--init", "--recursive",
        )

    logger.success("清单中的第三方源码已拉取并固定版本")
except Exception:
    logger.exception("拉取失败；已有目录和本地改动不会被强制覆盖")
    raise SystemExit(1)
PY
