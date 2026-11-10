"""生成场景查看页，并可调用项目预览命令渲染 Gaussian 场景。"""

import argparse
import html
import json
import subprocess
import sys
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional, Sequence

# -------------------------------------------
# 配置常量
# -------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
REPORT_NAME = "viewer.html"


# -------------------------------------------
# 场景读取
# -------------------------------------------
def resolve_scene_path(path: Path) -> Path:
    """将输出目录、scene 目录或 scene.json 解析为场景清单路径。"""
    path = path.expanduser().resolve()
    candidates = [path, path / "scene.json", path / "scene" / "scene.json"]
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "scene.json":
            return candidate
    raise FileNotFoundError(f"Cannot find scene.json from: {path}")


def load_scene(scene_path: Path) -> dict[str, Any]:
    """读取并校验场景清单。"""
    with scene_path.open(encoding="utf-8") as stream:
        scene = json.load(stream)
    if not isinstance(scene.get("objects"), list):
        raise ValueError(f"Invalid scene manifest: {scene_path}")
    return scene


# -------------------------------------------
# 查看页生成
# -------------------------------------------
def write_report(scene_path: Path) -> Path:
    """生成包含 Gaussian 预览、交互 mesh 和对象状态的 HTML 查看页。"""
    scene = load_scene(scene_path)
    scene_dir = scene_path.parent
    objects = scene["objects"]
    registered = [item for item in objects if item.get("status") == "registered"]
    previews = sorted((scene_dir / "previews").glob("*.png"))
    report_path = scene_dir / REPORT_NAME
    report_path.write_text(
        _report_html(scene, registered, previews, (scene_dir / "objects.glb").is_file()),
        encoding="utf-8",
    )
    return report_path


def _report_html(
    scene: dict[str, Any],
    registered: list[dict[str, Any]],
    previews: list[Path],
    has_glb: bool,
) -> str:
    object_rows = "\n".join(_object_row(item) for item in scene["objects"])
    preview_cards = "\n".join(
        f'<figure><img src="previews/{html.escape(path.name)}" alt="{html.escape(path.stem)}"><figcaption>{html.escape(path.stem)}</figcaption></figure>'
        for path in previews
    )
    mesh_viewer = ""
    if has_glb:
        mesh_viewer = """
        <section>
          <h2>交互式物体 Mesh</h2>
          <p>该视图只包含成功配准的物体 mesh，不包含 Gaussian 背景。组件脚本从 CDN 加载。</p>
          <model-viewer src="objects.glb" camera-controls auto-rotate shadow-intensity="1"></model-viewer>
        </section>
        """
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>scene_create viewer</title>
  <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
  <style>
    body {{ margin: 0 auto; max-width: 1200px; padding: 24px; color: #e8e8e8; background: #15171a; font: 15px/1.5 sans-serif; }}
    h1, h2 {{ color: #fff; }}
    .summary {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .metric {{ padding: 12px 18px; border-radius: 8px; background: #25292e; }}
    .previews {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; padding: 10px; border-radius: 8px; background: #25292e; }}
    img {{ width: 100%; display: block; }}
    model-viewer {{ width: 100%; height: 70vh; background: #25292e; }}
    table {{ width: 100%; border-collapse: collapse; background: #202328; }}
    th, td {{ padding: 8px; border: 1px solid #3a3f46; text-align: left; }}
    .ok {{ color: #70d68c; }}
    .failed {{ color: #ff9c86; }}
  </style>
</head>
<body>
  <h1>scene_create 场景查看</h1>
  <div class="summary">
    <div class="metric">尺度：{html.escape(str(scene.get("scale_status", "unknown")))}</div>
    <div class="metric">注册物体：{len(registered)} / {len(scene["objects"])}</div>
    <div class="metric">Gaussian：{html.escape(str(scene.get("gaussian_scene", "missing")))}</div>
    <div class="metric">背景：{html.escape(str(scene.get("background", {}).get("status", "unknown")))}</div>
  </div>
  <section>
    <h2>Gaussian 场景相机预览</h2>
    <div class="previews">{preview_cards or "<p>尚未生成预览，请使用 --render。</p>"}</div>
  </section>
  {mesh_viewer}
  <section>
    <h2>对象状态</h2>
    <table>
      <thead><tr><th>对象</th><th>类别</th><th>状态</th><th>可交互</th></tr></thead>
      <tbody>{object_rows}</tbody>
    </table>
  </section>
</body>
</html>
"""


def _object_row(item: dict[str, Any]) -> str:
    status = str(item.get("status", "unknown"))
    status_class = "ok" if status == "registered" else "failed"
    return (
        "<tr>"
        f"<td>{html.escape(str(item.get('object_id', '-')))}</td>"
        f"<td>{html.escape(str(item.get('category', '-')))}</td>"
        f'<td class="{status_class}">{html.escape(status)}</td>'
        f"<td>{'yes' if item.get('interactive') else 'no'}</td>"
        "</tr>"
    )


# -------------------------------------------
# 渲染与服务
# -------------------------------------------
def render_previews(scene_path: Path, config_path: Path) -> int:
    """调用项目 preview 子命令生成 Gaussian 相机预览。"""
    command = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "preview",
        "--scene",
        str(scene_path),
        "--config",
        str(config_path.expanduser().resolve()),
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


class QuietRequestHandler(SimpleHTTPRequestHandler):
    """仅输出 HTTP 错误，避免普通请求刷屏。"""

    def log_message(self, format: str, *args: Any) -> None:
        if str(args[1]).startswith(("4", "5")):
            super().log_message(format, *args)


def serve_report(report_path: Path, host: str, port: int, should_open: bool) -> int:
    """启动本地 HTTP 服务直到用户中断。"""
    handler = partial(QuietRequestHandler, directory=str(report_path.parent))
    server = ThreadingHTTPServer((host, port), handler)
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{display_host}:{server.server_port}/{report_path.name}"
    print(f"Scene viewer: {url}")
    if should_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
    return 0


# -------------------------------------------
# 命令行入口
# -------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """生成并可选托管场景查看页。"""
    parser = argparse.ArgumentParser(description="View scene_create Gaussian previews and registered object meshes")
    parser.add_argument("scene", type=Path, help="Output directory, scene directory, or scene.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--render", action="store_true", help="Render Gaussian camera previews before creating the page")
    parser.add_argument("--no-serve", action="store_true", help="Only create viewer.html")
    parser.add_argument("--open", action="store_true", dest="should_open", help="Open the viewer in the default browser")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    scene_path = resolve_scene_path(args.scene)
    if args.render:
        code = render_previews(scene_path, args.config)
        if code not in (0, 3):
            return code
    report_path = write_report(scene_path)
    print(f"Viewer page written to: {report_path}")
    if args.no_serve:
        return 0
    return serve_report(report_path, args.host, args.port, args.should_open)


if __name__ == "__main__":
    raise SystemExit(main())
