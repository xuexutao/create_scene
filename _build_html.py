"""生成 scene_create 算法流程与 output/view3 结果的交互式单文件 HTML 报告（临时生成脚本）。"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from plyfile import PlyData

ROOT = Path("/apdcephfs/private_xt/dev_workspace/scene_create")
OUT = ROOT / "output" / "view3"


def read_ply_vertex_count_fast(path: Path) -> int:
    """流式读取 PLY 文件头部获取顶点数量，不加载二进制数据块。"""
    if not path.is_file():
        return 0
    with path.open("rb") as stream:
        header_bytes = stream.read(2048)
    match = re.search(rb"element\s+vertex\s+(\d+)", header_bytes)
    return int(match.group(1)) if match else 0


def read_small_ply_points(path: Path, max_points: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """读取小型点云 PLY 并下采样，返回 (xyz, rgb)。"""
    if not path.is_file():
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    vertex = PlyData.read(str(path))["vertex"].data
    if len(vertex) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(vertex))
    if len(indices) > max_points:
        indices = np.sort(rng.choice(len(vertex), max_points, replace=False))
    xyz = np.column_stack([vertex["x"][indices], vertex["y"][indices], vertex["z"][indices]]).astype(np.float32)
    if "red" in vertex.dtype.names:
        rgb = np.column_stack([vertex["red"][indices], vertex["green"][indices], vertex["blue"][indices]]).astype(np.uint8)
    else:
        rgb = np.full((len(indices), 3), 180, dtype=np.uint8)
    return xyz, rgb


def image_to_webp_data_uri(image: Image.Image, max_side: int = 680, quality: int = 82) -> str:
    """将 PIL Image 缩放并编码为 WebP Base64 Data URI。"""
    w, h = image.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        image = image.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/webp;base64,{encoded}"


def path_to_webp_data_uri(path: Path, max_side: int = 680, quality: int = 82) -> str:
    """从路径读取图片并编码为 WebP Data URI。"""
    if not path.is_file():
        return ""
    with Image.open(path) as img:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        return image_to_webp_data_uri(img, max_side=max_side, quality=quality)


def colorize_scalar_map(values: np.ndarray, valid: np.ndarray, cmap: str = "turbo") -> Image.Image:
    """将二维标量场（深度或置信度）着色为直观的 RGB 图像。"""
    vals = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(vals)
    if mask.any():
        low = float(np.quantile(vals[mask], 0.02))
        high = float(np.quantile(vals[mask], 0.98))
    else:
        low, high = 0.0, 1.0
    norm = np.clip((vals - low) / max(high - low, 1e-6), 0.0, 1.0)
    if cmap == "depth":
        # 近处暖色（橙黄/红），远处冷色（青蓝/深紫）
        t = 1.0 - norm
        r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    else:
        # 置信度：低（暗紫蓝）-> 高（亮青绿黄）
        t = norm
        r = np.clip(t ** 1.5 * 0.95, 0.08, 1.0)
        g = np.clip(np.sin(t * np.pi * 0.85), 0.1, 1.0)
        b = np.clip(1.0 - 0.7 * (t ** 1.2), 0.15, 1.0)
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    rgb[~mask] = [18, 22, 32]
    return Image.fromarray(rgb)


PALETTE = [
    (56, 189, 248),   # sky
    (244, 114, 182),  # pink
    (52, 211, 153),   # emerald
    (251, 191, 36),   # amber
    (167, 139, 250),  # violet
    (248, 113, 113),  # red
    (45, 212, 191),   # teal
    (251, 146, 60),   # orange
    (129, 140, 248),  # indigo
    (163, 230, 53),   # lime
    (232, 121, 249),  # fuchsia
    (96, 165, 250),   # blue
    (74, 222, 128),   # green
    (250, 204, 21),   # yellow
    (244, 63, 94),    # rose
    (34, 211, 238),   # cyan
    (192, 132, 252),  # purple
    (253, 186, 116),  # light orange
    (110, 231, 183),  # light emerald
    (147, 197, 253),  # light blue
    (252, 165, 165),  # light red
]


def make_segmentation_overlay(base_image_path: Path, objects_meta: list[dict]) -> Image.Image:
    """把 21 个 SAM3 实例 mask 叠加到原图上并绘制边界与标签。"""
    with Image.open(base_image_path) as raw:
        base = raw.convert("RGB")
    canvas = np.asarray(base, dtype=np.float32) * 0.55
    height, width = canvas.shape[:2]
    boxes = []
    # 按像素面积从大到小画，避免大面积背景柜子遮住前景小植物
    ordered = sorted(enumerate(objects_meta), key=lambda pair: pair[1]["observations"][0]["pixel_count"], reverse=True)
    for idx, obj in ordered:
        color = np.array(PALETTE[idx % len(PALETTE)], dtype=np.float32)
        mask_path = Path(obj["reference_mask_path"])
        if not mask_path.is_file():
            continue
        with Image.open(mask_path) as m_img:
            mask = np.asarray(m_img.convert("L").resize((width, height), Image.Resampling.NEAREST)) > 127
        if not mask.any():
            continue
        canvas[mask] = canvas[mask] * 0.42 + color * 0.58
        # 1px 轮廓高亮
        edge = np.zeros_like(mask)
        edge[1:, :] |= mask[1:, :] != mask[:-1, :]
        edge[:, 1:] |= mask[:, 1:] != mask[:, :-1]
        canvas[edge] = np.clip(color * 1.25, 0, 255)
        ys, xs = np.where(mask)
        boxes.append((idx, obj["object_id"], obj["observations"][0]["score"], int(xs.min()), int(ys.min()), tuple(int(c) for c in color)))

    out_img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out_img)
    for _, oid, score, x0, y0, col in boxes:
        label = f"{oid} ({score:.2f})"
        tx = max(4, min(x0, width - 115))
        ty = max(4, min(y0, height - 18))
        draw.rectangle([tx, ty, tx + len(label) * 6 + 6, ty + 14], fill=(15, 23, 42))
        draw.rectangle([tx, ty, tx + len(label) * 6 + 6, ty + 14], outline=col, width=1)
        draw.text((tx + 3, ty + 1), label, fill=col)
    return out_img


def make_pixel_owner_overlay(pointmap_npz: Path, objects_meta: list[dict]) -> Image.Image:
    """可视化 Stage 4 在 504x336 几何网格上的最高置信度像素独占仲裁 (_assign_pixel_owners)。"""
    data = np.load(pointmap_npz)
    base = data["image"].astype(np.float32) * 0.45
    h, w = data["depth"].shape
    owner = np.full((h, w), -1, dtype=np.int32)
    by_score = sorted(range(len(objects_meta)), key=lambda i: objects_meta[i]["observations"][0]["score"], reverse=True)
    for idx in by_score:
        gmask_path = Path(objects_meta[idx]["observations"][0]["geometry_mask_path"])
        if not gmask_path.is_file():
            continue
        with Image.open(gmask_path) as gm:
            mask = np.asarray(gm.convert("L").resize((w, h), Image.Resampling.NEAREST)) > 127
        claim = mask & (owner < 0) & data["valid"]
        owner[claim] = idx
    for idx in range(len(objects_meta)):
        color = np.array(PALETTE[idx % len(PALETTE)], dtype=np.float32)
        region = owner == idx
        if region.any():
            base[region] = base[region] * 0.25 + color * 0.75
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def pack_points_quantized(xyz: np.ndarray, rgb: np.ndarray, inst_id: np.ndarray | None = None) -> dict:
    """将 3D 点云量化为紧凑数组以便内嵌至 HTML，支持前端零依赖 3D 交互渲染。"""
    if len(xyz) == 0:
        return {"n": 0, "xyz": [], "rgb": [], "inst": []}
    pts = np.round(xyz.astype(np.float64), 4).ravel().tolist()
    cols = rgb.astype(np.uint8).ravel().tolist()
    inst = inst_id.astype(np.int32).ravel().tolist() if inst_id is not None else []
    return {"n": int(len(xyz)), "xyz": pts, "rgb": cols, "inst": inst}


def collect_data() -> dict:
    """从 output/view3 提取构建 HTML 所需的全部真实数据、图像与下采样 3D 点。"""
    manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    views_meta = json.loads((OUT / "metadata" / "views.json").read_text(encoding="utf-8"))
    cameras_meta = json.loads((OUT / "metadata" / "cameras.json").read_text(encoding="utf-8"))
    objects_meta = json.loads((OUT / "metadata" / "objects.json").read_text(encoding="utf-8"))["objects"]
    scene_meta = json.loads((OUT / "scene" / "scene.json").read_text(encoding="utf-8"))
    quality_meta = json.loads((OUT / "reports" / "quality.json").read_text(encoding="utf-8"))
    bg_meta = json.loads((OUT / "background" / "metadata.json").read_text(encoding="utf-8"))
    config_resolved = json.loads((OUT / "metadata" / "config.resolved.json").read_text(encoding="utf-8"))

    # Stage 摘要
    stage_order = ["ingest", "segment", "depth", "objects", "complete", "assets", "register", "background", "assemble"]
    stages_info = []
    for sname in stage_order:
        st = manifest["stages"].get(sname, {})
        stages_info.append(
            {
                "name": sname,
                "status": st.get("status", "unknown"),
                "elapsed": round(float(st.get("elapsed_seconds", 0.0)), 2),
                "fingerprint": st.get("fingerprint", ""),
                "artifact_count": len(st.get("artifacts", [])),
                "directory": st.get("directory", ""),
            }
        )

    # 生成关键阶段图像 Data URI
    input_view_path = Path(views_meta["views"][0]["image_path"])
    processed_view_path = Path(cameras_meta["views"][0]["image_path"])
    pointmap_npz_path = OUT / "pointmaps" / "view_000.npz"
    pm = np.load(pointmap_npz_path)

    depth_img = colorize_scalar_map(pm["depth"], pm["valid"], cmap="depth")
    conf_img = colorize_scalar_map(pm["conf"], pm["valid"], cmap="conf")
    seg_overlay_img = make_segmentation_overlay(input_view_path, objects_meta)
    owner_overlay_img = make_pixel_owner_overlay(pointmap_npz_path, objects_meta)

    stage_images = {
        "input_rgb": path_to_webp_data_uri(input_view_path, max_side=680, quality=85),
        "seg_overlay": image_to_webp_data_uri(seg_overlay_img, max_side=680, quality=85),
        "processed_rgb": path_to_webp_data_uri(processed_view_path, max_side=504, quality=85),
        "depth_map": image_to_webp_data_uri(depth_img, max_side=504, quality=85),
        "conf_map": image_to_webp_data_uri(conf_img, max_side=504, quality=85),
        "owner_overlay": image_to_webp_data_uri(owner_overlay_img, max_side=504, quality=85),
        "bg_mask": path_to_webp_data_uri(OUT / "background" / "masks" / "view_000.png", max_side=504, quality=82),
        "bg_preview": path_to_webp_data_uri(OUT / "background" / "previews" / "view_000.png", max_side=504, quality=85),
        "scene_preview": path_to_webp_data_uri(OUT / "scene" / "previews" / "view_000.png", max_side=680, quality=86),
    }

    # 构建 DA3 全景下采样点云（带每个像素对应的实例 ID，用于 3D 切换 RGB / 实例分割着色）
    h, w = pm["depth"].shape
    owner_grid = np.full((h, w), -1, dtype=np.int32)
    by_score = sorted(range(len(objects_meta)), key=lambda i: objects_meta[i]["observations"][0]["score"], reverse=True)
    for idx in by_score:
        gmask_path = Path(objects_meta[idx]["observations"][0]["geometry_mask_path"])
        if gmask_path.is_file():
            with Image.open(gmask_path) as gm:
                m = np.asarray(gm.convert("L").resize((w, h), Image.Resampling.NEAREST)) > 127
            owner_grid[m & (owner_grid < 0)] = idx

    valid_mask = pm["valid"] & np.isfinite(pm["points_world"]).all(axis=-1)
    valid_pts = pm["points_world"][valid_mask]
    valid_rgb = pm["image"][valid_mask]
    valid_inst = owner_grid[valid_mask]
    rng = np.random.default_rng(42)
    sample_idx = np.sort(rng.choice(len(valid_pts), min(4500, len(valid_pts)), replace=False))
    da3_cloud_3d = pack_points_quantized(valid_pts[sample_idx], valid_rgb[sample_idx], valid_inst[sample_idx])

    # 背景点云 3D 下采样
    bg_xyz, bg_rgb = read_small_ply_points(OUT / "background" / "initial_points.ply", max_points=3500, seed=7)
    bg_cloud_3d = pack_points_quantized(bg_xyz, bg_rgb)

    # 21 个物体完整生命周期数据
    scene_obj_map = {item["object_id"]: item for item in scene_meta["objects"]}
    failure_map = {(f["object_id"], f["stage"]): f["error"] for f in scene_meta.get("failures", [])}
    objects_full = []
    object_3d_data = {}

    for idx, obj in enumerate(objects_meta):
        oid = obj["object_id"]
        obs = obj["observations"][0]
        sobj = scene_obj_map[oid]
        obj_dir = OUT / "objects" / oid
        color_rgb = list(PALETTE[idx % len(PALETTE)])

        # 补全信息
        comp_meta_path = obj_dir / "images" / "completion_meta.json"
        comp_meta = json.loads(comp_meta_path.read_text(encoding="utf-8")) if comp_meta_path.is_file() else None
        crop_meta_path = obj_dir / "images" / "crop_meta.json"
        crop_meta = json.loads(crop_meta_path.read_text(encoding="utf-8")) if crop_meta_path.is_file() else None

        # TRELLIS 资产信息
        asset_meta_path = obj_dir / "assets" / "asset_meta.json"
        asset_meta = json.loads(asset_meta_path.read_text(encoding="utf-8")) if asset_meta_path.is_file() else None
        gaussian_ply_path = obj_dir / "assets" / "gaussian.ply"
        gaussian_count = read_ply_vertex_count_fast(gaussian_ply_path)

        mesh_glb_path = obj_dir / "assets" / "mesh.glb"
        mesh_verts, mesh_faces = 0, 0
        if mesh_glb_path.is_file():
            tscene = trimesh.load(str(mesh_glb_path), force="scene", process=False)
            for geom in tscene.geometry.values():
                if hasattr(geom, "vertices") and hasattr(geom, "faces"):
                    mesh_verts += len(geom.vertices)
                    mesh_faces += len(geom.faces)

        # 配准信息
        transform_path = obj_dir / "registration" / "transform.json"
        transform_data = json.loads(transform_path.read_text(encoding="utf-8")) if transform_path.is_file() else None

        # 3D 点云：观测点云 (fused.ply) 与变换后的网格采样点 (mesh_samples.ply)
        obs_xyz, obs_rgb = read_small_ply_points(obj_dir / "clouds" / "fused.ply", max_points=500, seed=idx + 10)
        mesh_xyz_world = np.zeros((0, 3), dtype=np.float32)
        if transform_data is not None and (obj_dir / "registration" / "mesh_samples.ply").is_file():
            raw_mesh_xyz, _ = read_small_ply_points(obj_dir / "registration" / "mesh_samples.ply", max_points=550, seed=idx + 100)
            mat = np.asarray(transform_data["T_scene_from_asset"], dtype=np.float64)
            mesh_xyz_world = (raw_mesh_xyz @ mat[:3, :3].T + mat[:3, 3]).astype(np.float32)

        object_3d_data[oid] = {
            "obs": pack_points_quantized(obs_xyz, obs_rgb),
            "mesh": pack_points_quantized(mesh_xyz_world, np.tile(np.array(color_rgb, dtype=np.uint8), (len(mesh_xyz_world), 1))),
            "color": color_rgb,
        }

        # 各阶段缩略图 Data URI
        imgs = {
            "mask": path_to_webp_data_uri(Path(obj["reference_mask_path"]), max_side=240, quality=78),
            "geom_mask": path_to_webp_data_uri(Path(obs["geometry_mask_path"]), max_side=240, quality=78),
            "crop_context": path_to_webp_data_uri(obj_dir / "images" / "crop_context.png", max_side=220, quality=80),
            "crop": path_to_webp_data_uri(obj_dir / "images" / "crop.png", max_side=220, quality=80),
            "visible_mask": path_to_webp_data_uri(obj_dir / "images" / "visible_mask.png", max_side=220, quality=78),
            "completed": path_to_webp_data_uri(obj_dir / "images" / "completed.png", max_side=240, quality=82),
            "trellis_rgba": path_to_webp_data_uri(obj_dir / "assets" / "completed_rgba.png", max_side=220, quality=80),
            "trellis_input": path_to_webp_data_uri(obj_dir / "assets" / "trellis_input.png", max_side=220, quality=80),
            "alignment": path_to_webp_data_uri(obj_dir / "registration" / "view_000_alignment.png", max_side=320, quality=84),
        }

        fail_reason = None
        for st_name in ("complete", "assets", "register"):
            if (oid, st_name) in failure_map:
                fail_reason = {"stage": st_name, "error": failure_map[(oid, st_name)]}

        objects_full.append(
            {
                "index": idx,
                "object_id": oid,
                "category": obj["category"],
                "local_id": obs["local_id"],
                "score": round(float(obs["score"]), 4),
                "pixel_count": int(obs["pixel_count"]),
                "view_point_count": int(obs["point_count"]),
                "fused_point_count": int(obj["point_count"]),
                "voxel_size": round(float(obj["voxel_size"]), 6),
                "stage4_status": obj["status"],
                "final_status": sobj["status"],
                "interactive": bool(sobj["interactive"]),
                "preserved_in_background": bool(sobj["preserved_in_background"]),
                "color": color_rgb,
                "has_completion": comp_meta is not None,
                "prompt": comp_meta["metadata"]["prompt"] if comp_meta else None,
                "bbox_xyxy": crop_meta["bbox_xyxy"] if crop_meta else None,
                "has_asset": asset_meta is not None,
                "gaussian_count": gaussian_count,
                "mesh_vertices": mesh_verts,
                "mesh_faces": mesh_faces,
                "registration": {
                    "accepted": bool(transform_data["accepted"]),
                    "scale": round(float(transform_data["scale"]), 4),
                    "translation_xyz": [round(float(x), 4) for x in transform_data["translation_xyz"]],
                    "rotation_xyzw": [round(float(x), 4) for x in transform_data["rotation_quaternion_xyzw"]],
                    "matrix": [[round(float(v), 4) for v in row] for row in transform_data["T_scene_from_asset"]],
                    "metrics": {
                        "rmse": round(float(transform_data["metrics"]["rmse"]), 5),
                        "rmse_ratio": round(float(transform_data["metrics"]["rmse_ratio"]), 4),
                        "fitness": round(float(transform_data["metrics"]["fitness"]), 4),
                        "source_support": round(float(transform_data["metrics"]["source_support_fraction"]), 4),
                        "target_size": round(float(transform_data["metrics"]["target_size"]), 4),
                        "mask_iou": round(float(transform_data["metrics"]["mean_visible_mask_iou"]), 4),
                        "depth_error": round(float(transform_data["metrics"]["median_relative_depth_error"]), 4),
                        "observed_pixels": int(transform_data["metrics"]["per_view"][0]["observed_pixels"]),
                        "overlap_pixels": int(transform_data["metrics"]["per_view"][0]["overlap_pixels"]),
                        "reprojection_accepted": bool(transform_data["metrics"]["reprojection_accepted"]),
                    },
                }
                if transform_data
                else None,
                "failure": fail_reason,
                "images": imgs,
            }
        )

    return {
        "run_id": manifest.get("run_id"),
        "last_run_id": manifest.get("last_run_id"),
        "pipeline_status": manifest.get("status"),
        "stages": stages_info,
        "views": views_meta["views"],
        "camera": cameras_meta["views"][0],
        "da3_provenance": cameras_meta["provenance"],
        "background": bg_meta,
        "scene": {
            "world_frame": scene_meta["world_frame"],
            "scale_status": scene_meta["scale_status"],
            "gaussian_ranges": scene_meta["gaussian_ranges"],
            "total_gaussians": sum(r["count"] for r in scene_meta["gaussian_ranges"]),
            "quality": scene_meta["quality"],
        },
        "config": {
            "sam3_categories": config_resolved["models"]["sam3"]["categories"],
            "sam3_threshold": config_resolved["models"]["sam3"]["score_threshold"],
            "da3_res": config_resolved["models"]["da3"]["process_res"],
            "geometry": config_resolved["geometry"],
            "fusion": config_resolved["fusion"],
            "completion": config_resolved["completion"],
            "registration": config_resolved["registration"],
            "background": config_resolved["background"],
        },
        "stage_images": stage_images,
        "objects": objects_full,
        "clouds_3d": {
            "da3": da3_cloud_3d,
            "background": bg_cloud_3d,
            "objects": object_3d_data,
        },
    }


if __name__ == "__main__":
    payload = collect_data()
    out_json = OUT / "_temp_payload.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Payload extracted: {out_json.stat().st_size / 1024:.1f} KB")
