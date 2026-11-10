"""观测证据：输入解析、实例 mask、DA3 几何与物体点云。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from PIL import Image, ImageOps

from scene_create.geometry import (
    clean_cloud,
    cloud_overlap,
    depth_to_points,
    extent,
    map_mask,
    project_points,
    validate_transform,
    voxel_fuse,
)
from scene_create.io import (
    atomic_json,
    atomic_npz,
    file_digest,
    load_mask,
    load_npz,
    read_json,
    safe_name,
    save_image,
    validate_id,
    write_cloud,
)

if TYPE_CHECKING:
    from scene_create.runtime import StageContext

# -------------------------------------------
# 配置常量
# -------------------------------------------
# fmt: off
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}  # 目录输入时识别为图片的小写后缀
CALIBRATION_KEYS = ("intrinsics", "extrinsics", "distortion")                   # 普通图片清单暂不支持的外部标定字段
EXIF_ORIENTATION_TAG = 274                                                      # EXIF Orientation 标签编号
MIN_IMAGE_SIDE_PIXELS = 8                                                       # 输入图片短边的最小像素数
MASK_FOREGROUND_VALUE = 255                                                     # 保存 mask PNG 时的前景灰度值

PIXEL_MAPPING_MIN_DETERMINANT = 1e-12  # 原图到几何网格映射矩阵允许的最小行列式绝对值
VISIBILITY_DEPTH_TOLERANCE = 0.05      # 投影点被视为可见时允许超出观测深度的相对比例
DEPTH_EPSILON = 1e-6                   # 相对深度容差的下限，避免零深度
MIN_POINT_WEIGHT = 1e-6                # 融合权重下限，零置信度点仍保留极小权重
MIN_SUPPORT_POINTS = 3                 # 计算投影支持率所需的最少可见点数
MIN_CLOUD_POINTS = 3                   # 离群点清洗与跨视图比较所需的最少点数

RGB_DIAGONAL_DISTANCE = 255 * np.sqrt(3)  # RGB 空间对角线长度，用于把平均颜色距离归一化到 [0, 1]
MAX_CENTER_DISTANCE_RATIO = 1.5           # 两实例中位点距离超过物体尺寸该倍数时不再比较
OVERLAP_SAMPLE_POINTS = 2000              # 计算点云重叠前每个实例的目标采样点数
DEFAULT_MIN_PROJECTED_SUPPORT = 0.5       # fusion.min_projected_support 未配置时的默认值
COLOR_SCORE_PENALTY = 0.2                 # 颜色距离对关联得分的惩罚系数
# fmt: on

REQUIRED_GEOMETRY_FIELDS = frozenset({"depth", "conf", "intrinsics", "world_from_camera", "image", "original_to_processed"})


# -------------------------------------------
# 输入解析
# -------------------------------------------
def resolve_input(source: Mapping[str, Union[str, Path]]) -> dict:
    """解析单图、图片目录、视图清单或外部观测清单，并记录全部依赖文件的摘要。

    Args:
        source: 只含一个键的映射，键为 ``image``、``images``、``manifest`` 或 ``observations``。

    Returns:
        包含 ``views``、``provenance`` 和 ``dependencies`` 的输入描述。

    Raises:
        ValueError: 输入结构、标识或导入字段不合法。
        FileNotFoundError: 引用的图片不存在。
    """
    kind, path = next(iter(source.items()))
    path = Path(path).expanduser().resolve()
    is_imported = kind == "observations"
    records, base, metadata = _load_input_records(kind, path)
    if not records:
        raise ValueError("No input images were found")

    dependencies = []
    if kind in ("manifest", "observations"):
        dependencies.append(_file_record(path))

    views, seen = [], set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("Every view must be a mapping")
        view_id = validate_id(record.get("view_id", f"view_{index:03d}"))
        if view_id in seen:
            raise ValueError(f"Duplicate view_id: {view_id}")
        seen.add(view_id)

        view, files = _resolve_view(view_id, record, base)
        if is_imported:
            imported, imported_files = _resolve_imported_view(view_id, record, base, metadata)
            view.update(imported)
            files.extend(imported_files)
        views.append(view)
        dependencies.extend(files)

    return {
        "views": views,
        "provenance": "external_observations" if is_imported else "input_images",
        "dependencies": dependencies,
    }


def _load_input_records(kind: str, path: Path) -> tuple[list, Path, dict]:
    """返回视图记录、相对路径基准目录和清单顶层元数据。"""
    if kind == "image":
        return [{"image_path": str(path)}], path.parent, {}

    if kind == "images":
        if not path.is_dir():
            raise ValueError(f"Image directory does not exist: {path}")
        records = []
        for item in sorted(path.iterdir()):
            is_supported_image = item.suffix.lower() in IMAGE_EXTENSIONS and item.is_file()
            if is_supported_image:
                records.append({"image_path": str(item)})
        return records, path, {}

    metadata = read_json(path)
    has_views = isinstance(metadata, dict) and isinstance(metadata.get("views"), list)
    if not has_views:
        raise ValueError("Input manifest must contain a views list")
    return metadata["views"], path.parent, metadata


def _file_record(path: Path) -> dict:
    return {"path": str(path), "sha256": file_digest(path)}


def _resolve_view(view_id: str, record: Mapping, base: Path) -> tuple[dict, list[dict]]:
    """解析单个视图的原图路径，返回视图描述和依赖文件记录。"""
    # 绝对路径与基准目录拼接后仍是其自身，因此相对和绝对路径可以统一处理。
    image = (base / record["image_path"]).resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    image_record = _file_record(image)

    has_calibration = any(key in record for key in CALIBRATION_KEYS)
    if has_calibration:
        raise ValueError("External camera calibration must currently be supplied through --observations geometry NPZ; calibrated image-only input is not yet supported")

    view = {"view_id": view_id, "source_path": str(image), "source_sha256": image_record["sha256"]}
    return view, [image_record]


def _resolve_imported_view(view_id: str, record: Mapping, base: Path, metadata: Mapping) -> tuple[dict, list[dict]]:
    """解析外部观测视图的几何 NPZ、尺度来源和实例 mask。"""
    has_required_fields = "geometry_path" in record and "masks" in record
    if not has_required_fields:
        raise ValueError("Imported views require geometry_path and masks (an empty mask list is valid)")

    geometry = (base / record["geometry_path"]).resolve()
    files = [_file_record(geometry)]
    masks = []
    for mask_index, item in enumerate(record["masks"]):
        entry = _resolve_imported_mask(mask_index, item, base)
        masks.append(entry)
        files.append(_file_record(Path(entry["mask_path"])))

    local_ids = {mask["local_id"] for mask in masks}
    if len(local_ids) != len(masks):
        raise ValueError(f"Duplicate local instance IDs in {view_id}")

    imported = {
        "imported_geometry": str(geometry),
        "scale_status": record.get("scale_status", metadata.get("scale_status", "relative")),
        "scale_source": record.get("scale_source", metadata.get("scale_source", "external_unspecified")),
        "imported_masks": masks,
    }
    return imported, files


def _resolve_imported_mask(mask_index: int, item: Mapping, base: Path) -> dict:
    mask_path = (base / item["mask_path"]).resolve()
    local_id = validate_id(item.get("local_id", f"instance_{mask_index:03d}"))
    entry = {**item, "mask_path": str(mask_path), "local_id": local_id}

    category = entry.get("category")
    has_category = isinstance(category, str) and bool(category.strip())
    if not has_category:
        raise ValueError("Each imported mask requires a nonempty category")
    if entry.get("object_id"):
        validate_id(entry["object_id"])
    return entry


# -------------------------------------------
# 输入登记
# -------------------------------------------
def ingest(context: StageContext) -> tuple[dict, list]:
    """按 EXIF 方向校正并登记每张输入图片，写出 ``metadata/views.json``。"""
    inputs = context.inputs
    is_imported = inputs["provenance"] == "external_observations"
    views, artifacts = [], []
    for entry in inputs["views"]:
        view, target = _ingest_view(entry, context.directory, is_imported)
        views.append(view)
        artifacts.append(target)

    result = {"views": views, "provenance": inputs["provenance"]}
    metadata = context.output / "metadata" / "views.json"
    atomic_json(metadata, result)
    return result, artifacts + [metadata]


def _ingest_view(entry: Mapping, directory: Path, is_imported: bool) -> tuple[dict, Path]:
    with Image.open(entry["source_path"]) as original:
        orientation = original.getexif().get(EXIF_ORIENTATION_TAG, 1)
        # 外部几何和 mask 对应未旋转的像素网格，旋转原图会使它们错位。
        is_rotated = orientation not in (None, 1)
        if is_imported and is_rotated:
            raise ValueError("Imported geometry requires already EXIF-normalized images and masks")

        image = ImageOps.exif_transpose(original).convert("RGB")
        if min(image.size) < MIN_IMAGE_SIDE_PIXELS:
            raise ValueError(f"Image is too small: {entry['source_path']}")

        target = directory / f"{entry['view_id']}.png"
        save_image(target, image)
        view = {
            **entry,
            "image_path": str(target),
            "width": image.width,
            "height": image.height,
            "exif_orientation": orientation,
        }
    return view, target


# -------------------------------------------
# 实例分割
# -------------------------------------------
def segment(context: StageContext) -> tuple[dict, list]:
    """运行 SAM3 或导入外部 mask，并校验每个实例。"""
    ingested = context.results["ingest"]
    if context.imported:
        result = _import_instances(ingested["views"], context.directory)
    else:
        result = context.models.call("sam3", {"views": ingested["views"]}, context.directory)
    artifacts = _validate_instances(result["instances"], ingested["views"])
    return result, artifacts


def _mask_to_png(mask: np.ndarray) -> np.ndarray:
    return (mask * MASK_FOREGROUND_VALUE).astype(np.uint8)


def _import_instances(views: Sequence[dict], directory: Path) -> dict:
    instances = []
    for view in views:
        for item in view["imported_masks"]:
            instances.append(_import_instance(view, item, directory))
    return {"instances": instances, "provenance": "external_observations"}


def _import_instance(view: Mapping, item: Mapping, directory: Path) -> dict:
    mask = load_mask(item["mask_path"], (view["width"], view["height"]))
    if not mask.any():
        raise ValueError(f"Empty imported mask: {item['mask_path']}")

    target = directory / view["view_id"] / f"{item['local_id']}.png"
    save_image(target, _mask_to_png(mask))
    ys, xs = np.where(mask)
    return {
        **item,
        "view_id": view["view_id"],
        "mask_path": str(target),
        "score": float(item.get("score", 1.0)),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
    }


def _validate_instances(instances: Sequence[dict], views: Sequence[dict]) -> list[str]:
    """校验实例标识唯一、置信度合法且 mask 非空，返回 mask 路径。"""
    views_by_id = {view["view_id"]: view for view in views}
    seen, artifacts = set(), []
    for item in instances:
        key = (item["view_id"], validate_id(item["local_id"]))
        is_known_view = key[0] in views_by_id
        if key in seen or not is_known_view:
            raise ValueError(f"Duplicate or unknown segmentation instance: {key}")
        seen.add(key)

        score = float(item["score"])
        is_valid_score = np.isfinite(score) and 0 <= score <= 1
        if not is_valid_score:
            raise ValueError("Instance confidence must be in [0, 1]")

        view = views_by_id[key[0]]
        if not load_mask(item["mask_path"], (view["width"], view["height"])).any():
            raise ValueError("Segmentation contains an empty mask")
        artifacts.append(item["mask_path"])
    return artifacts


# -------------------------------------------
# 几何估计
# -------------------------------------------
@dataclass(frozen=True)
class _ViewGeometry:
    view: dict
    prediction: dict
    data: dict


def validate_geometry(data: Mapping[str, np.ndarray], original_size: tuple[int, int]) -> dict[str, np.ndarray]:
    """校验几何 NPZ 契约并返回规范化数组。

    Args:
        data: 含深度、置信度、内参、位姿、RGB 和像素映射的数组。
        original_size: 原图 ``(宽, 高)``；当前仅保留在接口中，不参与校验。

    Returns:
        规范化后的几何数组，``valid`` 已排除非有限、非正深度和负置信度像素。

    Raises:
        ValueError: 字段缺失、网格不一致或相机参数非法。
    """
    missing = REQUIRED_GEOMETRY_FIELDS - set(data)
    if missing:
        raise ValueError(f"Geometry NPZ is missing fields: {sorted(missing)}")

    depth = np.asarray(data["depth"], dtype=np.float32)
    confidence = np.asarray(data["conf"], dtype=np.float32)
    image = np.asarray(data["image"])
    is_shared_grid = depth.ndim == 2 and depth.shape == confidence.shape and image.shape == (*depth.shape, 3)
    if not is_shared_grid:
        raise ValueError("Depth, confidence and processed RGB must share a pixel grid")
    if image.dtype != np.uint8:
        raise ValueError("Processed RGB must be uint8")

    intrinsics = _validated_intrinsics(data["intrinsics"])
    matrix = np.asarray(data["world_from_camera"], dtype=np.float64)
    validate_transform(matrix)
    mapping = _validated_pixel_mapping(data["original_to_processed"])
    valid = _valid_depth_pixels(data, depth, confidence)
    return dict(
        depth=depth,
        conf=confidence,
        intrinsics=intrinsics,
        world_from_camera=matrix,
        image=image,
        original_to_processed=mapping,
        valid=valid,
    )


def _validated_intrinsics(values: np.ndarray) -> np.ndarray:
    intrinsics = np.asarray(values, dtype=np.float64)
    is_valid = intrinsics.shape == (3, 3) and np.isfinite(intrinsics).all() and min(intrinsics[0, 0], intrinsics[1, 1]) > 0 and np.allclose(intrinsics[2], [0, 0, 1])
    if not is_valid:
        raise ValueError("Invalid camera intrinsics")
    return intrinsics


def _validated_pixel_mapping(values: np.ndarray) -> np.ndarray:
    mapping = np.asarray(values, dtype=np.float64)
    is_valid = mapping.shape == (3, 3) and np.isfinite(mapping).all() and abs(np.linalg.det(mapping)) >= PIXEL_MAPPING_MIN_DETERMINANT
    if not is_valid:
        raise ValueError("Invalid original-to-processed pixel mapping")
    return mapping


def _valid_depth_pixels(data: Mapping[str, np.ndarray], depth: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """合并输入的有效区与有限正深度、非负置信度条件。"""
    valid = np.asarray(data.get("valid", np.ones(depth.shape, dtype=bool)), dtype=bool)
    if valid.shape != depth.shape:
        raise ValueError("Invalid depth validity mask")

    is_measured = np.isfinite(depth) & (depth > 0)
    is_confident = np.isfinite(confidence) & (confidence >= 0)
    valid = valid & is_measured & is_confident
    if not valid.any():
        raise ValueError("Geometry has no valid depth pixels")
    return valid


def estimate_depth(context: StageContext) -> tuple[dict, list]:
    """运行 DA3 或导入外部几何，统一到第一输入相机坐标系并写出 pointmap。"""
    ingested = context.results["ingest"]
    predicted = _predict_geometry(context)
    geometries = _load_predicted_geometry(ingested["views"], predicted)

    scene_from_world = np.linalg.inv(geometries[0].data["world_from_camera"])
    statuses = {geometry.prediction.get("scale_status", "relative") for geometry in geometries}
    if len(statuses) != 1:
        raise ValueError("Mixed scale conventions across views are not allowed")

    min_confidence = context.config["geometry"]["min_confidence"]
    records, artifacts = [], []
    for geometry in geometries:
        geometry.data["world_from_camera"] = scene_from_world @ geometry.data["world_from_camera"]
        geometry.data["valid"] &= geometry.data["conf"] >= min_confidence
        record, files = _write_view_geometry(context, geometry)
        records.append(record)
        artifacts.extend(files)

    result = {
        "views": records,
        "provenance": predicted.get("provenance"),
        "world_frame": "first_input_camera_opencv",
        "scale_status": next(iter(statuses)),
    }
    cameras = context.output / "metadata" / "cameras.json"
    atomic_json(cameras, result)
    return result, artifacts + [cameras]


def _predict_geometry(context: StageContext) -> dict:
    views = context.results["ingest"]["views"]
    if not context.imported:
        return context.models.call("da3", {"views": views}, context.directory)

    records = []
    for view in views:
        records.append(
            {
                "view_id": view["view_id"],
                "geometry_path": view["imported_geometry"],
                "scale_status": view["scale_status"],
                "scale_source": view["scale_source"],
            }
        )
    return {"views": records, "provenance": "external_observations"}


def _load_predicted_geometry(views: Sequence[dict], predicted: Mapping) -> list[_ViewGeometry]:
    """按输入视图顺序加载并校验几何，要求每个视图恰好一份结果。"""
    by_id = {item["view_id"]: item for item in predicted["views"]}
    expected_ids = {view["view_id"] for view in views}
    is_complete = len(by_id) == len(predicted["views"]) and set(by_id) == expected_ids
    if not is_complete:
        raise ValueError("DA3 must return geometry for each input view exactly once")

    geometries = []
    for view in views:
        prediction = by_id[view["view_id"]]
        data = validate_geometry(load_npz(prediction["geometry_path"]), (view["width"], view["height"]))
        geometries.append(_ViewGeometry(view, prediction, data))
    return geometries


def _write_view_geometry(context: StageContext, geometry: _ViewGeometry) -> tuple[dict, list[Path]]:
    """写出单视图的 pointmap、深度和处理后 RGB，返回相机记录与产物。"""
    view_id = geometry.view["view_id"]
    data = geometry.data
    safe_depth = np.where(data["valid"], data["depth"], 0)
    points = depth_to_points(safe_depth, data["intrinsics"], data["world_from_camera"])
    points[~data["valid"]] = np.nan
    camera_points = depth_to_points(safe_depth, data["intrinsics"], np.eye(4))
    camera_points[~data["valid"]] = np.nan

    point_path = context.output / "pointmaps" / f"{view_id}.npz"
    depth_path = context.output / "depth" / f"{view_id}.npz"
    image_path = context.directory / f"{view_id}_processed.png"
    atomic_npz(point_path, **data, points_world=points, points_camera=camera_points)
    atomic_npz(depth_path, depth=data["depth"], conf=data["conf"], valid=data["valid"])
    save_image(image_path, data["image"])

    record = {
        "view_id": view_id,
        "geometry_path": str(point_path),
        "image_path": str(image_path),
        "width": data["depth"].shape[1],
        "height": data["depth"].shape[0],
        "scale_status": geometry.prediction.get("scale_status", "relative"),
        "scale_source": geometry.prediction.get("scale_source", "unspecified"),
        "intrinsics": data["intrinsics"].tolist(),
        "world_from_camera": data["world_from_camera"].tolist(),
    }
    return record, [point_path, depth_path, image_path]


# -------------------------------------------
# 物体关联
# -------------------------------------------
@dataclass
class _InstanceGroups:
    """带约束的并查集：同一视图的实例或不同显式物体 ID 的实例永不合并。"""

    instances: Sequence[dict]
    parents: list[int] = field(init=False)
    members: dict[int, set[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.parents = list(range(len(self.instances)))
        self.members = {index: {index} for index in self.parents}

    def find(self, index: int) -> int:
        """返回实例所在分组的根索引，并压缩路径。"""
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def merge(self, first: int, second: int) -> None:
        """合并两个实例分组；同视图冲突或显式 ID 冲突时不合并。"""
        first, second = self.find(first), self.find(second)
        if first == second:
            return

        left, right = self.members[first], self.members[second]
        left_views = {self.instances[index]["view_id"] for index in left}
        right_views = {self.instances[index]["view_id"] for index in right}
        if left_views & right_views:
            return
        names = {self.instances[index].get("object_id") for index in left | right} - {None}
        if len(names) > 1:
            return

        self.parents[second] = first
        self.members[first] |= self.members.pop(second)

    def groups(self) -> list[list[int]]:
        """返回所有分组的实例索引。"""
        return [sorted(group) for group in self.members.values()]


def projected_support(points: np.ndarray, instance: Mapping) -> float:
    """返回 ``points`` 投影到实例视图后、可见部分落入其 mask 的比例。

    没有相机数据的实例视为完全支持；可见投影点不足时返回 0。
    """
    if "camera_data" not in instance:
        return 1.0

    data = instance["camera_data"]
    mask = instance["processed_mask"]
    uv, point_depth = project_points(points, data["intrinsics"], data["world_from_camera"])
    is_projected = np.isfinite(uv).all(axis=1) & (point_depth > 0)
    pixels = np.zeros((len(points), 2), dtype=int)
    pixels[is_projected] = np.rint(uv[is_projected]).astype(int)
    u, v = pixels.T
    is_inside = is_projected & (u >= 0) & (v >= 0) & (u < mask.shape[1]) & (v < mask.shape[0])
    indices = np.flatnonzero(is_inside)
    if not len(indices):
        return 0.0

    observed = data["depth"][v[indices], u[indices]]
    tolerance = VISIBILITY_DEPTH_TOLERANCE * np.maximum(observed, DEPTH_EPSILON)
    is_visible = data["valid"][v[indices], u[indices]] & (point_depth[indices] <= observed + tolerance)
    indices = indices[is_visible]
    if len(indices) < MIN_SUPPORT_POINTS:
        return 0.0
    return float(mask[v[indices], u[indices]].mean())


def associate_instances(instances: Sequence[dict], fusion: Mapping) -> tuple[list[list[int]], list[str]]:
    """把跨视图的同一物体实例分组。

    先合并显式 ``object_id`` 相同的实例，再按几何重叠、颜色和投影支持贪心合并；
    存在竞争且分差小于 ``ambiguity_margin`` 的分配保持分离。

    Args:
        instances: 带点云、颜色、视图和类别的实例。
        fusion: ``fusion`` 配置节。

    Returns:
        实例索引分组，以及需要人工复核的警告。

    Raises:
        ValueError: 显式物体 ID 在同一视图重复或类别冲突。
    """
    groups = _InstanceGroups(instances)
    _merge_explicit_ids(instances, groups)
    edges, competing = _candidate_edges(instances, fusion)
    ambiguous = _ambiguous_assignments(competing, fusion["ambiguity_margin"])
    for _, first, second in sorted(edges, reverse=True):
        is_ambiguous = (first, instances[second]["view_id"]) in ambiguous or (second, instances[first]["view_id"]) in ambiguous
        if is_ambiguous:
            continue
        groups.merge(first, second)

    warnings = []
    if ambiguous:
        warnings.append(f"{len(ambiguous)} ambiguous cross-view assignments were kept separate; provide explicit object_id values to review")
    return groups.groups(), warnings


def _merge_explicit_ids(instances: Sequence[dict], groups: _InstanceGroups) -> None:
    """合并人工确认为同一实体（相同显式 ``object_id``）的实例。"""
    explicit = defaultdict(list)
    for index, instance in enumerate(instances):
        if instance.get("object_id"):
            explicit[instance["object_id"]].append(index)

    for name, indices in explicit.items():
        view_ids = {instances[index]["view_id"] for index in indices}
        if len(view_ids) != len(indices):
            raise ValueError(f"The same explicit object ID occurs twice in one view: {name}")
        categories = {instances[index]["category"] for index in indices}
        if len(categories) != 1:
            raise ValueError(f"Conflicting categories for explicit object ID {name}")
        for index in indices[1:]:
            groups.merge(indices[0], index)


def _candidate_edges(instances: Sequence[dict], fusion: Mapping) -> tuple[list[tuple[float, int, int]], dict]:
    """返回可合并实例对 ``(得分, i, j)``，以及每个实例在各视图中的竞争得分。"""
    edges, competing = [], defaultdict(list)
    for first_index, first in enumerate(instances):
        for second_index in range(first_index + 1, len(instances)):
            second = instances[second_index]
            score = _pair_score(first, second, fusion)
            if score is None:
                continue
            edges.append((score, first_index, second_index))
            competing[(first_index, second["view_id"])].append(score)
            competing[(second_index, first["view_id"])].append(score)
    return edges, competing


def _stride_sample(points: np.ndarray) -> np.ndarray:
    return points[:: max(1, len(points) // OVERLAP_SAMPLE_POINTS)]


def _pair_score(first: Mapping, second: Mapping, fusion: Mapping) -> Optional[float]:
    """计算两个实例属于同一物体的得分；不满足任一关联条件时返回 ``None``。"""
    is_same_view = first["view_id"] == second["view_id"]
    is_other_category = first["category"].lower() != second["category"].lower()
    if is_same_view or is_other_category:
        return None
    has_conflicting_ids = bool(first.get("object_id")) and bool(second.get("object_id")) and first["object_id"] != second["object_id"]
    if has_conflicting_ids:
        return None
    if min(len(first["points"]), len(second["points"])) < MIN_CLOUD_POINTS:
        return None

    color_distance = np.linalg.norm(np.mean(first["colors"], axis=0) - np.mean(second["colors"], axis=0)) / RGB_DIAGONAL_DISTANCE
    if color_distance > fusion["max_color_distance"]:
        return None
    size = max(extent(first["points"]), extent(second["points"]))
    center_distance = np.linalg.norm(np.median(first["points"], axis=0) - np.median(second["points"], axis=0))
    if center_distance > MAX_CENTER_DISTANCE_RATIO * size:
        return None

    first_points, second_points = _stride_sample(first["points"]), _stride_sample(second["points"])
    score = float(cloud_overlap(first_points, second_points, size * fusion["match_distance_ratio"]))
    if score < fusion["min_overlap"]:
        return None
    support = min(projected_support(first_points, second), projected_support(second_points, first))
    if support < fusion.get("min_projected_support", DEFAULT_MIN_PROJECTED_SUPPORT):
        return None
    return score * (1 - COLOR_SCORE_PENALTY * float(color_distance))


def _ambiguous_assignments(competing: Mapping[tuple[int, str], list[float]], margin: float) -> set[tuple[int, str]]:
    """返回最佳与次佳候选得分差小于 ``margin`` 的 ``(实例, 视图)`` 分配。"""
    ambiguous = set()
    for key, scores in competing.items():
        ordered = sorted(scores, reverse=True)
        is_ambiguous = len(ordered) > 1 and ordered[0] - ordered[1] < margin
        if is_ambiguous:
            ambiguous.add(key)
    return ambiguous


# -------------------------------------------
# 物体点云融合
# -------------------------------------------
@dataclass
class _ObjectIdAllocator:
    """按“类别_三位序号”分配物体 ID，跳过已被显式占用的 ID。"""

    used: set[str]
    counts: dict[str, int] = field(default_factory=dict)

    def allocate(self, observations: Sequence[dict]) -> str:
        """为一组观测分配稳定的物体 ID，优先使用显式 ID。"""
        explicit_names = {item.get("object_id") for item in observations} - {None}
        if explicit_names:
            return next(iter(explicit_names))

        prefix = safe_name(observations[0]["category"])
        while True:
            self.counts[prefix] = self.counts.get(prefix, 0) + 1
            object_id = f"{prefix}_{self.counts[prefix]:03d}"
            if object_id not in self.used:
                break
        self.used.add(object_id)
        return object_id


def build_objects(context: StageContext) -> tuple[dict, list]:
    """提取各实例的观测点云，跨视图关联并融合为带稳定 ID 的物体点云。"""
    config, output = context.config, context.output
    views = {item["view_id"]: item for item in context.results["ingest"]["views"]}
    geometry_paths = {item["view_id"]: item["geometry_path"] for item in context.results["depth"]["views"]}
    by_view = defaultdict(list)
    for instance in context.results["segment"]["instances"]:
        by_view[instance["view_id"]].append(instance)

    instances, artifacts = [], []
    for view_id, entries in by_view.items():
        data = load_npz(geometry_paths[view_id])
        extracted, files = _extract_view_instances(entries, data, context.directory, config["geometry"])
        instances.extend(extracted)
        artifacts.extend(files)

    groups, warnings = associate_instances(instances, config["fusion"])
    groups.sort(key=lambda indices: _instance_order(instances[indices[0]]))
    allocator = _ObjectIdAllocator(used={item["object_id"] for item in instances if item.get("object_id")})
    objects, mask_records = [], []
    for group in groups:
        observations = [instances[index] for index in group]
        object_id = allocator.allocate(observations)
        record, masks, files = _write_object(context, object_id, observations, views)
        objects.append(record)
        mask_records.extend(masks)
        artifacts.extend(files)

    result = {"objects": objects, "warnings": warnings}
    metadata = output / "masks" / "metadata.json"
    object_metadata = output / "metadata" / "objects.json"
    representatives = {item["object_id"]: item["reference_view_id"] for item in objects}
    atomic_json(metadata, {"instances": mask_records, "representatives": representatives})
    atomic_json(object_metadata, result)
    return result, artifacts + [metadata, object_metadata]


def _instance_order(instance: Mapping) -> tuple[str, str, str]:
    return instance["category"], instance["view_id"], instance["local_id"]


def _extract_view_instances(entries: Sequence[dict], data: Mapping[str, np.ndarray], destination: Path, geometry_config: Mapping) -> tuple[list[dict], list[Path]]:
    """把一个视图内的实例 mask 映射到几何网格，并提取互不重叠的观测点云。"""
    shape = data["depth"].shape
    masks = [map_mask(load_mask(item["mask_path"]), data["original_to_processed"], shape) for item in entries]
    owner = _assign_pixel_owners(entries, masks, shape)
    has_finite_point = np.isfinite(data["points_world"]).all(axis=-1)

    instances, artifacts = [], []
    for index, item in enumerate(entries):
        keep = masks[index] & (owner == index) & data["valid"] & has_finite_point
        raw_path = destination / item["view_id"] / f"{item['local_id']}_raw.ply"
        cloud = _extract_instance_cloud(data, keep, raw_path, geometry_config)
        instances.append(
            {
                **item,
                **cloud,
                "processed_mask": masks[index],
                "pixel_count": int(load_mask(item["mask_path"]).sum()),
                "raw_cloud_path": str(raw_path),
                "camera_data": data,
            }
        )
        artifacts.append(raw_path)
    return instances, artifacts


def _assign_pixel_owners(entries: Sequence[dict], masks: Sequence[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """重叠像素只归属得分最高的实例；未归属像素为 -1。"""
    owner = np.full(shape, -1, dtype=np.int32)
    by_score = sorted(range(len(entries)), key=lambda index: entries[index]["score"], reverse=True)
    for index in by_score:
        owner[masks[index] & (owner < 0)] = index
    return owner


def _extract_instance_cloud(data: Mapping[str, np.ndarray], keep: np.ndarray, raw_path: Path, geometry_config: Mapping) -> dict:
    """写出清洗前的原始点云，返回清洗后的点、颜色、权重和像素索引。"""
    pixel_indices = np.flatnonzero(keep)
    points = data["points_world"][keep]
    colors = data["image"][keep]
    weights = np.maximum(data["conf"][keep], MIN_POINT_WEIGHT)
    write_cloud(raw_path, points, colors)
    if len(points) < MIN_CLOUD_POINTS:
        return {"points": points, "colors": colors, "weights": weights, "pixel_indices": pixel_indices}

    cleaned = clean_cloud(
        points,
        colors,
        weights,
        neighbors=geometry_config["outlier_neighbors"],
        std_ratio=geometry_config["outlier_std_ratio"],
    )
    return {
        "points": cleaned["points"],
        "colors": cleaned["colors"],
        "weights": cleaned["weights"],
        "pixel_indices": pixel_indices[cleaned["indices"]],
    }


def _write_object(context: StageContext, object_id: str, observations: Sequence[dict], views: Mapping[str, dict]) -> tuple[dict, list[dict], list[Path]]:
    """写出一个物体的 mask、各视图点云和融合点云，返回物体记录、mask 记录和产物。"""
    output, config = context.output, context.config
    base = output / "objects" / object_id
    category = observations[0]["category"]
    reference = max(observations, key=lambda item: item["score"] * np.sqrt(item["pixel_count"]))

    references, mask_records, artifacts = [], [], []
    for item in observations:
        record, files = _write_observation(output, base, object_id, item)
        view = views[item["view_id"]]
        references.append(record)
        mask_records.append({**record, "object_id": object_id, "category": category, "width": view["width"], "height": view["height"]})
        artifacts.extend(files)

    fused = _fuse_observations(observations, config["fusion"])
    cloud_path, provenance_path = _write_fused_cloud(base, fused)
    representative_mask = output / "masks" / f"{object_id}.png"
    save_image(representative_mask, _mask_to_png(load_mask(reference["mask_path"])))
    artifacts.extend([cloud_path, provenance_path, representative_mask])

    point_count = len(fused["fused_points"])
    has_enough_points = point_count >= config["geometry"]["min_points"]
    record = {
        "object_id": object_id,
        "category": category,
        "observations": references,
        "reference_view_id": reference["view_id"],
        "reference_mask_path": str(representative_mask),
        "cloud_path": str(cloud_path),
        "observations_path": str(provenance_path),
        "point_count": point_count,
        "voxel_size": fused["voxel_size"],
        "status": "observed" if has_enough_points else "insufficient_geometry",
    }
    return record, mask_records, artifacts


def _write_observation(output: Path, base: Path, object_id: str, item: Mapping) -> tuple[dict, list[Path]]:
    """以最终物体 ID 写出单视图原图 mask、几何网格 mask 和观测点云。"""
    view_id = item["view_id"]
    mask_path = output / "masks" / view_id / f"{object_id}.png"
    save_image(mask_path, _mask_to_png(load_mask(item["mask_path"])))
    processed_mask = base / "clouds" / "views" / f"{view_id}_mask.png"
    save_image(processed_mask, _mask_to_png(item["processed_mask"]))
    cloud_path = base / "clouds" / "views" / f"{view_id}.ply"
    write_cloud(cloud_path, item["points"], item["colors"])

    record = {
        "view_id": view_id,
        "local_id": item["local_id"],
        "mask_path": str(mask_path),
        "geometry_mask_path": str(processed_mask),
        "cloud_path": str(cloud_path),
        "score": item["score"],
        "pixel_count": item["pixel_count"],
        "point_count": len(item["points"]),
    }
    return record, [mask_path, processed_mask, cloud_path]


def _fuse_observations(observations: Sequence[dict], fusion: Mapping) -> dict:
    """拼接各视图观测并做加权体素融合；体素大小未配置时按物体尺寸比例估计。"""
    points = np.concatenate([item["points"] for item in observations])
    colors = np.concatenate([item["colors"] for item in observations])
    weights = np.concatenate([item["weights"] for item in observations])
    view_indices = np.concatenate([np.full(len(item["points"]), index, dtype=np.int32) for index, item in enumerate(observations)])
    pixel_indices = np.concatenate([item["pixel_indices"] for item in observations])
    voxel = fusion["voxel_size"] or extent(points) * fusion["voxel_ratio"]
    if len(points):
        fused = voxel_fuse(points, colors, weights, voxel)
    else:
        fused = {"points": points, "colors": colors, "weights": weights, "counts": np.empty(0, dtype=np.int32)}
    return {
        "points": points,
        "colors": colors,
        "weights": weights,
        "view_indices": view_indices,
        "pixel_indices": pixel_indices,
        "fused_points": fused["points"],
        "fused_colors": fused["colors"],
        "fused_weights": fused["weights"],
        "fused_counts": fused["counts"],
        "voxel_size": voxel,
    }


def _write_fused_cloud(base: Path, fused: Mapping) -> tuple[Path, Path]:
    """写出融合点云 PLY 和逐点来源 NPZ。"""
    cloud_path = base / "clouds" / "fused.ply"
    provenance_path = base / "clouds" / "observations.npz"
    write_cloud(cloud_path, fused["fused_points"], fused["fused_colors"])
    atomic_npz(
        provenance_path,
        points=fused["points"],
        colors=fused["colors"],
        weights=fused["weights"],
        view_indices=fused["view_indices"],
        pixel_indices=fused["pixel_indices"],
        fused_weights=fused["fused_weights"],
        fused_counts=fused["fused_counts"],
    )
    return cloud_path, provenance_path
