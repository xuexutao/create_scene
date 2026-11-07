from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from scene_create.io import atomic_json, file_digest, read_json, save_image, validate_id

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def resolve_input(source):
    kind, path = next(iter(source.items()))
    path = Path(path).expanduser().resolve()
    imported = kind == "observations"
    if kind == "image":
        records = [{"image_path": str(path)}]
        base = path.parent
        metadata = {}
    elif kind == "images":
        if not path.is_dir():
            raise ValueError(f"Image directory does not exist: {path}")
        records = [
            {"image_path": str(item)}
            for item in sorted(path.iterdir())
            if item.suffix.lower() in IMAGE_EXTENSIONS and item.is_file()
        ]
        base = path
        metadata = {}
    else:
        metadata = read_json(path)
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("views"), list
        ):
            raise ValueError("Input manifest must contain a views list")
        records = metadata["views"]
        base = path.parent
    if not records:
        raise ValueError("No input images were found")
    views, seen = [], set()
    dependencies = []
    if kind in ("manifest", "observations"):
        dependencies.append({"path": str(path), "sha256": file_digest(path)})
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("Every view must be a mapping")
        view_id = validate_id(record.get("view_id", f"view_{index:03d}"))
        if view_id in seen:
            raise ValueError(f"Duplicate view_id: {view_id}")
        seen.add(view_id)
        image = Path(record["image_path"])
        image = (base / image).resolve() if not image.is_absolute() else image.resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        digest = file_digest(image)
        view = {"view_id": view_id, "source_path": str(image), "source_sha256": digest}
        dependencies.append({"path": str(image), "sha256": digest})
        if any(key in record for key in ("intrinsics", "extrinsics", "distortion")):
            raise ValueError(
                "External camera calibration must currently be supplied through --observations geometry NPZ; calibrated image-only input is not yet supported"
            )
        if imported:
            if "geometry_path" not in record or "masks" not in record:
                raise ValueError(
                    "Imported views require geometry_path and masks (an empty mask list is valid)"
                )
            geometry = (base / record["geometry_path"]).resolve()
            view["imported_geometry"] = str(geometry)
            view["scale_status"] = record.get(
                "scale_status", metadata.get("scale_status", "relative")
            )
            view["scale_source"] = record.get(
                "scale_source", metadata.get("scale_source", "external_unspecified")
            )
            dependencies.append(
                {"path": str(geometry), "sha256": file_digest(geometry)}
            )
            masks = []
            for mask_index, item in enumerate(record["masks"]):
                mask_path = (base / item["mask_path"]).resolve()
                entry = dict(
                    item,
                    mask_path=str(mask_path),
                    local_id=validate_id(
                        item.get("local_id", f"instance_{mask_index:03d}")
                    ),
                )
                if (
                    "category" not in entry
                    or not isinstance(entry["category"], str)
                    or not entry["category"].strip()
                ):
                    raise ValueError("Each imported mask requires a nonempty category")
                if entry.get("object_id"):
                    validate_id(entry["object_id"])
                masks.append(entry)
                dependencies.append(
                    {"path": str(mask_path), "sha256": file_digest(mask_path)}
                )
            if len({mask["local_id"] for mask in masks}) != len(masks):
                raise ValueError(f"Duplicate local instance IDs in {view_id}")
            view["imported_masks"] = masks
        views.append(view)
    return {
        "views": views,
        "provenance": "external_observations" if imported else "input_images",
        "dependencies": dependencies,
    }


def ingest(inputs, destination, output):
    views, artifacts = [], []
    for entry in inputs["views"]:
        with Image.open(entry["source_path"]) as original:
            orientation = original.getexif().get(274, 1)
            if inputs["provenance"] == "external_observations" and orientation not in (
                None,
                1,
            ):
                raise ValueError(
                    "Imported geometry requires already EXIF-normalized images and masks"
                )
            image = ImageOps.exif_transpose(original).convert("RGB")
            if min(image.size) < 8:
                raise ValueError(f"Image is too small: {entry['source_path']}")
            target = destination / f"{entry['view_id']}.png"
            save_image(target, image)
            view = {
                **entry,
                "image_path": str(target),
                "width": image.width,
                "height": image.height,
                "exif_orientation": orientation,
            }
            views.append(view)
            artifacts.append(target)
    result = {"views": views, "provenance": inputs["provenance"]}
    metadata = output / "metadata" / "views.json"
    atomic_json(metadata, result)
    return result, artifacts + [metadata]
