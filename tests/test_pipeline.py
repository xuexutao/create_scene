import numpy as np
import pytest
from PIL import Image

from scene_create.config import load_config
from scene_create.io import atomic_json, atomic_npz, read_cloud, read_json
from scene_create.pipeline import ScenePipeline


def observations_fixture(root, views=2, objects=True):
    root.mkdir(parents=True, exist_ok=True)
    records = []
    intrinsic = np.array([[20, 0, 16], [0, 20, 12], [0, 0, 1]], dtype=float)
    for index in range(views):
        rgb = np.full((24, 32, 3), 180, np.uint8)
        masks = []
        if objects:
            for number, (start, color) in enumerate(
                [(4, [220, 20, 20]), (21, [20, 220, 20])]
            ):
                mask = np.zeros((24, 32), bool)
                mask[7:17, start - index : start + 6 - index] = True
                rgb[mask] = color
                mask_path = root / f"mask_{index}_{number}.png"
                Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
                masks.append({"category": "cup", "mask_path": mask_path.name})
        image = root / f"image_{index}.png"
        Image.fromarray(rgb).save(image)
        camera = np.eye(4)
        camera[0, 3] = 0.1 * index
        geometry = root / f"geometry_{index}.npz"
        atomic_npz(
            geometry,
            depth=np.full((24, 32), 2.0),
            conf=np.ones((24, 32)),
            intrinsics=intrinsic,
            world_from_camera=camera,
            image=rgb,
            original_to_processed=np.eye(3),
            valid=np.ones((24, 32), bool),
        )
        records.append(
            {
                "view_id": f"view_{index:03d}",
                "image_path": image.name,
                "geometry_path": geometry.name,
                "masks": masks,
            }
        )
    manifest = root / "observations.json"
    atomic_json(
        manifest,
        {
            "views": records,
            "scale_status": "relative",
            "scale_source": "analytic_test_fixture_not_model_output",
        },
    )
    return manifest


def test_multiview_masks_and_pointclouds(tmp_path):
    manifest = observations_fixture(tmp_path / "inputs")
    output = tmp_path / "output"
    pipeline = ScenePipeline(load_config(), output)
    result = pipeline.run({"observations": manifest}, through="objects")
    assert result["status"] == "completed"
    objects = result["result"]["objects"]
    assert len(objects) == 2
    assert {item["object_id"] for item in objects} == {"cup_001", "cup_002"}
    assert all(len(item["observations"]) == 2 for item in objects)
    assert all(item["status"] == "observed" for item in objects)
    assert (output / "masks" / "cup_001.png").exists()
    assert (output / "masks" / "view_001" / "cup_001.png").exists()
    for item in objects:
        points = read_cloud(item["cloud_path"])["points"]
        np.testing.assert_allclose(points[:, 2], 2)
    old = read_json(output / "run_manifest.json")["stages"]["objects"]["started_at"]
    pipeline.run({"observations": manifest}, through="objects", resume=True)
    assert (
        read_json(output / "run_manifest.json")["stages"]["objects"]["started_at"]
        == old
    )


def test_no_object_scene_explicit_init_only(tmp_path):
    manifest = observations_fixture(tmp_path / "inputs", views=1, objects=False)
    config = load_config()
    config["background"]["mode"] = "init-only"
    result = ScenePipeline(config, tmp_path / "output").run({"observations": manifest})
    assert result["status"] == "approximate"
    scene = read_json(result["result"]["scene_path"])
    assert scene["objects"] == []
    assert scene["background"]["trained"] is False
    assert scene["background"]["status"] == "initialization_only"
    assert scene["gaussian_scene"] == "scene_gaussian.ply"
    assert scene["quality"]["ground_truth_evaluated"] is False


def test_missing_models_fail_without_fake_masks(tmp_path):
    image = tmp_path / "input.png"
    Image.new("RGB", (32, 32), "gray").save(image)
    output = tmp_path / "output"
    with pytest.raises(Exception, match="model_path"):
        ScenePipeline(load_config(), output).run({"image": image}, through="segment")
    data = read_json(output / "run_manifest.json")
    assert data["stages"]["segment"]["status"] == "failed"
    assert not (output / "masks").exists()


def test_existing_output_not_overwritten(tmp_path):
    manifest = observations_fixture(tmp_path / "inputs", views=1)
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "user_data.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="nonempty"):
        ScenePipeline(load_config(), output).run({"observations": manifest}, "objects")
    assert marker.read_text() == "keep"
