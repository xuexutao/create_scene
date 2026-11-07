from __future__ import annotations

import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import binary_erosion

from scene_create.adapters.common import configure_device
from scene_create.io import (
    atomic_json,
    file_digest,
    fingerprint,
    load_mask,
    load_npz,
    read_cloud,
    read_json,
    save_image,
)


def projection_matrix(intrinsics, width, height, near=0.01, far=100.0):
    intrinsic = np.asarray(intrinsics, dtype=float)
    if (
        intrinsic.shape != (3, 3)
        or min(width, height) <= 0
        or not 0 < near < far
        or not np.isfinite(intrinsic).all()
    ):
        raise ValueError("Invalid camera projection parameters")
    if (
        not np.allclose(intrinsic[2], [0, 0, 1])
        or abs(intrinsic[0, 1]) > 1e-8
        or min(intrinsic[0, 0], intrinsic[1, 1]) <= 0
    ):
        raise ValueError("Gaussian rasterization requires a zero-skew pinhole camera")
    projection = np.zeros((4, 4), dtype=np.float32)
    projection[0, 0] = 2 * intrinsic[0, 0] / width
    projection[1, 1] = 2 * intrinsic[1, 1] / height
    projection[0, 2] = 2 * (intrinsic[0, 2] + 0.5) / width - 1
    projection[1, 2] = 2 * (intrinsic[1, 2] + 0.5) / height - 1
    projection[2, 2] = far / (far - near)
    projection[2, 3] = -far * near / (far - near)
    projection[3, 2] = 1
    return projection


def valid_window_mask(mask, window=11):
    if window < 1 or window % 2 == 0:
        raise ValueError("SSIM window must be odd and positive")
    return binary_erosion(
        np.asarray(mask, dtype=bool),
        structure=np.ones((window, window), bool),
        border_value=0,
    )


def masked_ssim(prediction, reference, valid_windows, window=11):
    import torch
    import torch.nn.functional as functional

    if not valid_windows.any():
        return prediction.sum() * 0
    coordinates = (
        torch.arange(window, device=prediction.device, dtype=prediction.dtype)
        - window // 2
    )
    kernel = torch.exp(-coordinates.square() / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    kernel = (
        (kernel[:, None] @ kernel[None, :]).expand(3, 1, window, window).contiguous()
    )
    a, b = prediction[None], reference[None]
    mean_a = functional.conv2d(a, kernel, padding=window // 2, groups=3)
    mean_b = functional.conv2d(b, kernel, padding=window // 2, groups=3)
    variance_a = (
        functional.conv2d(a * a, kernel, padding=window // 2, groups=3)
        - mean_a.square()
    )
    variance_b = (
        functional.conv2d(b * b, kernel, padding=window // 2, groups=3)
        - mean_b.square()
    )
    covariance = (
        functional.conv2d(a * b, kernel, padding=window // 2, groups=3)
        - mean_a * mean_b
    )
    similarity = ((2 * mean_a * mean_b + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (mean_a.square() + mean_b.square() + 0.01**2)
        * (variance_a + variance_b + 0.03**2)
    )
    return (1 - similarity[0][:, valid_windows]).mean()


def run(request):
    if request.get("mode") == "render_scene":
        from scene_create.adapters.gs_render import run as render_scene

        return render_scene(request)
    import torch
    from loguru import logger
    from gaussian_renderer import render
    from scene.cameras import MiniCam
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud
    from scene_create.assets.gaussian import transform_gaussians

    config, directory = request["config"], Path(request["output_dir"])
    device = config.get("device", "cuda")
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("MaskGaussian requires CUDA and its compiled mask rasterizer")
    device = configure_device(torch, device)
    torch.manual_seed(int(config.get("seed", 0)))
    random = np.random.default_rng(int(config.get("seed", 0)))
    dataset = read_json(request["dataset_path"])
    cloud = read_cloud(dataset["initial_ply"])
    if len(cloud["points"]) < 3:
        raise ValueError("MaskGaussian needs at least three initial points")
    extent = max(
        float(
            np.linalg.norm(
                np.quantile(cloud["points"], 0.95, axis=0)
                - np.quantile(cloud["points"], 0.05, axis=0)
            )
        ),
        1e-4,
    )
    near = float(config.get("near", max(extent * 1e-5, 1e-5)))
    far = float(config.get("far", max(extent * 100, 100)))
    iterations = int(config.get("iterations", 1000))
    degree = int(config.get("sh_degree", 0))
    if iterations < 1 or degree not in range(4):
        raise ValueError("Invalid training iteration count or SH degree")
    depth_weight = float(config.get("depth_weight", 0.1))
    if len(dataset["views"]) == 1 and depth_weight <= 0:
        raise ValueError(
            "Single-view MaskGaussian requires a positive depth_weight; otherwise use explicit init-only mode"
        )
    options = SimpleNamespace(
        percent_dense=0.01,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=iterations,
        feature_lr=0.0025,
        opacity_lr=0.05,
        mask_lr=0.01,
        scaling_lr=0.005,
        rotation_lr=0.001,
    )
    for key in vars(options):
        if key in config:
            setattr(options, key, config[key])
    pipe = SimpleNamespace(
        debug=False, compute_cov3D_python=False, convert_SHs_python=False
    )

    class CoherentGaussianModel(GaussianModel):
        cached_mask = None

        @property
        def get_mask(self):
            return super().get_mask if self.cached_mask is None else self.cached_mask

        def sample_iteration_mask(self):
            self.cached_mask = super().get_mask

    gaussian = CoherentGaussianModel(degree)
    dataset_signature = fingerprint(
        {
            "dataset": file_digest(request["dataset_path"]),
            "initial": file_digest(dataset["initial_ply"]),
            "files": [
                {
                    key: file_digest(view[key])
                    for key in ("geometry_path", "image_path", "mask_path")
                }
                for view in dataset["views"]
            ],
            "config": config,
        }
    )
    checkpoint = directory / "checkpoint.pt"
    first_iteration = 0
    if checkpoint.exists() and config.get("resume_checkpoint", True):
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        if state.get("signature") != dataset_signature:
            raise ValueError(
                "Existing MaskGaussian checkpoint belongs to different inputs/configuration; move it or set resume_checkpoint: false"
            )
        model_state = list(state["model"])
        for index in range(1, 11):
            if torch.is_tensor(model_state[index]):
                tensor = model_state[index].to(device)
                model_state[index] = (
                    torch.nn.Parameter(tensor) if index in range(1, 8) else tensor
                )
        gaussian.restore(tuple(model_state), options)
        first_iteration = int(state["iteration"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], device=device)
        random.bit_generator.state = state["numpy_rng"]
    else:
        gaussian.create_from_pcd(
            BasicPointCloud(
                points=cloud["points"],
                colors=cloud["colors"].astype(float) / 255,
                normals=np.zeros_like(cloud["points"]),
            ),
            extent,
        )
        gaussian.training_setup(options)
    views = []
    for index, record in enumerate(dataset["views"]):
        data = load_npz(record["geometry_path"])
        height, width = data["depth"].shape
        valid = load_mask(record["mask_path"], (width, height)) & data["valid"]
        if not valid.any():
            continue
        intrinsic = data["intrinsics"]
        camera_from_world = np.linalg.inv(data["world_from_camera"])
        world_view = torch.as_tensor(
            camera_from_world, dtype=torch.float32, device=device
        ).T.contiguous()
        projection = torch.as_tensor(
            projection_matrix(intrinsic, width, height, near, far), device=device
        ).T.contiguous()
        camera = MiniCam(
            width,
            height,
            2 * math.atan(height / (2 * intrinsic[1, 1])),
            2 * math.atan(width / (2 * intrinsic[0, 0])),
            near,
            far,
            world_view,
            world_view @ projection,
        )
        views.append(
            {
                "camera": camera,
                "view_id": record["view_id"],
                "rgb": torch.from_numpy(data["image"].copy()).permute(2, 0, 1).float()
                / 255,
                "valid": torch.from_numpy(valid),
                "ssim_valid": torch.from_numpy(valid_window_mask(valid)),
                "depth": torch.from_numpy(data["depth"].copy()).float(),
                "conf": torch.from_numpy(data["conf"].copy()).float(),
                "camera_from_world": torch.as_tensor(
                    camera_from_world, dtype=torch.float32, device=device
                ),
            }
        )
    if not views:
        raise ValueError("No valid background training pixels")
    black = torch.zeros(3, device=device)
    dssim_weight = float(config.get("dssim_weight", 0.2))
    mask_weight = float(config.get("mask_weight", 0.0005))
    maximum = int(config.get("max_gaussians", 500000))
    log_interval = max(1, int(config.get("log_interval", 50)))
    checkpoint_interval = max(1, int(config.get("checkpoint_interval", 250)))
    densify_from = int(config.get("densify_from", 100))
    densify_until = int(config.get("densify_until", max(0, int(iterations * 0.6))))
    history, started = [], time.monotonic()

    def save_checkpoint(iteration):
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(
            {
                "signature": dataset_signature,
                "iteration": iteration,
                "model": gaussian.capture(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device),
                "numpy_rng": random.bit_generator.state,
            },
            str(temporary),
        )
        os.replace(temporary, checkpoint)

    for iteration in range(first_iteration + 1, iterations + 1):
        view = views[int(random.integers(len(views)))]
        reference, valid = view["rgb"].to(device), view["valid"].to(device)
        ssim_valid = view["ssim_valid"].to(device)
        gaussian.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussian.oneupSHdegree()
        gaussian.sample_iteration_mask()
        package = render(view["camera"], gaussian, pipe, black)
        rendered = package["render"]
        l1 = (rendered[:, valid] - reference[:, valid]).abs().mean()
        ssim_loss = (
            masked_ssim(rendered, reference, ssim_valid)
            if dssim_weight
            else rendered.sum() * 0
        )
        loss = (
            (1 - dssim_weight) * l1
            + dssim_weight * ssim_loss
            + mask_weight * package["mask"].mean().square()
        )
        depth_loss = rendered.sum() * 0
        if depth_weight > 0:
            camera_from_world = view["camera_from_world"]
            depths = (
                gaussian.get_xyz @ camera_from_world[2, :3] + camera_from_world[2, 3]
            )
            depth_colors = depths.clamp_min(0)[:, None].expand(-1, 3).contiguous()
            depth_render = render(
                view["camera"], gaussian, pipe, black, override_color=depth_colors
            )["render"][0]
            alpha = render(
                view["camera"],
                gaussian,
                pipe,
                black,
                override_color=torch.ones_like(depth_colors),
            )["render"][0]
            measured_depth = view["depth"].to(device)
            depth_valid = (
                valid
                & (alpha > 0.05)
                & torch.isfinite(measured_depth)
                & (measured_depth > 0)
            )
            if depth_valid.any():
                predicted_depth = depth_render / alpha.clamp_min(1e-6)
                confidence = view["conf"].to(device)[depth_valid].clamp_min(0)
                confidence = confidence / confidence.mean().clamp_min(1e-6)
                depth_loss = (
                    (
                        (
                            predicted_depth[depth_valid] - measured_depth[depth_valid]
                        ).abs()
                        / measured_depth[depth_valid].clamp_min(1e-6)
                    )
                    * confidence
                ).mean()
                loss = loss + depth_weight * depth_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite MaskGaussian training loss")
        loss.backward()
        with torch.no_grad():
            visible = package["visibility_filter"]
            gaussian.max_radii2D[visible] = torch.maximum(
                gaussian.max_radii2D[visible], package["radii"][visible]
            )
            if package["viewspace_points"].grad is not None:
                gaussian.add_densification_stats(package["viewspace_points"], visible)
            gaussian.optimizer.step()
            gaussian.optimizer.zero_grad(set_to_none=True)
            gaussian.cached_mask = None
            if (
                densify_from <= iteration < densify_until
                and iteration % 100 == 0
                and len(gaussian.get_xyz) < maximum // 2
            ):
                gaussian.densify_and_prune(
                    float(config.get("densify_grad_threshold", 0.0002)),
                    0.005,
                    extent,
                    None,
                )
            elif iteration % 500 == 0:
                gaussian.mask_prune()
            if len(gaussian.get_xyz) == 0:
                raise RuntimeError("MaskGaussian pruned all background primitives")
        if iteration % log_interval == 0 or iteration == iterations:
            metrics = {
                "iteration": iteration,
                "l1": float(l1.detach()),
                "dssim": float(ssim_loss.detach()),
                "relative_depth_loss": float(depth_loss.detach()),
                "loss": float(loss.detach()),
                "gaussians": len(gaussian.get_xyz),
            }
            history.append(metrics)
            logger.info(
                "Background iteration {}: loss={:.5f}, gaussians={}",
                iteration,
                metrics["loss"],
                metrics["gaussians"],
            )
        if iteration % checkpoint_interval == 0 or iteration == iterations:
            save_checkpoint(iteration)
    with torch.no_grad():
        probability = torch.softmax(gaussian._mask_score, dim=-1)[:, 0]
        keep = (probability >= float(config.get("export_probability", 0.5))) & (
            gaussian.get_opacity[:, 0] > 0.005
        )
        if not keep.any():
            raise RuntimeError(
                "No background Gaussians passed the export probability threshold"
            )
        gaussian.cached_mask = None
        gaussian.prune_points(~keep)
        gaussian.cached_mask = torch.ones_like(gaussian.get_opacity)
        raw_path, output_path = (
            directory / "trained_with_masks.ply",
            directory / "background_gaussian.ply",
        )
        gaussian.save_ply(str(raw_path))
        transform_gaussians(raw_path, output_path, np.eye(4))
        for view in views:
            image = (
                render(view["camera"], gaussian, pipe, black)["render"]
                .clamp(0, 1)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            save_image(
                directory / "previews" / f"{view['view_id']}.png",
                (image * 255).astype(np.uint8),
            )
    result = {
        "gaussian_path": str(output_path),
        "checkpoint_path": str(checkpoint),
        "iterations": iterations,
        "elapsed_seconds": time.monotonic() - started,
        "gaussians": len(gaussian.get_xyz),
        "history": history,
        "mask_mechanism": "upstream_gumbel_straight_through_probability",
        "background_masked_supervision": True,
        "depth_weight": depth_weight,
        "single_view_approximate": len(views) == 1,
        "hidden_background_completed": False,
    }
    atomic_json(directory / "training_metrics.json", result)
    return result
