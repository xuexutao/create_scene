from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scene_create.adapters.common import configure_device
from scene_create.adapters.maskgaussian import projection_matrix
from scene_create.assets.gaussian import _read
from scene_create.io import atomic_json, read_json, save_image, validate_id


def run(request):
    import torch
    from gaussian_renderer import render
    from scene.cameras import MiniCam

    config = request["config"]
    device = config.get("device", "cuda")
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError(
            "Gaussian preview requires a configured CUDA MaskGaussian worker"
        )
    device = configure_device(torch, device)
    scene_path = Path(request["scene_path"])
    scene = read_json(scene_path)
    if not scene.get("gaussian_scene"):
        raise ValueError("Scene has no Gaussian representation to render")
    vertices, degree = _read(scene_path.parent / scene["gaussian_scene"])
    if not len(vertices):
        raise ValueError("Cannot render an empty Gaussian scene")

    def tensor(names):
        return torch.as_tensor(
            np.column_stack([vertices[name] for name in names]),
            dtype=torch.float32,
            device=device,
        )

    class FixedGaussians:
        pass

    model = FixedGaussians()
    model.get_xyz = tensor(["x", "y", "z"])
    model.get_scaling = tensor([f"scale_{i}" for i in range(3)]).exp()
    model.get_rotation = torch.nn.functional.normalize(
        tensor([f"rot_{i}" for i in range(4)]), dim=1
    )
    model.get_opacity = tensor(["opacity"]).sigmoid()
    model.get_mask = torch.ones_like(model.get_opacity)
    dc = tensor([f"f_dc_{i}" for i in range(3)])[:, None, :]
    count = (degree + 1) ** 2 - 1
    rest = (
        tensor([f"f_rest_{i}" for i in range(3 * count)])
        .reshape(-1, 3, count)
        .transpose(1, 2)
        if count
        else torch.empty((len(vertices), 0, 3), device=device)
    )
    model.get_features = torch.cat([dc, rest], dim=1).contiguous()
    model.active_sh_degree = model.max_sh_degree = degree
    pipe = SimpleNamespace(
        debug=False, compute_cov3D_python=False, convert_SHs_python=False
    )
    background = torch.zeros(3, device=device)
    minimum, maximum = model.get_xyz.min(dim=0).values, model.get_xyz.max(dim=0).values
    extent = max(float(torch.linalg.vector_norm(maximum - minimum)), 1e-4)
    near, far = max(extent * 1e-5, 1e-5), max(100, extent * 100)
    files = []
    with torch.no_grad():
        for view in scene["cameras"]:
            view_id = validate_id(view["view_id"])
            scale = min(
                1.0,
                int(config.get("render_max_side", 768))
                / max(view["width"], view["height"]),
            )
            width, height = (
                max(1, round(view["width"] * scale)),
                max(1, round(view["height"] * scale)),
            )
            sx, sy = width / view["width"], height / view["height"]
            intrinsic = np.array(
                [[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]]
            ) @ np.asarray(view["intrinsics"])
            world_view = torch.as_tensor(
                np.linalg.inv(view["world_from_camera"]),
                dtype=torch.float32,
                device=device,
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
            image = (
                render(camera, model, pipe, background)["render"]
                .clamp(0, 1)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            path = Path(request["output_dir"]) / f"{view_id}.png"
            save_image(path, (image * 255).astype(np.uint8))
            files.append(str(path))
    result = {
        "images": files,
        "gaussian_count": len(vertices),
        "sh_degree": degree,
        "renderer": "MaskGaussian CUDA rasterizer; fixed all-present masks",
        "background_color": "black",
    }
    atomic_json(Path(request["output_dir"]) / "previews.json", result)
    return result
