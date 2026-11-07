from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from scene_create.adapters.common import configure_device
from scene_create.io import atomic_json, read_json, save_image

GLB_FROM_NATIVE = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)


def validate_foreground(alpha, require_margin=True):
    alpha = np.asarray(alpha)
    foreground = alpha > 204
    if alpha.ndim != 2 or foreground.sum() < 16:
        raise ValueError("Completed object has no usable foreground")
    if require_margin and (
        foreground[0].any()
        or foreground[-1].any()
        or foreground[:, 0].any()
        or foreground[:, -1].any()
    ):
        raise ValueError(
            "Completed foreground touches the image boundary; regenerate with more margin"
        )
    ys, xs = np.where(foreground)
    if min(xs.max() - xs.min(), ys.max() - ys.min()) < 3:
        raise ValueError("Completed object foreground is geometrically degenerate")
    return foreground


def local_components(model_path):
    if not model_path:
        raise ValueError("TRELLIS requires a local model_path")
    root = Path(model_path).expanduser().resolve()
    manifest = root / "pipeline.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"TRELLIS local pipeline manifest is missing: {manifest}"
        )
    args = read_json(manifest).get("args")
    if not isinstance(args, dict) or not isinstance(args.get("models"), dict):
        raise ValueError("Invalid TRELLIS pipeline.json model mapping")
    required = (
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "slat_flow_model",
        "slat_decoder_mesh",
        "slat_decoder_gs",
    )
    components = {}
    for name in required:
        reference = args["models"].get(name)
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"TRELLIS pipeline is missing component {name}")
        stem = (root / reference).resolve()
        for suffix in (".json", ".safetensors"):
            if not Path(str(stem) + suffix).is_file():
                raise FileNotFoundError(
                    f"TRELLIS component {name} is incomplete: {stem}{suffix}; remote fallback is disabled"
                )
        components[name] = str(stem)
    return args, components


def normalize_export_rotations(rotations):
    rotations = np.asarray(rotations, dtype=np.float64)
    if (
        rotations.ndim != 2
        or rotations.shape[1] != 4
        or not len(rotations)
        or not np.isfinite(rotations).all()
    ):
        raise ValueError("Gaussian rotations must be nonempty finite Nx4 wxyz values")
    norm = np.linalg.norm(rotations, axis=1, keepdims=True)
    if not np.isfinite(norm).all() or (norm <= 1e-8).any():
        raise ValueError("Gaussian rotation quaternions must have nonzero finite norm")
    return rotations / norm


def local_rembg_session(rembg, model_path):
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Local U2Net ONNX model is missing: {path}")
    os.environ["U2NET_HOME"] = str(path.parent)
    os.environ["REMBG_HOME"] = str(path.parent)
    return rembg.new_session(
        "u2net_custom",
        model_path=str(path),
        providers=["CPUExecutionProvider"],
    )


def run(request):
    config = request["config"]
    args, component_paths = local_components(config.get("model_path"))
    if not str(config.get("device", "cuda")).startswith("cuda"):
        raise ValueError("TRELLIS requires a CUDA worker with compiled extensions")
    os.environ["ATTN_BACKEND"] = config.get("attention_backend", "xformers")
    os.environ["SPCONV_ALGO"] = config.get("spconv_algo", "native")
    import torch

    device = configure_device(torch, config.get("device", "cuda"))
    import rembg
    from safetensors.torch import load_file
    from torchvision import transforms
    from trellis import models as trellis_models
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.pipelines import samplers
    from trellis.utils import postprocessing_utils

    root = Path(request["root"])
    dino_repo = Path(config.get("dino_repo_path") or root / "third_parts" / "dinov2")
    dino_weights = config.get("dino_model_path")
    if (
        not dino_weights
        or not Path(dino_weights).is_file()
        or not (dino_repo / "hubconf.py").is_file()
    ):
        raise FileNotFoundError(
            "TRELLIS requires local dino_repo_path and dino_model_path; its upstream torch.hub auto-download is intentionally not used"
        )

    class LocalTrellisPipeline(TrellisImageTo3DPipeline):
        def _init_image_cond_model(self, name):
            model = torch.hub.load(
                str(dino_repo), name, source="local", pretrained=False
            )
            if str(dino_weights).endswith(".safetensors"):
                from safetensors.torch import load_file

                state = load_file(dino_weights)
            else:
                state = torch.load(dino_weights, map_location="cpu", weights_only=True)
            if "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            model.load_state_dict(state, strict=True)
            model.eval()
            self.models["image_cond_model"] = model
            self.image_cond_model_transform = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )
                ]
            )

    torch.manual_seed(config.get("seed", 0))
    models = {}
    for name, stem in component_paths.items():
        component_config = read_json(stem + ".json")
        model_type = getattr(trellis_models, component_config["name"])
        model = model_type(**component_config["args"])
        model.load_state_dict(load_file(stem + ".safetensors"), strict=True)
        models[name] = model
    sparse = args["sparse_structure_sampler"]
    slat = args["slat_sampler"]
    pipeline = LocalTrellisPipeline(
        models=models,
        sparse_structure_sampler=getattr(samplers, sparse["name"])(**sparse["args"]),
        slat_sampler=getattr(samplers, slat["name"])(**slat["args"]),
        slat_normalization=args["slat_normalization"],
        image_cond_model=args["image_cond_model"],
    )
    pipeline.sparse_structure_sampler_params = sparse["params"]
    pipeline.slat_sampler_params = slat["params"]
    pipeline.to(torch.device(device))
    with Image.open(request["object"]["image_path"]) as image:
        image = image.copy()
    if image.mode == "RGBA" and not np.all(np.asarray(image)[:, :, 3] == 255):
        rgba = image
    else:
        onnx_path = Path(
            config.get("rembg_model_path")
            or root / "checkpoints" / "rembg" / "u2net.onnx"
        )
        session = local_rembg_session(rembg, onnx_path)
        rgba = rembg.remove(image.convert("RGB"), session=session)
    alpha = np.asarray(rgba)[:, :, 3]
    foreground = validate_foreground(alpha, bool(config.get("require_margin", True)))
    directory = Path(request["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    save_image(directory / "completed_rgba.png", rgba)
    save_image(directory / "completed_mask.png", (foreground * 255).astype(np.uint8))
    conditioned = pipeline.preprocess_image(rgba)
    save_image(directory / "trellis_input.png", conditioned)
    with torch.inference_mode():
        output = pipeline.run(
            conditioned,
            seed=int(config.get("seed", 0)),
            formats=["mesh", "gaussian"],
            preprocess_image=False,
            sparse_structure_sampler_params=config.get(
                "sparse_structure_sampler_params", {}
            ),
            slat_sampler_params=config.get("slat_sampler_params", {}),
        )
    mesh, gaussian = output["mesh"][0], output["gaussian"][0]
    if (
        len(mesh.vertices) == 0
        or len(mesh.faces) == 0
        or not torch.isfinite(mesh.vertices).all()
    ):
        raise ValueError("TRELLIS returned an empty or non-finite mesh")
    if getattr(gaussian, "sh_degree", 0) != 0:
        raise ValueError(
            "Pinned TRELLIS save_ply only exports DC color; nonzero SH degree would lose information"
        )
    with torch.no_grad():
        raw_rotations = gaussian._rotation + gaussian.rots_bias[None, :]
        normalized = normalize_export_rotations(raw_rotations.detach().cpu().numpy())
        gaussian.from_rotation(
            torch.as_tensor(
                normalized,
                dtype=raw_rotations.dtype,
                device=raw_rotations.device,
            )
        )
    glb = postprocessing_utils.to_glb(
        gaussian,
        mesh,
        simplify=float(config.get("simplify", 0.5)),
        texture_size=int(config.get("texture_size", 1024)),
        fill_holes=bool(config.get("fill_holes", False)),
        verbose=True,
    )
    mesh_path, gaussian_path = directory / "mesh.glb", directory / "gaussian.ply"
    glb.export(str(mesh_path))
    gaussian.save_ply(str(gaussian_path), transform=GLB_FROM_NATIVE.tolist())
    metadata_path = directory / "asset_meta.json"
    metadata = {
        "model": "TRELLIS-image",
        "checkpoint": config["model_path"],
        "seed": int(config.get("seed", 0)),
        "dino_weights": str(dino_weights),
        "component_checkpoints": component_paths,
        "device": device,
        "remote_component_fallback": False,
        "rotation_quaternion_order": "wxyz",
        "rotations_normalized": True,
        "native_to_asset_rotation": GLB_FROM_NATIVE.tolist(),
        "same_latent_sample": True,
        "mesh_and_gaussian_same_frame": True,
        "asset_up": "Y",
        "asset_scale": "generative_normalized",
        "gaussian_format": "3dgs_sh0_logscale_logitopacity_wxyz",
        "synthetic": True,
        "semantic_quality_verified": False,
    }
    atomic_json(metadata_path, metadata)
    return {
        "mesh_path": str(mesh_path),
        "gaussian_path": str(gaussian_path),
        "metadata_path": str(metadata_path),
        "gaussian_format": metadata["gaussian_format"],
        "foreground_mask_path": str(directory / "completed_mask.png"),
    }
