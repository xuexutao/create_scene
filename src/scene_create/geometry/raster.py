from __future__ import annotations

import numpy as np

from scene_create.geometry.camera import validate_transform


def rasterize_depth(
    vertices, faces, intrinsics, world_from_camera, width, height, max_side=256
):
    """CPU pinhole z-buffer for alignment diagnostics, not appearance rendering."""
    vertices, faces = (
        np.asarray(vertices, dtype=float),
        np.asarray(faces, dtype=np.int64),
    )
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
    ):
        raise ValueError("Rasterization requires Nx3 vertices and Mx3 triangle indices")
    if (
        not np.isfinite(vertices).all()
        or (faces < 0).any()
        or (faces >= len(vertices)).any()
    ):
        raise ValueError("Invalid rasterization mesh")
    matrix = validate_transform(world_from_camera)
    scale = min(1.0, float(max_side) / max(width, height))
    target_w, target_h = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    sx, sy = target_w / width, target_h / height
    pixel_map = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    intrinsic = pixel_map @ np.asarray(intrinsics)
    camera = (vertices - matrix[:3, 3]) @ matrix[:3, :3]
    homogeneous = camera @ intrinsic.T
    pixels = np.zeros((len(vertices), 2))
    front = camera[:, 2] > 1e-6
    pixels[front] = homogeneous[front, :2] / homogeneous[front, 2:3]
    depth = np.full((target_h, target_w), np.inf, dtype=np.float64)
    for triangle in faces:
        if not front[triangle].all():
            continue
        xy = pixels[triangle]
        left, top = np.maximum(np.ceil(xy.min(axis=0)), [0, 0]).astype(int)
        right, bottom = np.minimum(
            np.floor(xy.max(axis=0)), [target_w - 1, target_h - 1]
        ).astype(int)
        if left > right or top > bottom:
            continue
        a, b, c = xy
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-12:
            continue
        y, x = np.mgrid[top : bottom + 1, left : right + 1]
        wa = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
        wb = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
        wc = 1 - wa - wb
        inside = (wa >= -1e-8) & (wb >= -1e-8) & (wc >= -1e-8)
        z = camera[triangle, 2]
        inverse_depth = wa / z[0] + wb / z[1] + wc / z[2]
        values = np.full(x.shape, np.inf)
        valid = inside & (inverse_depth > 0)
        values[valid] = 1 / inverse_depth[valid]
        tile = depth[top : bottom + 1, left : right + 1]
        np.minimum(tile, values, out=tile)
    return {
        "depth": depth,
        "mask": np.isfinite(depth),
        "pixel_map": pixel_map,
        "width": target_w,
        "height": target_h,
    }


def load_mesh_arrays(path, transform=None):
    import trimesh

    scene = trimesh.load(str(path), force="scene", process=False)
    parts = []
    for node in scene.graph.nodes_geometry:
        local, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        mesh.apply_transform(local)
        if transform is not None:
            mesh.apply_transform(transform)
        parts.append(mesh)
    if not parts:
        raise ValueError("No renderable triangle meshes")
    mesh = trimesh.util.concatenate(parts)
    return mesh.vertices, mesh.faces
