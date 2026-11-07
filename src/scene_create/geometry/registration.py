from __future__ import annotations

from itertools import permutations, product

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from scene_create.geometry.camera import validate_transform


def estimate_similarity(source, target, with_scale=True, weights=None):
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if (
        source.shape != target.shape
        or source.ndim != 2
        or source.shape[1] != 3
        or len(source) < 3
    ):
        raise ValueError("Similarity estimation needs at least three paired 3D points")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Similarity inputs must be finite")
    weights = (
        np.ones(len(source)) if weights is None else np.asarray(weights, dtype=float)
    )
    if (
        weights.shape != (len(source),)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("Invalid similarity weights")
    weights = weights / weights.sum()
    source_center, target_center = weights @ source, weights @ target
    source_zero, target_zero = source - source_center, target - target_center
    covariance = (target_zero * weights[:, None]).T @ source_zero
    u, singular, vt = np.linalg.svd(covariance)
    if np.count_nonzero(singular > max(singular[0], 1e-15) * 1e-10) < 2:
        raise ValueError(
            "Collinear or coincident correspondences do not determine a 3D pose"
        )
    sign = np.ones(3)
    sign[-1] = 1 if np.linalg.det(u @ vt) >= 0 else -1
    rotation = (u * sign) @ vt
    variance = np.sum(weights * np.sum(source_zero**2, axis=1))
    if variance < 1e-15:
        raise ValueError("Source variance is zero")
    scale = float(np.dot(singular, sign) / variance) if with_scale else 1.0
    if scale <= 0:
        raise ValueError("Similarity scale must be positive")
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = target_center - matrix[:3, :3] @ source_center
    return matrix


def sample_mesh(mesh_path, count=20000, seed=0):
    import trimesh

    if count < 3:
        raise ValueError("At least three mesh samples are required")
    loaded = trimesh.load(str(mesh_path), force="scene", process=False)
    parts = []
    for node in loaded.graph.nodes_geometry:
        transform, name = loaded.graph[node]
        mesh = loaded.geometry[name].copy()
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        mesh.apply_transform(transform)
        parts.append(mesh)
    if not parts:
        raise ValueError("Asset contains no triangle mesh")
    mesh = trimesh.util.concatenate(parts)
    areas = np.asarray(mesh.area_faces)
    if (
        not np.isfinite(mesh.vertices).all()
        or not np.isfinite(areas).all()
        or areas.sum() <= 0
    ):
        raise ValueError("Mesh geometry is empty, degenerate or non-finite")
    random = np.random.default_rng(seed)
    faces = random.choice(len(areas), size=int(count), p=areas / areas.sum())
    triangles = mesh.triangles[faces]
    a, b = np.sqrt(random.random((2, count)))
    b = b * b
    points = (
        (1 - a[:, None]) * triangles[:, 0]
        + (a * (1 - b))[:, None] * triangles[:, 1]
        + (a * b)[:, None] * triangles[:, 2]
    )
    return {"points": points, "normals": mesh.face_normals[faces]}


def _apply(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _basis(points):
    center = np.median(points, axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    basis = vt.T
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return center, basis


def _size(points):
    return float(
        np.linalg.norm(
            np.quantile(points, 0.95, axis=0) - np.quantile(points, 0.05, axis=0)
        )
    )


def _quality(source, target, matrix, radius, trim):
    transformed = _apply(source, matrix)
    distances = cKDTree(transformed).query(target, k=1)[0]
    cutoff = np.quantile(distances, trim)
    trimmed = distances[distances <= cutoff + 1e-12]
    rmse = float(np.sqrt(np.mean(trimmed**2)))
    coverage = float(np.mean(distances <= radius))
    source_coverage = float(
        np.mean(cKDTree(target).query(transformed, k=1)[0] <= radius)
    )
    return rmse, coverage, source_coverage


def _icp(source, target, initial, scale_bounds, iterations, trim=0.9, with_scale=True):
    matrix = initial.copy()
    last_error = np.inf
    for _ in range(iterations):
        transformed = _apply(source, matrix)
        distances, matches = cKDTree(transformed).query(target, k=1)
        threshold = np.quantile(distances, trim)
        keep = distances <= max(threshold, 1e-10)
        if keep.sum() < 6:
            break
        local_distances = distances[keep]
        robust = 1 / (
            1 + (local_distances / max(np.median(local_distances), 1e-8)) ** 2
        )
        try:
            increment = estimate_similarity(
                transformed[matches[keep]],
                target[keep],
                with_scale=with_scale,
                weights=robust,
            )
        except ValueError:
            break
        proposal = increment @ matrix
        scale = np.cbrt(np.linalg.det(proposal[:3, :3]))
        if not scale_bounds[0] <= scale <= scale_bounds[1]:
            increment = estimate_similarity(
                transformed[matches[keep]],
                target[keep],
                with_scale=False,
                weights=robust,
            )
            proposal = increment @ matrix
        matrix = proposal
        error = float(np.mean(local_distances**2))
        if np.isfinite(last_error) and abs(last_error - error) < max(
            1e-14, last_error * 1e-7
        ):
            break
        last_error = error
    return matrix


def register_points(source, target, config=None):
    """Robust observed-target to full-source Sim(3), followed by rigid ICP.

    Config: coarse_points=800, fine_points=6000, coarse_iterations=10,
    iterations=60, trim_fraction=.9, scale_candidates=(.75,1,1.4),
    min_fitness=.5, max_rmse_ratio=.04, distance_ratio=.06, seed=0.
    Scores are geometric diagnostics, not proof of semantic identity.
    """
    options = dict(config or {})
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    for name, points in (("source", source), ("target", target)):
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or len(points) < 6
            or not np.isfinite(points).all()
        ):
            raise ValueError(f"Registration {name} needs at least six finite 3D points")
        if np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-8) < 2:
            raise ValueError(f"Registration {name} is geometrically degenerate")
    random = np.random.default_rng(options.get("seed", 0))

    def subset(points, count):
        return (
            points
            if len(points) <= count
            else points[random.choice(len(points), count, replace=False)]
        )

    fine_source = subset(source, options.get("fine_points", 6000))
    fine_target = subset(target, options.get("fine_points", 6000))
    coarse_source = subset(fine_source, options.get("coarse_points", 800))
    coarse_target = subset(fine_target, options.get("coarse_points", 800))
    source_size, target_size = _size(source), _size(target)
    if min(source_size, target_size) < 1e-9:
        raise ValueError("Registration size is zero")
    base_scale = target_size / source_size
    lower, upper = base_scale * 0.4, base_scale * 3.5
    trim = float(options.get("trim_fraction", 0.9))
    if not 0.5 <= trim <= 1:
        raise ValueError("trim_fraction must be in [0.5, 1]")
    radius = target_size * options.get("distance_ratio", 0.06)
    source_center, source_basis = _basis(coarse_source)
    target_center, target_basis = _basis(coarse_target)
    candidates = []
    if options.get("initial_transform") is not None:
        initial = validate_transform(options["initial_transform"], allow_scale=True)
        candidates.append(initial)
    for order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            axes = np.eye(3)[:, order] * np.asarray(signs)
            if np.linalg.det(axes) < 0:
                continue
            rotation = target_basis @ axes @ source_basis.T
            for multiplier in options.get("scale_candidates", [0.75, 1.0, 1.4]):
                initial = np.eye(4)
                initial[:3, :3] = base_scale * multiplier * rotation
                initial[:3, 3] = target_center - initial[:3, :3] @ source_center
                candidates.append(initial)
    ranked = []
    for initial in candidates:
        matrix = _icp(
            coarse_source,
            coarse_target,
            initial,
            (lower, upper),
            options.get("coarse_iterations", 10),
            trim,
        )
        rmse, fitness, support = _quality(
            coarse_source, coarse_target, matrix, radius, trim
        )
        rank = rmse / target_size + 0.02 * (1 - fitness) + 0.005 * (1 - support)
        ranked.append((rank, matrix))
    fine = []
    for _, initial in sorted(ranked, key=lambda value: value[0])[
        : options.get("refine_candidates", 4)
    ]:
        matrix = _icp(
            fine_source,
            fine_target,
            initial,
            (lower, upper),
            options.get("iterations", 60),
            trim,
        )
        matrix = _icp(
            fine_source, fine_target, matrix, (lower, upper), 20, trim, with_scale=False
        )
        rmse, fitness, support = _quality(
            fine_source, fine_target, matrix, radius, trim
        )
        fine.append(
            (
                rmse / target_size + 0.02 * (1 - fitness) + 0.005 * (1 - support),
                matrix,
                rmse,
                fitness,
                support,
            )
        )
    fine.sort(key=lambda value: value[0])
    _, transform, rmse, fitness, support = fine[0]
    validate_transform(transform, allow_scale=True)
    scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
    rotation = transform[:3, :3] / scale
    metrics = {
        "rmse": rmse,
        "rmse_ratio": rmse / target_size,
        "fitness": fitness,
        "observed_surface_coverage": fitness,
        "source_support_fraction": support,
        "target_size": target_size,
        "correspondence_distance": radius,
        "method": "pca_multistart_sim3_trimmed_observation_to_asset_icp",
        "orientation_ambiguity_checked": False,
        "silhouette_checked": False,
    }
    accepted = (
        fitness >= options.get("min_fitness", 0.5)
        and rmse / target_size <= options.get("max_rmse_ratio", 0.04)
        and support >= options.get("min_source_support", 0.03)
    )
    return {
        "transform": transform,
        "scale": scale,
        "rotation_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(),
        "translation_xyz": transform[:3, 3].tolist(),
        "metrics": metrics,
        "accepted": bool(accepted),
    }
