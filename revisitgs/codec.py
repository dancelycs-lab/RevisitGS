"""Final RevisitGS transport, allocation, revision coding, and decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

from .bitstream import (
    dequantize_q4,
    dequantize_quaternion,
    dequantize_symmetric,
    dequantize_xyz,
    deserialize,
    quantize_q4,
    quantize_quaternion,
    quantize_symmetric,
    quantize_xyz,
    quaternion_multiply,
    quaternion_to_rotvec,
    rotvec_to_quaternion,
    serialize,
)
from .gaussian import GaussianMap
from .matching import resolve


@dataclass(frozen=True)
class CodecConfig:
    radius_fraction: float = 0.01
    candidate_neighbors: int = 8
    transport_fraction: float = 0.50
    sh_rank: int = 4
    opacity_threshold: float = 0.02
    random_seed: int = 2026
    revision_profile: str = "default_rank4_q4"
    allocation_score: str = "full"


def canonical_target(source: GaussianMap, target: GaussianMap) -> GaussianMap:
    """Common finite-precision fidelity reference for conditional and full intra coding."""
    union = np.concatenate([source.means, target.means])
    low = np.quantile(union, .0001, axis=0).astype(np.float32)
    high = np.quantile(union, .9999, axis=0).astype(np.float32)
    return GaussianMap(
        dequantize_xyz(quantize_xyz(target.means, low, high), low, high),
        dequantize_quaternion(quantize_quaternion(target.quats)),
        target.scales.astype(np.float16).astype(np.float32),
        target.opacities.astype(np.float16).astype(np.float32),
        target.sh.astype(np.float16).astype(np.float32),
    )


def robust_extent(source: GaussianMap, target: GaussianMap) -> float:
    points = np.concatenate([source.means, target.means])
    center = np.median(points, axis=0)
    return 2.0 * float(np.quantile(np.linalg.norm(points - center, axis=1), 0.95))


def _candidate_assignment(
    source: GaussianMap, target: GaussianMap, radius: float, neighbors: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    distances, indices = cKDTree(target.means).query(
        source.means, k=neighbors, workers=-1
    )
    if neighbors == 1:
        distances, indices = distances[:, None], indices[:, None]
    edge_source = np.repeat(np.arange(len(source), dtype=np.int64), neighbors)
    edge_target = indices.reshape(-1).astype(np.int64)
    edge_distance = distances.reshape(-1)
    valid = (edge_distance <= radius) & (edge_target < len(target))
    edge_source = edge_source[valid]
    edge_target = edge_target[valid]
    edge_distance = edge_distance[valid]
    order = np.argsort(edge_distance, kind="stable")
    source_used, target_used, assignment = resolve(
        order, edge_source, edge_target, len(source), len(target)
    )
    return source_used, target_used, assignment, int(valid.sum())


def _allocate(
    source: GaussianMap,
    target: GaussianMap,
    source_used: np.ndarray,
    target_used: np.ndarray,
    assignment: np.ndarray,
    fraction: float,
    score: str = "full",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    source_indices = np.flatnonzero(source_used)
    target_indices = assignment[source_indices]
    if len(source_indices) == 0:
        return source_used, np.empty(0, dtype=np.int64), np.ones(len(target), dtype=bool), {
            "transport_candidates": 0, "transported_matches": 0,
            "birth_gaussians": len(target), "mean_selected_complexity": 0.0,
        }

    geometry = np.linalg.norm(
        target.means[target_indices] - source.means[source_indices], axis=1
    )
    dc = np.linalg.norm(
        target.sh[target_indices, 0] - source.sh[source_indices, 0], axis=1
    )
    shape = np.max(
        np.abs(target.scales[target_indices] - source.scales[source_indices]), axis=1
    )
    opacity = np.abs(
        target.opacities[target_indices] - source.opacities[source_indices]
    )
    quaternion_dot = np.abs(np.sum(
        source.quats[source_indices] * target.quats[target_indices], axis=1
    ))
    rotation = 2.0 * np.arccos(np.clip(quaternion_dot, 0.0, 1.0))

    if score == "geometry":
        components, weights = (geometry,), (1.0,)
    elif score == "full":
        components, weights = (geometry, dc, shape, opacity, rotation), (1.0, .75, .75, .25, .5)
    else:
        raise ValueError(f"Unknown allocation score: {score}")
    difficulty = np.zeros(len(source_indices), dtype=np.float64)
    for values, weight in zip(components, weights):
        difficulty += weight * values / max(float(np.median(values)), 1e-8)
    retain_count = max(1, int(round(fraction * len(source_indices))))
    # Serialize predictors and their matched targets in source-index order.
    selected = np.sort(np.argsort(difficulty, kind="stable")[:retain_count])
    kept_source = source_indices[selected]
    matched_target = target_indices[selected]
    birth_mask = ~target_used.copy()
    rejected = np.ones(len(source_indices), dtype=bool)
    rejected[selected] = False
    birth_mask[target_indices[rejected]] = True
    keep_mask = np.zeros(len(source), dtype=bool)
    keep_mask[kept_source] = True
    return keep_mask, matched_target, birth_mask, {
        "transport_candidates": int(len(source_indices)),
        "transported_matches": int(len(kept_source)),
        "birth_gaussians": int(birth_mask.sum()),
        "mean_selected_complexity": float(np.mean(difficulty[selected])),
    }


def _encode_directional(residual: np.ndarray, rank: int, seed: int, precision: int = 4, nested_rank: int | None = None) -> dict[str, np.ndarray]:
    directional = residual[:, 1:, :].reshape(len(residual), -1).astype(np.float32)
    rank = min(rank, directional.shape[1], max(0, len(directional) - 1))
    if rank == 0:
        return {
            "revision_sh_mean": directional.mean(axis=0).astype(np.float16),
            "revision_sh_basis": np.zeros((0, directional.shape[1]), np.float16),
            "revision_sh_constant": np.asarray([1], np.uint8),
            "sh_coefficients": np.asarray([residual.shape[1]], np.int16),
        }
    fit_count = min(len(directional), 100000)
    fit_indices = np.linspace(0, len(directional) - 1, fit_count).astype(np.int64)
    model = PCA(n_components=rank, svd_solver="randomized", random_state=seed)
    model.fit(directional[fit_indices])
    coefficients = model.transform(directional).astype(np.float32)
    basis = model.components_
    if nested_rank is not None:
        coefficients = coefficients[:, :nested_rank].copy()
        basis = basis[:nested_rank].copy()
    if precision == 8:
        codes, step = quantize_symmetric(coefficients, 127)
        return {"revision_sh_mean": model.mean_.astype(np.float16),
                "revision_sh_basis": basis.astype(np.float16),
                "revision_sh_codes": codes, "revision_sh_step": step}
    packed, shape, step = quantize_q4(coefficients)
    return {
        "revision_sh_mean": model.mean_.astype(np.float16),
        "revision_sh_basis": basis.astype(np.float16),
        "revision_sh_q4": packed,
        "revision_sh_q4_shape": shape,
        "revision_sh_q4_step": step.astype(np.float32),
    }


def _decode_directional(arrays: dict[str, np.ndarray], count: int) -> np.ndarray:
    if "revision_sh_constant" in arrays:
        return np.broadcast_to(arrays["revision_sh_mean"].astype(np.float32),
                               (count, len(arrays["revision_sh_mean"]))).reshape(count, -1, 3).copy()
    if "revision_sh_codes" in arrays:
        coefficients = dequantize_symmetric(arrays["revision_sh_codes"], arrays["revision_sh_step"])
    else:
        coefficients = dequantize_q4(
            arrays["revision_sh_q4"], arrays["revision_sh_q4_shape"], arrays["revision_sh_q4_step"],
        )
    directional = (
        coefficients @ arrays["revision_sh_basis"].astype(np.float32)
        + arrays["revision_sh_mean"].astype(np.float32)
    )
    coefficients_count = arrays["revision_sh_mean"].size // 3 + 1
    return directional.reshape(count, coefficients_count - 1, 3)


def encode_revision(
    source: GaussianMap, target: GaussianMap, config: CodecConfig = CodecConfig()
) -> tuple[bytes, GaussianMap, dict]:
    profiles = {"default_rank4_q4", "without_directional_revision", "rank4_q8", "uniform_q8",
                "nested_rank2_q4", "without_scale_revision", "without_opacity_revision", "without_rotation_revision"}
    if config.revision_profile not in profiles:
        raise ValueError(f"Unknown revision profile: {config.revision_profile}")
    if source.sh.shape[1:] != target.sh.shape[1:]:
        raise ValueError("Source and target must use the same spherical-harmonic degree.")
    extent = robust_extent(source, target)
    radius = config.radius_fraction * extent
    source_used, target_used, assignment, candidate_edges = _candidate_assignment(
        source, target, radius, config.candidate_neighbors
    )
    keep, matched, birth_mask, allocation = _allocate(
        source, target, source_used, target_used, assignment, config.transport_fraction, config.allocation_score
    )
    source_indices = np.flatnonzero(keep)

    if not len(source_indices):
        payload = serialize({"schema": np.asarray([4], np.uint8),
                             "source_count": np.asarray([len(source)], np.int64),
                             "empty_revision": np.asarray([1], np.uint8)})
        return payload, target.subset(np.arange(len(target))), {
            "configuration": asdict(config), "source_gaussians": len(source),
            "target_gaussians": len(target), "revision_bytes": len(payload),
            "recommended_mode": "intra", **allocation,
        }

    mean_delta = target.means[matched] - source.means[source_indices]
    scale_delta = target.scales[matched] - source.scales[source_indices]
    opacity_delta = (target.opacities[matched] - source.opacities[source_indices])[:, None]
    sh_delta = target.sh[matched] - source.sh[source_indices]

    mean_q4, mean_shape, mean_step = quantize_q4(mean_delta)
    dc_codes, dc_step = quantize_symmetric(sh_delta[:, 0, :], 127)
    scale_codes, scale_step = quantize_symmetric(scale_delta, 127)
    opacity_codes, opacity_step = quantize_symmetric(opacity_delta, 127)

    source_inverse = source.quats[source_indices].copy()
    source_inverse[:, 1:] *= -1.0
    canonical_quats = dequantize_quaternion(quantize_quaternion(target.quats[matched]))
    relative = quaternion_multiply(canonical_quats, source_inverse)
    rotvec = quaternion_to_rotvec(relative)
    rot_step = np.asarray([np.pi / 127.0], dtype=np.float32)
    rot_codes = np.rint(np.clip(rotvec / rot_step[0], -127.0, 127.0)).astype(np.int8)

    arrays = {
        "schema": np.asarray([3], dtype=np.uint8),
        "source_count": np.asarray([len(source)], dtype=np.int64),
        "keep_bits": np.packbits(keep, bitorder="little"),
        "revision_bits": np.packbits(np.ones(len(source_indices), dtype=bool), bitorder="little"),
        "revision_mean_q4": mean_q4,
        "revision_mean_q4_shape": mean_shape,
        "revision_mean_q4_step": mean_step.astype(np.float32),
        "revision_dc_codes": dc_codes,
        "revision_dc_step": dc_step.astype(np.float32),
        "revision_scale_codes": scale_codes,
        "revision_scale_step": scale_step.astype(np.float32),
        "revision_opacity_codes": opacity_codes,
        "revision_opacity_step": opacity_step.astype(np.float32),
    }
    profile = config.revision_profile
    if profile == "uniform_q8":
        for key in ("revision_mean_q4", "revision_mean_q4_shape", "revision_mean_q4_step"):
            arrays.pop(key)
        arrays["revision_mean_codes"], arrays["revision_mean_step"] = quantize_symmetric(mean_delta, 127)
    if profile != "without_directional_revision":
        arrays.update(_encode_directional(sh_delta, config.sh_rank, config.random_seed,
                                         precision=8 if profile in ("rank4_q8", "uniform_q8") else 4,
                                         nested_rank=2 if profile == "nested_rank2_q4" else None))
    arrays.update(revision_rotvec_q8=rot_codes, revision_rotvec_q8_step=rot_step)
    for label, group in (("without_scale_revision", "scale"), ("without_opacity_revision", "opacity"),
                         ("without_rotation_revision", "rotvec_q8")):
        if profile == label:
            arrays.pop(f"revision_{group}_codes" if group != "rotvec_q8" else "revision_rotvec_q8")
            arrays.pop(f"revision_{group}_step")
    payload = serialize(arrays)

    union = np.concatenate([source.means, target.means])
    low = np.quantile(union, 0.0001, axis=0).astype(np.float32)
    high = np.quantile(union, 0.9999, axis=0).astype(np.float32)
    birth_indices = np.flatnonzero(birth_mask)
    birth = GaussianMap(
        dequantize_xyz(quantize_xyz(target.means[birth_indices], low, high), low, high),
        dequantize_quaternion(quantize_quaternion(target.quats[birth_indices])),
        target.scales[birth_indices].astype(np.float16).astype(np.float32),
        target.opacities[birth_indices].astype(np.float16).astype(np.float32),
        target.sh[birth_indices].astype(np.float16).astype(np.float32),
    )
    metadata = {
        "configuration": asdict(config),
        "source_gaussians": len(source),
        "target_gaussians": len(target),
        "scene_extent": extent,
        "candidate_radius": radius,
        "candidate_edges": candidate_edges,
        "revision_bytes": len(payload),
        **allocation,
    }
    return payload, birth, metadata


def decode_revision(payload: bytes, source: GaussianMap) -> GaussianMap:
    arrays = deserialize(payload)
    source_count = int(arrays["source_count"][0])
    if source_count != len(source):
        raise ValueError("The stored map does not match the revision stream.")
    if "empty_revision" in arrays:
        return source.subset(np.empty(0, dtype=np.int64))
    keep = np.unpackbits(
        arrays["keep_bits"], count=source_count, bitorder="little"
    ).astype(bool)
    retained = source.subset(np.flatnonzero(keep))
    revise = np.unpackbits(
        arrays["revision_bits"], count=len(retained), bitorder="little"
    ).astype(bool)
    index = np.flatnonzero(revise)
    if "revision_mean_q4" in arrays:
        retained.means[index] += dequantize_q4(arrays["revision_mean_q4"], arrays["revision_mean_q4_shape"], arrays["revision_mean_q4_step"])
    elif "revision_mean_codes" in arrays:
        retained.means[index] += dequantize_symmetric(arrays["revision_mean_codes"], arrays["revision_mean_step"])
    retained.sh[index, 0, :] += dequantize_symmetric(arrays["revision_dc_codes"], arrays["revision_dc_step"])
    if "revision_scale_codes" in arrays:
        retained.scales[index] += dequantize_symmetric(arrays["revision_scale_codes"], arrays["revision_scale_step"])
    if "revision_opacity_codes" in arrays:
        retained.opacities[index] += dequantize_symmetric(arrays["revision_opacity_codes"], arrays["revision_opacity_step"]).reshape(-1)
    if "revision_sh_mean" in arrays:
        retained.sh[index, 1:, :] += _decode_directional(arrays, len(index))
    if "revision_rotvec_q8" in arrays:
        relative = rotvec_to_quaternion(arrays["revision_rotvec_q8"].astype(np.float32) * float(arrays["revision_rotvec_q8_step"][0]))
        quaternions = quaternion_multiply(relative, retained.quats[index])
        retained.quats[index] = quaternions / np.maximum(np.linalg.norm(quaternions, axis=1, keepdims=True), 1e-8)
    return retained
