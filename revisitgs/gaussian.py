"""Minimal INRIA 3DGS PLY I/O used by the released RevisitGS codec."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


@dataclass
class GaussianMap:
    means: np.ndarray
    quats: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    sh: np.ndarray

    def __len__(self) -> int:
        return int(len(self.means))

    def subset(self, indices: np.ndarray) -> "GaussianMap":
        return GaussianMap(
            self.means[indices].copy(), self.quats[indices].copy(),
            self.scales[indices].copy(), self.opacities[indices].copy(),
            self.sh[indices].copy(),
        )


def _stack(vertex, names: list[str]) -> np.ndarray:
    return np.stack([np.asarray(vertex[name], dtype=np.float32) for name in names], axis=1)


def load_ply(path: str | Path, opacity_threshold: float | None = None) -> GaussianMap:
    vertex = PlyData.read(str(path))["vertex"].data
    names = vertex.dtype.names or ()
    means = _stack(vertex, ["x", "y", "z"])
    quats = _stack(vertex, [f"rot_{i}" for i in range(4)])
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    scales = _stack(vertex, [f"scale_{i}" for i in range(3)])
    opacities = np.asarray(vertex["opacity"], dtype=np.float32)
    dc = _stack(vertex, [f"f_dc_{i}" for i in range(3)])[:, None, :]
    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if rest_names:
        rest = _stack(vertex, rest_names)
        if rest.shape[1] % 3:
            raise ValueError("The number of higher-order SH fields must be divisible by three.")
        rest = rest.reshape(len(rest), 3, rest.shape[1] // 3).transpose(0, 2, 1)
        sh = np.concatenate([dc, rest], axis=1)
    else:
        sh = dc
    result = GaussianMap(means, quats, scales, opacities, sh)
    if opacity_threshold is not None:
        alpha = 1.0 / (1.0 + np.exp(-opacities))
        result = result.subset(np.flatnonzero(alpha >= opacity_threshold))
    return result


def save_ply(value: GaussianMap, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[tuple[str, np.ndarray]] = [
        ("x", value.means[:, 0]), ("y", value.means[:, 1]), ("z", value.means[:, 2]),
        ("nx", np.zeros(len(value), np.float32)),
        ("ny", np.zeros(len(value), np.float32)),
        ("nz", np.zeros(len(value), np.float32)),
        ("f_dc_0", value.sh[:, 0, 0]),
        ("f_dc_1", value.sh[:, 0, 1]),
        ("f_dc_2", value.sh[:, 0, 2]),
    ]
    if value.sh.shape[1] > 1:
        rest = value.sh[:, 1:, :].transpose(0, 2, 1).reshape(len(value), 3 * (value.sh.shape[1] - 1))
        fields.extend((f"f_rest_{i}", rest[:, i]) for i in range(rest.shape[1]))
    fields.extend([
        ("opacity", value.opacities),
        ("scale_0", value.scales[:, 0]),
        ("scale_1", value.scales[:, 1]),
        ("scale_2", value.scales[:, 2]),
        ("rot_0", value.quats[:, 0]),
        ("rot_1", value.quats[:, 1]),
        ("rot_2", value.quats[:, 2]),
        ("rot_3", value.quats[:, 3]),
    ])
    dtype = [(name, "<f4") for name, _ in fields]
    array = np.empty(len(value), dtype=dtype)
    for name, column in fields:
        array[name] = np.asarray(column, dtype=np.float32)
    PlyData([PlyElement.describe(array, "vertex")], text=False).write(str(path))


def concatenate(first: GaussianMap, second: GaussianMap) -> GaussianMap:
    if first.sh.shape[1:] != second.sh.shape[1:]:
        raise ValueError("Both maps must use the same SH degree.")
    return GaussianMap(
        np.concatenate([first.means, second.means]),
        np.concatenate([first.quats, second.quats]),
        np.concatenate([first.scales, second.scales]),
        np.concatenate([first.opacities, second.opacities]),
        np.concatenate([first.sh, second.sh]),
    )
