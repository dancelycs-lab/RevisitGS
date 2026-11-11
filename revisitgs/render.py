"""Rasterization and image metrics used for decoded-scene evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from skimage.metrics import structural_similarity
import torch

from .gaussian import GaussianMap


@dataclass(frozen=True)
class Camera:
    view: torch.Tensor
    projection: torch.Tensor
    center: torch.Tensor
    tan_x: float
    tan_y: float


def make_camera(record: dict, extent: float, device: torch.device) -> Camera:
    camera_to_world = np.asarray(record["rotation"], dtype=np.float64)
    position = np.asarray(record["position"], dtype=np.float64)
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = camera_to_world.T
    world_to_camera[:3, 3] = -camera_to_world.T @ position
    view = torch.as_tensor(world_to_camera.T.copy(), dtype=torch.float32, device=device)
    fov_x = 2.0 * math.atan(float(record["width"]) / (2.0 * float(record["fx"])))
    fov_y = 2.0 * math.atan(float(record["height"]) / (2.0 * float(record["fy"])))
    near, far = 0.002 * extent, 15.0 * extent
    projection = torch.zeros((4, 4), dtype=torch.float32, device=device)
    projection[0, 0] = 1.0 / math.tan(fov_x / 2.0)
    projection[1, 1] = 1.0 / math.tan(fov_y / 2.0)
    projection[3, 2] = 1.0
    projection[2, 2] = far / (far - near)
    projection[2, 3] = -(far * near) / (far - near)
    return Camera(
        view=view,
        projection=view @ projection.T,
        center=torch.as_tensor(position, dtype=torch.float32, device=device),
        tan_x=math.tan(fov_x / 2.0),
        tan_y=math.tan(fov_y / 2.0),
    )


@torch.inference_mode()
def render(value: GaussianMap, camera: Camera, width: int, height: int, device) -> np.ndarray:
    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    except ImportError as exc:
        raise RuntimeError(
            "Install the standard 3DGS diff-gaussian-rasterization extension to evaluate images."
        ) from exc
    alpha = 1.0 / (1.0 + np.exp(-value.opacities))
    keep = (alpha > 0.01) & np.isfinite(value.means).all(axis=1)
    means = torch.as_tensor(value.means[keep], dtype=torch.float32, device=device)
    opacities = torch.as_tensor(alpha[keep, None], dtype=torch.float32, device=device)
    scales = torch.as_tensor(np.exp(value.scales[keep]), dtype=torch.float32, device=device)
    quats = torch.as_tensor(value.quats[keep], dtype=torch.float32, device=device)
    sh = torch.as_tensor(value.sh[keep], dtype=torch.float32, device=device)
    settings = dict(
        image_height=height, image_width=width,
        tanfovx=camera.tan_x, tanfovy=camera.tan_y,
        bg=torch.zeros(3, dtype=torch.float32, device=device),
        scale_modifier=1.0, viewmatrix=camera.view, projmatrix=camera.projection,
        sh_degree=int(round(math.sqrt(value.sh.shape[1]) - 1)),
        campos=camera.center, prefiltered=False, debug=False,
    )
    if hasattr(GaussianRasterizationSettings, "_fields") and "antialiasing" in GaussianRasterizationSettings._fields:
        settings["antialiasing"] = False
    rasterizer = GaussianRasterizer(GaussianRasterizationSettings(**settings))
    image = rasterizer(
        means3D=means,
        means2D=torch.zeros_like(means),
        shs=sh,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        rotations=quats,
        cov3D_precomp=None,
    )[0].clamp(0, 1)
    return image.permute(1, 2, 0).cpu().numpy()


def image_metrics(reference: np.ndarray, decoded: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((reference - decoded) ** 2))
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    ssim = structural_similarity(reference, decoded, channel_axis=2, data_range=1.0)
    return psnr, float(ssim)
