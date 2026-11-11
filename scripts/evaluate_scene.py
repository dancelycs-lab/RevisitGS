#!/usr/bin/env python3
"""Render and evaluate one decoded Gaussian scene against its canonical target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revisitgs import load_ply  # noqa: E402
from revisitgs.render import image_metrics, make_camera, render  # noqa: E402


def payload_size(path: Path) -> int:
    """Count only an existing, nonempty serialized payload."""
    if not path.exists():
        raise FileNotFoundError(f"Bitstream does not exist: {path}")
    if path.is_file():
        size = path.stat().st_size
    elif path.is_dir():
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    else:
        raise ValueError(f"Not a payload file or directory: {path}")
    if size <= 0:
        raise ValueError(f"Bitstream is empty: {path}")
    return size


def transform_camera(record: dict, transform: dict | None) -> dict:
    if transform is None:
        return record
    values = transform.get("ground_truth_rescan_to_reference", transform)
    scale = float(values["scale"])
    rotation = np.asarray(values["rotation"], dtype=np.float32)
    translation = np.asarray(values["translation"], dtype=np.float32)
    result = dict(record)
    result["position"] = (
        scale * rotation @ np.asarray(record["position"]) + translation
    ).tolist()
    result["rotation"] = (
        rotation @ np.asarray(record["rotation"], dtype=np.float32)
    ).tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--decoded", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--transform", type=Path, help="optional Sim(3) JSON for native cameras")
    parser.add_argument("--views", type=int, default=4, help="0 evaluates every camera")
    parser.add_argument("--exclude-uniform-views", type=int, default=0, help="exclude this many uniformly spaced selection cameras before sampling")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--bitstream", type=Path, help="intra stream, or revision stream with --birth-bitstream")
    parser.add_argument("--birth-bitstream", type=Path, help="public birth payload for conditional evaluation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.birth_bitstream is not None and args.bitstream is None:
        parser.error("--birth-bitstream requires --bitstream")
    stream_bytes = None
    if args.bitstream is not None:
        stream_bytes = payload_size(args.bitstream) + 1  # mode/profile byte
        if args.birth_bitstream is not None:
            stream_bytes += payload_size(args.birth_bitstream)

    target, decoded = load_ply(args.target), load_ply(args.decoded)
    center = np.median(target.means, axis=0)
    extent = 2.0 * float(np.quantile(np.linalg.norm(target.means - center, axis=1), 0.95))
    cameras = json.loads(args.cameras.read_text())
    transform = None if args.transform is None else json.loads(args.transform.read_text())
    eligible = np.arange(len(cameras))
    if args.exclude_uniform_views > 0:
        excluded = np.unique(np.rint(np.linspace(0, len(cameras)-1, args.exclude_uniform_views)).astype(int))
        eligible = np.setdiff1d(eligible, excluded)
    if not len(eligible):
        raise ValueError("No cameras remain after exclusions")
    if args.views > 0:
        indices = eligible[np.unique(np.rint(np.linspace(0, len(eligible)-1, min(args.views,len(eligible)))).astype(int))]
    else:
        indices = eligible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for index in indices:
        record = transform_camera(cameras[int(index)], transform)
        camera = make_camera(record, extent, device)
        target_image = render(target, camera, args.width, args.height, device)
        decoded_image = render(decoded, camera, args.width, args.height, device)
        psnr, ssim = image_metrics(target_image, decoded_image)
        rows.append({"index": int(index), "camera": record.get("img_name", str(index)), "psnr_db": psnr, "ssim": ssim})
    summary = {
        "target_gaussians": len(target), "decoded_gaussians": len(decoded),
        "cardinality_match": len(target) == len(decoded),
        "bitstream_bytes": stream_bytes,
        "mean_psnr_db": float(np.mean([row["psnr_db"] for row in rows])),
        "mean_view_mse": float(np.mean([10.0 ** (-row["psnr_db"] / 10.0) for row in rows])),
        "psnr_derived_distortion": float(10.0 ** (-np.mean([row["psnr_db"] for row in rows]) / 10.0)),
        "mean_ssim": float(np.mean([row["ssim"] for row in rows])),
        "views": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "decoded_gaussians", "cardinality_match", "bitstream_bytes", "mean_psnr_db", "mean_ssim"
    )}, indent=2))


if __name__ == "__main__":
    main()
