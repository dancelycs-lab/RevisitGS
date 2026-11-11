#!/usr/bin/env python3
"""Apply an encoder-side Sim(3) to a 3DGS PLY, including directional SH rotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        from splatreg.api import apply_transform
        from splatreg.io import load_ply, save_ply
    except ImportError as exc:
        raise RuntimeError(
            "Install splatreg from https://github.com/Archerkattri/splatreg to canonicalize SH fields."
        ) from exc
    record = json.loads(args.transform.read_text())
    values = record.get("ground_truth_rescan_to_reference", record)
    scale = float(values["scale"])
    rotation = np.asarray(values["rotation"], dtype=np.float32)
    translation = np.asarray(values["translation"], dtype=np.float32)
    matrix = torch.eye(4, dtype=torch.float32)
    matrix[:3, :3] = torch.from_numpy(scale * rotation)
    matrix[:3, 3] = torch.from_numpy(translation)
    save_ply(apply_transform(load_ply(args.input, device="cpu"), matrix, scale), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
