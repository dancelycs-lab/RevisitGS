#!/usr/bin/env python3
"""Encode one aligned Gaussian-map pair with the final RevisitGS configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revisitgs import CodecConfig, encode_revision, load_ply, save_ply  # noqa: E402
from revisitgs.codec import canonical_target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="stored canonical 3DGS PLY")
    parser.add_argument("--target", type=Path, required=True, help="aligned revisit 3DGS PLY")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-fraction", type=float, default=0.01)
    parser.add_argument("--candidate-neighbors", type=int, default=8)
    parser.add_argument("--transport-fraction", type=float, default=0.50)
    parser.add_argument("--sh-rank", type=int, default=4)
    parser.add_argument("--allocation-score", choices=("full", "geometry"), default="full")
    parser.add_argument("--revision-profile", default="default_rank4_q4", help="Default or a fixed representation-ablation profile")
    parser.add_argument("--opacity-threshold", type=float, default=0.02)
    args = parser.parse_args()

    config = CodecConfig(
        radius_fraction=args.radius_fraction,
        candidate_neighbors=args.candidate_neighbors,
        transport_fraction=args.transport_fraction,
        sh_rank=args.sh_rank,
        opacity_threshold=args.opacity_threshold,
        revision_profile=args.revision_profile,
        allocation_score=args.allocation_score,
    )
    source = load_ply(args.source, config.opacity_threshold)
    target = load_ply(args.target, config.opacity_threshold)
    payload, birth, metadata = encode_revision(source, target, config)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "revision.bin").write_bytes(payload)
    save_ply(birth, args.output / "birth_input.ply")
    canonical = canonical_target(source, target)
    save_ply(canonical, args.output / "target_canonical.ply")
    metadata.update({
        "source": str(args.source),
        "target": str(args.target),
        "revision_stream": "revision.bin",
        "birth_input": "birth_input.ply",
    })
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
