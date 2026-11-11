#!/usr/bin/env python3
"""Prepare RevisitGS revision and birth inputs for a JSON-defined map-pair cohort."""

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
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius-fraction", type=float, default=0.01)
    parser.add_argument("--candidate-neighbors", type=int, default=8)
    parser.add_argument("--transport-fraction", type=float, default=0.50)
    parser.add_argument("--sh-rank", type=int, default=4)
    parser.add_argument("--revision-profile", default="default_rank4_q4")
    parser.add_argument("--allocation-score", choices=("full", "geometry"), default="full")
    parser.add_argument("--opacity-threshold", type=float, default=0.02)
    args = parser.parse_args()

    entries = json.loads(args.pairs.read_text())
    if not isinstance(entries, list):
        raise ValueError("The pair manifest must be a JSON array.")
    config = CodecConfig(
        args.radius_fraction, args.candidate_neighbors, args.transport_fraction,
        args.sh_rank, args.opacity_threshold,
        revision_profile=args.revision_profile, allocation_score=args.allocation_score,
    )
    cohort = []
    for entry in entries:
        name = str(entry["name"])
        output = args.output / name
        source = load_ply(entry["source"], config.opacity_threshold)
        target = load_ply(entry["target"], config.opacity_threshold)
        payload, birth, metadata = encode_revision(source, target, config)
        output.mkdir(parents=True, exist_ok=True)
        (output / "revision.bin").write_bytes(payload)
        save_ply(birth, output / "birth_input.ply")
        save_ply(canonical_target(source, target), output / "target_canonical.ply")
        metadata.update({"name": name, "source": entry["source"], "target": entry["target"]})
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        cohort.append(metadata)
        print(f"{name}: revisions={metadata['transported_matches']} births={metadata['birth_gaussians']}")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cohort_summary.json").write_text(json.dumps(cohort, indent=2) + "\n")


if __name__ == "__main__":
    main()
