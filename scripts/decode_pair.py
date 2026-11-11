#!/usr/bin/env python3
"""Decode a RevisitGS revision stream and concatenate a decoded birth map."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revisitgs import concatenate, decode_revision, load_ply, save_ply  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", type=Path, required=True)
    parser.add_argument("--birth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opacity-threshold", type=float, default=0.02)
    args = parser.parse_args()

    source = load_ply(args.source, args.opacity_threshold)
    revised = decode_revision(args.revision.read_bytes(), source)
    decoded = concatenate(revised, load_ply(args.birth))
    save_ply(decoded, args.output)
    print(f"decoded_gaussians={len(decoded)} output={args.output}")


if __name__ == "__main__":
    main()
