#!/usr/bin/env python3
"""Choose conditional or full-intra coding from measured rate and distortion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def value(record: dict, *names: str) -> float:
    for name in names:
        if name in record:
            return float(record[name])
    raise KeyError(f"None of {names} occurs in the evaluation record.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditional", type=Path, required=True)
    parser.add_argument("--intra", type=Path, required=True)
    parser.add_argument("--target-gaussians", type=int, required=True)
    parser.add_argument("--beta", type=float, default=4.232736869584208e-6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode-output", type=Path, required=True, help="one-byte mode/profile signal")
    parser.add_argument("--profile-id", type=int, default=0, choices=range(128))
    args = parser.parse_args()

    conditional, intra = load(args.conditional), load(args.intra)
    if args.target_gaussians <= 0:
        parser.error("--target-gaussians must be positive")
    if not math.isfinite(args.beta) or args.beta < 0:
        parser.error("--beta must be finite and nonnegative")
    for record in (conditional, intra):
        if "target_gaussians" in record and record["target_gaussians"] != args.target_gaussians:
            parser.error("Target population differs from the evaluation record")
        if record.get("cardinality_match") is False:
            parser.error("A decoded candidate does not preserve the target population")
    if "views" in conditional and "views" in intra:
        def view_ids(record):
            return sorted((int(v["index"]), str(v.get("camera", ""))) for v in record["views"])
        if not conditional["views"] or view_ids(conditional) != view_ids(intra):
            parser.error("Conditional and intra records must use the same nonempty view set")
    rows = {}
    for name, record in (("conditional", conditional), ("intra", intra)):
        stream_bytes = value(record, "total_bytes", "bitstream_bytes", "bytes")
        psnr = value(record, "mean_psnr_db", "psnr_db")
        if not math.isfinite(stream_bytes) or stream_bytes < 1 or not stream_bytes.is_integer():
            parser.error("Serialized bytes must be a positive integer including the mode byte")
        if not math.isfinite(psnr):
            parser.error("PSNR must be finite")
        mse = 10.0 ** (-psnr / 10.0)  # geometric mean of per-view MSE, not arithmetic MSE
        bpg = 8.0 * stream_bytes / args.target_gaussians
        rows[name] = {"bytes": int(stream_bytes), "psnr_db": psnr, "bpg": bpg,
                      "objective": mse + args.beta * bpg}
    selected = "conditional" if rows["conditional"]["objective"] < rows["intra"]["objective"] else "intra"
    args.mode_output.parent.mkdir(parents=True, exist_ok=True)
    args.mode_output.write_bytes(bytes([2 * args.profile_id + int(selected == "conditional")]))
    result = {"selected_mode": selected, "beta": args.beta, "modes": rows}
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
