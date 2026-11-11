#!/usr/bin/env python3
"""Check the tested FCGS revision and apply its PyTorch C++17 build compatibility."""
from pathlib import Path
import argparse
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fcgs", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads((ROOT / "configs/codec_versions.json").read_text())["FCGS"]["commit"]
    actual = subprocess.check_output(["git", "-C", str(args.fcgs), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise SystemExit(f"Expected FCGS commit {expected}, found {actual}")
    for module in ("freqencoder", "gridcreater", "gridencoder"):
        path = args.fcgs / "submodules" / module / "setup.py"
        text = path.read_text()
        changed = text.replace("-std=c++14", "-std=c++17")
        if changed != text:
            path.write_text(changed)
    print("FCGS revision verified; CUDA-extension build flags use C++17. Coding sources unchanged.")


if __name__ == "__main__":
    main()
