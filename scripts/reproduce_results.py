#!/usr/bin/env python3
"""Regenerate manuscript-facing tables and plots from released result records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import subprocess
import sys
from audit_results import audit


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def write_summary(records: dict[str, dict], output: Path) -> None:
    rd = records["rd"]
    reuse = records["reuse"]
    external = records["external"]
    runtime = records["runtime"]
    alignment = records["alignment"]
    sh = records["sh"]
    recent_public = load("recent_public_codecs.json")["statistics"]
    recent_external = load("recent_external_codecs.json")["means"]
    lines = [
        "# Reproduced quantitative summary", "",
        "## Complete-cohort multi-rate results", "",
        "| Backend | Mode | BD-Rate (%) | BD-PSNR (dB) |", "|---|---:|---:|---:|",
    ]
    for backend in ("FCGS", "L-GSC"):
        row = rd[backend]
        lines.append(f"| {backend} | Raw conditional | {row['raw_mean_bd_rate_percent']:.3f} | {row['raw_mean_bd_psnr_db']:+.3f} |")
        lines.append(f"| {backend} | Selected streams | {row['mean_bd_rate_percent']:.3f} | {row['mean_bd_psnr_db']:+.3f} |")
    lines += ["", "## Reference-reuse controls", "",
              "| Configuration | Rate change (%) | MR PSNR gain (dB) | Positive/nonnegative |",
              "|---|---:|---:|---:|"]
    for row in reuse["variants"]:
        lines.append(f"| {row['variant'].replace('_', ' ')} | {row['mean_rate_change_percent']:+.3f} | {row['mean_matched_rate_psnr_gain_db']:+.3f} | {row['positive_psnr_pairs']}/{row['pairs']} |")
    row = reuse["global_selection"]
    lines.append(f"| global selection | {row['mean_rate_change_percent']:+.3f} | {row['mean_matched_rate_psnr_gain_db']:+.3f} | {row['nonnegative_psnr_pairs']}/{row['pairs']} |")
    lines += ["", "## External evaluation", "",
              "| Backend | Regime | Rate change (%) | PSNR change (dB) | SSIM change |",
              "|---|---|---:|---:|---:|"]
    for backend, values in external["backends"].items():
        for regime in ("clsplats_captured", "clsplats_target", "scannet_target"):
            row = values[regime]["selected"]
            lines.append(f"| {backend} | {regime.replace('clsplats_captured', 'CL-Splats reference RGB').replace('_', ' ')} | {row['mean_rate_change_percent']:+.3f} | {row['mean_psnr_change_db']:+.3f} | {row['mean_ssim_change']:+.6f} |")
    lines += ["", "## Recent public fixed-map codecs", "",
              "| Dataset | Method / point | Rate (bits/Gaussian) | PSNR (dB) | SSIM |",
              "|---|---|---:|---:|---:|"]
    for point, row in sorted(recent_public["FlexGaussian"].items(), key=lambda item: item[1]["rate_bpg"]):
        lines.append(f"| 3RScan | FlexGaussian {point} | {row['rate_bpg']:.3f} | {row['psnr_db']:.3f} | {row['ssim']:.6f} |")
    for name in ("RevisitGS_FCGS_selected_middle", "GSCodec_rp1"):
        row = recent_public[name]
        lines.append(f"| 3RScan | {name.replace('_', ' ')} | {row['rate_bpg']:.3f} | {row['psnr_db']:.3f} | {row['ssim']:.6f} |")
    for dataset, points in recent_external.items():
        for name, row in sorted(points.items(), key=lambda item: item[1]["rate_bpg"]):
            lines.append(f"| {dataset} | {name} | {row['rate_bpg']:.3f} | {row['psnr_db']:.3f} | {row['ssim']:.6f} |")
    lines += ["", "## Runtime and sensitivity", "",
              f"Mean selected-path encoding ratio: {runtime['encode_ratio']['mean']:.3f}x.",
              f"Mean conditional decoding ratio: {runtime['decode_ratio']['mean']:.3f}x.",
              f"Mean two-stream construction ratio (excluding decision evaluation): {runtime['dual_mode_encode_ratio']['mean']:.3f}x.",
              f"Mild alignment perturbation: {alignment['by_condition']['mild']['mean_psnr_change_db']:+.3f} dB.",
              f"Moderate alignment perturbation: {alignment['by_condition']['moderate']['mean_psnr_change_db']:+.3f} dB.",
              "", "## Directional-radiance rank", "",
              "| Rank | Total bytes | PSNR (dB) | Change from full rank (dB) |", "|---:|---:|---:|---:|",
    ]
    for row in sh["rows"]:
        lines.append(f"| {row['rank']} | {row['total_bytes']} | {row['mean_psnr_db']:.3f} | {row['psnr_change_vs_full_db']:+.3f} |")
    representation = load("representation_ablation.json")
    lines += ["", "## Complete-cohort representation ablations", "",
              "| Profile | Rate (bits/Gaussian) | PSNR (dB) | Matched-rate gain (dB) |",
              "|---|---:|---:|---:|"]
    for row in representation["profiles"]:
        lines.append(f"| {row['profile']} | {row['mean_rate_bpg']:.3f} | {row['mean_psnr_db']:.3f} | {row['mean_matched_rate_psnr_gain_db']:+.3f} |")
    views = {r["pair"]: r for r in load("independent_views.json")["rows"]}
    lines += ["", "## Runtime stage sums", "", "The sums combine separately measured stages, not a fused wall-time benchmark.", "",
              "| Target Gaussians (M) | Full intra encode (s) | Both encoders (s) | Decision evaluation (s) | Sum (s) |", "|---:|---:|---:|---:|---:|"]
    for row in sorted(runtime["rows"], key=lambda r: r["target_gaussians"]):
        v = views[row["pair"]]
        stage = sum(v[k] for k in ("decode_intra_wall_seconds", "decode_birth_wall_seconds", "core_decode_seconds", "selection_render_metric_seconds"))
        lines.append(f"| {row['target_gaussians']/1e6:.3f} | {row['fcgs_encode_seconds']:.2f} | {row['dual_mode_encode_seconds']:.2f} | {stage:.2f} | {stage+row['dual_mode_encode_seconds']:.2f} |")
    lines += ["", "## Independent-view evaluation", "",
              "These comparisons retain the original codec setting and mode decision; they are not matched-rate comparisons.", "",
              "| Backend | Mean PSNR change (dB) | 95% CI (dB) | Mean rate change (%) |",
              "|---|---:|---|---:|"]
    for backend, filename in (("FCGS", "independent_views.json"), ("L-GSC L1", "independent_views_lgsc.json")):
        row = load(filename)
        lo, hi = row["selected_gain_ci95"]
        lines.append(f"| {backend} | {row['mean_selected_gain_db']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {row['mean_selected_rate_change_percent']:+.3f} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audited = audit(RESULTS)
    (args.output / "metric_audit.json").write_text(json.dumps(audited, indent=2) + "\n")
    records = {
        "rd": load("multirate_rd.json"),
        "reuse": load("reference_reuse.json"),
        "external": load("external_multicodec.json"),
        "runtime": load("runtime.json"),
        "alignment": load("alignment_sensitivity.json"),
        "sh": load("sh_rank.json"),
    }
    for backend in ("FCGS", "L-GSC"):
        for mode, prefix in (("conditional", "raw_"), ("selected", "")):
            row = audited["multirate"][backend][mode]
            for key in ("mean_bd_rate_percent", "mean_bd_psnr_db"):
                records["rd"][backend][prefix + key] = row[key]
    subprocess.run([sys.executable, str(ROOT / "scripts/plot_results.py"), "--output", str(args.output)], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/plot_visual_comparisons.py"), "--output", str(args.output)], check=True)
    write_summary(records, args.output)
    for name in ("independent_views.json", "independent_views_lgsc.json"):
        views = load(name)
        (args.output / name).write_text(json.dumps({k: v for k, v in views.items() if k != "rows"}, indent=2) + "\n")
    print(f"Wrote reproduced tables and figures to {args.output}")


if __name__ == "__main__":
    main()
