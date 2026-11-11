#!/usr/bin/env python3
"""Recalculate statistics from per-pair measured records, without GPU or datasets."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "revisitgs"))
from rd_metrics import bd_metrics, matched_quality, bootstrap_ci


def close(actual, expected, name):
    if not np.allclose(actual, expected, atol=1e-9, rtol=1e-9):
        raise AssertionError(f"{name}: recalculated {actual}, saved {expected}")


def audit_multirate(data):
    result = {}
    beta = data["beta"]
    for backend in ("FCGS", "L-GSC"):
        original = data[backend]
        values = {"conditional": [], "selected": []}
        modes = []
        for row in original["rows"]:
            selected, pair_modes = [], []
            for intra, conditional, saved in zip(row["baseline"], row["conditional"], row["selected"]):
                objective = lambda p: 10 ** (-p["psnr_db"] / 10) + beta * 8 * p["bytes"] / row["target_gaussians"]
                use = objective(conditional) < objective(intra)
                choice = conditional if use else intra
                for field in ("bytes", "psnr_db", "ssim"):
                    close(choice[field], saved[field], f"{backend}/{row['pair']}/{field}")
                selected.append(choice)
                pair_modes.append(use)
            modes.append(pair_modes)
            for mode, points in (("conditional", row["conditional"]), ("selected", selected)):
                bd = bd_metrics(row["baseline"], points)
                if bd is not None:
                    values[mode].append(bd)
        summary = {}
        for mode, prefix in (("conditional", "raw_"), ("selected", "")):
            numbers = np.asarray(values[mode])
            rate, quality = numbers.mean(0)
            close(rate, original[prefix + "mean_bd_rate_percent"], backend + mode + " BD-rate")
            close(quality, original[prefix + "mean_bd_psnr_db"], backend + mode + " BD-PSNR")
            if mode == "conditional":
                rate_ci = bootstrap_ci(numbers[:, 0], 2028)
                quality_ci = bootstrap_ci(numbers[:, 1], 2038)
                rate_ci_key = "raw_mean_bd_rate_95ci_percent"
                quality_ci_key = "raw_mean_bd_psnr_95ci_db"
            else:
                rate_ci = bootstrap_ci(numbers[:, 0], 940)
                quality_ci = bootstrap_ci(numbers[:, 1], 941)
                rate_ci_key = "bd_rate_ci95"
                quality_ci_key = "bd_psnr_ci95"
            close(rate_ci, original[rate_ci_key], backend + mode + " BD-rate CI")
            close(quality_ci, original[quality_ci_key], backend + mode + " BD-PSNR CI")
            summary[mode] = {"valid_pairs": len(numbers), "mean_bd_rate_percent": float(rate),
                             "mean_bd_psnr_db": float(quality),
                             "bd_rate_ci95": rate_ci,
                             "bd_psnr_ci95": quality_ci}
        counts = np.asarray(modes).sum(0).tolist()
        close(counts, original["conditional_counts"], backend + " mode counts")
        summary["conditional_counts"] = counts
        result[backend] = summary
    return result


def audit_reuse(data):
    result = {}
    for index, group in enumerate(data["variants"] + [data["global_selection"]]):
        quality, structural, rates = [], [], []
        for row in group["rows"]:
            curve = row["baseline"]
            x = [p["bytes"] for p in curve]
            q = matched_quality(row["bytes"], x, [p["psnr_db"] for p in curve])
            s = matched_quality(row["bytes"], x, [p["ssim"] for p in curve])
            quality.append(row["psnr_db"] - q)
            structural.append(row["ssim"] - s)
            rates.append(100 * (row["bytes"] / row["reference_bytes"] - 1))
        close(np.mean(quality), group["mean_matched_rate_psnr_gain_db"], group["variant"] + " MR PSNR")
        close(np.mean(structural), group["mean_matched_rate_ssim_gain"], group["variant"] + " MR SSIM")
        close(np.mean(rates), group["mean_rate_change_percent"], group["variant"] + " rate")
        ci = bootstrap_ci(quality, 2027 if group["variant"] == "mixed" else 2040 + index)
        close(ci, group["matched_rate_psnr_gain_95ci_db"], group["variant"] + " CI")
        positive = int(np.sum(np.asarray(quality) > 0))
        close(positive, group["positive_psnr_pairs"], group["variant"] + " positive count")
        result[group["variant"]] = {"pairs": len(quality), "mean_matched_rate_psnr_gain_db": float(np.mean(quality)),
                                    "mean_matched_rate_ssim_gain": float(np.mean(structural)),
                                    "mean_rate_change_percent": float(np.mean(rates)),
                                    "matched_rate_psnr_gain_95ci_db": ci, "positive_pairs": positive}
    return result


def audit_representation(data, rd):
    curves = {r["pair"]: r["baseline"] for r in rd["FCGS"]["rows"]}
    expected_pairs = set(curves)
    if {r["pair"] for r in data["rows"]} != expected_pairs:
        raise AssertionError("Representation ablations do not cover the complete test cohort")
    by_profile = {}
    for row in data["rows"]:
        default = row["profiles"][0]
        for profile in row["profiles"]:
            close(profile["birth_bytes"], default["birth_bytes"], "Fixed birth stream")
            close(profile["total_bytes"], profile["core_bytes"] + profile["birth_bytes"] + 1, "Complete rate")
            close(profile["decoded_gaussians"], row["target_gaussians"], "Decoded cardinality")
            close(profile["mean_psnr_db"], np.mean([v["psnr_db"] for v in profile["views"]]), "View averaging")
            base = curves[row["pair"]]
            gain = profile["mean_psnr_db"] - matched_quality(profile["total_bytes"],
                [p["bytes"] for p in base], [p["psnr_db"] for p in base])
            close(gain, profile["matched_rate_psnr_gain_db"], "Representation MR gain")
            by_profile.setdefault(profile["profile"], []).append((profile, row["target_gaussians"]))
    result = {}
    for saved in data["profiles"]:
        values = by_profile[saved["profile"]]
        close(len(values), len(expected_pairs), "Representation cohort size")
        gains = [v["matched_rate_psnr_gain_db"] for v,n in values]
        rate = np.mean([8*v["total_bytes"]/n for v,n in values])
        psnr = np.mean([v["mean_psnr_db"] for v,n in values])
        close(rate, saved["mean_rate_bpg"], "Representation rate")
        close(psnr, saved["mean_psnr_db"], "Representation PSNR")
        close(np.mean(gains), saved["mean_matched_rate_psnr_gain_db"], "Representation mean gain")
        close(bootstrap_ci(gains, 2044), saved["matched_rate_gain_ci95"], "Representation gain interval")
        result[saved["profile"]] = {"pairs": len(values), "mean_rate_bpg": float(rate),
                                    "mean_psnr_db": float(psnr), "mean_matched_rate_psnr_gain_db": float(np.mean(gains))}
    return result


def audit_views(data, label):
    gains, rates = [], []
    for row in data["rows"]:
        selection = set(row["selection_indices"])
        independent = set(row["independent_indices"])
        if selection & independent:
            raise AssertionError(label + " selection and independent cameras overlap")
        views = [v for v in row["views"] if v["role"] == "independent"]
        if {v["index"] for v in views} != independent:
            raise AssertionError(label + " independent camera indices differ")
        active = row["selected_mode"] == "conditional"
        gain = float(np.mean([v["conditional_psnr"] - v["intra_psnr"] for v in views])) if active else 0.0
        rate = 100 * (row["conditional_bytes"] / row["intra_bytes"] - 1) if active else 0.0
        close(gain, row["selected_gain_db"], label + " independent gain")
        close(rate, row["selected_rate_change_percent"], label + " rate")
        gains.append(gain); rates.append(rate)
    close(np.mean(gains), data["mean_selected_gain_db"], label + " mean gain")
    close(np.mean(rates), data["mean_selected_rate_change_percent"], label + " mean rate")
    close(sum(g > 0 for g in gains), data["positive"], label + " positive count")
    close(sum(g < 0 for g in gains), data["negative"], label + " negative count")
    return {"pairs": len(gains), "mean_selected_gain_db": float(np.mean(gains)),
            "mean_selected_rate_change_percent": float(np.mean(rates))}


def _check_public_record(row, label, rate_key, psnr_key, ssim_key):
    values = (row[rate_key], row[psnr_key], row[ssim_key])
    if not np.all(np.isfinite(values)):
        raise AssertionError(label + " contains a nonfinite metric")
    if row["bytes"] <= 0 or row[rate_key] <= 0:
        raise AssertionError(label + " contains a nonpositive stream size")
    if not 0 <= row[ssim_key] <= 1:
        raise AssertionError(label + " contains SSIM outside [0, 1]")


def audit_recent_public(data):
    pairs = data["test_pairs"]
    if not pairs or len(pairs) != len(set(pairs)):
        raise AssertionError("Recent public-codec pair identifiers are empty or repeated")
    close(len(pairs), data["statistics"]["cohort_pairs"], "Recent public-codec cohort size")
    summary = {}
    for method, rows in data["methods"].items():
        grouped = {}
        for row in rows:
            _check_public_record(row, method, "bits_per_target_gaussian", "psnr_db", "ssim")
            if row["pair"] not in pairs:
                raise AssertionError(method + " contains an unknown pair")
            grouped.setdefault(row["point"], []).append(row)
        summary[method] = {}
        for point, point_rows in grouped.items():
            if len(point_rows) != len(pairs) or len({r["pair"] for r in point_rows}) != len(pairs):
                raise AssertionError(f"{method}/{point} does not cover the complete cohort")
            values = {
                "rate_bpg": float(np.mean([r["bits_per_target_gaussian"] for r in point_rows])),
                "psnr_db": float(np.mean([r["psnr_db"] for r in point_rows])),
                "ssim": float(np.mean([r["ssim"] for r in point_rows])),
            }
            saved = (data["statistics"]["FlexGaussian"][point] if method == "FlexGaussian"
                     else data["statistics"][method + "_" + point])
            for key, value in values.items():
                close(value, saved[key], f"{method}/{point}/{key}")
            summary[method][point] = values
    return {"pairs": len(pairs), "methods": summary}


def audit_recent_external(data):
    grouped = {}
    scene_ids = set()
    dataset_counts = {}
    for scene in data["scenes"]:
        key = (scene["dataset"], scene["scene"])
        if key in scene_ids:
            raise AssertionError("Repeated external public-codec scene: " + "/".join(key))
        scene_ids.add(key)
        dataset_counts[scene["dataset"]] = dataset_counts.get(scene["dataset"], 0) + 1
        seen = set()
        for row in scene["records"]:
            method = row["method"] + ":" + row["point"]
            if method in seen:
                raise AssertionError("Repeated method point in external scene: " + method)
            seen.add(method)
            _check_public_record(row, "/".join(key) + "/" + method,
                                 "rate_bpg", "mean_psnr_db", "mean_ssim")
            grouped.setdefault((scene["dataset"], method), []).append(row)
    summary = {}
    for (dataset, method), rows in grouped.items():
        if len(rows) != dataset_counts[dataset]:
            raise AssertionError(f"{dataset}/{method} does not cover every released scene")
        values = {
            "rate_bpg": float(np.mean([r["rate_bpg"] for r in rows])),
            "psnr_db": float(np.mean([r["mean_psnr_db"] for r in rows])),
            "ssim": float(np.mean([r["mean_ssim"] for r in rows])),
        }
        for key, value in values.items():
            close(value, data["means"][dataset][method][key], f"{dataset}/{method}/{key}")
        summary.setdefault(dataset, {})[method] = values
    return {"scenes": len(scene_ids), "datasets": summary}


def audit(directory):
    load = lambda name: json.loads((directory / name).read_text())
    # Guard against the historical silent endpoint clamp, including a one-byte excursion.
    for rate in (9, 21):
        try:
            matched_quality(rate, [10, 20], [1, 2])
        except ValueError:
            pass
        else:
            raise AssertionError("Out-of-domain interpolation was accepted")
    result = {"multirate": audit_multirate(load("multirate_rd.json")),
            "reference_reuse": audit_reuse(load("reference_reuse.json")),
            "status": "PASS: selections, BD integration, matched-rate domains, means, counts and reuse intervals"}
    if (directory / "representation_ablation.json").exists():
        result["representation"] = audit_representation(load("representation_ablation.json"), load("multirate_rd.json"))
        result["status"] += "; complete-cohort representation statistics"
    result["independent_views"] = {
        "FCGS": audit_views(load("independent_views.json"), "FCGS"),
        "L-GSC": audit_views(load("independent_views_lgsc.json"), "L-GSC")}
    result["recent_public_codecs"] = audit_recent_public(load("recent_public_codecs.json"))
    result["recent_external_codecs"] = audit_recent_external(load("recent_external_codecs.json"))
    result["status"] += "; independent views for both codecs; recent public-codec cohorts"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])
    for backend, row in result["multirate"].items():
        print(backend, row["selected"])


if __name__ == "__main__":
    main()
