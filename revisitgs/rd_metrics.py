"""Rate-distortion statistics with explicit interpolation domains."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator


def matched_quality(rate: float, rates, qualities) -> float:
    rates, qualities = np.asarray(rates, float), np.asarray(qualities, float)
    order = np.argsort(rates, kind="stable")
    rates, qualities = rates[order], qualities[order]
    if len(rates) < 2 or not np.all(np.isfinite(rates)) or not np.all(np.isfinite(qualities)):
        raise ValueError("At least two finite reference points are required")
    if np.any(np.diff(rates) <= 0):
        raise ValueError("Reference rates must be distinct")
    if not np.isfinite(rate) or rate < rates[0] or rate > rates[-1]:
        raise ValueError(f"Rate {rate} is outside measured interval [{rates[0]}, {rates[-1]}]")
    return float(np.interp(rate, rates, qualities))


def pareto(points):
    result, best = [], -np.inf
    for row in sorted(points, key=lambda p: p["bytes"]):
        if row["psnr_db"] > best + 1e-10:
            result.append(row)
            best = row["psnr_db"]
    return result


def bd_metrics(reference, test):
    ref, cur = pareto(reference), pareto(test)
    if min(len(ref), len(cur)) < 3:
        return None
    rq = np.array([p["psnr_db"] for p in ref])
    tq = np.array([p["psnr_db"] for p in cur])
    rr = np.log([p["bytes"] for p in ref])
    tr = np.log([p["bytes"] for p in cur])
    qlo, qhi = max(rq[0], tq[0]), min(rq[-1], tq[-1])
    rlo, rhi = max(rr.min(), tr.min()), min(rr.max(), tr.max())
    if qhi <= qlo or rhi <= rlo:
        return None
    dr = (PchipInterpolator(tq, tr).integrate(qlo, qhi)
          - PchipInterpolator(rq, rr).integrate(qlo, qhi)) / (qhi - qlo)
    dq = (PchipInterpolator(tr, tq).integrate(rlo, rhi)
          - PchipInterpolator(rr, rq).integrate(rlo, rhi)) / (rhi - rlo)
    return float(100 * np.expm1(dr)), float(dq)


def bootstrap_ci(values, seed=2026, samples=10000):
    values = np.asarray(values, float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Bootstrap requires finite observations")
    means = np.random.default_rng(seed).choice(values, (samples, len(values)), replace=True).mean(1)
    return np.quantile(means, [.025, .975]).tolist()
