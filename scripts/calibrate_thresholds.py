"""
Threshold Calibration — Task D
================================
Calibrates drift thresholds from a no-drift (iid) validation stream using
bootstrap resampling. Estimates the KL, JS, Wasserstein, and KS statistic
distributions under the null hypothesis of no drift, then sets thresholds
at the (1 - target_fpr) quantile.

Outputs
-------
  results/drift_v2/calibrated_thresholds.json

Usage
-----
  python scripts/calibrate_thresholds.py \\
      --input results/drift_v2/streams/iid_control.jsonl \\
      --out results/drift_v2/calibrated_thresholds.json \\
      --target_fpr 0.05 \\
      --n_bootstrap 500 \\
      --window_size 200 \\
      --bins 20 \\
      --binning equal_width

The output JSON can be passed directly to run_drift_experiments_v2.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))
from drift_detector_v2 import (
    load_and_normalise, NormalisedRecord,
    _build_histogram, kl_divergence_hist, js_divergence_hist,
    ks_test, wasserstein_dist,
)


# ---------------------------------------------------------------------------
# Bootstrap calibration
# ---------------------------------------------------------------------------

def calibrate(
    records: List[NormalisedRecord],
    window_size: int = 200,
    bins: int = 20,
    binning: str = "equal_width",
    n_bootstrap: int = 500,
    target_fpr: float = 0.05,
    seed: int = 42,
) -> Dict:
    """
    Estimate null distributions of drift metrics by repeatedly splitting
    the iid stream into two random windows of size `window_size` and
    computing metrics between them.

    Returns
    -------
    dict with threshold values at the (1 - target_fpr) quantile.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    uc_vals = [r.unsafe_confidence for r in records if r.unsafe_confidence is not None]
    n = len(uc_vals)

    if n < window_size * 2:
        warnings.warn(
            f"Only {n} records with confidence values — need at least {window_size * 2} "
            f"for reliable calibration (2 × window_size). Results may be noisy.",
            stacklevel=2,
        )

    # Build shared bin edges from the full no-drift stream
    _, shared_edges = _build_histogram(uc_vals, bins=bins, binning=binning)

    kls, jss, wasss, ks_stats = [], [], [], []

    for _ in range(n_bootstrap):
        # Random split: draw two non-overlapping windows from shuffled data
        if n >= window_size * 2:
            shuffled = rng.sample(uc_vals, len(uc_vals))
            ref = shuffled[:window_size]
            win = shuffled[window_size: window_size * 2]
        else:
            # Smaller stream: sample with replacement
            ref = rng.choices(uc_vals, k=window_size)
            win = rng.choices(uc_vals, k=window_size)

        ref_hist, _ = _build_histogram(ref, bins=bins, binning=binning, bin_edges=shared_edges)
        win_hist, _ = _build_histogram(win, bins=bins, binning=binning, bin_edges=shared_edges)

        kls.append(kl_divergence_hist(ref_hist, win_hist))
        jss.append(js_divergence_hist(ref_hist, win_hist))
        wasss.append(wasserstein_dist(ref, win))
        ks_stat, _ = ks_test(ref, win)
        ks_stats.append(ks_stat)

    q = 1.0 - target_fpr

    thresholds = {
        "target_fpr": target_fpr,
        "n_bootstrap": n_bootstrap,
        "window_size": window_size,
        "bins": bins,
        "binning": binning,
        "n_records_used": n,
        "kl_threshold_95": round(float(np.quantile(kls, q)), 6),
        "kl_mean_null": round(float(np.mean(kls)), 6),
        "kl_std_null": round(float(np.std(kls)), 6),
        "js_threshold_95": round(float(np.quantile(jss, q)), 6),
        "js_mean_null": round(float(np.mean(jss)), 6),
        "js_std_null": round(float(np.std(jss)), 6),
        "wasserstein_threshold_95": round(float(np.quantile(wasss, q)), 6),
        "wasserstein_mean_null": round(float(np.mean(wasss)), 6),
        "wasserstein_std_null": round(float(np.std(wasss)), 6),
        "ks_stat_threshold_95": round(float(np.quantile(ks_stats, q)), 6),
        "ks_stat_mean_null": round(float(np.mean(ks_stats)), 6),
        "ks_stat_std_null": round(float(np.std(ks_stats)), 6),
        "ks_alpha": target_fpr,   # p-value threshold = FPR target
        # Null distribution percentiles for reporting
        "kl_null_percentiles": {
            str(int(p)): round(float(np.percentile(kls, p)), 6)
            for p in [50, 75, 90, 95, 99]
        },
        "js_null_percentiles": {
            str(int(p)): round(float(np.percentile(jss, p)), 6)
            for p in [50, 75, 90, 95, 99]
        },
        "wasserstein_null_percentiles": {
            str(int(p)): round(float(np.percentile(wasss, p)), 6)
            for p in [50, 75, 90, 95, 99]
        },
        "ks_null_percentiles": {
            str(int(p)): round(float(np.percentile(ks_stats, p)), 6)
            for p in [50, 75, 90, 95, 99]
        },
    }

    return thresholds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate drift thresholds from a no-drift (iid) stream."
    )
    parser.add_argument("--input", required=True,
                        help="Path to iid_control.jsonl (no-drift stream)")
    parser.add_argument("--out", default="results/drift_v2/calibrated_thresholds.json",
                        help="Output JSON path")
    parser.add_argument("--target_fpr", type=float, default=0.05,
                        help="Target false positive rate (default 0.05 = 5%%)")
    parser.add_argument("--n_bootstrap", type=int, default=500,
                        help="Number of bootstrap resamples (default 500)")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--binning", default="equal_width",
                        choices=["equal_width", "quantile"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading records from: {input_path}")
    records = load_and_normalise(input_path)
    uc_available = sum(1 for r in records if r.unsafe_confidence is not None)
    print(f"  Total records: {len(records)},  with unsafe_confidence: {uc_available}")

    if uc_available == 0:
        print("ERROR: No unsafe_confidence values found. Run evaluator_v2.py first.")
        sys.exit(1)

    print(f"\nRunning bootstrap calibration (n={args.n_bootstrap}, "
          f"target_fpr={args.target_fpr}, window_size={args.window_size}) ...")

    thresholds = calibrate(
        records=records,
        window_size=args.window_size,
        bins=args.bins,
        binning=args.binning,
        n_bootstrap=args.n_bootstrap,
        target_fpr=args.target_fpr,
        seed=args.seed,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    print("\nCalibrated thresholds:")
    print(f"  KL     : {thresholds['kl_threshold_95']:.6f}  "
          f"(null mean={thresholds['kl_mean_null']:.4f} ± {thresholds['kl_std_null']:.4f})")
    print(f"  JS     : {thresholds['js_threshold_95']:.6f}  "
          f"(null mean={thresholds['js_mean_null']:.4f} ± {thresholds['js_std_null']:.4f})")
    print(f"  Wass   : {thresholds['wasserstein_threshold_95']:.6f}  "
          f"(null mean={thresholds['wasserstein_mean_null']:.4f} ± "
          f"{thresholds['wasserstein_std_null']:.4f})")
    print(f"  KS-stat: {thresholds['ks_stat_threshold_95']:.6f}  "
          f"(KS p-value threshold α={thresholds['ks_alpha']})")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
