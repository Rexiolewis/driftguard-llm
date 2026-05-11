"""
Sensitivity and Ablation Runner — Task F
==========================================
Runs ALL synthetic drift streams through DriftDetectorV2 across a full grid of
hyperparameter combinations, then saves three tables:

  results/drift_v2/tables/sensitivity_analysis.csv   — window_size × bins × binning
  results/drift_v2/tables/ablation_detector_rules.csv — detector rule × reference_mode
  results/drift_v2/tables/stream_detection_summary.csv — per stream, best config

Usage
-----
  python scripts/run_drift_experiments_v2.py \\
      --streams_dir results/drift_v2/streams \\
      --thresholds results/drift_v2/calibrated_thresholds.json \\
      --out results/drift_v2 \\
      --window_sizes 50 100 200 500 \\
      --bins 10 20 30 50 \\
      --step_fractions 1.0 0.5 \\
      --reference_modes fixed_first rolling multi_reference adaptive_clean \\
      --detector_rules KL_only KS_only KL_AND_KS JS_only Wasserstein_only Ensemble_majority

Output columns (all tables)
----------------------------
  stream_name, detector_rule, window_size, step_size, bins, binning,
  reference_mode, calibrated_threshold_used, drift_detected,
  first_detection_window, detection_delay, false_alarm_count_before_drift,
  true_detection_after_drift, max_kl, min_ks_pvalue, max_js, max_wasserstein, n_windows
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from drift_detector_v2 import DriftDetectorV2, load_and_normalise, NormalisedRecord


# ---------------------------------------------------------------------------
# Default grid
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_SIZES = [50, 100, 200, 500]
DEFAULT_BINS = [10, 20, 30, 50]
DEFAULT_BINNINGS = ["equal_width", "quantile"]
DEFAULT_STEP_FRACTIONS = [1.0, 0.5]
DEFAULT_REFERENCE_MODES = ["fixed_first", "rolling", "multi_reference", "adaptive_clean"]
DEFAULT_DETECTOR_RULES = [
    "KL_only", "KS_only", "KL_AND_KS", "JS_only", "Wasserstein_only", "Ensemble_majority"
]


# ---------------------------------------------------------------------------
# Load stream + metadata
# ---------------------------------------------------------------------------

def load_stream(jsonl_path: Path) -> Dict[str, Any]:
    """Load stream JSONL and companion metadata JSON."""
    records = load_and_normalise(jsonl_path)

    meta_path = jsonl_path.with_name(jsonl_path.stem + "_meta.json")
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    drift_point = meta.get("drift_point")
    return {"records": records, "meta": meta, "drift_point": drift_point,
            "stream_name": jsonl_path.stem}


def load_thresholds(path: Optional[Path]) -> Dict:
    """Load calibrated thresholds JSON, or return defaults."""
    defaults = {
        "kl_threshold_95": 0.05,
        "js_threshold_95": 0.03,
        "wasserstein_threshold_95": 0.05,
        "ks_alpha": 0.05,
    }
    if path and path.exists():
        with open(path) as f:
            d = json.load(f)
        defaults.update(d)
        print(f"Loaded calibrated thresholds from {path}")
    else:
        print("Using default thresholds (calibrate_thresholds.py not run yet).")
    return defaults


# ---------------------------------------------------------------------------
# Run single experiment
# ---------------------------------------------------------------------------

def run_single(
    records: List[NormalisedRecord],
    stream_name: str,
    drift_point: Optional[int],
    window_size: int,
    step_fraction: float,
    bins: int,
    binning: str,
    reference_mode: str,
    detector_rule: str,
    thresholds: Dict,
) -> Dict[str, Any]:
    """Run one detector configuration and return a result row."""
    step_size = max(1, int(window_size * step_fraction))

    detector = DriftDetectorV2(
        window_size=window_size,
        step_size=step_size,
        bins=bins,
        binning=binning,
        reference_mode=reference_mode,
        reference_size=window_size,
        detector_rule=detector_rule,
        kl_warn=thresholds.get("kl_threshold_95", 0.05),
        kl_crit=thresholds.get("kl_threshold_95", 0.05) * 3,
        ks_alpha=thresholds.get("ks_alpha", 0.05),
        js_warn=thresholds.get("js_threshold_95", 0.03),
        wasserstein_warn=thresholds.get("wasserstein_threshold_95", 0.05),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = detector.analyse(
            records=records,
            drift_point=drift_point,
            source_file=stream_name,
        )

    # Detection delay = first_detection_window - window_at_drift_point
    detection_delay = None
    if report.first_detection_window is not None and drift_point is not None:
        # Estimated window index where drift_point falls
        drift_window_idx = max(0, (drift_point - window_size) // step_size)
        detection_delay = report.first_detection_window - drift_window_idx

    # True detection: did we detect drift *after* the drift point?
    true_detection = False
    if drift_point is not None and report.first_detection_window is not None:
        for w in report.windows:
            if w.drift_detected and w.start_record >= drift_point:
                true_detection = True
                break

    return {
        "stream_name": stream_name,
        "detector_rule": detector_rule,
        "window_size": window_size,
        "step_size": step_size,
        "step_fraction": step_fraction,
        "bins": bins,
        "binning": binning,
        "reference_mode": reference_mode,
        "calibrated_threshold_used": "kl_threshold_95" in thresholds,
        "drift_detected": report.drift_detected,
        "first_detection_window": report.first_detection_window,
        "detection_delay": detection_delay,
        "false_alarm_count_before_drift": report.false_alarm_count_before_drift,
        "true_detection_after_drift": true_detection,
        "max_kl": report.max_kl,
        "min_ks_pvalue": report.min_ks_pvalue,
        "max_js": report.max_js,
        "max_wasserstein": report.max_wasserstein,
        "n_windows": len(report.windows),
        "n_records": report.n_total_records,
        "constant_confidence_warning": report.constant_confidence_warning,
    }


# ---------------------------------------------------------------------------
# Save tables
# ---------------------------------------------------------------------------

COLS = [
    "stream_name", "detector_rule", "window_size", "step_size", "step_fraction",
    "bins", "binning", "reference_mode", "calibrated_threshold_used",
    "drift_detected", "first_detection_window", "detection_delay",
    "false_alarm_count_before_drift", "true_detection_after_drift",
    "max_kl", "min_ks_pvalue", "max_js", "max_wasserstein",
    "n_windows", "n_records", "constant_confidence_warning",
]


def save_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved → {path}  ({len(rows)} rows)")


def save_reports(results: List[Dict], reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Group by stream
    by_stream: Dict[str, List] = {}
    for r in results:
        by_stream.setdefault(r["stream_name"], []).append(r)
    for sname, rows in by_stream.items():
        p = reports_dir / f"{sname}_results.json"
        with open(p, "w") as f:
            json.dump(rows, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run sensitivity + ablation drift experiments.")
    parser.add_argument("--streams_dir", default="results/drift_v2/streams")
    parser.add_argument("--thresholds", default=None,
                        help="Path to calibrated_thresholds.json")
    parser.add_argument("--out", default="results/drift_v2")
    parser.add_argument("--window_sizes", nargs="+", type=int, default=DEFAULT_WINDOW_SIZES)
    parser.add_argument("--bins", nargs="+", type=int, default=DEFAULT_BINS)
    parser.add_argument("--binnings", nargs="+", default=DEFAULT_BINNINGS)
    parser.add_argument("--step_fractions", nargs="+", type=float, default=DEFAULT_STEP_FRACTIONS)
    parser.add_argument("--reference_modes", nargs="+", default=DEFAULT_REFERENCE_MODES)
    parser.add_argument("--detector_rules", nargs="+", default=DEFAULT_DETECTOR_RULES)
    args = parser.parse_args()

    streams_dir = Path(args.streams_dir)
    out_dir = Path(args.out)

    # Load streams (skip meta and adversarial input files)
    stream_paths = [
        p for p in sorted(streams_dir.glob("*.jsonl"))
        if "_meta" not in p.stem and "adversarial_text_stream_input" not in p.stem
    ]
    if not stream_paths:
        print(f"No stream JSONL files found in {streams_dir}. Run generate_drift_streams.py first.")
        sys.exit(1)

    print(f"Found {len(stream_paths)} stream(s): {[p.stem for p in stream_paths]}")

    thresholds = load_thresholds(
        Path(args.thresholds) if args.thresholds else None
    )

    # Total experiment count
    n_configs = (len(args.window_sizes) * len(args.bins) * len(args.binnings) *
                 len(args.step_fractions) * len(args.reference_modes) * len(args.detector_rules))
    n_total = n_configs * len(stream_paths)
    print(f"\nGrid: {n_configs} configs × {len(stream_paths)} streams = {n_total} experiments\n")

    all_results: List[Dict] = []
    t0 = time.time()
    done = 0

    for sp in stream_paths:
        stream_data = load_stream(sp)
        stream_name = stream_data["stream_name"]
        records = stream_data["records"]
        drift_point = stream_data["drift_point"]

        print(f"Stream: {stream_name}  ({len(records)} records, drift_point={drift_point})")

        for ws, nb, binning, sf, rm, dr in product(
            args.window_sizes, args.bins, args.binnings,
            args.step_fractions, args.reference_modes, args.detector_rules
        ):
            if len(records) < ws * 2:
                # Can't form even 2 windows — skip with a note
                all_results.append({
                    "stream_name": stream_name,
                    "detector_rule": dr,
                    "window_size": ws,
                    "step_size": int(ws * sf),
                    "step_fraction": sf,
                    "bins": nb,
                    "binning": binning,
                    "reference_mode": rm,
                    "calibrated_threshold_used": "kl_threshold_95" in thresholds,
                    "drift_detected": None,
                    "first_detection_window": None,
                    "detection_delay": None,
                    "false_alarm_count_before_drift": None,
                    "true_detection_after_drift": None,
                    "max_kl": None, "min_ks_pvalue": None,
                    "max_js": None, "max_wasserstein": None,
                    "n_windows": 0,
                    "n_records": len(records),
                    "constant_confidence_warning": None,
                    "_skip_reason": f"n_records({len(records)}) < window_size*2({ws*2})",
                })
                done += 1
                continue

            result = run_single(
                records=records,
                stream_name=stream_name,
                drift_point=drift_point,
                window_size=ws,
                step_fraction=sf,
                bins=nb,
                binning=binning,
                reference_mode=rm,
                detector_rule=dr,
                thresholds=thresholds,
            )
            all_results.append(result)
            done += 1

        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  {done}/{n_total} runs done  ({rate:.0f}/s)")

    print(f"\nAll experiments complete in {time.time()-t0:.1f}s")

    # ---- Build output tables ----
    tables_dir = out_dir / "tables"

    # 1. Full sensitivity analysis (window_size × bins × binning)
    sensitivity_cols = ["stream_name", "detector_rule", "window_size", "step_size", "bins",
                        "binning", "reference_mode"]
    sensitivity = [r for r in all_results if r.get("drift_detected") is not None]
    save_csv(sensitivity, tables_dir / "sensitivity_analysis.csv")

    # 2. Ablation: detector rule × reference_mode (fixed window_size=200, bins=20, equal_width)
    ablation = [
        r for r in all_results
        if r.get("window_size") == 200 and r.get("bins") == 20
        and r.get("binning") == "equal_width" and r.get("step_fraction") == 1.0
        and r.get("drift_detected") is not None
    ]
    save_csv(ablation, tables_dir / "ablation_detector_rules.csv")

    # 3. Stream detection summary (best config per stream by true_detection and min detection_delay)
    summary = _best_per_stream(all_results)
    save_csv(summary, tables_dir / "stream_detection_summary.csv")

    # 4. Per-stream JSON reports
    reports_dir = out_dir / "reports"
    save_reports(all_results, reports_dir)

    # Print quick ablation table
    print("\n" + "=" * 90)
    print("ABLATION TABLE (W=200, bins=20, equal_width, step=1.0) — detector_rule × reference_mode")
    print("=" * 90)
    _print_ablation_summary(ablation)

    print(f"\nOutputs:")
    print(f"  Sensitivity : {tables_dir}/sensitivity_analysis.csv")
    print(f"  Ablation    : {tables_dir}/ablation_detector_rules.csv")
    print(f"  Summary     : {tables_dir}/stream_detection_summary.csv")
    print(f"  Reports     : {reports_dir}/")


def _best_per_stream(results: List[Dict]) -> List[Dict]:
    """For each stream, pick the config with most true detections & smallest detection delay."""
    by_stream: Dict[str, List] = {}
    for r in results:
        if r.get("drift_detected") is None:
            continue
        by_stream.setdefault(r["stream_name"], []).append(r)

    summary = []
    for sname, rows in by_stream.items():
        # Prefer true_detection, then smallest detection_delay, then smallest false_alarms
        drift_rows = [r for r in rows if r.get("true_detection_after_drift")]
        pool = drift_rows if drift_rows else rows
        pool_with_delay = [r for r in pool if r.get("detection_delay") is not None]
        if pool_with_delay:
            best = min(pool_with_delay, key=lambda r: (
                -int(r.get("true_detection_after_drift") or 0),
                r.get("detection_delay", 999),
                r.get("false_alarm_count_before_drift", 999),
            ))
        else:
            best = pool[0] if pool else rows[0]
        summary.append(best)

    return summary


def _print_ablation_summary(rows: List[Dict]) -> None:
    rules = sorted(set(r["detector_rule"] for r in rows))
    streams = sorted(set(r["stream_name"] for r in rows))
    ref_modes = sorted(set(r["reference_mode"] for r in rows))

    for rm in ref_modes:
        print(f"\n  reference_mode = {rm}")
        print(f"  {'stream':<30} " + " ".join(f"{dr[:12]:<12}" for dr in rules))
        for sname in streams:
            row_vals = []
            for dr in rules:
                match = [r for r in rows if r["stream_name"] == sname
                         and r["detector_rule"] == dr and r["reference_mode"] == rm]
                if match:
                    m = match[0]
                    val = "✓" if m.get("true_detection_after_drift") else (
                          "FA" if m.get("drift_detected") else "—")
                    row_vals.append(f"{val:<12}")
                else:
                    row_vals.append(f"{'N/A':<12}")
            print(f"  {sname:<30} " + " ".join(row_vals))

    print("\n  Key: ✓=true detection after drift  FA=false alarm only  —=no detection  N/A=skipped")


if __name__ == "__main__":
    main()
