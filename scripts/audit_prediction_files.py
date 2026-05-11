"""
Audit Prediction Files — Task G
=================================
Inspects all JSONL prediction files and reports:
  - Row count and column inventory
  - Language, true-label, predicted-label distributions
  - Confidence statistics (min, max, mean, std, # unique values)
  - Warnings for constant confidence, all-0.5 pred_confidence, wrong row counts
  - Warns if Phase 2 files have < 400 rows (too small for W=200 drift)
  - Warns if any file has ≤ 3 rows (unusable)

Outputs
-------
  results/drift_v2/audit/prediction_file_audit.csv
  results/drift_v2/audit/prediction_file_audit.json

Usage
-----
  python scripts/audit_prediction_files.py \\
      --input_dir predictions \\
      --out results/drift_v2/audit

  # Or scan a whole results tree:
  python scripts/audit_prediction_files.py \\
      --input_dir results \\
      --out results/drift_v2/audit
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Expected row counts per experiment (Phase 1 = 4282 for baselines)
# ---------------------------------------------------------------------------
EXPECTED_PHASE1_ROWS = 4282
PHASE2_MIN_ROWS_FOR_W200 = 400  # less than this → cannot run W=200 drift claims

EXPERIMENT_LABELS = {
    "e0_zero_shot": "E0  Zero-shot LLaMA-3.1 8B (proposed, zero-shot)",
    "e1_qlora_ft": "E1  QLoRA Fine-tuned (proposed, fine-tuned)",
    "e2_qlora_la": "E2  QLoRA Language-Aware (proposed, fine-tuned)",
    "e3_qlora_mm": "E3  QLoRA Multimodal BLIP (proposed, multimodal)",
    "e4_qlora_rag": "E4  QLoRA RAG+FAISS (proposed, fine-tuned+RAG)",
    "e5_lg4": "E5  LlamaGuard-4 12B (zero-shot off-the-shelf baseline)",
    "lg3_baseline": "B2  LlamaGuard-3 8B (zero-shot off-the-shelf baseline)",
    "xlmr_baseline": "B1  XLM-R (fine-tuned encoder baseline)",
    "transfer_phase2": "T1  Transfer Phase1→Phase2 (proposed)",
    "e3_phase2_captioned": "E3P2  QLoRA BLIP Phase2 (proposed, multimodal)",
}


# ---------------------------------------------------------------------------
# Core auditor
# ---------------------------------------------------------------------------

def audit_file(path: Path) -> Dict:
    rows = []
    parse_errors = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                parse_errors += 1

    n = len(rows)
    result: Dict = {
        "file": path.name,
        "path": str(path),
        "n_rows": n,
        "parse_errors": parse_errors,
        "columns": [],
        "warnings": [],
    }

    if n == 0:
        result["warnings"].append("EMPTY FILE — no valid JSON rows found.")
        return result

    # Columns
    all_keys: Counter = Counter()
    for r in rows:
        all_keys.update(r.keys())
    result["columns"] = sorted(all_keys.keys())

    # ---- Label distributions ----
    true_col = _pick(rows[0], ["label", "true_label", "ground_truth"])
    pred_col = _pick(rows[0], ["pred_label", "predicted_label", "prediction", "predicted"])
    lang_col = _pick(rows[0], ["lang", "language", "lang_code"])

    true_labels = [_norm_label(r.get(true_col)) for r in rows] if true_col else []
    pred_labels = [_norm_label(r.get(pred_col)) for r in rows] if pred_col else []
    languages = [str(r.get(lang_col, "unknown")) for r in rows] if lang_col else []

    result["true_label_col"] = true_col
    result["pred_label_col"] = pred_col
    result["lang_col"] = lang_col
    result["true_label_distribution"] = dict(Counter(true_labels))
    result["pred_label_distribution"] = dict(Counter(pred_labels))
    result["language_distribution"] = dict(Counter(languages))

    # ---- Confidence statistics ----
    # pred_confidence
    pred_confs = _safe_floats(rows, ["pred_confidence", "confidence", "score"])
    result["pred_confidence_stats"] = _conf_stats(pred_confs)

    # unsafe_confidence / unsafe_probability
    unsafe_confs = _safe_floats(rows, ["unsafe_confidence", "unsafe_probability"])
    result["unsafe_confidence_stats"] = _conf_stats(unsafe_confs)

    # ---- Warnings ----
    warnings: List[str] = []

    if n <= 3:
        warnings.append(
            f"CRITICAL: Only {n} rows — file is unusable for any multi-window drift experiment."
        )

    # Phase 1 row count check
    is_phase1 = "phase2" not in path.name and "transfer" not in path.name
    is_phase2 = "phase2" in path.name or "transfer" in path.name
    if is_phase1 and n not in (4217, EXPECTED_PHASE1_ROWS):
        warnings.append(
            f"Row count mismatch: {n} rows (expected {EXPECTED_PHASE1_ROWS} for baselines "
            f"or 4217 for fine-tuned experiments). Check for dropped records."
        )

    # Phase 2 too small for W=200
    if is_phase2 and n < PHASE2_MIN_ROWS_FOR_W200:
        warnings.append(
            f"Phase 2 file has only {n} rows — window_size must be reduced below "
            f"{n//3} for multi-window drift. Do NOT make W=200 drift claims on this data "
            f"without marking results as 'exploratory' in the paper."
        )

    # pred_confidence constant check
    if pred_confs:
        pc_unique = set(round(v, 6) for v in pred_confs)
        if len(pc_unique) == 1:
            val = next(iter(pc_unique))
            warnings.append(
                f"pred_confidence is CONSTANT = {val} for ALL {len(pred_confs)} rows. "
                f"This file is NOT suitable for confidence-based drift analysis. "
                f"Must rerun inference with real token logit probabilities."
            )
            if abs(val - 0.5) < 1e-6:
                warnings.append(
                    "STOP — pred_confidence all 0.5 detected. "
                    "Need to rerun inference with real unsafe_probability before using "
                    "confidence-based drift results in the paper."
                )

    # unsafe_confidence constant check
    if unsafe_confs:
        uc_unique = set(round(v, 6) for v in unsafe_confs)
        if len(uc_unique) == 1:
            warnings.append(
                f"unsafe_confidence is CONSTANT = {next(iter(uc_unique))} — "
                f"confidence-histogram drift will be uninformative."
            )

    # Warn on near-constant confidence
    if pred_confs and len(set(round(v, 4) for v in pred_confs)) < 4:
        warnings.append(
            f"pred_confidence has only {len(set(round(v,4) for v in pred_confs))} unique "
            f"values (low variance). Check that inference used real logit probabilities."
        )

    # Missing unsafe_confidence
    if not unsafe_confs and "unsafe_confidence" not in (result.get("columns") or []):
        warnings.append(
            "No unsafe_confidence / unsafe_probability column found. "
            "Re-run with evaluator_v2.py to get real token probabilities for drift analysis."
        )

    # Baseline fairness note
    stem = path.stem.lower()
    if "lg3" in stem or "lg4" in stem or "e5" in stem:
        warnings.append(
            "FAIRNESS NOTE: This is a zero-shot off-the-shelf baseline. "
            "Do NOT present direct performance comparison with fine-tuned models as if controlled. "
            "Mark as 'off-the-shelf operational baseline' in all tables."
        )
    if "xlmr" in stem:
        warnings.append(
            "FAIRNESS NOTE: XLM-R is a fine-tuned encoder baseline (B1). "
            "Comparison with QLoRA fine-tuned LLaMA is cross-architecture and should be noted."
        )

    result["warnings"] = warnings
    result["n_warnings"] = len(warnings)
    result["experiment_label"] = _match_label(path.name)

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick(row: dict, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in row:
            return c
    return None


def _norm_label(v) -> str:
    if v is None:
        return "UNKNOWN"
    s = str(v).strip().upper()
    if s in ("1", "UNSAFE", "HATE", "TOXIC", "HARMFUL"):
        return "UNSAFE"
    if s in ("0", "SAFE"):
        return "SAFE"
    return s


def _safe_floats(rows: List[dict], cols: List[str]) -> List[float]:
    col = None
    for c in cols:
        if c in (rows[0] if rows else {}):
            col = c
            break
    if col is None:
        return []
    vals = []
    for r in rows:
        v = r.get(col)
        if v is not None:
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                pass
    return vals


def _conf_stats(vals: List[float]) -> Dict:
    if not vals:
        return {"n": 0, "min": None, "max": None, "mean": None, "std": None, "n_unique": 0}
    arr = np.array(vals)
    return {
        "n": len(vals),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std()), 6),
        "n_unique": len(set(round(v, 6) for v in vals)),
    }


def _match_label(filename: str) -> str:
    fname = filename.lower()
    for key, label in EXPERIMENT_LABELS.items():
        if key in fname:
            return label
    return filename


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Audit prediction JSONL files.")
    parser.add_argument("--input_dir", default="predictions",
                        help="Directory containing prediction JSONL files (searched recursively).")
    parser.add_argument("--out", default="results/drift_v2/audit",
                        help="Output directory for audit reports.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(input_dir.rglob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found under {input_dir}. Exiting.")
        sys.exit(1)

    print(f"\nAuditing {len(jsonl_files)} JSONL file(s) from '{input_dir}'...\n")
    print("=" * 80)

    all_results = []
    stop_flag = False

    for path in jsonl_files:
        result = audit_file(path)
        all_results.append(result)

        exp = result.get("experiment_label", path.name)
        print(f"\n{'─'*80}")
        print(f"  {exp}")
        print(f"  File   : {path.name}")
        print(f"  Rows   : {result['n_rows']:,}   Columns: {len(result.get('columns', []))}")

        # Distributions
        td = result.get("true_label_distribution", {})
        pd = result.get("pred_label_distribution", {})
        ld = result.get("language_distribution", {})
        if td:
            print(f"  True labels   : {td}")
        if pd:
            print(f"  Pred labels   : {pd}")
        if ld:
            print(f"  Languages     : {dict(sorted(ld.items(), key=lambda x: -x[1])[:8])}")

        # Confidence stats
        pcs = result.get("pred_confidence_stats", {})
        if pcs.get("n", 0) > 0:
            print(f"  pred_conf     : mean={pcs['mean']:.4f}  std={pcs['std']:.4f}  "
                  f"min={pcs['min']}  max={pcs['max']}  unique={pcs['n_unique']}")
        ucs = result.get("unsafe_confidence_stats", {})
        if ucs.get("n", 0) > 0:
            print(f"  unsafe_conf   : mean={ucs['mean']:.4f}  std={ucs['std']:.4f}  "
                  f"unique={ucs['n_unique']}")

        for w in result.get("warnings", []):
            tag = "  ⛔ STOP" if "STOP" in w else ("  ⚠  WARN" if "CRITICAL" not in w else "  🔴 CRIT")
            print(f"{tag}: {w}")
            if "STOP" in w:
                stop_flag = True

    print("\n" + "=" * 80)

    # Write CSV
    csv_path = out_dir / "prediction_file_audit.csv"
    csv_fields = [
        "file", "n_rows", "parse_errors", "experiment_label", "n_warnings",
        "true_label_distribution", "pred_label_distribution", "language_distribution",
        "pred_confidence_mean", "pred_confidence_std", "pred_confidence_n_unique",
        "unsafe_confidence_mean", "unsafe_confidence_std", "unsafe_confidence_n_unique",
        "warnings",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            pcs = r.get("pred_confidence_stats", {})
            ucs = r.get("unsafe_confidence_stats", {})
            writer.writerow({
                "file": r["file"],
                "n_rows": r["n_rows"],
                "parse_errors": r["parse_errors"],
                "experiment_label": r.get("experiment_label", ""),
                "n_warnings": r.get("n_warnings", 0),
                "true_label_distribution": json.dumps(r.get("true_label_distribution", {})),
                "pred_label_distribution": json.dumps(r.get("pred_label_distribution", {})),
                "language_distribution": json.dumps(r.get("language_distribution", {})),
                "pred_confidence_mean": pcs.get("mean", ""),
                "pred_confidence_std": pcs.get("std", ""),
                "pred_confidence_n_unique": pcs.get("n_unique", ""),
                "unsafe_confidence_mean": ucs.get("mean", ""),
                "unsafe_confidence_std": ucs.get("std", ""),
                "unsafe_confidence_n_unique": ucs.get("n_unique", ""),
                "warnings": " | ".join(r.get("warnings", [])),
            })

    # Write JSON
    json_path = out_dir / "prediction_file_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAudit saved:")
    print(f"  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")

    if stop_flag:
        print("\n" + "!" * 80)
        print("STOP — One or more files require inference rerun before paper results are valid.")
        print("Need to rerun inference with real unsafe_probability before using")
        print("E1 confidence-based drift results in the paper.")
        print("!" * 80)
        sys.exit(2)  # exit code 2 signals stop condition to shell scripts

    print(f"\nAudit complete. {len(all_results)} file(s) inspected.")


if __name__ == "__main__":
    main()
