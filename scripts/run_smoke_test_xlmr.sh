#!/usr/bin/env bash
# =============================================================================
# run_smoke_test_xlmr.sh — Pipeline smoke test using XLM-R baseline predictions
# =============================================================================
# PURPOSE: Verifies that the entire code pipeline (stream generation →
#          threshold calibration → sensitivity/ablation experiments) runs
#          end-to-end without errors. This is NOT a paper result.
#
# IMPORTANT LABEL: XLM-R is a fine-tuned encoder baseline (B1).
#   - It is NOT the proposed DriftGuard-LLM system.
#   - Results from this script must NEVER be labelled as DriftGuard-LLM results.
#   - Use only to confirm the code pipeline produces correct outputs.
#
# Run: bash scripts/run_smoke_test_xlmr.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

SOURCE_PREDICTIONS="predictions/xlmr_baseline_predictions.jsonl"
OUTPUT_DIR="results/drift_v2_xlmr_smoke_test"
LABEL="XLM-R Baseline (SMOKE TEST ONLY — not a paper result)"

echo "======================================================================"
echo "DriftGuard-LLM v2 — XLM-R SMOKE TEST"
echo "  Source : $SOURCE_PREDICTIONS"
echo "  Output : $OUTPUT_DIR"
echo "  NOTE   : This is a code pipeline test only."
echo "           XLM-R results MUST NOT appear as DriftGuard-LLM paper results."
echo "======================================================================"

# Load shared validation
source scripts/_validate_predictions.sh

# ---- Pre-flight validation ----
validate_predictions "$SOURCE_PREDICTIONS" "$LABEL"
VALID_EXIT=$?
if [ $VALID_EXIT -ne 0 ]; then
    exit $VALID_EXIT
fi

mkdir -p "$OUTPUT_DIR"/{streams,tables,reports,audit}

# ---- Step 0: Audit this file ----
echo ""
echo "--- STEP 0: Audit prediction file ---"
python scripts/audit_prediction_files.py \
    --input_dir predictions \
    --out "$OUTPUT_DIR/audit" 2>/dev/null || true
# Non-fatal: audit may warn but we already validated manually

# ---- Step 1: Generate streams ----
echo ""
echo "--- STEP 1: Generate drift streams ---"
python scripts/generate_drift_streams.py \
    --input "$SOURCE_PREDICTIONS" \
    --out "$OUTPUT_DIR/streams" \
    --target_n 2000 \
    --drift_fraction 0.5 \
    --post_unsafe_frac 0.75 \
    --confidence_shift_delta 0.20 \
    --seed 42

# ---- Step 2: Calibrate thresholds ----
echo ""
echo "--- STEP 2: Calibrate thresholds ---"
python scripts/calibrate_thresholds.py \
    --input "$OUTPUT_DIR/streams/iid_control.jsonl" \
    --out "$OUTPUT_DIR/calibrated_thresholds.json" \
    --target_fpr 0.05 \
    --n_bootstrap 500 \
    --window_size 200 \
    --bins 20 \
    --binning equal_width

# ---- Step 3: Run experiments ----
echo ""
echo "--- STEP 3: Sensitivity and ablation experiments ---"
python scripts/run_drift_experiments_v2.py \
    --streams_dir "$OUTPUT_DIR/streams" \
    --thresholds "$OUTPUT_DIR/calibrated_thresholds.json" \
    --out "$OUTPUT_DIR" \
    --window_sizes 50 100 200 500 \
    --bins 10 20 30 50 \
    --step_fractions 1.0 0.5 \
    --reference_modes fixed_first rolling multi_reference adaptive_clean \
    --detector_rules KL_only KS_only KL_AND_KS JS_only Wasserstein_only Ensemble_majority

# ---- Validation: confirm output files exist ----
echo ""
echo "======================================================================"
echo "OUTPUT VALIDATION:"
PASS=true
for f in \
    "$OUTPUT_DIR/tables/sensitivity_analysis.csv" \
    "$OUTPUT_DIR/tables/ablation_detector_rules.csv" \
    "$OUTPUT_DIR/tables/stream_detection_summary.csv" \
    "$OUTPUT_DIR/calibrated_thresholds.json"
do
    if [ -f "$f" ]; then
        ROWS=$(wc -l < "$f" 2>/dev/null || echo "?")
        echo "  ✓  $f  ($ROWS lines)"
    else
        echo "  ✗  MISSING: $f"
        PASS=false
    fi
done

echo ""
if [ "$PASS" = true ]; then
    echo "SMOKE TEST PASSED — All output files generated."
    echo ""
    echo "REMINDER: These results are for code verification only."
    echo "          XLM-R (B1) is NOT the DriftGuard-LLM proposed system."
    echo "          Do NOT use these tables in the paper as DriftGuard results."
else
    echo "SMOKE TEST FAILED — Some output files are missing."
    exit 1
fi
echo "======================================================================"
