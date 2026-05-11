#!/usr/bin/env bash
# =============================================================================
# run_final_e1_drift.sh — Final DriftGuard-LLM result using E1 QLoRA predictions
# =============================================================================
# PURPOSE: Produces the main paper result for the DriftGuard-LLM drift
#          detection section using E1 QLoRA fine-tuned LLaMA-3.1 8B predictions
#          that have been regenerated with real unsafe_probability values.
#
# REQUIREMENTS (must be satisfied before this script will proceed):
#   1. predictions/e1_qlora_ft_predictions_v2.jsonl must exist
#   2. File must have unsafe_confidence or unsafe_probability column
#   3. Confidence must NOT be constant
#   4. Confidence must NOT be all 0.5
#   5. Row count should be ≥ 4000 (Phase 1 full test set)
#
# If any requirement fails, this script STOPS and tells you exactly what to do.
#
# Run: bash scripts/run_final_e1_drift.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

SOURCE_PREDICTIONS="predictions/e1_qlora_ft_predictions_v2.jsonl"
OUTPUT_DIR="results/drift_v2_e1_final"
LABEL="E1 QLoRA Fine-tuned — FINAL PAPER RESULT"

echo "======================================================================"
echo "DriftGuard-LLM v2 — FINAL E1 EXPERIMENT"
echo "  Source : $SOURCE_PREDICTIONS"
echo "  Output : $OUTPUT_DIR"
echo "  This output WILL be used as the paper's main drift result."
echo "======================================================================"

# Load shared validation
source scripts/_validate_predictions.sh

# ---- Strict pre-flight validation ----
validate_predictions "$SOURCE_PREDICTIONS" "$LABEL"
VALID_EXIT=$?
if [ $VALID_EXIT -ne 0 ]; then
    echo ""
    echo "======================================================================"
    echo "PIPELINE BLOCKED — E1 predictions did not pass validation."
    echo ""
    echo "REQUIRED ACTION:"
    echo "  1. Run inference with evaluator_v2.py to generate real token probabilities:"
    echo "     bash scripts/run_e1_inference_v2.sh"
    echo ""
    echo "  2. After inference completes, verify the output:"
    echo "     python scripts/audit_prediction_files.py \\"
    echo "         --input_dir predictions \\"
    echo "         --out results/drift_v2_e1_final/audit"
    echo ""
    echo "  3. Re-run this script:"
    echo "     bash scripts/run_final_e1_drift.sh"
    echo "======================================================================"
    exit $VALID_EXIT
fi

mkdir -p "$OUTPUT_DIR"/{streams,tables,reports,audit}

# ---- Step 0: Full audit of E1 v2 predictions ----
echo ""
echo "--- STEP 0: Audit E1 v2 prediction file ---"
python scripts/audit_prediction_files.py \
    --input_dir predictions \
    --out "$OUTPUT_DIR/audit"
AUDIT_EXIT=$?
if [ $AUDIT_EXIT -eq 2 ]; then
    echo ""
    echo "STOP: Audit failed for E1 v2 predictions."
    echo "      Fix the issues above, then re-run this script."
    exit 2
fi

# ---- Step 1: Generate drift streams ----
echo ""
echo "--- STEP 1: Generate drift streams from E1 QLoRA ---"
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
echo "--- STEP 2: Calibrate thresholds from iid_control ---"
python scripts/calibrate_thresholds.py \
    --input "$OUTPUT_DIR/streams/iid_control.jsonl" \
    --out "$OUTPUT_DIR/calibrated_thresholds.json" \
    --target_fpr 0.05 \
    --n_bootstrap 500 \
    --window_size 200 \
    --bins 20 \
    --binning equal_width

# ---- Step 3: Sensitivity and ablation ----
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

# ---- Validation: confirm all output files exist ----
echo ""
echo "======================================================================"
echo "OUTPUT VALIDATION:"
PASS=true
declare -A EXPECTED_FILES=(
    ["$OUTPUT_DIR/tables/sensitivity_analysis.csv"]="sensitivity analysis table"
    ["$OUTPUT_DIR/tables/ablation_detector_rules.csv"]="ablation table"
    ["$OUTPUT_DIR/tables/stream_detection_summary.csv"]="stream summary table"
    ["$OUTPUT_DIR/calibrated_thresholds.json"]="calibrated thresholds"
)
for f in "${!EXPECTED_FILES[@]}"; do
    label="${EXPECTED_FILES[$f]}"
    if [ -f "$f" ]; then
        ROWS=$(wc -l < "$f" 2>/dev/null || echo "?")
        echo "  ✓  [$label]  $f  ($ROWS lines)"
    else
        echo "  ✗  MISSING [$label]: $f"
        PASS=false
    fi
done

echo ""
if [ "$PASS" = true ]; then
    echo "FINAL E1 EXPERIMENT COMPLETE — All output files generated."
    echo ""
    echo "These results use real E1 QLoRA unsafe_probability values."
    echo "They are valid for inclusion in the DriftGuard-LLM paper."
else
    echo "EXPERIMENT INCOMPLETE — Some output files are missing."
    exit 1
fi
echo "======================================================================"
