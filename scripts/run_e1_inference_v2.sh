#!/usr/bin/env bash
# =============================================================================
# run_e1_inference_v2.sh — Rerun E1 QLoRA inference with real token probabilities
# =============================================================================
# PURPOSE: Regenerate E1 QLoRA fine-tuned predictions with real unsafe_probability
#          and safe_probability extracted from model logits via evaluator_v2.py.
#          Output: predictions/e1_qlora_ft_predictions_v2.jsonl
#
# BEFORE RUNNING: fill in the three path variables below.
#   - MODEL_PATH   : path to base LLaMA-3.1 8B Instruct model weights
#   - ADAPTER_PATH : path to E1 QLoRA adapter checkpoint
#   - DATA_PATH    : path to Phase 1 test set (processed_v2.jsonl)
#
# Run: bash scripts/run_e1_inference_v2.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# =============================================================================
# !! FILL IN THESE PATHS BEFORE RUNNING !!
# =============================================================================
MODEL_PATH=""       # e.g. /home/saravanan/models/Llama-3.1-8B-Instruct
ADAPTER_PATH=""     # e.g. /home/saravanan/experiments/e1_qlora/adapter
DATA_PATH=""        # e.g. /home/saravanan/data/processed/processed_v2.jsonl
# =============================================================================

OUTPUT_JSONL="predictions/e1_qlora_ft_predictions_v2.jsonl"
OUTPUT_DIR="results/e1_v2_inference"

echo "======================================================================"
echo "E1 QLoRA Inference v2 — real token probability extraction"
echo "======================================================================"

# ---- Check paths are set ----
MISSING=false
if [ -z "$MODEL_PATH" ]; then
    echo "  MISSING: MODEL_PATH is not set."
    echo "    Set it to the directory containing the LLaMA-3.1 8B Instruct model weights."
    echo "    On DICC HPC, check: ls /home/\$USER/models/ or /scratch/models/"
    MISSING=true
fi
if [ -z "$ADAPTER_PATH" ]; then
    echo "  MISSING: ADAPTER_PATH is not set."
    echo "    Set it to the directory containing the E1 QLoRA LoRA adapter checkpoint."
    echo "    On DICC HPC, check: ls /home/\$USER/experiments/ or your training output dir."
    MISSING=true
fi
if [ -z "$DATA_PATH" ]; then
    echo "  MISSING: DATA_PATH is not set."
    echo "    Set it to the Phase 1 processed test JSONL."
    echo "    On DICC HPC, check: ls /home/\$USER/data/processed/"
    MISSING=true
fi

if [ "$MISSING" = true ]; then
    echo ""
    echo "======================================================================"
    echo "ACTION REQUIRED: Open scripts/run_e1_inference_v2.sh and fill in:"
    echo ""
    echo "  MODEL_PATH=\"/path/to/Llama-3.1-8B-Instruct\""
    echo "  ADAPTER_PATH=\"/path/to/e1_qlora_adapter\""
    echo "  DATA_PATH=\"/path/to/processed_v2.jsonl\""
    echo ""
    echo "Then re-run: bash scripts/run_e1_inference_v2.sh"
    echo "======================================================================"
    exit 1
fi

# ---- Check files exist ----
FOUND_MISSING=false
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "  ERROR: MODEL_PATH not found on disk: $MODEL_PATH"
    FOUND_MISSING=true
fi
if [ ! -d "$ADAPTER_PATH" ] && [ ! -f "$ADAPTER_PATH" ]; then
    echo "  ERROR: ADAPTER_PATH not found on disk: $ADAPTER_PATH"
    FOUND_MISSING=true
fi
if [ ! -f "$DATA_PATH" ]; then
    echo "  ERROR: DATA_PATH not found on disk: $DATA_PATH"
    FOUND_MISSING=true
fi
if [ "$FOUND_MISSING" = true ]; then
    echo ""
    echo "One or more paths do not exist. Check the paths above and try again."
    exit 1
fi

echo "  Model   : $MODEL_PATH"
echo "  Adapter : $ADAPTER_PATH"
echo "  Data    : $DATA_PATH"
echo "  Output  : $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR" predictions

# ---- Run evaluator_v2 inference ----
echo "Running E1 inference with evaluator_v2..."
python -c "
import sys
sys.path.insert(0, 'scripts')
from evaluator_v2 import run_evaluation_v2

metrics = run_evaluation_v2(
    input_path='$DATA_PATH',
    output_dir='$OUTPUT_DIR',
    model_path='$MODEL_PATH',
    local_files_only=True,
    adapter_path='$ADAPTER_PATH',
    use_mock_model=False,
    load_in_4bit=True,
    bf16=True,
    max_new_tokens=4,
    temperature=0.0,
    do_sample=False,
    uncertainty_conf_threshold=0.7,
    use_rag=False,
)
print('Metrics:', metrics)
"

# ---- Copy to canonical predictions path ----
if [ -f "$OUTPUT_DIR/predictions.jsonl" ]; then
    cp "$OUTPUT_DIR/predictions.jsonl" "$OUTPUT_JSONL"
    echo ""
    echo "======================================================================"
    echo "Inference complete."
    echo "  Raw output  : $OUTPUT_DIR/predictions.jsonl"
    echo "  Canonical   : $OUTPUT_JSONL"
    echo "  Conf summary: $OUTPUT_DIR/confidence_summary.json"
    echo ""
    echo "Next step: run the final drift experiment:"
    echo "  bash scripts/run_final_e1_drift.sh"
    echo "======================================================================"
else
    echo "ERROR: inference did not produce predictions.jsonl"
    echo "Check $OUTPUT_DIR/ for error logs."
    exit 1
fi

# ---- Quick confidence check ----
echo ""
echo "Quick confidence check on new predictions:"
python3 - <<PYEOF
import json, statistics
vals = []
with open("$OUTPUT_JSONL") as f:
    for line in f:
        r = json.loads(line)
        v = r.get("unsafe_confidence") or r.get("unsafe_probability")
        if v is not None:
            vals.append(float(v))

if not vals:
    print("  WARNING: no unsafe_confidence values found — extraction may have failed.")
else:
    n_unique = len(set(round(v, 6) for v in vals))
    print(f"  unsafe_confidence: n={len(vals)} unique={n_unique} "
          f"mean={sum(vals)/len(vals):.4f} "
          f"min={min(vals):.4f} max={max(vals):.4f}")
    if n_unique == 1:
        print("  WARNING: Still constant! Check logit extraction in evaluator_v2.py.")
    else:
        print("  Confidence values look varied — ready for drift analysis.")
PYEOF
