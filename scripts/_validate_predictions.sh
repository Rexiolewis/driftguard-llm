#!/usr/bin/env bash
# =============================================================================
# _validate_predictions.sh  — shared pre-flight validation
# Source this file; call validate_predictions "$FILE" "$LABEL"
# Returns exit 0 if valid, exit 2 if STOP condition met.
# =============================================================================

validate_predictions() {
    local FILE="$1"
    local LABEL="${2:-predictions}"

    echo ""
    echo "----------------------------------------------------------------------"
    echo "PRE-FLIGHT VALIDATION: $LABEL"
    echo "  File: $FILE"
    echo "----------------------------------------------------------------------"

    # 1. File must exist
    if [ ! -f "$FILE" ]; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "STOP: $LABEL file not found."
        echo "  Expected: $FILE"
        echo "  If this is E1, run:  bash scripts/run_e1_inference_v2.sh"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        return 2
    fi

    # 2. Row count
    local NROWS
    NROWS=$(wc -l < "$FILE" | tr -d ' ')
    echo "  Rows: $NROWS"
    if [ "$NROWS" -lt 100 ]; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "STOP: $LABEL has only $NROWS rows — too few for drift experiments."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        return 2
    fi
    if [ "$NROWS" -lt 4000 ] && echo "$FILE" | grep -qv "phase2\|transfer"; then
        echo "  WARNING: Phase 1 file has only $NROWS rows (expected ≥4000). Check for dropped records."
    fi

    # 3. Check for unsafe_confidence or unsafe_probability column
    local HAS_UNSAFE_CONF
    HAS_UNSAFE_CONF=$(python3 - <<PYEOF
import json, sys
found = False
with open("$FILE") as f:
    for i, line in enumerate(f):
        if i >= 5: break
        try:
            r = json.loads(line)
            if "unsafe_confidence" in r or "unsafe_probability" in r:
                found = True
                break
        except: pass
print("yes" if found else "no")
PYEOF
)
    echo "  unsafe_confidence / unsafe_probability column: $HAS_UNSAFE_CONF"

    # 4. Confidence value analysis
    local CONF_ANALYSIS
    CONF_ANALYSIS=$(python3 - <<PYEOF
import json, sys, statistics

vals = []
fields_tried = ["unsafe_confidence", "unsafe_probability", "pred_confidence"]
field_used = None

with open("$FILE") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            r = json.loads(line)
        except: continue
        for fld in fields_tried:
            v = r.get(fld)
            if v is not None:
                try:
                    vals.append(float(v))
                    if field_used is None:
                        field_used = fld
                    break
                except: pass

if not vals:
    print("NO_CONFIDENCE_VALUES")
    sys.exit(0)

n_unique = len(set(round(v, 6) for v in vals))
mn = min(vals)
mx = max(vals)
mean = sum(vals) / len(vals)
std = statistics.stdev(vals) if len(vals) > 1 else 0.0
is_constant = n_unique == 1
all_half = is_constant and abs(mean - 0.5) < 1e-6
all_zero = is_constant and abs(mean) < 1e-6

print(f"field={field_used} n={len(vals)} unique={n_unique} min={mn:.4f} max={mx:.4f} mean={mean:.4f} std={std:.4f} constant={is_constant} all_half={all_half} all_zero={all_zero}")
PYEOF
)
    echo "  Confidence: $CONF_ANALYSIS"

    # 5. Stop conditions
    if echo "$CONF_ANALYSIS" | grep -q "NO_CONFIDENCE_VALUES"; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "STOP: $LABEL has no confidence values (unsafe_confidence, unsafe_probability,"
        echo "      or pred_confidence). This prediction file cannot be used for"
        echo "      confidence-based drift analysis."
        echo ""
        echo "  Action required: Rerun inference with evaluator_v2.py."
        echo "  Command:         bash scripts/run_e1_inference_v2.sh"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        return 2
    fi

    if echo "$CONF_ANALYSIS" | grep -q "all_half=True"; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "STOP: E1 predictions are not valid for confidence-based drift."
        echo "      pred_confidence is constant 0.5 for ALL rows."
        echo "      This means the evaluator did not extract real token probabilities."
        echo ""
        echo "  Action required: Rerun evaluator_v2.py to generate real unsafe_probability"
        echo "                   and safe_probability from model logits."
        echo "  Command:         bash scripts/run_e1_inference_v2.sh"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        return 2
    fi

    if echo "$CONF_ANALYSIS" | grep -q "all_zero=True"; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "STOP: $LABEL predictions are not valid — confidence is constant 0.0."
        echo "      Cannot run confidence-based drift."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        return 2
    fi

    if echo "$CONF_ANALYSIS" | grep -q "constant=True"; then
        echo ""
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "STOP: $LABEL has constant confidence (all identical values)."
        echo "      Confidence-histogram drift will be uninformative."
        echo "      Rerun inference with evaluator_v2.py."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        return 2
    fi

    echo "  VALIDATION PASSED — confidence values look usable."
    echo "----------------------------------------------------------------------"
    return 0
}
