#!/usr/bin/env bash
# =============================================================================
# run_all_drift_v2.sh — DriftGuard-LLM v2 pipeline dispatcher
# =============================================================================
# Usage:
#   bash scripts/run_all_drift_v2.sh              # default: final E1 mode
#   bash scripts/run_all_drift_v2.sh --smoke      # XLM-R smoke test
#   bash scripts/run_all_drift_v2.sh --final      # Final E1 (explicit)
#
# Delegates to the appropriate single-mode script:
#   Smoke test  → scripts/run_smoke_test_xlmr.sh
#   Final E1    → scripts/run_final_e1_drift.sh
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="final"  # default
if [ "${1:-}" = "--smoke" ]; then
    MODE="smoke"
elif [ "${1:-}" = "--final" ]; then
    MODE="final"
elif [ -n "${1:-}" ]; then
    echo "Unknown option: $1"
    echo "Usage: $0 [--smoke | --final]"
    exit 1
fi

if [ "$MODE" = "smoke" ]; then
    echo "Mode: XLM-R SMOKE TEST"
    bash scripts/run_smoke_test_xlmr.sh
else
    echo "Mode: FINAL E1 QLoRA (paper result)"
    bash scripts/run_final_e1_drift.sh
fi
