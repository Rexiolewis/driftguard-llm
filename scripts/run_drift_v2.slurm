#!/bin/bash
#SBATCH --job-name=driftguard_v2
#SBATCH --output=logs/drift_v2_%j.out
#SBATCH --error=logs/drift_v2_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1          # Only needed if rerunning inference (evaluator_v2.py)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nanavaras.s11@gmail.com

# =============================================================================
# DriftGuard-LLM v2 — DICC HPC SLURM Job Script
# =============================================================================
# Submit with: sbatch scripts/run_drift_v2.slurm
# Monitor   : squeue -u $USER
# Cancel    : scancel <JOBID>
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
echo "Job started: $(date)"
echo "Node: $HOSTNAME  SLURM_JOB_ID: $SLURM_JOB_ID"

module purge
module load python/3.10
module load cuda/12.1         # adjust to DICC available CUDA version

# Project root (edit this path to match your HPC workspace)
PROJECT_DIR="$HOME/moderation_exp/drift_v2_project"
cd "$PROJECT_DIR"

# Activate virtual environment
source venv/bin/activate
echo "Python: $(python --version)"
echo "Working dir: $(pwd)"

mkdir -p logs results/drift_v2/{audit,streams,tables,reports}

# ---------------------------------------------------------------------------
# Step 0: Audit
# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 0: Audit ---"
python scripts/audit_prediction_files.py \
    --input_dir predictions \
    --out results/drift_v2/audit
AUDIT_EXIT=$?

if [ $AUDIT_EXIT -eq 2 ]; then
    echo "STOP: Constant confidence detected. Rerun inference before submitting paper results."
    exit 2
fi

# ---------------------------------------------------------------------------
# (Optional) Step 0b: Re-run inference for E1 with real token probabilities
# Uncomment if you have rerun inference with evaluator_v2.py
# ---------------------------------------------------------------------------
# echo "--- STEP 0b: Re-run E1 inference ---"
# python -m moderation_exp.experiments.run \
#     --experiment e1_qlora_ft \
#     --evaluator evaluator_v2 \
#     --model_path /path/to/llama3_1_8b \
#     --adapter_path /path/to/e1_adapter \
#     --input data/processed/processed_v2.jsonl \
#     --output results/e1_v2

# ---------------------------------------------------------------------------
# Step 1: Generate streams
# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 1: Generate drift streams ---"
python scripts/generate_drift_streams.py \
    --input predictions/e0_zero_shot_predictions.jsonl \
    --out results/drift_v2/streams \
    --target_n 2000 \
    --drift_fraction 0.5 \
    --post_unsafe_frac 0.75 \
    --confidence_shift_delta 0.20 \
    --seed 42

# ---------------------------------------------------------------------------
# Step 2: Calibrate thresholds
# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 2: Calibrate thresholds ---"
python scripts/calibrate_thresholds.py \
    --input results/drift_v2/streams/iid_control.jsonl \
    --out results/drift_v2/calibrated_thresholds.json \
    --target_fpr 0.05 \
    --n_bootstrap 1000 \
    --window_size 200 \
    --bins 20 \
    --binning equal_width

# ---------------------------------------------------------------------------
# Step 3: Run full sensitivity and ablation grid
# ---------------------------------------------------------------------------
echo ""
echo "--- STEP 3: Run experiments ---"
python scripts/run_drift_experiments_v2.py \
    --streams_dir results/drift_v2/streams \
    --thresholds results/drift_v2/calibrated_thresholds.json \
    --out results/drift_v2 \
    --window_sizes 50 100 200 500 \
    --bins 10 20 30 50 \
    --binnings equal_width quantile \
    --step_fractions 1.0 0.5 \
    --reference_modes fixed_first rolling multi_reference adaptive_clean \
    --detector_rules KL_only KS_only KL_AND_KS JS_only Wasserstein_only Ensemble_majority

echo ""
echo "Job complete: $(date)"
echo "Results in: $PROJECT_DIR/results/drift_v2/"
