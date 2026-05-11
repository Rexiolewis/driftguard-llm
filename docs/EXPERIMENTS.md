# Reproducing the DriftGuard-LLM Drift Experiments

This guide walks through every step to reproduce the drift detection results from scratch — from raw predictions to the final paper tables.

---

## Prerequisites

```bash
git clone https://github.com/YOUR_USERNAME/driftguard-llm.git
cd driftguard-llm
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

You will also need prediction files. The experiment uses real token-level probabilities (`unsafe_probability`, `safe_probability`) extracted from LLaMA-3.1 8B model outputs. See **Step 0** for how these are generated.

---

## Step 0 — Generate E1 QLoRA Predictions (HPC)

> Skip this step if you already have `predictions/e1_qlora_ft_predictions_v2.jsonl`.

The E1 QLoRA fine-tuned model runs on GPU only. We provide a self-contained SLURM script for the DICC HPC cluster.

**Required paths on HPC:**
```
models/llama31_8b_instruct/           ← LLaMA-3.1 8B Instruct weights
artifacts/checkpoints/e1_qlora/adapter/  ← QLoRA LoRA adapter
data/processed_v2/text_only_test.jsonl   ← 4,282-row test set
```

**Submit the job:**
```bash
# From your HPC project directory
sbatch scripts/jobs/e1_v2_inference_v2.sh
```

The script creates an isolated venv (`.venv_e1_infer`), installs torch+cu121, loads the model in fp16, and runs batched inference (batch_size=16) with `output_scores=True` to extract real first-token logit probabilities.

**Resume support:** If the job times out, resubmit — it resumes from the last checkpoint automatically.

**Output:**
```
results/e1_v2_inference/predictions.jsonl   ← 4,282 rows, real unsafe_confidence
results/e1_v2_inference/confidence_summary.json
```

**Validate the output:**
```bash
python scripts/audit_prediction_files.py \
    --input_dir results/e1_v2_inference \
    --out results/e1_v2_inference/audit
```

**Copy to local predictions folder:**
```bash
cp results/e1_v2_inference/predictions.jsonl \
   predictions/e1_qlora_ft_predictions_v2.jsonl
```

---

## Step 1 — Validate Predictions

Before running any drift experiment, audit all prediction files:

```bash
python scripts/audit_prediction_files.py \
    --input_dir predictions \
    --out results/audit
```

The audit checks:
- Required fields present (`id`, `lang`, `label`, `pred_label`, `unsafe_confidence`, ...)
- `unsafe_confidence` is not constant, not all 0.5, not all zero
- Row count matches expected (~4,282 for Phase 1 test set)

---

## Step 2 — Run the Full E1 Final Experiment

This is the single command that runs the complete pipeline:

```bash
bash scripts/run_final_e1_drift.sh
```

It executes four sub-steps automatically:

| Sub-step | Script | What it does |
|----------|--------|--------------|
| 0 | `audit_prediction_files.py` | Pre-flight validation of all prediction files |
| 1 | `generate_drift_streams.py` | Generates 8 synthetic drift streams (2,000 records each) |
| 2 | `calibrate_thresholds.py` | Bootstrap threshold calibration on IID control stream |
| 3 | `run_drift_experiments_v2.py` | 12,288 experiments (1,536 configs × 8 streams) |

**Output files:**
```
results/drift_v2_e1_final/
├── calibrated_thresholds.json          ← Bootstrap-calibrated KL/JS/Wass/KS thresholds
├── streams/                            ← 8 JSONL drift streams (2,000 rows each)
├── tables/
│   ├── sensitivity_analysis.csv        ← 12,288 rows: all configs × streams
│   ├── ablation_detector_rules.csv     ← 192 rows: ablation subset
│   ├── stream_detection_summary.csv    ← 8-row stream summary
│   ├── paper_primary_results.csv       ← Primary config results (paper Table 1)
│   └── paper_ablation_summary.csv      ← Ablation summary (paper Table 2)
└── reports/                            ← Per-stream detection reports
```

---

## Step 3 — Generate Paper Tables

```bash
python3 - <<'EOF'
import pandas as pd
from pathlib import Path

sa = pd.read_csv("results/drift_v2_e1_final/tables/sensitivity_analysis.csv")

# Primary config: Ensemble_majority, fixed_first, W=200, step=200, bins=20
primary = sa[
    (sa.detector_rule == "Ensemble_majority") &
    (sa.reference_mode == "fixed_first") &
    (sa.window_size == 200) &
    (sa.step_size == 200) &
    (sa.bins == 20) &
    (sa.binning == "equal_width")
]
print(primary[["stream_name","drift_detected","detection_delay","false_alarm_count_before_drift"]].to_string())
EOF
```

Or simply open `results/drift_v2_e1_final/tables/paper_primary_results.csv` — it contains the pre-computed paper table.

---

## Step 4 — Run XLM-R Smoke Test (Optional Baseline Validation)

```bash
bash scripts/run_smoke_test_xlmr.sh
```

Verifies the XLM-R baseline prediction file is valid and generates its drift results. Used as a sanity check that the pipeline works correctly end-to-end.

---

## Synthetic Drift Stream Types

| Stream | Drift Type | Drift Point | Description |
|--------|-----------|-------------|-------------|
| `iid_control` | None | — | IID sample, no drift; used for threshold calibration |
| `class_prior_gradual` | Label shift | 700 | UNSAFE fraction increases gradually after window 700 |
| `class_prior_sudden` | Label shift | 1000 | Sudden jump in UNSAFE fraction at record 1000 |
| `confidence_shift_gradual` | Confidence | 600 | Model confidence distribution shifts gradually |
| `confidence_shift_sudden` | Confidence | 1000 | Sudden confidence distribution change |
| `language_gradual` | Covariate | 600 | Gradual shift toward a minority language |
| `language_sudden` | Covariate | 1000 | Sudden language distribution change |
| `source_sudden` | Covariate | 1000 | Sudden change in data source distribution |

All streams use 2,000 records with `seed=42`.

---

## Detector Configurations

The sensitivity grid sweeps:

| Parameter | Values |
|-----------|--------|
| `window_size` | 50, 100, 200, 500 |
| `step_size` | `window_size × step_fraction` |
| `step_fraction` | 1.0, 0.5 |
| `bins` | 10, 20, 30, 50 |
| `binning` | `equal_width`, `quantile` |
| `reference_mode` | `fixed_first`, `rolling`, `multi_reference`, `adaptive_clean` |
| `detector_rule` | `KL_only`, `KS_only`, `KL_AND_KS`, `JS_only`, `Wasserstein_only`, `Ensemble_majority` |

**Recommended configuration** (used in paper):  
`Ensemble_majority` + `fixed_first` + W=200 + step=200 + bins=20 + `equal_width`

This achieves **7/7 drift stream detection**, **0 IID false alarms**, and **0 pre-drift false alarms**.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `scripts/drift_detector_v2.py` | Core histogram-based drift detector |
| `scripts/evaluator_v2.py` | Token-logit probability extraction from LLaMA predictions |
| `scripts/generate_drift_streams.py` | Synthetic drift stream generator |
| `scripts/calibrate_thresholds.py` | Bootstrap threshold calibration |
| `scripts/run_drift_experiments_v2.py` | Sensitivity + ablation experiment runner |
| `scripts/audit_prediction_files.py` | Pre-flight validation of prediction files |
| `sample_data/` | Small sample files for testing the pipeline |

---

## Sample Data Quick Test

To verify the pipeline works without a full prediction file:

```bash
# Test drift stream generation on sample data
python scripts/generate_drift_streams.py \
    --input sample_data/sample_predictions.jsonl \
    --out /tmp/test_streams \
    --target_n 25

# Test threshold calibration
python scripts/calibrate_thresholds.py \
    --input sample_data/sample_iid_control.jsonl \
    --out /tmp/test_thresholds.json \
    --n_bootstrap 50 \
    --window_size 10
```
