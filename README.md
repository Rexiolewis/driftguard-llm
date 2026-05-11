# DriftGuard-LLM v2 — Concept Drift Detection for Multilingual Content Moderation

---

## ⛔ Important Experimental Rules (Read First)

These rules govern what is and is not a valid paper result. Violating them will
produce misleading drift claims that reviewers will reject.

| Rule | Requirement |
|------|-------------|
| **Main result must be E1** | The DriftGuard-LLM paper result must use E1 QLoRA fine-tuned LLaMA-3.1 8B predictions with real `unsafe_probability` values extracted from token logits. |
| **E0 must not be the main result** | E0 is a zero-shot baseline, not the proposed DriftGuard-LLM system. Drift results from E0 cannot represent DriftGuard-LLM capability. |
| **XLM-R and LlamaGuard-3 are smoke-test / baseline models only** | These can be used to verify the pipeline runs. They must never be labelled as DriftGuard-LLM results in paper tables or text. |
| **Constant 0.5 confidence = invalid** | Any prediction file where `pred_confidence` or `unsafe_confidence` is constant 0.5 is not valid for confidence-based drift analysis. The pipeline will STOP if this is detected. |
| **E3 (3 rows) must be excluded** | `e3_qlora_mm_predictions.jsonl` has only 3 rows and is unusable for any multi-window drift experiment. It must not appear in any drift result table. |
| **Phase 2 (208 rows) is exploratory only** | Phase 2 files have 208 rows — too few for W=200 multi-window drift. Reduce window size to ≤ 69 and explicitly label results as "exploratory small-sample analysis" in the paper. |
| **Require real token probabilities for paper submission** | Before submitting the revised paper, E1 must be rerun with `evaluator_v2.py` to generate real `unsafe_probability` and `safe_probability`. Files with constant confidence cannot be used. |

**Scripts enforce these rules automatically.** If validation fails, the script
stops with a clear STOP message and tells you exactly what to do next.

---

## Overview

This repository accompanies the paper *"DriftGuard-LLM: Concept Drift Monitoring for QLoRA Fine-Tuned LLaMA-3.1 8B Harmful Content Moderation"* and contains all code needed to reproduce the drift detection experiments, sensitivity analysis, and ablation tables.

---

## ⚠️ Critical Data Limitation — Read Before Reproducing

The downloaded prediction files have **constant confidence values** due to the original evaluator saving a fixed threshold rather than real token probabilities:

| File | Issue |
|------|-------|
| `e0_zero_shot_predictions.jsonl` | `pred_confidence` = 0.7 (constant — threshold artefact) |
| `e1_qlora_ft_predictions.jsonl` | `pred_confidence` = 0.5 (constant — no logit extraction) |
| `e2_qlora_la_predictions.jsonl` | `pred_confidence` = 0.7 (constant) |
| `e4_qlora_rag_predictions.jsonl` | `pred_confidence` = 0.7 (constant) |

**Consequence:** Confidence-histogram drift results from these files are not valid for paper claims. The `audit_prediction_files.py` script will catch this and print a STOP warning.

**Fix required:** Rerun inference using `evaluator_v2.py` which extracts real `unsafe_probability` and `safe_probability` from the model's first-token logits.

---

## Baseline Fairness Note

> **This is mandatory disclosure for the paper.**

The experiments include two types of baselines that must **not** be compared as if controlled:

| Model | Type | Fair comparison label |
|-------|------|-----------------------|
| XLM-R (fine-tuned) | Encoder baseline B1 | Fine-tuned cross-architecture |
| LlamaGuard-3 8B | Zero-shot off-the-shelf | Operational baseline B2 |
| LlamaGuard-4 12B (E5) | Zero-shot off-the-shelf | Operational baseline (larger model) |
| LLaMA-3.1 8B zero-shot (E0) | Zero-shot | Ablation reference |
| LLaMA-3.1 8B QLoRA (E1–E4) | Fine-tuned (proposed) | Proposed system |

**LlamaGuard is a zero-shot off-the-shelf safety classifier.** It was not fine-tuned on the same data as QLoRA. Presenting QLoRA as "outperforming" LlamaGuard does not constitute a fair controlled comparison — it reflects the advantage of in-domain fine-tuning. This must be clearly stated in the paper's experimental section.

---

## Project Structure

```
drift_v2_project/
├── predictions/                     # Downloaded prediction JSONL files
│   ├── e0_zero_shot_predictions.jsonl
│   ├── e1_qlora_ft_predictions.jsonl
│   ├── e2_qlora_la_predictions.jsonl
│   ├── e3_phase2_captioned_predictions.jsonl
│   ├── e3_qlora_mm_predictions.jsonl      ← only 3 rows — unusable
│   ├── e4_qlora_rag_predictions.jsonl
│   ├── e5_lg4_predictions.jsonl
│   ├── lg3_baseline_predictions.jsonl
│   ├── transfer_phase2_predictions.jsonl
│   └── xlmr_baseline_predictions.jsonl
├── scripts/
│   ├── drift_detector_v2.py             # Core detector (confidence-histogram KL/JS/KS/Wass)
│   ├── evaluator_v2.py                  # Extended evaluator (saves unsafe_probability)
│   ├── audit_prediction_files.py        # Step 0: audit all JSONL files
│   ├── calibrate_thresholds.py          # Step 2: bootstrap threshold calibration
│   ├── generate_drift_streams.py        # Step 1: create 8 synthetic drift streams
│   ├── run_drift_experiments_v2.py      # Step 3: sensitivity + ablation grid
│   ├── run_all_drift_v2.sh              # Shell: run full pipeline locally
│   └── run_drift_v2.slurm              # HPC: SLURM batch job
├── results/
│   └── drift_v2/
│       ├── audit/                       # prediction_file_audit.csv / .json
│       ├── streams/                     # synthetic drift stream JSONL + metadata
│       ├── tables/                      # sensitivity_analysis.csv, ablation_*.csv
│       └── reports/                     # per-stream JSON reports
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

**Required:** `scipy>=1.10.0` for `ks_2samp` and `wasserstein_distance`.

---

## Step-by-Step Reproduction

### Step 0 — Audit prediction files

Always run the audit first. It will:
- Report row counts, confidence distributions, label distributions
- Warn if confidence is constant (invalid for drift analysis)
- Print a **STOP** message if E1 has all-0.5 confidence
- Generate `results/drift_v2/audit/prediction_file_audit.csv`

```bash
python scripts/audit_prediction_files.py \
  --input_dir predictions \
  --out results/drift_v2/audit
```

**If the audit prints STOP:** You must rerun inference before proceeding.

---

### Step 0b — Regenerate unsafe_probability (required before paper submission)

The original evaluator does not save token-level probabilities. Run `evaluator_v2.py` on the HPC cluster to regenerate predictions with real `unsafe_probability` and `safe_probability`:

```bash
# On DICC HPC — requires GPU and model checkpoint
python -m moderation_exp.experiments.run \
    --experiment e1_qlora_ft \
    --evaluator v2 \
    --model_path /path/to/llama3_1_8b_instruct \
    --adapter_path /path/to/e1_qlora_adapter \
    --input data/processed/processed_v2.jsonl \
    --output results/e1_v2/
```

The new `predictions.jsonl` will include:
- `unsafe_probability` — real P(UNSAFE) from first token logits
- `safe_probability` — real P(SAFE)
- `unsafe_confidence` — == unsafe_probability
- `pred_confidence` — max(safe_prob, unsafe_prob)

---

### Step 1 — Generate synthetic drift streams

```bash
python scripts/generate_drift_streams.py \
  --input predictions/e0_zero_shot_predictions.jsonl \
  --out results/drift_v2/streams \
  --seed 42
```

This creates 8 streams in `results/drift_v2/streams/`:

| Stream | Drift Type |
|--------|-----------|
| `iid_control` | No drift — stratified shuffle |
| `language_sudden` | EN/MS → ZH/TA sudden shift at 50% |
| `language_gradual` | ZH/TA fraction grows 0%→80% |
| `class_prior_sudden` | Balanced → 75% UNSAFE sudden |
| `class_prior_gradual` | UNSAFE ratio 30%→70% gradual |
| `source_sudden` | Source distribution sudden shift |
| `confidence_shift_sudden` | unsafe_confidence +0.20 after 50% |
| `confidence_shift_gradual` | unsafe_confidence grows gradually |
| `adversarial_text_stream_input` | Perturbed text — requires re-inference |

**Note:** All streams except `adversarial_text_stream_input` are **prediction-level simulations**. They manipulate existing predictions — the LLM is not re-run. Mark all results from these streams as "controlled prediction-level simulation" in the paper. Never present them as real deployment drift.

---

### Step 2 — Calibrate thresholds

```bash
python scripts/calibrate_thresholds.py \
  --input results/drift_v2/streams/iid_control.jsonl \
  --out results/drift_v2/calibrated_thresholds.json \
  --target_fpr 0.05 \
  --n_bootstrap 500
```

Output `calibrated_thresholds.json` contains statistically grounded thresholds:
- `kl_threshold_95` — KL divergence at 95th percentile under no-drift
- `js_threshold_95`, `wasserstein_threshold_95`, `ks_stat_threshold_95`
- All set at the (1 - target_fpr) quantile of the bootstrap null distribution

These replace the arbitrary KL=0.05/0.15 thresholds from the original paper.

---

### Step 3 — Run sensitivity analysis and ablation

```bash
python scripts/run_drift_experiments_v2.py \
  --streams_dir results/drift_v2/streams \
  --thresholds results/drift_v2/calibrated_thresholds.json \
  --out results/drift_v2 \
  --window_sizes 50 100 200 500 \
  --bins 10 20 30 50 \
  --step_fractions 1.0 0.5 \
  --reference_modes fixed_first rolling multi_reference adaptive_clean \
  --detector_rules KL_only KS_only KL_AND_KS JS_only Wasserstein_only Ensemble_majority
```

**Output tables:**

| File | Purpose |
|------|---------|
| `sensitivity_analysis.csv` | Full grid: window_size × bins × binning × step |
| `ablation_detector_rules.csv` | Rule × reference_mode at W=200, bins=20 |
| `stream_detection_summary.csv` | Best config per stream |

---

### Run everything at once (local)

```bash
bash scripts/run_all_drift_v2.sh
```

### Run on DICC HPC

```bash
sbatch scripts/run_drift_v2.slurm
squeue -u $USER  # monitor
```

---

## Drift Detector Architecture (v2)

`drift_detector_v2.py` replaces the original detector with the following improvements:

### Input
- `unsafe_confidence` values in [0, 1] derived from token logits

### Signal
- Histogram over `unsafe_confidence` (not label counts)

### Metrics
| Metric | Method | Notes |
|--------|--------|-------|
| KL divergence | Histogram-based D(P‖Q) | Primary metric |
| Jensen-Shannon | Symmetric JS divergence | Bounded [0, ln 2] |
| KS test | `scipy.stats.ks_2samp` | Returns statistic **and p-value** |
| Wasserstein | `scipy.stats.wasserstein_distance` | Earth mover's distance |

### Window modes
| Parameter | Options |
|-----------|---------|
| `window_size` | 50, 100, 200, 500 |
| `step_size` | window_size (non-overlapping), window_size//2 (50% overlap) |
| `bins` | 10, 20, 30, 50 |
| `binning` | equal_width, quantile |

### Reference modes
| Mode | Description |
|------|-------------|
| `fixed_first` | Reference = first window, never updated |
| `rolling` | Reference = most recent window |
| `multi_reference` | Reference = all previous windows combined |
| `adaptive_clean` | Update reference only when no drift detected |

### Detector rules
| Rule | Condition |
|------|-----------|
| `KL_only` | KL ≥ threshold |
| `KS_only` | KS p-value < α |
| `KL_AND_KS` | Both conditions |
| `JS_only` | JS ≥ threshold |
| `Wasserstein_only` | W ≥ threshold |
| `Ensemble_majority` | ≥ 2 of 4 detectors fire |

---

## Data Sharing Limitation

The prediction JSONL files are derived outputs from HuggingFace public datasets:
- `nahiar/HS_df_bahasa_inggris_hs` (EN)
- `nahiar/HS_df_tambahan_bahasa_inggris` (EN)
- `textdetox/multilingual_toxicity_dataset` (ZH/EN)
- `krishan-CSE/Tamil_Hate_Speech` (TA)
- `mohanrj/MYBully` (MS)

The original text content cannot be redistributed under the source dataset licenses. Researchers wishing to reproduce results from raw text must download the datasets from HuggingFace and re-run inference. A **sample of 100 records per experiment** (without original text, only predictions and metadata) is available in `predictions/sample_100/` for code testing purposes.

---

## Known Issues and Limitations

1. **E3 multimodal (e3_qlora_mm_predictions.jsonl):** Only 3 rows. Unusable for drift experiments. Do not include in any paper tables without this explicit disclosure.

2. **Phase 2 files (208 rows):** Too small for W=200 non-overlapping drift with multiple windows. Reduce to W=30–50 and mark results as "exploratory" in the paper.

3. **Constant confidence (E0, E1, E2, E4):** All Phase 1 experiments have constant `pred_confidence`. This is an evaluator bug — see Step 0b above.

4. **Row count mismatch:** E0/E1/E2/E4/E5 have 4217 rows; baselines (XLM-R, LG-3) have 4282 rows. Investigate whether 65 records were dropped in pre-processing.

---

## Citation

If you use this code, please cite:

```bibtex
@article{saravanan2026driftguard,
  title={DriftGuard-LLM: Concept Drift Monitoring for QLoRA Fine-Tuned LLaMA-3.1 8B Harmful Content Moderation},
  author={Saravanan, ...},
  note={Under review},
  year={2026}
}
```
