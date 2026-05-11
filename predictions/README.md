# predictions/

This directory holds model prediction files used for drift analysis.

Prediction files are **not committed** to this repository (they are large and contain research data).

## Required file for the E1 final experiment

```
predictions/e1_qlora_ft_predictions_v2.jsonl
```

**Schema** (one JSON object per line):

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique record identifier |
| `lang` | str | Language code (`en`, `ms`, `zh`, `ta`) |
| `modality` | str | `text` |
| `label` | str | Ground truth: `SAFE` or `UNSAFE` |
| `pred_label` | str | Model prediction: `SAFE` or `UNSAFE` |
| `safe_probability` | float | Softmax probability of SAFE token |
| `unsafe_probability` | float | Softmax probability of UNSAFE token |
| `unsafe_confidence` | float | Same as `unsafe_probability` (canonical drift field) |
| `pred_confidence` | float | `max(safe_probability, unsafe_probability)` |
| `raw_output` | str | Raw model generation |
| `source` | str | Dataset source |
| `prob_extraction_failed` | bool | True if token logit extraction failed |

## How to generate

See `docs/EXPERIMENTS.md` — Step 0 (HPC inference with QLoRA E1 model).

## Sample data

Small example files for pipeline testing are in `sample_data/`.
