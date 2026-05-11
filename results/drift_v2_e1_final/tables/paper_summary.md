# DriftGuard-LLM — Final E1 Experiment Results

**Model:** E1 QLoRA Fine-tuned LLaMA-3.1 8B Instruct  
**Predictions:** `predictions/e1_qlora_ft_predictions_v2.jsonl` (4,282 rows, 3,698 unique `unsafe_confidence` values, range 0.0001–0.9979)  
**Drift analysis:** Confidence-histogram drift detection over synthetic streams derived from real E1 token-level probabilities

---

## Table 1 — Primary Configuration Results

**Configuration:** `Ensemble_majority` detector · `fixed_first` reference · window = 200 · step = 200 · bins = 20 · `equal_width` binning

| Stream | Drift Detected | 1st Detection Window | Detection Delay | FA Before Drift | True Detection | Max KL | Min KS p-value | Max JS | Max Wasserstein |
| :--- | :---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| IID Control (no drift) | ✗ | — | — | 0 | ✗ | 0.5122 | 0.006094 | 0.0308 | 0.0649 |
| Class-Prior Gradual | ✓ | 3 | 1 | 0 | ✓ | 0.3158 | 0.000000 | 0.0757 | 0.2310 |
| Class-Prior Sudden | ✓ | 4 | 0 | 0 | ✓ | 1.2672 | 0.000000 | 0.0822 | 0.2258 |
| Confidence-Shift Gradual | ✓ | 2 | 0 | 0 | ✓ | 3.5302 | 0.000000 | 0.1363 | 0.0978 |
| Confidence-Shift Sudden | ✓ | 4 | 0 | 0 | ✓ | 5.2365 | 0.000000 | 0.1738 | 0.1801 |
| Language Gradual | ✓ | 2 | 0 | 0 | ✓ | 0.5557 | 0.000000 | 0.1323 | 0.2133 |
| Language Sudden | ✓ | 4 | 0 | 0 | ✓ | 12.4463 | 0.000000 | 0.3326 | 0.3320 |
| Source Sudden | ✓ | 4 | 0 | 0 | ✓ | 4.0272 | 0.000000 | 0.3553 | 0.3556 |

> **Streams:** 7 drift types + 1 IID control (no drift).  
> **Detection Delay** = number of windows after drift point before first alarm (0 = detected in first post-drift window).  
> **FA Before Drift** = false alarms triggered before the drift point.  
> `✓` = detected / true detection after drift point. `✗` = not detected.

---

## Table 2 — Detector Ablation

**Configuration:** `fixed_first` reference · window = 200 · step = 200 · bins = 20 · `equal_width` binning  
Compared across all 6 detector rules. IID Control false alarm = whether the detector fired on the no-drift stream.  
Bold = recommended configuration.

| Detector Rule | Detected Streams | IID FA | Total FA Before Drift | Mean Detection Delay (windows) |
| :--- | :---: | :---: | ---: | ---: |
| KL_only | 5/7 | No | 0 | 0.0 |
| KS_only | 7/7 | Yes | 0 | 0.14 |
| KL_AND_KS | 5/7 | No | 0 | 0.0 |
| JS_only | 7/7 | No | 1 | 0.14 |
| Wasserstein_only | 7/7 | No | 1 | 0.29 |
| **Ensemble_majority** | 7/7 | No | 0 | 0.14 |

> **Detected Streams** = true detections after drift point out of 7 drift streams.  
> **IID FA** = false alarm on IID control stream.  
> **Total FA Before Drift** = sum of pre-drift false alarms across all 7 drift streams.  
> **Mean Detection Delay** = mean number of windows from drift point to first true alarm (lower is better).

---

## Key Findings

- **`Ensemble_majority` with `fixed_first` reference** achieves 7/7 drift stream detection with **zero false alarms** on IID control and **zero pre-drift false alarms** — the only rule to satisfy all three criteria simultaneously.
- `KS_only` also detects 7/7 streams but produces a **false alarm on IID control**, making it unsuitable as a standalone detector.
- `JS_only` and `Wasserstein_only` detect 7/7 streams but each incur **1 pre-drift false alarm**.
- `KL_only` and `KL_AND_KS` miss **2 gradual drift streams** (class-prior gradual, language gradual), indicating KL divergence alone is insufficient for slow distributional shifts.
- Detection delay is **0 windows** for 6/7 drift streams under `Ensemble_majority`; the one exception (class-prior gradual) has delay of 1 window = 200 records.
- **`language_sudden`** produces the highest KL (12.45) and JS (0.33), confirming it as the most detectable drift type.

---

*Generated from `results/drift_v2_e1_final/tables/sensitivity_analysis.csv` (12,288 experiments: 1,536 configs × 8 streams).*  
*Thresholds calibrated via bootstrap (n=500, target FPR=0.05) on IID control stream.*
