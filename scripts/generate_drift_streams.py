"""
Generate Synthetic Drift Streams — Task E
==========================================
Creates 8 types of controlled drift streams from Phase 1 prediction JSONL files.
All streams are *prediction-level simulations* (no model re-inference required)
except stream 9 (adversarial_text_stream) which only creates the input text file.

Stream types
------------
1  iid_control              — Stratified shuffle, similar distribution throughout
2  language_sudden          — EN/MS → ZH/TA shift at drift_point
3  language_gradual         — ZH/TA fraction increases linearly across stream
4  class_prior_sudden       — Balanced SAFE/UNSAFE → UNSAFE-heavy at drift_point
5  class_prior_gradual      — UNSAFE fraction grows linearly
6  source_sudden            — Source distribution shifts suddenly at drift_point
7  confidence_shift_sudden  — unsafe_confidence shifted upward after drift_point
8  confidence_shift_gradual — unsafe_confidence shifts upward gradually

All outputs include metadata JSON with:
  stream_name, drift_type, drift_point, n_records,
  language_distribution_before_after, label_distribution_before_after,
  confidence_mean_before_after

Usage
-----
  python scripts/generate_drift_streams.py \\
      --input predictions/e0_zero_shot_predictions.jsonl \\
      --out results/drift_v2/streams \\
      --seed 42

NOTE: e1_qlora_ft_predictions.jsonl has constant pred_confidence=0.5.
      Streams 7 and 8 will artificially inject confidence variation as documented.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from drift_detector_v2 import load_and_normalise, NormalisedRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _distribution(records: List[NormalisedRecord], field: str) -> Dict[str, float]:
    vals = [getattr(r, field, None) or "unknown" for r in records]
    c = Counter(vals)
    total = len(vals)
    return {k: round(v / total, 4) for k, v in c.items()} if total else {}


def _conf_mean(records: List[NormalisedRecord]) -> Optional[float]:
    vals = [r.unsafe_confidence for r in records if r.unsafe_confidence is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def _save_stream(
    records: List[NormalisedRecord],
    out_dir: Path,
    stream_name: str,
    drift_type: str,
    drift_point: Optional[int],
    extra_meta: Optional[Dict] = None,
) -> None:
    """Save stream JSONL + metadata JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{stream_name}.jsonl"
    meta_path = out_dir / f"{stream_name}_meta.json"

    n = len(records)
    dp = drift_point if drift_point is not None else n

    before = records[:dp]
    after = records[dp:]

    meta = {
        "stream_name": stream_name,
        "drift_type": drift_type,
        "drift_point": drift_point,
        "n_records": n,
        "simulation_level": "prediction-level (no model re-inference)",
        "language_distribution_before": _distribution(before, "language"),
        "language_distribution_after": _distribution(after, "language"),
        "label_distribution_before": _distribution(before, "predicted"),
        "label_distribution_after": _distribution(after, "predicted"),
        "confidence_mean_before": _conf_mean(before),
        "confidence_mean_after": _conf_mean(after),
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            obj = {
                "id": r.id,
                "lang": r.language,
                "label": r.true_label,
                "pred_label": r.predicted,
                "pred_confidence": r.pred_confidence,
                "unsafe_confidence": r.unsafe_confidence,
                "confidence_source": r.confidence_source,
                "source": r.source,
                "stream_name": stream_name,
                "drift_type": drift_type,
            }
            if r.text:
                obj["text"] = r.text
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved {stream_name}: {n} records, drift_point={drift_point} → {jsonl_path}")


# ---------------------------------------------------------------------------
# Stream generators
# ---------------------------------------------------------------------------

def gen_iid_control(
    records: List[NormalisedRecord],
    rng: random.Random,
    target_n: int,
    out_dir: Path,
) -> None:
    """
    Stratified shuffle: language × label cells preserved.
    """
    cells: Dict[Tuple, List] = defaultdict(list)
    for r in records:
        cells[(r.language, r.predicted)].append(r)

    # Sample proportionally from each cell
    sampled = []
    for cell, cell_recs in cells.items():
        n_cell = int(round(len(cell_recs) / len(records) * target_n))
        sampled.extend(rng.choices(cell_recs, k=n_cell))

    # Trim / pad to target_n
    rng.shuffle(sampled)
    sampled = sampled[:target_n]

    _save_stream(sampled, out_dir, "iid_control", "no_drift", drift_point=None,
                 extra_meta={"note": "Stratified shuffled iid stream. No drift injected."})


def gen_language_sudden(
    records: List[NormalisedRecord],
    rng: random.Random,
    target_n: int,
    drift_fraction: float,
    out_dir: Path,
) -> None:
    """Before drift: EN/MS-heavy. After drift: ZH/TA-heavy."""
    drift_point = int(target_n * drift_fraction)

    pre_langs = {"en", "ms"}
    post_langs = {"zh", "ta", "zh-cn", "zh-tw"}

    pre_recs = [r for r in records if r.language.lower() in pre_langs]
    post_recs = [r for r in records if r.language.lower() in post_langs]

    # Fallback if insufficient data
    if len(pre_recs) < drift_point // 2:
        pre_recs = records
    if len(post_recs) < (target_n - drift_point) // 2:
        post_recs = records

    before = rng.choices(pre_recs, k=drift_point)
    after = rng.choices(post_recs, k=target_n - drift_point)
    stream = before + after

    _save_stream(stream, out_dir, "language_sudden", "language_covariate_shift_sudden",
                 drift_point=drift_point,
                 extra_meta={
                     "pre_drift_langs": sorted(pre_langs),
                     "post_drift_langs": sorted(post_langs),
                     "drift_fraction": drift_fraction,
                 })


def gen_language_gradual(
    records: List[NormalisedRecord],
    rng: random.Random,
    target_n: int,
    out_dir: Path,
) -> None:
    """ZH/TA fraction increases linearly from 0% to 80% across the stream."""
    pre_langs = {"en", "ms"}
    post_langs = {"zh", "ta", "zh-cn", "zh-tw"}

    pre_recs = [r for r in records if r.language.lower() in pre_langs] or records
    post_recs = [r for r in records if r.language.lower() in post_langs] or records

    stream = []
    for i in range(target_n):
        post_frac = (i / target_n) * 0.8  # 0% → 80%
        if rng.random() < post_frac:
            stream.append(rng.choice(post_recs))
        else:
            stream.append(rng.choice(pre_recs))

    drift_point = int(target_n * 0.3)  # noticeable drift starts around 30%

    _save_stream(stream, out_dir, "language_gradual", "language_covariate_shift_gradual",
                 drift_point=drift_point,
                 extra_meta={
                     "transition": "ZH/TA fraction 0% → 80% linearly over full stream",
                     "approximate_drift_point": drift_point,
                 })


def gen_class_prior_sudden(
    records: List[NormalisedRecord],
    rng: random.Random,
    target_n: int,
    drift_fraction: float,
    post_unsafe_frac: float,
    out_dir: Path,
) -> None:
    """Before: ~50% UNSAFE. After: post_unsafe_frac UNSAFE."""
    drift_point = int(target_n * drift_fraction)

    safe_recs = [r for r in records if r.predicted == "SAFE"] or records
    unsafe_recs = [r for r in records if r.predicted == "UNSAFE"] or records

    # Before: balanced
    n_before = drift_point
    n_safe_pre = n_before // 2
    n_unsafe_pre = n_before - n_safe_pre
    before = rng.choices(safe_recs, k=n_safe_pre) + rng.choices(unsafe_recs, k=n_unsafe_pre)
    rng.shuffle(before)

    # After: UNSAFE-heavy
    n_after = target_n - drift_point
    n_unsafe_post = int(n_after * post_unsafe_frac)
    n_safe_post = n_after - n_unsafe_post
    after = rng.choices(unsafe_recs, k=n_unsafe_post) + rng.choices(safe_recs, k=n_safe_post)
    rng.shuffle(after)

    stream = before + after

    _save_stream(stream, out_dir, "class_prior_sudden", "label_prior_shift_sudden",
                 drift_point=drift_point,
                 extra_meta={
                     "pre_unsafe_frac": 0.5,
                     "post_unsafe_frac": post_unsafe_frac,
                     "drift_fraction": drift_fraction,
                 })


def gen_class_prior_gradual(
    records: List[NormalisedRecord],
    rng: random.Random,
    target_n: int,
    out_dir: Path,
) -> None:
    """UNSAFE fraction grows from 30% to 70% across stream."""
    safe_recs = [r for r in records if r.predicted == "SAFE"] or records
    unsafe_recs = [r for r in records if r.predicted == "UNSAFE"] or records

    stream = []
    for i in range(target_n):
        unsafe_frac = 0.3 + (i / target_n) * 0.4  # 30% → 70%
        if rng.random() < unsafe_frac:
            stream.append(rng.choice(unsafe_recs))
        else:
            stream.append(rng.choice(safe_recs))

    drift_point = int(target_n * 0.35)

    _save_stream(stream, out_dir, "class_prior_gradual", "label_prior_shift_gradual",
                 drift_point=drift_point,
                 extra_meta={"transition": "UNSAFE fraction 30% → 70% linearly"})


def gen_source_sudden(
    records: List[NormalisedRecord],
    rng: random.Random,
    target_n: int,
    drift_fraction: float,
    out_dir: Path,
) -> None:
    """Source distribution shifts suddenly at drift_point."""
    by_source: Dict[str, List] = defaultdict(list)
    for r in records:
        by_source[r.source].append(r)

    sources = sorted(by_source.keys())
    if len(sources) < 2:
        # Not enough sources — fall back to language split
        gen_language_sudden(records, rng, target_n, drift_fraction, out_dir)
        print("    (source_sudden fell back to language_sudden — only 1 source in data)")
        return

    drift_point = int(target_n * drift_fraction)
    mid = len(sources) // 2
    pre_sources = sources[:mid]
    post_sources = sources[mid:]

    pre_pool = [r for r in records if r.source in pre_sources] or records
    post_pool = [r for r in records if r.source in post_sources] or records

    before = rng.choices(pre_pool, k=drift_point)
    after = rng.choices(post_pool, k=target_n - drift_point)
    stream = before + after

    _save_stream(stream, out_dir, "source_sudden", "source_distribution_shift_sudden",
                 drift_point=drift_point,
                 extra_meta={
                     "pre_drift_sources": pre_sources,
                     "post_drift_sources": post_sources,
                 })


def gen_confidence_shift_sudden(
    records: List[NormalisedRecord],
    rng: random.Random,
    target_n: int,
    drift_fraction: float,
    shift_delta: float,
    out_dir: Path,
) -> None:
    """
    After drift_point, add shift_delta to unsafe_confidence (clipped to [0,1]).
    If unsafe_confidence is constant (e.g. all 0.5 from E1), inject uniform
    noise ± 0.05 before shifting so the stream is non-trivial.
    """
    drift_point = int(target_n * drift_fraction)

    pool = rng.choices(records, k=target_n)
    before = pool[:drift_point]
    after = pool[drift_point:]

    # Detect constant confidence
    all_uc = [r.unsafe_confidence for r in pool if r.unsafe_confidence is not None]
    is_constant = len(set(round(v, 4) for v in all_uc)) == 1 if all_uc else False

    def perturb_before(r: NormalisedRecord, noise: float) -> NormalisedRecord:
        r2 = deepcopy(r)
        uc = r2.unsafe_confidence
        if uc is None:
            uc = 0.5
        if is_constant:
            uc += rng.uniform(-noise, noise)
        r2.unsafe_confidence = float(np.clip(uc, 0.0, 1.0))
        return r2

    def perturb_after(r: NormalisedRecord, delta: float, noise: float) -> NormalisedRecord:
        r2 = deepcopy(r)
        uc = r2.unsafe_confidence
        if uc is None:
            uc = 0.5
        if is_constant:
            uc += rng.uniform(-noise, noise)
        uc += delta
        r2.unsafe_confidence = float(np.clip(uc, 0.0, 1.0))
        return r2

    stream = ([perturb_before(r, 0.05) for r in before] +
              [perturb_after(r, shift_delta, 0.05) for r in after])

    _save_stream(stream, out_dir, "confidence_shift_sudden",
                 "confidence_distribution_shift_sudden",
                 drift_point=drift_point,
                 extra_meta={
                     "shift_delta": shift_delta,
                     "artificial_noise_injected": is_constant,
                     "note": (
                         "unsafe_confidence artificially shifted. "
                         "This is a prediction-level simulation — not real model output."
                         + (" Uniform noise added because original confidence was constant." if is_constant else "")
                     ),
                 })


def gen_confidence_shift_gradual(
    records: List[NormalisedRecord],
    rng: random.Random,
    target_n: int,
    max_delta: float,
    out_dir: Path,
) -> None:
    """unsafe_confidence shifts linearly upward over the full stream."""
    pool = rng.choices(records, k=target_n)
    all_uc = [r.unsafe_confidence for r in pool if r.unsafe_confidence is not None]
    is_constant = len(set(round(v, 4) for v in all_uc)) == 1 if all_uc else False

    stream = []
    for i, r in enumerate(pool):
        r2 = deepcopy(r)
        uc = r2.unsafe_confidence
        if uc is None:
            uc = 0.5
        if is_constant:
            uc += rng.uniform(-0.05, 0.05)
        delta = (i / target_n) * max_delta
        uc += delta
        r2.unsafe_confidence = float(np.clip(uc, 0.0, 1.0))
        stream.append(r2)

    drift_point = int(target_n * 0.3)

    _save_stream(stream, out_dir, "confidence_shift_gradual",
                 "confidence_distribution_shift_gradual",
                 drift_point=drift_point,
                 extra_meta={
                     "max_delta": max_delta,
                     "artificial_noise_injected": is_constant,
                     "note": (
                         "unsafe_confidence shifts linearly from 0 to max_delta. "
                         "Prediction-level simulation — not real model output."
                         + (" Uniform noise added because original confidence was constant." if is_constant else "")
                     ),
                 })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic drift streams.")
    parser.add_argument("--input", required=True,
                        help="Phase 1 prediction JSONL (e.g. e0_zero_shot_predictions.jsonl)")
    parser.add_argument("--out", default="results/drift_v2/streams",
                        help="Output directory for stream files")
    parser.add_argument("--target_n", type=int, default=2000,
                        help="Total records per stream (default 2000)")
    parser.add_argument("--drift_fraction", type=float, default=0.5,
                        help="Fraction of stream before drift point (default 0.5)")
    parser.add_argument("--post_unsafe_frac", type=float, default=0.75,
                        help="UNSAFE fraction after class_prior_sudden drift (default 0.75)")
    parser.add_argument("--confidence_shift_delta", type=float, default=0.20,
                        help="unsafe_confidence shift magnitude for sudden/gradual (default 0.20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print(f"Loading records from: {input_path}")
    records = load_and_normalise(input_path)
    print(f"  Loaded {len(records)} records")

    if len(records) < 100:
        print(f"ERROR: Too few records ({len(records)}) to generate meaningful drift streams.")
        sys.exit(1)

    target_n = min(args.target_n, len(records) * 3)  # safety cap on choices()
    print(f"  Target stream length: {target_n}")
    print(f"  Output dir: {out_dir}\n")

    print("Generating streams...")

    gen_iid_control(records, rng, target_n, out_dir)
    gen_language_sudden(records, rng, target_n, args.drift_fraction, out_dir)
    gen_language_gradual(records, rng, target_n, out_dir)
    gen_class_prior_sudden(records, rng, target_n, args.drift_fraction,
                           args.post_unsafe_frac, out_dir)
    gen_class_prior_gradual(records, rng, target_n, out_dir)
    gen_source_sudden(records, rng, target_n, args.drift_fraction, out_dir)
    gen_confidence_shift_sudden(records, rng, target_n, args.drift_fraction,
                                args.confidence_shift_delta, out_dir)
    gen_confidence_shift_gradual(records, rng, target_n, args.confidence_shift_delta, out_dir)

    # Stream 9: adversarial text (save input file for re-inference)
    _gen_adversarial_text_input(records, rng, target_n, out_dir)

    print(f"\nDone. All streams saved to: {out_dir}")


def _gen_adversarial_text_input(
    records: List[NormalisedRecord],
    rng: random.Random,
    target_n: int,
    out_dir: Path,
) -> None:
    """
    Stream 9: adversarial_text_stream
    Creates a new INPUT file (for re-inference, not prediction-level simulation).
    Text perturbations: typos, repeated chars, symbols, emoji, spacing, leetspeak.
    Only records that have a 'text' field are included.
    """
    text_records = [r for r in records if r.text]
    if not text_records:
        print("  [adversarial_text] No text field in records — skipping stream 9.")
        return

    pool = rng.choices(text_records, k=min(target_n, len(text_records) * 3))
    drift_point = len(pool) // 2

    def perturb(text: str, severity: float) -> str:
        """Apply random text perturbations proportional to severity ∈ [0,1]."""
        if not text:
            return text

        chars = list(text)
        # 1. Typo insertion
        if rng.random() < severity * 0.4:
            i = rng.randint(0, max(0, len(chars) - 1))
            chars.insert(i, rng.choice("!@#$%^&*"))

        # 2. Repeated characters
        if rng.random() < severity * 0.4:
            i = rng.randint(0, max(0, len(chars) - 1))
            chars.insert(i, chars[i])

        # 3. Emoji insertion
        emojis = ["😈", "💀", "🔥", "😡", "🤬", "💩"]
        if rng.random() < severity * 0.3:
            chars.insert(rng.randint(0, len(chars)), rng.choice(emojis))

        # 4. Leetspeak substitutions
        leet = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
        if rng.random() < severity * 0.2:
            chars = [leet.get(c.lower(), c) if rng.random() < 0.3 else c for c in chars]

        # 5. Code-switch marker
        if rng.random() < severity * 0.2:
            markers = ["[MS]", "[ZH]", "[TA]"]
            chars = [rng.choice(markers)] + chars

        return "".join(chars)

    out_path = out_dir / "adversarial_text_stream_input.jsonl"
    meta_path = out_dir / "adversarial_text_stream_meta.json"

    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(pool):
            severity = 0.0 if i < drift_point else (i - drift_point) / max(1, len(pool) - drift_point)
            perturbed = perturb(r.text, severity)
            obj = {
                "id": r.id + "_adv",
                "lang": r.language,
                "label": r.true_label,
                "modality": r.modality or "text",
                "text": perturbed,
                "original_text": r.text,
                "source": r.source,
                "stream_name": "adversarial_text_stream",
                "adversarial_severity": round(severity, 4),
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    meta = {
        "stream_name": "adversarial_text_stream",
        "drift_type": "adversarial_text_perturbation",
        "drift_point": drift_point,
        "n_records": len(pool),
        "simulation_level": "INPUT-level — must re-run inference before using as drift evidence",
        "perturbations": ["typo_insertion", "char_repetition", "emoji", "leetspeak", "code_switch_markers"],
        "note": (
            "This file must be passed through the LLaMA-3.1 8B model (evaluator_v2.py) "
            "before being used as adversarial drift detection evidence. "
            "Do NOT use the original pred_label values from Phase 1 for this stream."
        ),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved adversarial_text_stream_input: {len(pool)} records (re-inference needed) "
          f"→ {out_path}")


if __name__ == "__main__":
    main()
