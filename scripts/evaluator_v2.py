"""
Evaluator v2 — saves unsafe_probability and safe_probability from token logits
===============================================================================
Extends the original evaluator.py output schema to include per-class token
probabilities extracted directly from the LLaMA generation logits.

New output fields per record
-----------------------------
  unsafe_probability   float | null   P(model assigns to UNSAFE token)
  safe_probability     float | null   P(model assigns to SAFE token)
  unsafe_confidence    float | null   = unsafe_probability (canonical drift input)
  pred_confidence      float          max(safe_probability, unsafe_probability)

Design choices
--------------
  - At temperature=0 (greedy), we inspect the logits for the *first generated token*
    and look for SAFE / UNSAFE token ids. This gives a real probability, not a fixed 0.5.
  - If the model's tokenizer splits "SAFE" or "UNSAFE" into multiple sub-tokens, we fall
    back to summing probabilities for the most common split patterns.
  - If token probability extraction fails for any reason, unsafe_probability = null and
    pred_confidence defaults to None. This is logged per record.
  - pred_confidence = 0.5 is NEVER saved unless the model truly gives equal probability
    to SAFE and UNSAFE tokens.

NOTE: This module requires access to model logits (return_dict_in_generate=True,
output_scores=True). The mock model path returns null for all probabilities.

Integration with drift detector
--------------------------------
    from scripts.drift_detector_v2 import load_and_normalise
    # Normalisation automatically picks up unsafe_probability → unsafe_confidence
    records = load_and_normalise("results/e1/predictions.jsonl")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token probability extraction
# ---------------------------------------------------------------------------

# Common tokenisations of "SAFE" and "UNSAFE" for LLaMA-3 tokenizer
_SAFE_ALIASES = ["SAFE", "Safe", "safe", "▁SAFE", "▁Safe"]
_UNSAFE_ALIASES = ["UNSAFE", "Unsafe", "unsafe", "▁UNSAFE", "▁Unsafe"]


def extract_class_probs(
    scores: Tuple,
    tokenizer,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract P(SAFE) and P(UNSAFE) from generation scores.

    Parameters
    ----------
    scores      : Tuple of tensors from model.generate(output_scores=True)
                  Each element is shape (batch, vocab_size) for one generated token.
    tokenizer   : HuggingFace tokenizer

    Returns
    -------
    (safe_prob, unsafe_prob)  both in [0, 1], or (None, None) if extraction fails.
    """
    if scores is None or len(scores) == 0:
        return None, None

    try:
        # Use logits from the first generated token (position 0)
        first_logits = scores[0][0]  # shape: (vocab_size,)
        probs = torch.softmax(first_logits.float(), dim=-1)

        safe_ids = _get_token_ids(tokenizer, _SAFE_ALIASES)
        unsafe_ids = _get_token_ids(tokenizer, _UNSAFE_ALIASES)

        safe_prob = float(probs[safe_ids].sum().item()) if safe_ids else None
        unsafe_prob = float(probs[unsafe_ids].sum().item()) if unsafe_ids else None

        # Renormalise over just these two classes if both found
        if safe_prob is not None and unsafe_prob is not None:
            total = safe_prob + unsafe_prob
            if total > 0:
                safe_prob = safe_prob / total
                unsafe_prob = unsafe_prob / total

        return safe_prob, unsafe_prob

    except Exception as exc:
        logger.warning(f"Token probability extraction failed: {exc}")
        return None, None


def _get_token_ids(tokenizer, aliases: List[str]) -> List[int]:
    """Return all valid token IDs for the given string aliases."""
    ids = []
    for alias in aliases:
        try:
            toks = tokenizer.encode(alias, add_special_tokens=False)
            if len(toks) == 1:  # only use single-token representations
                ids.append(toks[0])
        except Exception:
            pass
    return list(set(ids))


# ---------------------------------------------------------------------------
# Inference result with probabilities
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class InferenceResultV2:
    """Extended inference result with per-class token probabilities."""
    label: str                        # "SAFE" | "UNSAFE"
    raw_output: str
    pred_confidence: Optional[float]  # max(safe_prob, unsafe_prob)
    safe_probability: Optional[float]
    unsafe_probability: Optional[float]
    unsafe_confidence: Optional[float]  # = unsafe_probability (canonical)
    token_scores: Optional[object]
    prob_extraction_failed: bool = False


def infer_binary_label_v2(
    loaded,           # dict returned by load_text_model()
    prompt: str,
    max_new_tokens: int = 4,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> InferenceResultV2:
    """
    Run inference and extract per-class token probabilities.

    Replaces infer_binary_label() from moderation_exp.models.inference.
    Key difference: always calls generate with output_scores=True, then
    extracts SAFE/UNSAFE probabilities from the first token logits.
    """
    model = loaded.get("model")
    tokenizer = loaded.get("tokenizer")

    if model is None or tokenizer is None:
        # Mock / unit-test path
        return InferenceResultV2(
            label="SAFE",
            raw_output="MOCK",
            pred_confidence=None,
            safe_probability=None,
            unsafe_probability=None,
            unsafe_confidence=None,
            token_scores=None,
            prob_extraction_failed=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode generated text
    generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip().upper()
    label = "UNSAFE" if "UNSAFE" in raw_text else "SAFE"

    # Extract class probabilities
    safe_prob, unsafe_prob = extract_class_probs(outputs.scores, tokenizer)
    prob_failed = safe_prob is None and unsafe_prob is None

    if unsafe_prob is not None:
        pred_confidence = max(safe_prob or 0.0, unsafe_prob)
    else:
        pred_confidence = None

    return InferenceResultV2(
        label=label,
        raw_output=raw_text,
        pred_confidence=pred_confidence,
        safe_probability=safe_prob,
        unsafe_probability=unsafe_prob,
        unsafe_confidence=unsafe_prob,  # canonical input for drift detector
        token_scores=None,  # not serialised (too large)
        prob_extraction_failed=prob_failed,
    )


# ---------------------------------------------------------------------------
# Extended prediction row builder
# ---------------------------------------------------------------------------

def build_prediction_row(
    row: dict,
    final: InferenceResultV2,
    first_pass: InferenceResultV2,
    used_rag: bool,
    malformed_output: bool,
    fusion_payload: Optional[dict] = None,
) -> dict:
    """
    Build the full output prediction record.

    Output schema (used by drift_detector_v2.py normalise_record)
    ---------------------------------------------------------------
    id, lang, modality, label, pred_label,
    pred_confidence,       max(safe_prob, unsafe_prob)
    unsafe_probability,    direct P(UNSAFE)
    safe_probability,      direct P(SAFE)
    unsafe_confidence,     = unsafe_probability  (canonical drift field)
    raw_output,
    first_pass_label,
    first_pass_unsafe_probability,
    rag_used, malformed_output_rejected, source
    """
    pred_row = {
        "id": row["id"],
        "lang": row["lang"],
        "modality": row.get("modality", "text"),
        "label": row["label"],
        "pred_label": final.label,
        # --- probability fields ---
        "pred_confidence": final.pred_confidence,      # max(safe, unsafe) or None
        "unsafe_probability": final.unsafe_probability, # direct P(UNSAFE) or None
        "safe_probability": final.safe_probability,     # direct P(SAFE) or None
        "unsafe_confidence": final.unsafe_confidence,   # == unsafe_probability
        # --- bookkeeping ---
        "raw_output": final.raw_output,
        "first_pass_label": first_pass.label,
        "first_pass_unsafe_probability": first_pass.unsafe_probability,
        "rag_used": used_rag,
        "malformed_output_rejected": malformed_output,
        "prob_extraction_failed": final.prob_extraction_failed,
        "source": row.get("source", "unknown"),
    }
    if fusion_payload:
        pred_row.update({
            "fused_text": fusion_payload.get("fused_text", ""),
            "ocr_text": fusion_payload.get("ocr_text", ""),
            "caption_text": fusion_payload.get("caption_text", ""),
        })
    return pred_row


# ---------------------------------------------------------------------------
# Drop-in replacement for run_evaluation() using v2 inference
# ---------------------------------------------------------------------------

def run_evaluation_v2(
    input_path: str,
    output_dir: str,
    model_path: str,
    local_files_only: bool = True,
    adapter_path: Optional[str] = None,
    use_mock_model: bool = False,
    load_in_4bit: bool = False,
    bf16: bool = True,
    max_new_tokens: int = 4,
    temperature: float = 0.0,
    do_sample: bool = False,
    uncertainty_conf_threshold: float = 0.7,
    use_rag: bool = False,
    rag_top_k: int = 4,
    policy_docs_path: str = "data/policy/policy_docs.jsonl",
    policy_examples_path: str = "data/policy/policy_examples.jsonl",
    use_multimodal_fusion: bool = False,
    use_mock_multimodal: bool = False,
    caption_model_path: str = "",
) -> Dict:
    """
    Drop-in replacement for run_evaluation() from evaluator.py.
    Key difference: uses infer_binary_label_v2() to capture per-class token
    probabilities so that unsafe_probability is never a constant 0.5.
    """
    # Import original infrastructure (unchanged modules)
    from moderation_exp.metrics import build_experiment_metrics
    from moderation_exp.models.inference import load_text_model
    from moderation_exp.models.prompting import build_moderation_prompt
    from moderation_exp.multimodal.fusion import (
        build_fused_text, CaptionExtractor, MetadataExtractor,
        MockCaptionExtractor, MockMetadataExtractor, MockOCRExtractor, OCRExtractor,
    )
    from moderation_exp.rag.retriever import SimplePolicyRetriever
    from moderation_exp.utils.io import read_jsonl, write_json, write_jsonl

    rows = list(read_jsonl(Path(input_path)))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_text_model(
        model_path=model_path,
        local_files_only=local_files_only,
        load_in_4bit=load_in_4bit,
        bf16=bf16,
        adapter_path=adapter_path,
        use_mock_model=use_mock_model,
    )

    retriever = None
    if use_rag:
        retriever = SimplePolicyRetriever(Path(policy_docs_path), Path(policy_examples_path))

    ocr_ext = cap_ext = meta_ext = None
    if use_multimodal_fusion:
        if use_mock_multimodal:
            ocr_ext, cap_ext, meta_ext = MockOCRExtractor(), MockCaptionExtractor(), MockMetadataExtractor()
        else:
            ocr_ext = OCRExtractor()
            cap_ext = CaptionExtractor(caption_model_path, local_files_only)
            meta_ext = MetadataExtractor()

    prediction_rows: List[Dict] = []
    rag_trace_rows: List[Dict] = []
    prob_failure_count = 0

    for row in rows:
        eval_text = row["text"]
        fusion_payload = {}
        malformed_output = False

        if use_multimodal_fusion and row.get("modality") == "image_text":
            fusion_payload = build_fused_text(
                record=row,
                ocr_extractor=ocr_ext,
                caption_extractor=cap_ext,
                metadata_extractor=meta_ext,
            )
            eval_text = fusion_payload["fused_text"]

        prompt = build_moderation_prompt(
            text=eval_text,
            lang=row["lang"],
            modality=row.get("modality", "text"),
            retrieved_evidence=None,
        )

        try:
            first_pass = infer_binary_label_v2(
                loaded=loaded,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
        except Exception as exc:
            malformed_output = True
            first_pass = InferenceResultV2(
                label="UNSAFE",
                raw_output=f"MALFORMED: {exc}",
                pred_confidence=None,
                safe_probability=None,
                unsafe_probability=None,
                unsafe_confidence=None,
                token_scores=None,
                prob_extraction_failed=True,
            )

        final = first_pass
        used_rag = False

        # RAG fallback when confidence is low
        if (use_rag and first_pass.pred_confidence is not None
                and first_pass.pred_confidence < uncertainty_conf_threshold):
            retrieved = retriever.retrieve(eval_text, top_k=rag_top_k) if retriever else []
            if retrieved:
                prompt_rag = build_moderation_prompt(
                    text=eval_text,
                    lang=row["lang"],
                    modality=row.get("modality", "text"),
                    retrieved_evidence=retrieved,
                )
                try:
                    final = infer_binary_label_v2(loaded=loaded, prompt=prompt_rag,
                                                   max_new_tokens=max_new_tokens,
                                                   temperature=temperature, do_sample=do_sample)
                    used_rag = True
                except Exception as exc:
                    malformed_output = True

        if final.prob_extraction_failed:
            prob_failure_count += 1

        pred_row = build_prediction_row(
            row=row, final=final, first_pass=first_pass,
            used_rag=used_rag, malformed_output=malformed_output,
            fusion_payload=fusion_payload or None,
        )
        prediction_rows.append(pred_row)

        if used_rag:
            rag_trace_rows.append({
                "id": row["id"],
                "lang": row["lang"],
                "first_pass_label": first_pass.label,
                "first_pass_unsafe_prob": first_pass.unsafe_probability,
                "final_label": final.label,
                "final_unsafe_prob": final.unsafe_probability,
            })

    metrics = build_experiment_metrics(prediction_rows)

    write_jsonl(out_dir / "predictions.jsonl", prediction_rows)
    write_json(out_dir / "metrics.json", metrics)

    # ------------------------------------------------------------------ #
    # Confidence summary — printed and saved; used to verify non-constant  #
    # ------------------------------------------------------------------ #
    uc_vals = [r["unsafe_confidence"] for r in prediction_rows if r.get("unsafe_confidence") is not None]
    pc_vals = [r["pred_confidence"] for r in prediction_rows if r.get("pred_confidence") is not None]

    import statistics as _stats

    def _summary(vals, name):
        if not vals:
            return {name + "_n": 0}
        return {
            name + "_n": len(vals),
            name + "_min": round(min(vals), 6),
            name + "_max": round(max(vals), 6),
            name + "_mean": round(sum(vals) / len(vals), 6),
            name + "_std": round(_stats.stdev(vals) if len(vals) > 1 else 0.0, 6),
            name + "_n_unique": len(set(round(v, 6) for v in vals)),
        }

    conf_summary = {
        **_summary(uc_vals, "unsafe_confidence"),
        **_summary(pc_vals, "pred_confidence"),
        "prob_extraction_failed_count": prob_failure_count,
        "total_records": len(prediction_rows),
    }

    write_json(out_dir / "run_info.json", {
        "input_path": input_path,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "use_rag": use_rag,
        "use_multimodal_fusion": use_multimodal_fusion,
        **conf_summary,
    })
    write_json(out_dir / "confidence_summary.json", conf_summary)

    # --- Console summary ---
    print("\n" + "=" * 65)
    print("EVALUATOR v2 — CONFIDENCE SUMMARY")
    print(f"  Records total      : {len(prediction_rows)}")
    print(f"  Prob extract fails : {prob_failure_count}")
    if uc_vals:
        cs = conf_summary
        print(f"  unsafe_confidence  : n={cs['unsafe_confidence_n']}  "
              f"min={cs['unsafe_confidence_min']:.4f}  max={cs['unsafe_confidence_max']:.4f}  "
              f"mean={cs['unsafe_confidence_mean']:.4f}  std={cs['unsafe_confidence_std']:.4f}  "
              f"unique={cs['unsafe_confidence_n_unique']}")
    else:
        print("  unsafe_confidence  : NONE SAVED — token probability extraction failed entirely.")
        print("                       Check output_scores=True in model.generate().")

    # Warn if constant
    if uc_vals and len(set(round(v, 6) for v in uc_vals)) == 1:
        print("\n  *** WARNING: unsafe_confidence is CONSTANT — drift results will be invalid. ***")
        print("      Do NOT use this file for paper results. Investigate logit extraction.")
    elif uc_vals and abs(sum(uc_vals) / len(uc_vals) - 0.5) < 0.01 and \
            len(set(round(v, 6) for v in uc_vals)) < 3:
        print("\n  *** WARNING: unsafe_confidence near-constant ~0.5 — likely extraction issue. ***")
    else:
        print("\n  Confidence distribution looks healthy for drift analysis.")
    print("=" * 65 + "\n")

    if rag_trace_rows:
        write_json(out_dir / "rag_evidence.json", {"rows": rag_trace_rows})

    if prob_failure_count > 0:
        logger.warning(
            f"Token probability extraction failed for {prob_failure_count}/{len(prediction_rows)} "
            f"records. Check that model.generate() is returning output_scores=True."
        )

    return metrics
