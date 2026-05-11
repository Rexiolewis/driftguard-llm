"""
run_e1_standalone_infer.py
==========================
Fully self-contained E1 QLoRA inference — NO moderation_exp dependency.

Loads LLaMA-3.1 8B + QLoRA adapter directly via transformers + peft,
runs batched generation with output_scores=True, and extracts real
unsafe_probability / safe_probability from the first generated token's logits.

Features:
  - Batched inference (batch_size rows per forward pass)
  - Incremental JSONL saving (checkpoint every N rows)
  - Resume: skips rows already in output file
  - Fails fast if model lands on CPU

Output schema per row:
  id, lang, modality, label, pred_label,
  safe_probability, unsafe_probability,
  unsafe_confidence (= unsafe_probability),
  pred_confidence (= max(safe_prob, unsafe_prob)),
  raw_output, source, prob_extraction_failed
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BATCH_SIZE       = 16
CHECKPOINT_EVERY = 200
SAFE_LABEL       = "SAFE"
UNSAFE_LABEL     = "UNSAFE"

# LLaMA-3 prompt template (matches moderation_exp's build_moderation_prompt)
SYSTEM_PROMPT = (
    "You are a multilingual content moderation assistant. "
    "Classify the following content as SAFE or UNSAFE. "
    "Reply with exactly one word: SAFE or UNSAFE."
)

def build_prompt(text: str, lang: str = "en") -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Language: {lang}\nContent: {text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_completed_ids(output_jsonl: Path) -> Set[str]:
    if not output_jsonl.exists():
        return set()
    done = set()
    try:
        with open(output_jsonl, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line.strip())
                if "id" in r:
                    done.add(str(r["id"]))
    except Exception as e:
        log.warning(f"Could not load checkpoint: {e}")
    return done


# ---------------------------------------------------------------------------
# Token ID lookup
# ---------------------------------------------------------------------------

def get_safe_unsafe_token_ids(tokenizer) -> tuple[int, int]:
    def first_id(word):
        for candidate in [word, f" {word}", f"▁{word}"]:
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if ids:
                return ids[0]
        raise ValueError(f"No token ID for '{word}'")
    safe_id   = first_id(SAFE_LABEL)
    unsafe_id = first_id(UNSAFE_LABEL)
    log.info(f"Token IDs — SAFE={safe_id}  UNSAFE={unsafe_id}")
    return safe_id, unsafe_id


# ---------------------------------------------------------------------------
# Batched inference
# ---------------------------------------------------------------------------

def infer_batch(
    model,
    tokenizer,
    prompts: List[str],
    safe_id: int,
    unsafe_id: int,
    max_new_tokens: int,
    device: str,
) -> List[Dict]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    first_logits = out.scores[0]                    # [B, vocab]
    probs        = F.softmax(first_logits, dim=-1)  # [B, vocab]
    safe_probs   = probs[:, safe_id].tolist()
    unsafe_probs = probs[:, unsafe_id].tolist()

    input_len = enc["input_ids"].shape[1]
    results = []
    for i in range(len(prompts)):
        gen_ids = out.sequences[i, input_len:]
        raw     = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        sp      = float(safe_probs[i])
        up      = float(unsafe_probs[i])
        failed  = sp < 1e-9 and up < 1e-9
        if failed:
            sp, up = None, None
        results.append({"safe_prob": sp, "unsafe_prob": up, "raw": raw, "failed": failed})
    return results


def decode_label(raw: str) -> str:
    up = raw.upper()
    if "UNSAFE" in up:
        return UNSAFE_LABEL
    return SAFE_LABEL


def build_row(row, pred_label, sp, up, raw, failed):
    unsafe_conf = up
    pred_conf   = max(sp or 0.0, up or 0.0) if (sp is not None or up is not None) else None
    return {
        "id":                     row["id"],
        "lang":                   row.get("lang", row.get("language", "unknown")),
        "modality":               row.get("modality", "text"),
        "label":                  row["label"],
        "pred_label":             pred_label,
        "safe_probability":       round(sp, 8)          if sp   is not None else None,
        "unsafe_probability":     round(up, 8)          if up   is not None else None,
        "unsafe_confidence":      round(unsafe_conf, 8) if unsafe_conf is not None else None,
        "pred_confidence":        round(pred_conf, 8)   if pred_conf  is not None else None,
        "raw_output":             raw,
        "source":                 row.get("source", "unknown"),
        "prob_extraction_failed": failed,
    }


def print_summary(rows):
    uc = [r["unsafe_confidence"] for r in rows if r.get("unsafe_confidence") is not None]
    n_failed = sum(1 for r in rows if r.get("prob_extraction_failed"))
    log.info("=" * 65)
    log.info("CONFIDENCE SUMMARY")
    log.info(f"  Total  : {len(rows)}  |  Failures: {n_failed}")
    if uc:
        n_unique = len(set(round(v, 6) for v in uc))
        log.info(f"  unsafe_confidence: n={len(uc)} unique={n_unique} "
                 f"min={min(uc):.4f} max={max(uc):.4f} mean={sum(uc)/len(uc):.4f}")
        valid = n_unique > 1
        if not valid:
            log.warning("  CONSTANT confidence — extraction failed!")
        else:
            log.info("  Distribution looks valid for drift analysis.")
    else:
        log.error("  No unsafe_confidence values — extraction failed entirely.")
        valid = False
    log.info("=" * 65)
    return valid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    model_path, adapter_path, data_path, output_jsonl,
    output_dir, batch_size=BATCH_SIZE, max_new_tokens=4,
    load_in_4bit=False,
):
    log.info("=" * 65)
    log.info(f"E1 Standalone Inference  batch_size={batch_size}")
    log.info(f"  Model   : {model_path}")
    log.info(f"  Adapter : {adapter_path}")
    log.info(f"  Data    : {data_path}")
    log.info(f"  Output  : {output_jsonl}")
    log.info("=" * 65)

    # GPU check
    if not torch.cuda.is_available():
        log.error("STOP: CUDA not available. GPU required for timely inference.")
        sys.exit(1)
    log.info(f"GPU: {torch.cuda.get_device_name(0)} | "
             f"{torch.cuda.get_device_properties(0).total_memory//1024**3}GB")

    device = "cuda"

    # Load tokenizer
    log.info("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    log.info(f"Loading model (load_in_4bit={load_in_4bit})...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    if load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_cfg,
            device_map="auto",
            local_files_only=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )

    model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True)
    model.eval()
    log.info(f"Model loaded in {time.time()-t0:.1f}s  |  device: {next(model.parameters()).device}")

    # Fail fast if still on CPU
    if next(model.parameters()).device.type == "cpu":
        log.error("STOP: Model is on CPU. Cannot proceed.")
        sys.exit(1)

    safe_id, unsafe_id = get_safe_unsafe_token_ids(tokenizer)

    # Load data + resume
    rows = read_jsonl(Path(data_path))
    log.info(f"Loaded {len(rows)} records")
    out_path  = Path(output_jsonl)
    fail_path = Path(output_dir) / "prob_extraction_failures.json"
    done_ids  = load_completed_ids(out_path)
    if done_ids:
        log.info(f"RESUME: {len(done_ids)} rows already done — skipping.")
    pending = [r for r in rows if str(r["id"]) not in done_ids]
    log.info(f"Rows to process: {len(pending)}")

    # Inference loop
    failed_list: List[dict] = []
    checkpoint_buf: List[dict] = []
    n_done = len(done_ids)
    t_start = time.time()
    batch_n = 0

    for b_start in range(0, len(pending), batch_size):
        batch = pending[b_start: b_start + batch_size]
        prompts = [
            build_prompt(r["text"], r.get("lang", r.get("language", "en")))
            for r in batch
        ]

        try:
            results = infer_batch(model, tokenizer, prompts, safe_id, unsafe_id,
                                  max_new_tokens, device)
        except Exception as exc:
            log.warning(f"Batch {batch_n} failed ({exc}), falling back to single-row")
            results = []
            for prompt in prompts:
                try:
                    results.extend(infer_batch(model, tokenizer, [prompt], safe_id, unsafe_id,
                                               max_new_tokens, device))
                except Exception as exc2:
                    results.append({"safe_prob": None, "unsafe_prob": None,
                                    "raw": f"ERROR:{exc2}", "failed": True})

        for row, res in zip(batch, results):
            sp, up = res["safe_prob"], res["unsafe_prob"]
            raw    = res["raw"]
            failed = res["failed"]
            label  = decode_label(raw)
            if failed:
                failed_list.append({"id": row["id"], "raw": raw})
            checkpoint_buf.append(build_row(row, label, sp, up, raw, failed))

        batch_n += 1
        rows_done = b_start + len(batch)

        if len(checkpoint_buf) >= CHECKPOINT_EVERY:
            append_jsonl(out_path, checkpoint_buf)
            checkpoint_buf = []

        if batch_n % 5 == 0 or rows_done >= len(pending):
            elapsed = time.time() - t_start
            rate = rows_done / elapsed if elapsed else 0
            eta  = (len(pending) - rows_done) / rate if rate else 0
            log.info(f"  [{n_done+rows_done}/{len(rows)}] "
                     f"{rate:.2f} rows/s  ETA: {eta/60:.1f}min  fails: {len(failed_list)}")

    if checkpoint_buf:
        append_jsonl(out_path, checkpoint_buf)

    # Save failures
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if failed_list:
        existing = []
        if fail_path.exists():
            try:
                existing = json.loads(fail_path.read_text())
            except Exception:
                pass
        fail_path.write_text(json.dumps(existing + failed_list, indent=2))
        log.warning(f"{len(existing+failed_list)} total failures → {fail_path}")

    # Final summary
    all_rows = read_jsonl(out_path)
    log.info(f"Total rows written: {len(all_rows)}")
    is_valid = print_summary(all_rows)

    uc = [r["unsafe_confidence"] for r in all_rows if r.get("unsafe_confidence") is not None]
    summary = {
        "total_records": len(all_rows),
        "prob_extraction_failures": sum(1 for r in all_rows if r.get("prob_extraction_failed")),
        "unsafe_confidence_n":        len(uc),
        "unsafe_confidence_n_unique": len(set(round(v,6) for v in uc)) if uc else 0,
        "unsafe_confidence_min":  min(uc)         if uc else None,
        "unsafe_confidence_max":  max(uc)         if uc else None,
        "unsafe_confidence_mean": sum(uc)/len(uc) if uc else None,
        "is_valid_for_drift": is_valid,
        "output_path": str(output_jsonl),
    }
    (Path(output_dir) / "confidence_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"Summary saved → {output_dir}/confidence_summary.json")

    if not is_valid:
        log.error("STOP: Output not valid for drift analysis.")
        sys.exit(2)

    log.info("SUCCESS: Valid for drift analysis.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",   required=True)
    p.add_argument("--adapter_path", required=True)
    p.add_argument("--data_path",    required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--output_dir",   required=True)
    p.add_argument("--batch_size",   type=int, default=BATCH_SIZE)
    p.add_argument("--load_in_4bit", action="store_true")
    args = p.parse_args()

    run(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        data_path=args.data_path,
        output_jsonl=args.output_jsonl,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        load_in_4bit=args.load_in_4bit,
    )
