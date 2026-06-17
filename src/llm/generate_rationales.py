"""
Phase A of distillation: the 7B teacher writes a chain-of-thought rationale for
every training example, conditioned on the GOLD label (rationalization).

Conditioning on the answer guarantees the rationale is label-consistent -- the
student then learns to reproduce *correct* reasoning, not the teacher's mistakes.
This is the "distilling step-by-step" recipe: the student's training target is
``{reasoning (from teacher), label (gold)}`` given the question.

Output: one JSONL row per (participant_id, item_id) with the teacher rationale.
Generated over the FULL dataset; the per-fold student trainer later uses only the
rationales of its training participants (leakage handled downstream).

    python -m src.llm.generate_rationales \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --output outputs/cot/teacher_rationales_hybw3.jsonl
"""

from pathlib import Path
import argparse
import json
import re
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import load_item_dataset
from src.llm.cot_probe import ANCHORS, clean_evidence, DEFAULT_DATASET, parse_output

RATIONALIZE_SYSTEM = (
    "You are a careful clinical-research assistant. For a single PHQ-8 item you are "
    "given evidence snippets from a person's interview AND the correct severity score. "
    "The snippets are keyword-like, out of order, separated by '///'.\n\n"
    "PHQ-8 frequency scale:\n" + ANCHORS + "\n\n"
    "Write a SHORT explanation (2-4 sentences) of why the given score is correct: cite the "
    "specific snippets that support it and map them to the matching frequency anchor. Do not "
    "contradict the given score and do not invent symptoms beyond the snippets. "
    "Respond with ONLY JSON: {\"reasoning\": \"...\", \"label\": <the given score>}."
)

# Few-shot rationalization demos (label is given in the user turn).
DEMOS = [
    ("Feeling tired or having little energy",
     "plenty of energy for work and the gym /// out hiking on weekends /// I sleep fine", 0,
     "Snippets show normal-to-high energy ('plenty of energy', 'out hiking') and no fatigue cue, "
     "which matches 'Not at all' (score 0)."),
    ("Little interest or pleasure in doing things",
     "don't really enjoy things anymore /// can't be bothered with painting /// most days feel flat", 2,
     "Loss of interest spans activities ('don't enjoy things anymore', \"can't be bothered\") and "
     "'most days feel flat' signals high frequency, matching 'More than half the days' (score 2)."),
]


def build_messages(item_text, evidence, label):
    msgs = [{"role": "system", "content": RATIONALIZE_SYSTEM}]
    for it, ev, lb, rat in DEMOS:
        msgs.append({"role": "user",
                     "content": f"PHQ-8 item: {it}\nEvidence: {ev}\nCorrect score: {lb}"})
        msgs.append({"role": "assistant",
                     "content": json.dumps({"reasoning": rat, "label": lb})})
    msgs.append({"role": "user",
                 "content": f"PHQ-8 item: {item_text}\nEvidence: {evidence}\nCorrect score: {label}"})
    return msgs


def main():
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    p = argparse.ArgumentParser(description="Teacher rationale generation (Phase A)")
    p.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    p.add_argument("--evidence-column", default="retrieved_context_windows_hybrid_pack")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "cot" /
                                           "teacher_rationales_hybw3.jsonl"))
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_item_dataset(args.dataset_path)
    df["participant_id"] = df["participant_id"].astype(str)
    if args.limit:
        df = df.head(args.limit)

    # resume support: skip rows already written
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((str(r["participant_id"]), int(r["item_id"])))
            except (json.JSONDecodeError, KeyError):
                pass
    print(f"[A] {len(df)} rows | already done {len(done)} | model {args.model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=device).eval()

    t0 = time.time()
    n_new = 0
    with out_path.open("a") as fh:
        for i, r in df.reset_index(drop=True).iterrows():
            kk = (r["participant_id"], int(r["item_id"]))
            if kk in done:
                continue
            evidence = clean_evidence(r[args.evidence_column])
            messages = build_messages(r["item_text"], evidence, int(r["label"]))
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            _, reasoning = parse_output(text)  # label is the gold one
            rec = {"participant_id": r["participant_id"], "item_id": int(r["item_id"]),
                   "item_name": r["item_name"], "item_text": r["item_text"],
                   "label": int(r["label"]), "evidence": evidence,
                   "reasoning": reasoning.strip()}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_new += 1
            if n_new % 50 == 0:
                rate = (time.time() - t0) / n_new
                left = (len(df) - len(done) - n_new) * rate / 60
                print(f"  +{n_new} ({rate:.1f}s/ex, ~{left:.0f} min left)")

    print(f"[A] done: wrote {n_new} new rationales to {out_path} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
