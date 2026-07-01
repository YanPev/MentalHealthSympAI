"""
LLM-scored evidence relevance: an alternative to the keyword-hit rate.

retrieval_window_analysis.py measures whether retrieved evidence is "on topic"
for a PHQ-8 item with a heuristic (does any glossary token appear as a
substring?). That's a topical proxy: it can't tell a genuine symptom mention
from an incidental one, and it can't credit relevant evidence phrased without
any glossary word.

This script asks an LLM the same question directly: "does this evidence
contain information relevant to judging this PHQ-8 item?" -- zero-shot,
binary judgment, one call per (participant, item) example -- so the result is
comparable 1:1 to keyword_hit_rate (both are "fraction of examples with
on-topic evidence"). It does NOT score symptom severity (that's cot_probe.py);
it only judges topical relevance of the evidence, same question the keyword
heuristic answers, just judged by the model instead of by substring match.

Needs a GPU (Qwen2.5-7B-Instruct via transformers) -- run on the cluster:

    sbatch run_llm_evidence_relevance.sbatch

or locally with CUDA:

    python -m src.evaluation.llm_evidence_relevance --samples-per-item 40
"""

from pathlib import Path
import argparse
import json
import re
import time

import pandas as pd

from src.llm.cot_probe import format_evidence

PR = Path(__file__).resolve().parents[2]
DEFAULT_DS = PR / "data" / "processed" / "phq8_item_dataset_context_windows_hybrid_w3.csv"
KEYWORD_JSON = PR / "outputs" / "cot" / "retrieval_window_analysis.json"
OUT_CSV = PR / "outputs" / "cot" / "llm_evidence_relevance_examples.csv"
OUT_JSON = PR / "outputs" / "cot" / "llm_evidence_relevance.json"

SYSTEM_PROMPT = (
    "You are helping audit a retrieval system used in a depression-screening "
    "pipeline. You will be shown a PHQ-8 item (a symptom being screened for) and "
    "a set of short evidence snippets retrieved from a person's interview "
    "transcript for that item.\n\n"
    "Your ONLY job is to judge whether the evidence snippets, taken together, "
    "contain ANY information relevant to assessing that specific symptom -- "
    "not whether the symptom is present or absent, just whether the snippets "
    "speak to it at all (directly or in clearly-related lay language). "
    "Evidence about a different symptom, or generic interview chit-chat with no "
    "bearing on this item, counts as NOT relevant.\n\n"
    "Respond with ONLY a JSON object: "
    '{"relevant": true|false, "reason": "<one short phrase>"}.'
)


def build_messages(item_text: str, evidence: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"PHQ-8 item: {item_text}\nEvidence:\n{evidence}"},
    ]


def parse_relevant(text: str):
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            val = obj.get("relevant")
            if isinstance(val, bool):
                return val, str(obj.get("reason", ""))
            if isinstance(val, str):
                return val.strip().lower().startswith("t"), str(obj.get("reason", ""))
        except json.JSONDecodeError:
            pass
    # last-ditch: look for a bare true/false
    m = re.search(r'"?relevant"?\s*[:=]\s*(true|false)', text, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true", "[unparsed reason]"
    return None, "[unparseable output]"


def main():
    ap = argparse.ArgumentParser(description="LLM-scored evidence-relevance hit rate")
    ap.add_argument("--dataset", default=str(DEFAULT_DS))
    ap.add_argument("--evidence-column", default="retrieved_context_windows_hybrid_list")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--samples-per-item", type=int, default=0,
                     help="0 = score every example; >0 = a random subsample per item "
                          "(cost control, mirrors the other offline sanity checks).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    df = pd.read_csv(args.dataset)
    df["participant_id"] = df["participant_id"].astype(str)
    if args.samples_per_item:
        df = (df.groupby("item_id", group_keys=False)
                .apply(lambda g: g.sample(min(args.samples_per_item, len(g)),
                                           random_state=args.seed)))

    print("=" * 64)
    print("LLM-scored evidence relevance (binary, zero-shot)")
    print(f"  model   : {args.model_name}")
    print(f"  dataset : {Path(args.dataset).name}  ({len(df)} examples)")
    print("=" * 64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, quantization_config=qcfg, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    rows = []
    t0 = time.time()
    for i, r in df.reset_index(drop=True).iterrows():
        evidence = format_evidence(r[args.evidence_column])
        messages = build_messages(r["item_text"], evidence)
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                  do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        relevant, reason = parse_relevant(text)
        rows.append({"participant_id": r["participant_id"], "item_id": int(r["item_id"]),
                     "item_name": r["item_name"], "label": int(r["label"]),
                     "llm_relevant": relevant, "llm_reason": reason})
        if (i + 1) % 25 == 0:
            dt = time.time() - t0
            print(f"  [{i+1}/{len(df)}] {dt:.0f}s elapsed, {dt/(i+1):.1f}s/ex")

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(OUT_CSV, index=False)

    unparsed = pred_df["llm_relevant"].isna().sum()
    if unparsed:
        print(f"[warn] {unparsed} unparseable outputs (excluded from hit-rate mean)")

    per_item = []
    for iid in sorted(pred_df.item_id.unique()):
        g = pred_df[pred_df.item_id == iid]
        per_item.append({"item_id": int(iid), "item": g.item_name.iloc[0],
                         "n": int(len(g)),
                         "llm_hit_rate": float(g["llm_relevant"].mean(skipna=True))})

    kw_rate = {}
    if KEYWORD_JSON.exists():
        kw = {r["item"]: r["keyword_hit_rate"] for r in json.loads(KEYWORD_JSON.read_text())["per_item"]}
        for row in per_item:
            row["keyword_hit_rate"] = kw.get(row["item"])
            kw_rate[row["item"]] = kw.get(row["item"])

    OUT_JSON.write_text(json.dumps({"dataset": Path(args.dataset).name,
                                    "model": args.model_name,
                                    "samples_per_item": args.samples_per_item,
                                    "per_item": per_item}, indent=2))

    print("-" * 64)
    print(f"{'item':14s} {'llm_hit_rate':>13} {'keyword_hit_rate':>17}")
    for row in sorted(per_item, key=lambda r: r["llm_hit_rate"]):
        kw = row.get("keyword_hit_rate")
        print(f"{row['item']:14s} {row['llm_hit_rate']:13.2f} {'' if kw is None else f'{kw:17.2f}'}")
    print(f"\nSaved examples: {OUT_CSV}")
    print(f"Saved summary : {OUT_JSON}")


if __name__ == "__main__":
    main()
