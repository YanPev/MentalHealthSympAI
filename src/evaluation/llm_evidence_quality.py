"""
4-way LLM evidence-quality judge (extends llm_evidence_relevance.py).

For each (participant, item) the judge classifies the retrieved evidence as:

  supports   -- clear evidence the symptom is present / bothers the person
  against    -- clear evidence the symptom is absent (explicit denial, or the
                topic is discussed and clearly not a problem)
  ambiguous  -- touches the symptom but is unclear, mixed, or insufficient
  none       -- no relevant content for this symptom at all

plus an optional severity_hint (0-3 PHQ-8 frequency anchor, null when not
inferable) and a one-phrase reason. Rows whose evidence list is empty are
labelled `none` deterministically without an LLM call.

The judge sees ONLY the item text and the retrieved evidence -- never gold
labels -- so one global run per dataset is fold-safe: it can drive
retrieval-variant selection, training-label masking (--missing-policy) and the
clinician report without leakage.

Self-consistency: --n-samples chains at temperature --temperature, majority
vote on status (ties -> ambiguous), median severity_hint of the winning-status
chains.

    sbatch run_evidence_quality.sbatch          # array over datasets
    python -m src.evaluation.llm_evidence_quality --dataset ... --tag ...
"""

from pathlib import Path
import argparse
import json
import re
import statistics
import time

import pandas as pd

from src.llm.cot_probe import format_evidence

PR = Path(__file__).resolve().parents[2]
OUT_DIR = PR / "outputs" / "cot"

STATUSES = ("supports", "against", "ambiguous", "none")

SYSTEM_PROMPT = (
    "You are auditing evidence quality in a depression-screening pipeline. "
    "You will be shown a PHQ-8 item (a symptom being screened for) and short "
    "evidence snippets retrieved from a person's interview transcript for that "
    "item.\n\n"
    "Classify the evidence, taken together, into exactly one category:\n"
    '  "supports"  - clear evidence the symptom is present or bothers the '
    "person (stated directly or in clearly-related lay language).\n"
    '  "against"   - clear evidence the symptom is NOT a problem (explicit '
    "denial, or the topic comes up and is clearly fine).\n"
    '  "ambiguous" - the snippets touch on the symptom but are unclear, '
    "mixed, or too thin to judge reliably.\n"
    '  "none"      - nothing in the snippets is relevant to this symptom '
    "(other symptoms or generic chit-chat do not count).\n\n"
    "Also give severity_hint: if the evidence clearly maps to a PHQ-8 "
    "frequency (0 = not at all, 1 = several days, 2 = more than half the "
    "days, 3 = nearly every day), give that number; otherwise null. Do NOT "
    "guess -- only when the evidence itself indicates frequency or clear "
    "absence.\n\n"
    "Respond with ONLY a JSON object: "
    '{"status": "supports|against|ambiguous|none", '
    '"severity_hint": 0|1|2|3|null, "reason": "<one short phrase>"}.'
)


def build_messages(item_text: str, evidence: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"PHQ-8 item: {item_text}\nEvidence:\n{evidence}"},
    ]


def parse_judgment(text: str):
    """Return (status|None, severity_hint|None, reason)."""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            status = str(obj.get("status", "")).strip().lower()
            if status in STATUSES:
                hint = obj.get("severity_hint")
                hint = int(hint) if isinstance(hint, (int, float)) and 0 <= int(hint) <= 3 else None
                return status, hint, str(obj.get("reason", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    m = re.search(r'"?status"?\s*[:=]\s*"?(supports|against|ambiguous|none)', text,
                  flags=re.IGNORECASE)
    if m:
        return m.group(1).lower(), None, "[unparsed reason]"
    return None, None, "[unparseable output]"


def vote(statuses, hints):
    """Majority status (ties -> ambiguous); median hint of winning chains."""
    valid = [s for s in statuses if s]
    if not valid:
        return None, None, {}
    counts = {s: valid.count(s) for s in set(valid)}
    top = max(counts.values())
    winners = sorted(s for s, c in counts.items() if c == top)
    status = winners[0] if len(winners) == 1 else "ambiguous"
    win_hints = [h for s, h in zip(statuses, hints) if s == status and h is not None]
    hint = int(statistics.median(win_hints)) if win_hints else None
    return status, hint, counts


def main():
    ap = argparse.ArgumentParser(description="4-way LLM evidence-quality judge")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True,
                    help="output tag -> outputs/cot/evidence_quality_<tag>.{csv,json}")
    ap.add_argument("--evidence-column", default="retrieved_context_windows_hybrid_list")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--samples-per-item", type=int, default=0,
                    help="0 = all examples; >0 = random subsample per item")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    out_csv = OUT_DIR / f"evidence_quality_{args.tag}.csv"
    out_json = OUT_DIR / f"evidence_quality_{args.tag}.json"

    df = pd.read_csv(args.dataset)
    df["participant_id"] = df["participant_id"].astype(str)
    if args.samples_per_item:
        df = (df.groupby("item_id", group_keys=False)
                .apply(lambda g: g.sample(min(args.samples_per_item, len(g)),
                                          random_state=args.seed)))

    print("=" * 64)
    print("LLM evidence-quality judge (4-way, self-consistency)")
    print(f"  model    : {args.model_name}")
    print(f"  dataset  : {Path(args.dataset).name}  ({len(df)} examples)")
    print(f"  chains   : {args.n_samples} @ T={args.temperature}")
    print(f"  tag      : {args.tag}")
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
    n_llm_calls = 0
    t0 = time.time()
    for i, r in df.reset_index(drop=True).iterrows():
        evidence = format_evidence(r[args.evidence_column])
        n_windows = evidence.count("Excerpt ")
        if not evidence.strip() or n_windows == 0:
            rows.append({"participant_id": r["participant_id"], "item_id": int(r["item_id"]),
                         "item_name": r["item_name"], "label": int(r["label"]),
                         "evidence_status": "none", "status_votes": json.dumps({}),
                         "severity_hint": None, "reason": "[empty evidence list]",
                         "n_windows": 0, "dataset_tag": args.tag})
            continue

        messages = build_messages(r["item_text"], evidence)
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=args.n_samples > 1, pad_token_id=tok.eos_token_id,
                temperature=args.temperature if args.n_samples > 1 else None,
                num_return_sequences=args.n_samples)
        statuses, hints, reasons = [], [], []
        for seq in out:
            text = tok.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            s, h, reason = parse_judgment(text)
            statuses.append(s); hints.append(h); reasons.append(reason)
        n_llm_calls += 1
        status, hint, counts = vote(statuses, hints)
        reason = next((rs for s, rs in zip(statuses, reasons) if s == status),
                      reasons[0] if reasons else "")
        rows.append({"participant_id": r["participant_id"], "item_id": int(r["item_id"]),
                     "item_name": r["item_name"], "label": int(r["label"]),
                     "evidence_status": status, "status_votes": json.dumps(counts),
                     "severity_hint": hint, "reason": reason,
                     "n_windows": int(n_windows), "dataset_tag": args.tag})
        if (i + 1) % 25 == 0:
            dt = time.time() - t0
            print(f"  [{i+1}/{len(df)}] {dt:.0f}s elapsed, {dt/(i+1):.1f}s/ex")

    pred_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_csv, index=False)

    unparsed = int(pred_df["evidence_status"].isna().sum())
    parse_rate = 1.0 - unparsed / max(len(pred_df), 1)
    if parse_rate < 0.95:
        print(f"[warn] parse rate {parse_rate:.3f} < 0.95 ({unparsed} unparseable)")

    per_item = []
    for iid in sorted(pred_df.item_id.unique()):
        g = pred_df[pred_df.item_id == iid]
        counts = g["evidence_status"].value_counts(dropna=True).to_dict()
        n_valid = int(g["evidence_status"].notna().sum())
        informative = (counts.get("supports", 0) + counts.get("against", 0)) / max(n_valid, 1)
        per_item.append({
            "item_id": int(iid), "item": g.item_name.iloc[0], "n": int(len(g)),
            "status_counts": {s: int(counts.get(s, 0)) for s in STATUSES},
            "informative_rate": float(informative),
            "none_rate": float(counts.get("none", 0) / max(n_valid, 1)),
            "unparsed": int(g["evidence_status"].isna().sum()),
        })

    out_json.write_text(json.dumps({
        "dataset": Path(args.dataset).name, "tag": args.tag,
        "model": args.model_name, "n_samples": args.n_samples,
        "temperature": args.temperature, "samples_per_item": args.samples_per_item,
        "n_examples": int(len(pred_df)), "n_llm_calls": n_llm_calls,
        "parse_rate": parse_rate, "per_item": per_item}, indent=2))

    print("-" * 64)
    print(f"{'item':14s} {'supports':>9} {'against':>8} {'ambig':>6} {'none':>5} {'informative':>12}")
    for row in sorted(per_item, key=lambda r: r["informative_rate"]):
        c = row["status_counts"]
        print(f"{row['item']:14s} {c['supports']:9d} {c['against']:8d} "
              f"{c['ambiguous']:6d} {c['none']:5d} {row['informative_rate']:12.2f}")
    print(f"\nSaved examples: {out_csv}")
    print(f"Saved summary : {out_json}")


if __name__ == "__main__":
    main()
