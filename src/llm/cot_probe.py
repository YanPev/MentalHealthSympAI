"""
Few-shot chain-of-thought (CoT) feasibility probe for PHQ-8 item severity.

This is a *go / no-go* experiment, not a leaderboard entry. It asks one
question: can an off-the-shelf open-weight instruct model, given the same
Hybrid-W3 evidence our best encoder sees, reason about the PHQ-8 frequency
anchors and predict severity 0-3 well enough to be worth a full
teacher-generation + distillation effort?

Design choices that make it a *fair, paired* comparison to the encoder:

* It evaluates on the EXACT rows of one CV fold, taken from an existing
  out-of-fold (OOF) prediction file (default: the best config,
  MentalBERT + CORN + Hybrid-W3). Same participants, same items, same labels,
  so the two models' predictions can be compared row-for-row.
* The few-shot exemplars are crafted demonstrations (synthetic evidence), so
  no real participant from any fold leaks into the prompt.
* The reasoning is forced to map quoted evidence to the standard PHQ-8
  frequency anchors (0 not at all .. 3 nearly every day) -- the adjacent-
  severity boundary our CORN / near-miss work identified as the bottleneck.

Output is a predictions CSV in the same shape as the encoder OOF files
(participant_id, item_id, item_name, label, prediction, prob_0..3) plus a
`reasoning` column, so it drops straight into the existing metric code.

    python -m src.llm.cot_probe \
        --baseline-oof outputs/cv/oof_predictions_ctxm_corn_hybw3.csv \
        --fold 1 --n-samples 1
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

NUM_LABELS = 4
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / \
    "phq8_item_dataset_context_windows_hybrid_w3.csv"

# Standard PHQ-8 response anchors. The whole point of CoT here is to make the
# model reason explicitly about which of these the evidence supports.
ANCHORS = (
    "0 = Not at all (the symptom is absent, or the person reports the opposite)\n"
    "1 = Several days (occasional / mild; present some of the time)\n"
    "2 = More than half the days (frequent; clearly affecting the person)\n"
    "3 = Nearly every day (persistent / severe; dominant in their account)"
)

SYSTEM_PROMPT = (
    "You are a careful clinical-research assistant scoring a single PHQ-8 "
    "depression-screening item from evidence taken from a person's interview.\n\n"
    "The evidence is a set of short, retrieved snippets from the interview; they "
    "are keyword-like and out of order, separated by '///'. Treat them as weak "
    "cues, not full sentences. Do NOT invent symptoms that are not supported by "
    "the snippets.\n\n"
    "Score the item on the PHQ-8 frequency scale:\n" + ANCHORS + "\n\n"
    "Reason step by step: (1) note which snippets are relevant to THIS item, "
    "(2) judge whether they indicate the symptom is absent, occasional, frequent, "
    "or persistent, (3) map that to the closest anchor. When evidence is thin or "
    "ambiguous, prefer the lower adjacent score. "
    "Respond with ONLY a JSON object: {\"reasoning\": \"...\", \"label\": <0|1|2|3>}."
)

# Crafted few-shot demonstrations (synthetic evidence -- no real participant).
# One per severity level, each quoting snippets and mapping to an anchor.
FEWSHOT = [
    {
        "item": "Feeling tired or having little energy",
        "evidence": "I sleep fine /// plenty of energy for work and the gym /// "
                    "weekends I'm out hiking /// no I don't really get worn out",
        "reasoning": "The snippets ('plenty of energy', 'out hiking', \"don't get "
                     "worn out\") all indicate normal energy and contradict fatigue. "
                     "No cue suggests tiredness. This maps to 'Not at all'.",
        "label": 0,
    },
    {
        "item": "Trouble falling or staying asleep, or sleeping too much",
        "evidence": "some nights I toss and turn /// usually okay though /// "
                    "couple bad nights last week /// mostly fall asleep fine",
        "reasoning": "Snippets show occasional difficulty ('some nights I toss and "
                     "turn', 'couple bad nights') against a mostly-fine baseline "
                     "('usually okay', 'mostly fall asleep fine'). Present some of "
                     "the time, not most days -> 'Several days'.",
        "label": 1,
    },
    {
        "item": "Little interest or pleasure in doing things",
        "evidence": "stopped seeing friends /// don't really enjoy things anymore /// "
                    "used to love painting now I can't be bothered /// most days feel flat",
        "reasoning": "Multiple snippets show loss of interest across activities "
                     "('don't enjoy things anymore', \"can't be bothered\" with a "
                     "former hobby, 'stopped seeing friends') and 'most days feel "
                     "flat' signals high frequency. This is frequent -> 'More than "
                     "half the days'.",
        "label": 2,
    },
    {
        "item": "Feeling down, depressed, or hopeless",
        "evidence": "I feel down every single day /// nothing is going to get better /// "
                    "cry most mornings /// can't see a way out /// hopeless",
        "reasoning": "The snippets are pervasive and severe: 'every single day', "
                     "'cry most mornings', explicit 'hopeless' and 'nothing is going "
                     "to get better'. Persistent and dominant -> 'Nearly every day'.",
        "label": 3,
    },
]


def clean_evidence(raw: str) -> str:
    """Make the retrieved-window pack readable for the LLM."""
    if not isinstance(raw, str):
        return ""
    txt = raw.replace("[WINDOW_SEP]", " /// ").replace("[TURN_SEP]", "; ")
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def format_evidence(value) -> str:
    """Preferred formatter: if the column is a JSON list of windows, present each
    retrieved window as its own numbered excerpt (consecutive turns joined), which
    keeps local context coherent instead of one shuffled keyword blob. Falls back
    to ``clean_evidence`` for the flat '_pack' string column."""
    if not isinstance(value, str):
        return ""
    s = value.strip()
    if s.startswith("["):
        try:
            wins = json.loads(s)
        except json.JSONDecodeError:
            return clean_evidence(value)
        out = []
        for i, w in enumerate(wins, 1):
            turns = [t.strip() for t in str(w).split("[TURN_SEP]") if t.strip()]
            if turns:
                out.append(f"Excerpt {i}: " + "; ".join(turns))
        if out:
            return "\n".join(out)
    return clean_evidence(value)


def build_messages(item_text: str, evidence: str):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEWSHOT:
        msgs.append({"role": "user", "content":
                     f"PHQ-8 item: {ex['item']}\nEvidence: {ex['evidence']}"})
        msgs.append({"role": "assistant", "content": json.dumps(
            {"reasoning": ex["reasoning"], "label": ex["label"]})})
    msgs.append({"role": "user", "content":
                 f"PHQ-8 item: {item_text}\nEvidence: {evidence}"})
    return msgs


def parse_output(text: str):
    """Extract (label, reasoning) from a model completion. Robust to extra prose."""
    label, reasoning = None, ""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj.get("label"), (int, float)):
                label = int(obj["label"])
            elif isinstance(obj.get("label"), str):
                d = re.search(r"[0-3]", obj["label"])
                label = int(d.group(0)) if d else None
            reasoning = str(obj.get("reasoning", ""))
        except json.JSONDecodeError:
            pass
    if label is None:  # last-ditch: first standalone 0-3 in the text
        d = re.search(r'"?label"?\s*[:=]\s*([0-3])', text)
        if not d:
            d = re.search(r"\b([0-3])\b", text)
        label = int(d.group(1)) if d else None
    if label is None or label not in (0, 1, 2, 3):
        label = 1  # safe fallback to the modal-ish middle-low class
        reasoning = "[unparseable output -> fallback] " + reasoning
    return label, reasoning


def main():
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    p = argparse.ArgumentParser(description="Few-shot CoT probe for PHQ-8 severity")
    p.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    p.add_argument("--evidence-column", default="retrieved_context_windows_hybrid_pack")
    p.add_argument("--baseline-oof",
                   default=str(PROJECT_ROOT / "outputs" / "cv" /
                               "oof_predictions_ctxm_corn_hybw3.csv"),
                   help="OOF file that defines the exact eval slice (and the "
                        "encoder baseline we compare against).")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--n-samples", type=int, default=1,
                   help="1 = greedy (deterministic). >1 = self-consistency vote.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=0, help="Debug: cap #examples (0=all).")
    p.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "cot" /
                                           "cot_probe_qwen_hybw3_fold1.csv"))
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- define the eval slice from the baseline OOF (paired comparison) -----
    base = pd.read_csv(args.baseline_oof)
    base["participant_id"] = base["participant_id"].astype(str)
    slice_keys = base[base["fold"] == args.fold][
        ["participant_id", "item_id"]].copy()

    df = load_item_dataset(args.dataset_path)
    df["participant_id"] = df["participant_id"].astype(str)
    eval_df = slice_keys.merge(df, on=["participant_id", "item_id"], how="left")
    missing = eval_df["item_text"].isna().sum()
    assert missing == 0, f"{missing} eval rows not found in dataset"
    if args.limit:
        eval_df = eval_df.head(args.limit)

    print("=" * 64)
    print("PHQ-8 few-shot CoT feasibility probe")
    print(f"  model        : {args.model_name}")
    print(f"  dataset      : {Path(args.dataset_path).name}")
    print(f"  evidence col : {args.evidence_column}")
    print(f"  eval slice   : fold {args.fold} of {Path(args.baseline_oof).name}")
    print(f"  n examples   : {len(eval_df)}  ({eval_df['participant_id'].nunique()} participants)")
    print(f"  decoding     : {'greedy' if args.n_samples == 1 else f'self-consistency x{args.n_samples} @ T={args.temperature}'}")
    print("=" * 64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()

    rows = []
    t0 = time.time()
    for i, r in eval_df.reset_index(drop=True).iterrows():
        evidence = format_evidence(r[args.evidence_column])
        messages = build_messages(r["item_text"], evidence)
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(device)

        gen_kw = dict(max_new_tokens=args.max_new_tokens,
                      pad_token_id=tok.eos_token_id,
                      num_return_sequences=args.n_samples)
        if args.n_samples == 1:
            gen_kw.update(do_sample=False)
        else:
            # batched sampling: all N chains in one call (shares prompt encoding)
            gen_kw.update(do_sample=True, temperature=args.temperature, top_p=0.9)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kw)
        plen = inputs["input_ids"].shape[1]
        votes, first_reasoning = [], ""
        for s in range(out.shape[0]):
            lbl, reasoning = parse_output(
                tok.decode(out[s][plen:], skip_special_tokens=True))
            votes.append(lbl)
            if s == 0:
                first_reasoning = reasoning

        counts = np.bincount(votes, minlength=NUM_LABELS).astype(float)
        prediction = int(counts.argmax())
        probs = counts / counts.sum()

        rec = {"participant_id": r["participant_id"], "item_id": int(r["item_id"]),
               "item_name": r["item_name"], "label": int(r["label"]),
               "prediction": prediction}
        for c in range(NUM_LABELS):
            rec[f"prob_{c}"] = float(probs[c])
        rec["reasoning"] = first_reasoning
        rows.append(rec)

        if (i + 1) % 25 == 0 or i + 1 == len(eval_df):
            done = i + 1
            rate = (time.time() - t0) / done
            print(f"  {done:4d}/{len(eval_df)}  "
                  f"({rate:.1f}s/ex, ~{rate * (len(eval_df) - done) / 60:.1f} min left)")

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_path, index=False)
    print(f"\nSaved predictions: {out_path}  ({time.time() - t0:.0f}s total)")

    # ---- quick metrics (full report comes from the figure script) -----------
    from sklearn.metrics import (accuracy_score, f1_score,
                                  mean_absolute_error, cohen_kappa_score)
    yt, yp = pred_df["label"].to_numpy(), pred_df["prediction"].to_numpy()
    print("-" * 64)
    print(f"  accuracy : {accuracy_score(yt, yp):.3f}")
    print(f"  macro_f1 : {f1_score(yt, yp, average='macro', zero_division=0):.3f}")
    print(f"  MAE      : {mean_absolute_error(yt, yp):.3f}")
    print(f"  QWK      : {cohen_kappa_score(yt, yp, weights='quadratic', labels=[0,1,2,3]):.3f}")


if __name__ == "__main__":
    main()
