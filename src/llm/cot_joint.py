"""Joint Chain-of-Thought: score all 8 PHQ-8 items together, reasoning about the
connections between symptoms.

The per-item probe scores each symptom in isolation. But the 8 items are highly
correlated (mean r = 0.56), and the thin-evidence items (Appetite, Sleep) are
predicted far better from the *other* symptoms than from their own evidence. So
here the frozen LLM sees ALL eight symptoms' evidence for one person at once,
forms an overall impression, and scores each — using the whole clinical picture
to calibrate thin-evidence items, while being told NOT to force agreement
(symptoms genuinely dissociate).

Reuses the model + evidence helpers from cot_probe. Writes one row per (item)
in the cot_probe schema so the ensemble / figures can consume it.

    python -m src.llm.cot_joint --fold 1
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

import pandas as pd

from src.data.dataset_loader import load_item_dataset
from src.llm.cot_probe import format_evidence, fit_transcript

NUM_LABELS = 4
ITEM_QUESTIONS = {
    1: "Little interest or pleasure in doing things",
    2: "Feeling down, depressed, or hopeless",
    3: "Trouble falling or staying asleep, or sleeping too much",
    4: "Feeling tired or having little energy",
    5: "Poor appetite or overeating",
    6: "Feeling bad about yourself — or that you are a failure",
    7: "Trouble concentrating on things",
    8: "Moving/speaking slowly, or being fidgety / restless",
}
ANCHORS = ("0 = Not at all (absent, or the opposite is reported)\n"
           "1 = Several days (occasional / mild)\n"
           "2 = More than half the days (frequent)\n"
           "3 = Nearly every day (persistent / dominant)")

JOINT_SYSTEM = (
    "You are a careful clinical-research assistant scoring ALL EIGHT PHQ-8 "
    "depression items for ONE person. For each item you are given short, "
    "keyword-like snippets retrieved from that person's interview.\n\n"
    "Depression symptoms are correlated, so FIRST form a brief overall impression "
    "of how depressed this person seems, THEN score each of the 8 items. Use the "
    "overall picture to help judge an item whose own snippets are thin or "
    "ambiguous. BUT symptoms can genuinely dissociate (e.g. low mood with normal "
    "appetite) — so ground each score primarily in that item's own snippets and "
    "do NOT force the items to agree. Do not invent symptoms not in the snippets.\n\n"
    "Score each item on the PHQ-8 frequency scale:\n" + ANCHORS + "\n\n"
    "Respond with ONLY a JSON object of this exact shape:\n"
    '{"overall": "<one sentence>", "scores": {"1": <0-3>, "2": <0-3>, "3": <0-3>, '
    '"4": <0-3>, "5": <0-3>, "6": <0-3>, "7": <0-3>, "8": <0-3>}}'
)


STAGED_SYSTEM = (
    "You are a careful clinical-research assistant scoring ALL EIGHT PHQ-8 "
    "depression items for ONE person, from short retrieved interview snippets per "
    "item. Work in EXACTLY this order:\n\n"
    "1. CONFIDENT ITEMS: go through the 8 items and identify the ones whose "
    "snippets give CLEAR evidence (strong presence, or clear absence). Assign each "
    "of those a score you are confident in.\n"
    "2. CLINICAL CONTEXT: from ONLY those confident items, summarise this person's "
    "overall depression picture in one or two sentences (how depressed they seem, "
    "which areas are affected).\n"
    "3. DIFFICULT ITEMS: for the remaining items whose snippets are thin or "
    "ambiguous, decide each score using BOTH its own snippets AND the clinical "
    "context — let the context tip genuinely ambiguous calls, but do NOT override "
    "clear evidence, and remember symptoms can dissociate (e.g. low mood with "
    "normal appetite).\n"
    "4. FINAL SCORES: give all 8 final scores.\n\n"
    "Score each item on the PHQ-8 frequency scale:\n" + ANCHORS + "\n\n"
    "Respond with ONLY a JSON object of this exact shape:\n"
    '{"confident": {"<item number>": <0-3>, ...}, '
    '"clinical_context": "<one or two sentences>", '
    '"difficult": {"<item number>": <0-3>, ...}, '
    '"scores": {"1": <0-3>, "2": <0-3>, "3": <0-3>, "4": <0-3>, "5": <0-3>, '
    '"6": <0-3>, "7": <0-3>, "8": <0-3>}}'
)


# Severe-class-nudge variant. Staged CoT is too reluctant to call 3 (its severe
# class-3 F1 collapses to ~0.13, well below per-item CoT's 0.30): it compresses
# the top of the scale, preferring 2 over 3. This prompt tells the difficult stage
# that UNDER-calling a clearly persistent symptom is as much an error as
# over-calling, so 3 should be used when both the item's own evidence and the
# clinical context support a persistent/near-daily symptom — while still requiring
# that bar (we do NOT want to trade the severe gain for a flood of over-calls).
STAGED_SEVERE_SYSTEM = (
    "You are a careful clinical-research assistant scoring ALL EIGHT PHQ-8 "
    "depression items for ONE person, from short retrieved interview snippets per "
    "item. Work in EXACTLY this order:\n\n"
    "1. CONFIDENT ITEMS: go through the 8 items and identify the ones whose "
    "snippets give CLEAR evidence (strong presence, or clear absence). Assign each "
    "of those a score you are confident in. When the snippets show a symptom is "
    "PERSISTENT or dominant (e.g. 'every day', 'all the time', 'constantly'), do "
    "not hesitate to assign 3.\n"
    "2. CLINICAL CONTEXT: from ONLY those confident items, summarise this person's "
    "overall depression picture in one or two sentences (how depressed they seem, "
    "which areas are affected, and whether the overall severity is mild, moderate, "
    "or severe).\n"
    "3. DIFFICULT ITEMS: for the remaining items whose snippets are thin or "
    "ambiguous, decide each score using BOTH its own snippets AND the clinical "
    "context. Let the context tip genuinely ambiguous calls, but do NOT override "
    "clear evidence, and remember symptoms can dissociate (e.g. low mood with "
    "normal appetite). IMPORTANT — do not be reluctant to score 3: UNDER-calling a "
    "symptom (scoring 2 when the item's evidence shows it is persistent/near-daily "
    "AND the overall picture is moderate-to-severe) is just as wrong as "
    "over-calling. Reserve 3 for genuinely persistent symptoms, but when that bar "
    "is met, assign 3 rather than defaulting to 2.\n"
    "4. FINAL SCORES: give all 8 final scores, using the full 0-3 range.\n\n"
    "Score each item on the PHQ-8 frequency scale:\n" + ANCHORS + "\n\n"
    "Respond with ONLY a JSON object of this exact shape:\n"
    '{"confident": {"<item number>": <0-3>, ...}, '
    '"clinical_context": "<one or two sentences>", '
    '"difficult": {"<item number>": <0-3>, ...}, '
    '"scores": {"1": <0-3>, "2": <0-3>, "3": <0-3>, "4": <0-3>, "5": <0-3>, '
    '"6": <0-3>, "7": <0-3>, "8": <0-3>}}'
)


# Tolerance-aware variant. Optimises for the OFF-BY-≥2 objective: an answer is
# only "wrong" if it misses the true severity by two or more bands (pred 1 / true 2
# is fine; pred 2 / true 0 is the error to avoid). The prompt keeps the staged
# confident→context→difficult→final structure (our best config) but reframes the
# scoring goal as: never be off by two — so avoid extreme calls (0 or 3) unless the
# evidence is strong, and when genuinely uncertain across a wide range, hedge toward
# the MIDDLE of the plausible range instead of gambling on an extreme.
STAGED_TOLERANT_SYSTEM = (
    "You are a careful clinical-research assistant scoring ALL EIGHT PHQ-8 "
    "depression items for ONE person, from short retrieved interview snippets per "
    "item.\n\n"
    "SCORING GOAL — you are judged ONLY on whether you land within ONE band of the "
    "true severity. Being off by one level is acceptable; being off by TWO or more "
    "(e.g. scoring 0 when it is 2, or 3 when it is 1) is the only real error. So: "
    "make an extreme call (0 or 3) ONLY when the evidence clearly supports it; when "
    "you are genuinely uncertain across a wide range, choose a MIDDLE value (1 or 2) "
    "to stay close to the truth, rather than gambling on 0 or 3.\n\n"
    "Work in EXACTLY this order:\n"
    "1. CONFIDENT ITEMS: identify the items whose snippets give CLEAR evidence and "
    "assign each a score you are confident in (use 0 or 3 here only when the "
    "evidence is unambiguous).\n"
    "2. CLINICAL CONTEXT: from ONLY those confident items, summarise this person's "
    "overall depression picture in one or two sentences.\n"
    "3. DIFFICULT ITEMS: for the remaining thin/ambiguous items, pick the score "
    "MOST LIKELY to be within one band of the truth, using the item's own snippets "
    "and the clinical context. When torn between two adjacent levels either is fine; "
    "when the range is wide and unclear, take the middle (1 or 2). Remember symptoms "
    "can dissociate.\n"
    "4. FINAL SCORES: give all 8 final scores.\n\n"
    "Score each item on the PHQ-8 frequency scale:\n" + ANCHORS + "\n\n"
    "Respond with ONLY a JSON object of this exact shape:\n"
    '{"confident": {"<item number>": <0-3>, ...}, '
    '"clinical_context": "<one or two sentences>", '
    '"difficult": {"<item number>": <0-3>, ...}, '
    '"scores": {"1": <0-3>, "2": <0-3>, "3": <0-3>, "4": <0-3>, "5": <0-3>, '
    '"6": <0-3>, "7": <0-3>, "8": <0-3>}}'
)


def parse_staged(text):
    """Merge confident + difficult + final scores ('scores' takes precedence)."""
    out = {}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return out
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        for k, v in re.findall(r'"?(\d)"?\s*:\s*([0-3])\b', m.group(0)):
            out[int(k)] = int(v)
        return out
    for key in ("confident", "difficult", "scores"):   # scores processed last = wins
        d = obj.get(key, {})
        if isinstance(d, dict):
            for k, v in d.items():
                ks = re.sub(r"\D", "", str(k))
                if ks and isinstance(v, (int, float)) and 0 <= int(v) <= 3:
                    out[int(ks)] = int(v)
    return out


def parse_staged_buckets(text):
    """Return (confident_ids, difficult_ids) the model declared, as int sets.
    Used to derive the DISJOINT difficult set (difficult − confident) — the items
    the model singled out as hard, which is the semantic gate the cascade routes on."""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return set(), set()
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return set(), set()

    def ids(key):
        d = obj.get(key, {})
        out = set()
        if isinstance(d, dict):
            for k in d:
                ks = re.sub(r"\D", "", str(k))
                if ks:
                    out.add(int(ks))
        return out
    return ids("confident"), ids("difficult")


def build_user(part_df, evidence_column, transcript_items=None, transcript=""):
    """Per-symptom evidence block. For any item id in ``transcript_items`` the
    block points to the full interview transcript (appended once at the end)
    instead of the retrieved snippets — these are the thin-evidence items
    (Appetite/NoInterest) that W3 retrieval starves, where the whole transcript
    recovers symptoms retrieval hid."""
    transcript_items = set(transcript_items or ())
    use_tx = bool(transcript_items and transcript)
    blocks = []
    by_item = {int(r["item_id"]): r for _, r in part_df.iterrows()}
    for iid in range(1, 9):
        if use_tx and iid in transcript_items:
            ev = "(judge this item from the FULL INTERVIEW TRANSCRIPT at the end)"
        else:
            ev = format_evidence(by_item[iid][evidence_column]) if iid in by_item else ""
            ev = ev.strip() or "(no snippets retrieved)"
        blocks.append(f"ITEM {iid} — {ITEM_QUESTIONS[iid]}:\n{ev}")
    user = "Retrieved evidence for this person, by symptom:\n\n" + "\n\n".join(blocks)
    if use_tx:
        tags = ", ".join(f"ITEM {i}" for i in sorted(transcript_items))
        user += (f"\n\n=== FULL INTERVIEW TRANSCRIPT (use this for {tags}) ===\n"
                 + transcript)
    return user


def parse_joint(text):
    """Return {item_id: label} from the model's JSON. Robust to extra prose."""
    out = {}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return out
    try:
        obj = json.loads(m.group(0))
        scores = obj.get("scores", obj)
        for k, v in scores.items():
            ks = re.sub(r"\D", "", str(k))
            if ks and isinstance(v, (int, float)) and 0 <= int(v) <= 3:
                out[int(ks)] = int(v)
    except json.JSONDecodeError:
        # fallback: scrape "item N": v patterns
        for k, v in re.findall(r'"?(\d)"?\s*:\s*([0-3])', m.group(0)):
            out[int(k)] = int(v)
    return out


def main():
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    ap = argparse.ArgumentParser(description="Joint 8-item CoT")
    ap.add_argument("--dataset-path",
                    default="data/processed/phq8_item_dataset_context_windows_hybrid_w3.csv")
    ap.add_argument("--evidence-column", default="retrieved_context_windows_hybrid_list")
    ap.add_argument("--baseline-oof",
                    default=str(PROJECT_ROOT / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"))
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--mode", choices=["joint", "staged"], default="joint",
                    help="joint = single-shot all-8 with an overall impression; "
                         "staged = confident items -> clinical context -> difficult "
                         "items -> final 8 scores (uses item connections explicitly).")
    ap.add_argument("--severe-nudge", action="store_true",
                    help="staged only: use the severe-class-nudge prompt that tells "
                         "the difficult stage not to under-call persistent symptoms "
                         "(counteracts staged's reluctance to score 3).")
    ap.add_argument("--tolerant", action="store_true",
                    help="staged only: tolerance-aware prompt optimised for the "
                         "off-by->=2 objective (off-by-1 forgiven) — hedge toward the "
                         "middle, avoid extreme misses. Takes precedence over "
                         "--severe-nudge (opposite objectives).")
    ap.add_argument("--full-transcript-items", default="",
                    help="Comma-list of item ids (e.g. '1,5') to score from the FULL "
                         "transcript instead of retrieved snippets — for thin-evidence "
                         "items W3 retrieval starves (NoInterest=1, Appetite=5).")
    ap.add_argument("--max-transcript-tokens", type=int, default=6000,
                    help="Token budget (head+tail) for the appended full transcript.")
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--load-4bit", action="store_true",
                    help="Load weights in 4-bit (nf4) — needed to fit 14B/32B/72B Qwen "
                         "on a single 3090. Matches cot_probe's --load-4bit.")
    ap.add_argument("--n-samples", type=int, default=1,
                    help="1 = greedy one-hot. >1 = self-consistency: sample N chains "
                         "per participant and store per-item vote fractions as prob_0..3 "
                         "(the graded confidence the cascade router needs).")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--sc-batch-size", type=int, default=0,
                    help="Generate the N self-consistency chains in micro-batches "
                         "of this size instead of all N at once (0 = all at once, "
                         "the original behaviour). Use 1-2 to bound peak GPU memory "
                         "on long-transcript folds that OOM a 24GB 3090 at N=5.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "cot" /
                                            "folds_joint" / "cot_joint_fold1.csv"))
    args = ap.parse_args()

    # staged emits more text (confident + clinical context + difficult + final),
    # so it needs a bigger budget and the staged-aware parser.
    if args.mode == "staged":
        if args.tolerant:
            system_prompt = STAGED_TOLERANT_SYSTEM
        elif args.severe_nudge:
            system_prompt = STAGED_SEVERE_SYSTEM
        else:
            system_prompt = STAGED_SYSTEM
        parse_scores = parse_staged
        reasoning_cap = 1500
    else:
        system_prompt = JOINT_SYSTEM
        parse_scores = parse_joint
        reasoning_cap = 600
    transcript_items = {int(x) for x in args.full_transcript_items.split(",")
                        if x.strip().isdigit()}
    out_path = Path(args.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = pd.read_csv(args.baseline_oof)
    base["participant_id"] = base["participant_id"].astype(str)
    keys = base[base["fold"] == args.fold][["participant_id", "item_id"]]
    df = load_item_dataset(args.dataset_path)
    df["participant_id"] = df["participant_id"].astype(str)
    eval_df = keys.merge(df, on=["participant_id", "item_id"], how="left")
    parts = list(eval_df.groupby("participant_id"))
    if args.limit:
        parts = parts[:args.limit]
    tag = args.mode.capitalize()
    if args.mode == "staged" and args.tolerant:
        tag += "+tolerant"
    elif args.mode == "staged" and args.severe_nudge:
        tag += "+severe-nudge"
    if transcript_items:
        tag += f" | full-transcript items={sorted(transcript_items)}"
    decode = "greedy" if args.n_samples == 1 else f"self-consistency×{args.n_samples}@T{args.temperature}"
    print(f"{tag} CoT | model {args.model_name} | fold {args.fold} | {decode} | "
          f"{len(parts)} participants × 8 items | device {device}")

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

    rows, n_fail, t0 = [], 0, time.time()
    for pid, pdf in parts:
        transcript = ""
        if transcript_items:
            raw = str(pdf.iloc[0].get("transcript_text", "") or "")
            transcript, _, _ = fit_transcript(tok, raw, 0, args.max_transcript_tokens)
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user(pdf, args.evidence_column,
                                                       transcript_items, transcript)}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        plen = inputs["input_ids"].shape[1]
        # Generate the N self-consistency chains in micro-batches so peak GPU
        # memory tracks the micro-batch size, not N. Five 1100-token chains in
        # one batch OOM'd long-transcript folds on a 24GB 3090 (jobs 19371881 /
        # 19376654); the chains are i.i.d. samples, so batching only changes
        # memory, never the vote distribution.
        chunk = args.sc_batch_size if args.sc_batch_size > 0 else args.n_samples
        texts, remaining = [], args.n_samples
        while remaining > 0:
            k = min(chunk, remaining)
            gen_kw = dict(max_new_tokens=args.max_new_tokens,
                          pad_token_id=tok.eos_token_id, num_return_sequences=k)
            if args.n_samples == 1:
                gen_kw.update(do_sample=False)
            else:  # self-consistency: sampled chains
                gen_kw.update(do_sample=True, temperature=args.temperature, top_p=0.9)
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kw)
            texts.extend(tok.decode(out[s][plen:], skip_special_tokens=True)
                         for s in range(out.shape[0]))
            del out
            remaining -= k
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # tally per-item votes across the N chains (N=1 => a single one-hot vote).
        # diff_counts = how many chains placed the item in the DISJOINT difficult set
        # (difficult − confident) -> the semantic confidence the cascade gates on,
        # now aggregated over all chains instead of just chain-0.
        votes = {iid: [] for iid in range(1, 9)}
        diff_counts = {iid: 0 for iid in range(1, 9)}
        first_text = ""
        for s, txt in enumerate(texts):
            if s == 0:
                first_text = txt
            for iid, lbl in parse_scores(txt).items():
                if 0 <= int(lbl) <= 3:
                    votes[iid].append(int(lbl))
            conf, diff = parse_staged_buckets(txt)
            for iid in (diff - conf):
                if 1 <= iid <= 8:
                    diff_counts[iid] += 1
        for _, r in pdf.iterrows():
            iid = int(r["item_id"])
            v = votes.get(iid, [])
            if v:
                counts = np.bincount(v, minlength=NUM_LABELS).astype(float)
                pred = int(counts.argmax())
                probs = (counts / counts.sum()).tolist()
            else:  # no chain produced a parseable score for this item
                n_fail += 1
                pred = -1
                probs = [0.0] * NUM_LABELS
            rows.append({"participant_id": pid, "item_id": iid,
                         "item_name": r["item_name"], "label": int(r["label"]),
                         "prediction": pred if pred >= 0 else 1,
                         **{f"prob_{c}": probs[c] for c in range(NUM_LABELS)},
                         "difficult_frac": round(diff_counts[iid] / len(texts), 4),
                         "reasoning": first_text.strip()[:reasoning_cap]})
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_path, index=False)
    dt = time.time() - t0
    print(f"Saved {out_path}  ({len(pred_df)} rows, {n_fail} parse-misses, {dt:.0f}s)")

    from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                                 cohen_kappa_score)
    ok = pred_df[pred_df.prediction >= 0]
    yt, yp = ok["label"].to_numpy(), ok["prediction"].to_numpy()
    print(f"  acc {accuracy_score(yt, yp):.3f} | macroF1 "
          f"{f1_score(yt, yp, average='macro', labels=[0,1,2,3], zero_division=0):.3f} | "
          f"MAE {mean_absolute_error(yt, yp):.3f} | "
          f"QWK {cohen_kappa_score(yt, yp, weights='quadratic', labels=[0,1,2,3]):.3f}")


if __name__ == "__main__":
    main()
