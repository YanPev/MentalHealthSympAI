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


# --- Full-transcript mode -------------------------------------------------
# Zero-shot prompt that scores ALL eight PHQ-8 items in one pass from the whole
# interview transcript. No few-shot exemplars: the transcript is long, so we
# spend the context budget on the participant's own words instead of crafted
# snippet demos. This is the natural-advantage setup -- the LLM sees everything,
# unlike the encoder, which only sees retrieved windows.
SYSTEM_PROMPT_TRANSCRIPT = (
    "You are a careful clinical-research assistant. You are given the full "
    "transcript of a person's interview. Score ALL EIGHT PHQ-8 depression items "
    "from the transcript.\n\n"
    "Score each item on the PHQ-8 frequency scale:\n" + ANCHORS + "\n\n"
    "For each item: briefly note the supporting evidence from the transcript, "
    "judge whether the symptom is absent, occasional, frequent, or persistent, "
    "and map it to the closest anchor. Base every score ONLY on what the "
    "transcript supports -- do not invent symptoms. If an item is never "
    "discussed, score it 0. When evidence is thin or ambiguous, prefer the "
    "lower adjacent score.\n\n"
    "Respond with ONLY a JSON object of this exact form, one entry per item id:\n"
    '{"scores": [{"id": 1, "reasoning": "...", "label": <0|1|2|3>}, ... id 8]}'
)


def build_transcript_messages(item_lines: str, transcript: str):
    """One zero-shot turn: the 8 item texts + the full transcript."""
    user = (f"PHQ-8 items:\n{item_lines}\n\n"
            f"Interview transcript:\n{transcript}")
    return [{"role": "system", "content": SYSTEM_PROMPT_TRANSCRIPT},
            {"role": "user", "content": user}]


def coerce_label(value):
    """Best-effort coerce a model-emitted label into {0,1,2,3} or None."""
    if isinstance(value, (int, float)):
        v = int(value)
        return v if v in (0, 1, 2, 3) else None
    if isinstance(value, str):
        d = re.search(r"[0-3]", value)
        return int(d.group(0)) if d else None
    return None


def parse_transcript_output(text: str, item_ids):
    """Extract per-item (label, reasoning) from an all-8-items completion.
    Robust to a ``{"scores": [...]}`` object, a bare list, or a dict keyed by
    item id. Missing/unparseable items come back as None for the caller to
    fall back on."""
    labels = {iid: None for iid in item_ids}
    reasons = {iid: "" for iid in item_ids}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    entries = []
    if isinstance(obj, dict):
        if isinstance(obj.get("scores"), list):
            entries = obj["scores"]
        else:  # maybe a dict keyed by id: {"1": 2, "2": {...}, ...}
            for k, v in obj.items():
                if re.fullmatch(r"\d+", str(k)):
                    entries.append(v if isinstance(v, dict)
                                   else {"id": int(k), "label": v})
                    if isinstance(v, dict):
                        entries[-1].setdefault("id", int(k))
    elif isinstance(obj, list):
        entries = obj
    for e in entries:
        if not isinstance(e, dict):
            continue
        try:
            iid = int(e.get("id"))
        except (TypeError, ValueError):
            continue
        if iid in labels:
            labels[iid] = coerce_label(e.get("label"))
            reasons[iid] = str(e.get("reasoning", ""))
    return labels, reasons


def fit_transcript(tok, transcript: str, base_overhead: int, max_input: int):
    """Token-truncate a transcript to fit ``max_input - base_overhead``.
    Keeps head + tail (interview opening and most recent turns), dropping the
    middle, since depressive content is spread throughout. Returns
    (text, n_tokens, was_truncated)."""
    ids = tok(transcript, add_special_tokens=False)["input_ids"]
    budget = max(0, max_input - base_overhead)
    if len(ids) <= budget:
        return transcript, len(ids), False
    head = budget // 2
    tail = budget - head
    kept = ids[:head] + ids[-tail:]
    return tok.decode(kept, skip_special_tokens=True), len(kept), True


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
    p.add_argument("--full-transcript", action="store_true",
                   help="Per-participant zero-shot mode: score all 8 PHQ items in "
                        "one pass from the whole transcript (instead of per-item "
                        "retrieved snippets).")
    p.add_argument("--transcript-column", default="transcript_text")
    p.add_argument("--max-context", type=int, default=4096,
                   help="Model context window (LLaMA-2 / MentaLLaMA = 4096). Used "
                        "to size transcript truncation in --full-transcript mode.")
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
    p.add_argument("--load-4bit", action="store_true",
                   help="Load weights in 4-bit (nf4) — needed to fit 13B/32B on one 3090.")
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
    if args.full_transcript:
        print(f"  mode         : full-transcript zero-shot (all 8 items / pass)")
        print(f"  evidence col : {args.transcript_column}  (max_context={args.max_context})")
    else:
        print(f"  mode         : per-item few-shot")
        print(f"  evidence col : {args.evidence_column}")
    print(f"  eval slice   : fold {args.fold} of {Path(args.baseline_oof).name}")
    print(f"  n examples   : {len(eval_df)}  ({eval_df['participant_id'].nunique()} participants)")
    print(f"  decoding     : {'greedy' if args.n_samples == 1 else f'self-consistency x{args.n_samples} @ T={args.temperature}'}")
    print("=" * 64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # LLaMA-2-era tokenizers (e.g. MentaLLaMA) ship no chat template; supply the
    # canonical LLaMA-2 [INST] template so few-shot system/user/assistant turns render.
    if tok.chat_template is None:
        tok.chat_template = (
            "{% if messages[0]['role'] == 'system' %}{% set sys = messages[0]['content'] %}"
            "{% set messages = messages[1:] %}{% else %}{% set sys = '' %}{% endif %}"
            "{% for m in messages %}{% if m['role'] == 'user' %}"
            "{{ bos_token + '[INST] ' + (('<<SYS>>\\n' + sys + '\\n<</SYS>>\\n\\n') if loop.first and sys else '') + m['content'] + ' [/INST]' }}"
            "{% elif m['role'] == 'assistant' %}{{ ' ' + m['content'] + ' ' + eos_token }}{% endif %}{% endfor %}")
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

    if args.full_transcript:
        rows = run_full_transcript(args, eval_df, tok, model)
    else:
        rows = run_per_item(args, eval_df, tok, model)

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_path, index=False)
    print(f"\nSaved predictions: {out_path}  ({len(pred_df)} rows)")

    # ---- quick metrics (full report comes from the figure script) -----------
    from sklearn.metrics import (accuracy_score, f1_score,
                                  mean_absolute_error, cohen_kappa_score)
    yt, yp = pred_df["label"].to_numpy(), pred_df["prediction"].to_numpy()
    print("-" * 64)
    print(f"  accuracy : {accuracy_score(yt, yp):.3f}")
    print(f"  macro_f1 : {f1_score(yt, yp, average='macro', zero_division=0):.3f}")
    print(f"  MAE      : {mean_absolute_error(yt, yp):.3f}")
    print(f"  QWK      : {cohen_kappa_score(yt, yp, weights='quadratic', labels=[0,1,2,3]):.3f}")


def run_per_item(args, eval_df, tok, model):
    import numpy as np
    import torch

    rows = []
    t0 = time.time()
    for i, r in eval_df.reset_index(drop=True).iterrows():
        evidence = format_evidence(r[args.evidence_column])
        messages = build_messages(r["item_text"], evidence)
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        # chat templates already emit special tokens as text -> don't double-add
        inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

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
    print(f"  ({time.time() - t0:.0f}s total)")
    return rows


def run_full_transcript(args, eval_df, tok, model):
    """Per-participant, zero-shot: one pass scores all 8 PHQ items from the full
    transcript. Returns rows in the same schema as the per-item path."""
    import numpy as np
    import torch

    # canonical 8 items (id -> text/name), taken from the eval slice itself
    items = (eval_df.drop_duplicates("item_id")
             .sort_values("item_id")[["item_id", "item_name", "item_text"]])
    item_ids = [int(x) for x in items["item_id"].tolist()]
    name_by_id = dict(zip(items["item_id"].astype(int), items["item_name"]))
    item_lines = "\n".join(f"{int(r.item_id)}. {r.item_text}"
                           for r in items.itertuples())

    # base prompt overhead (everything except the transcript) -> truncation budget
    empty_msgs = build_transcript_messages(item_lines, "")
    empty_prompt = tok.apply_chat_template(
        empty_msgs, tokenize=False, add_generation_prompt=True)
    base_overhead = len(tok(empty_prompt, add_special_tokens=False)["input_ids"])
    max_input = args.max_context - args.max_new_tokens - 16  # safety margin
    print(f"  prompt overhead {base_overhead} tok | transcript budget "
          f"{max(0, max_input - base_overhead)} tok (max_input {max_input})")

    # ground-truth labels: (participant, item) -> label
    label_by_key = {(str(r.participant_id), int(r.item_id)): int(r.label)
                    for r in eval_df.itertuples()}
    transcript_by_pid = (eval_df.drop_duplicates("participant_id")
                         .set_index("participant_id")["transcript_text"].to_dict())
    participants = list(transcript_by_pid.keys())

    rows, n_trunc = [], 0
    t0 = time.time()
    for pi, pid in enumerate(participants):
        transcript = str(transcript_by_pid[pid] or "")
        transcript, ntok, truncated = fit_transcript(
            tok, transcript, base_overhead, max_input)
        n_trunc += int(truncated)
        messages = build_transcript_messages(item_lines, transcript)
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt",
                     add_special_tokens=False).to(model.device)

        gen_kw = dict(max_new_tokens=args.max_new_tokens,
                      pad_token_id=tok.eos_token_id,
                      num_return_sequences=args.n_samples)
        if args.n_samples == 1:
            gen_kw.update(do_sample=False)
        else:
            gen_kw.update(do_sample=True, temperature=args.temperature, top_p=0.9)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kw)
        plen = inputs["input_ids"].shape[1]

        # per-item votes across the N sampled chains (greedy => single chain)
        votes = {iid: [] for iid in item_ids}
        reasons = {iid: "" for iid in item_ids}
        for s in range(out.shape[0]):
            lbls, rsns = parse_transcript_output(
                tok.decode(out[s][plen:], skip_special_tokens=True), item_ids)
            for iid in item_ids:
                if lbls[iid] is not None:
                    votes[iid].append(lbls[iid])
                if s == 0:
                    reasons[iid] = rsns[iid]

        for iid in item_ids:
            v = votes[iid]
            if v:
                counts = np.bincount(v, minlength=NUM_LABELS).astype(float)
                prediction = int(counts.argmax())
                probs = counts / counts.sum()
                reasoning = reasons[iid]
            else:  # model never produced a parseable score for this item
                prediction = 1  # same safe fallback as the per-item path
                probs = np.zeros(NUM_LABELS); probs[1] = 1.0
                reasoning = "[no parseable score -> fallback]"
            rec = {"participant_id": str(pid), "item_id": iid,
                   "item_name": name_by_id[iid],
                   "label": label_by_key[(str(pid), iid)],
                   "prediction": prediction}
            for c in range(NUM_LABELS):
                rec[f"prob_{c}"] = float(probs[c])
            rec["reasoning"] = reasoning
            rows.append(rec)

        done = pi + 1
        rate = (time.time() - t0) / done
        print(f"  {done:3d}/{len(participants)} participants  "
              f"({rate:.1f}s/part, ~{rate * (len(participants) - done) / 60:.1f} min left)")
    print(f"  ({time.time() - t0:.0f}s total | {n_trunc}/{len(participants)} "
          f"transcripts truncated)")
    return rows


if __name__ == "__main__":
    main()
