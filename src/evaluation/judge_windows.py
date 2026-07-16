"""
Stage D — independent evidence judge (per-window + set-level).

INDEPENDENCE (brief §6, constraint 11): the PHQ predictor is Qwen2.5-7B-Instruct,
so the judge must be a DIFFERENT model family. Default judge is
`klyang/MentaLLaMA-chat-7B` (LLaMA-2 family, mental-health instruction-tuned,
bf16). It sees only the item text and retrieved evidence — never gold labels.

4-way task: {supports, against, ambiguous, none} + severity_hint (0-3 | null) +
reason. Self-consistency: 3 samples @ T (default 0.4, in [0.3,0.5]); majority
vote; ties -> ambiguous; median severity_hint of winning-status chains.

Two modes:
  --mode window : judge each candidate window separately. A per-window judgment
                  depends only on (item, window_text), so unique
                  (participant, item, window) are judged ONCE and reused across
                  configs/retrievers. Covers the Top-N (default 10) of every
                  (config, retriever) ranking.
  --mode set    : judge the COMBINED evidence for the Top-3, Top-5 and
                  TOKEN_BUDGET prefixes of each (config, retriever, participant,
                  item). TOKEN_BUDGET is derived from the MentalBERT tokenizer +
                  max_length 256 (see token_budget()).

Raw generations, parsed outputs, vote counts and final decisions are all saved.

    sbatch run_r2_judge.sbatch                                  # array by fold
    .venv/bin/python -m src.evaluation.judge_windows --mode window --limit 2 --dry-run
"""

from pathlib import Path
import argparse
import json
import re
import statistics
import time

import pandas as pd

PR = Path(__file__).resolve().parents[2]
RANKINGS = PR / "outputs" / "r2_systematic" / "retrieval" / "retrieval_window_scores.parquet"
BASE_DS = PR / "data" / "processed" / "phq8_item_dataset_context_windows_hybrid_w3.csv"
OUT_DIR = PR / "outputs" / "r2_systematic" / "judge"
MBERT = "mental/mental-bert-base-uncased"
DEFAULT_JUDGE = "klyang/MentaLLaMA-chat-7B"

STATUSES = ("supports", "against", "ambiguous", "none")
MAX_LENGTH = 256          # MentalBERT configured max_length (encoder token budget)
SAFETY_MARGIN = 8         # reserved tokens beyond prompt + special tokens
WINDOW_SEP = " [WINDOW_SEP] "

SYSTEM_PROMPT = (
    "You audit whether a short interview excerpt is EVIDENCE about one specific "
    "PHQ-8 depression symptom, for the participant themselves. Choose exactly one "
    "status and reply with ONLY a JSON object.\n\n"
    "CRUCIAL: 'supports' means the excerpt shows the person HAS the symptom. If the "
    "excerpt shows the OPPOSITE (they are fine, they enjoy things, they deny it), "
    "that is 'against', NOT 'supports'. A topic being merely mentioned is not "
    "'supports'.\n"
    "  supports  = the person clearly HAS / experiences this symptom.\n"
    "  against   = the person clearly does NOT (they are fine, deny it, or show the "
    "opposite).\n"
    "  ambiguous = about this symptom but unclear, mixed, or too thin to tell.\n"
    "  none      = the excerpt is not about this symptom at all (other topics or "
    "small talk).\n"
    "Use only what the participant says about themselves; ignore the interviewer; a "
    "keyword alone is not evidence; do not infer this symptom from a different one.\n"
    "severity_hint: 0-3 (0=not at all, 1=several days, 2=more than half the days, "
    "3=nearly every day) ONLY if frequency is explicit, else null. If status is "
    "'against' severity is 0; if 'none' or 'ambiguous' it is null.\n\n"
    "Examples (symptom = 'Little interest or pleasure in doing things'):\n"
    'Excerpt: "I love painting and I still go hiking every weekend, I really enjoy '
    'my hobbies" -> {"status":"against","severity_hint":0,"reason":"still enjoys hobbies"}\n'
    'Excerpt: "we talked about the weather and I took the bus downtown" -> '
    '{"status":"none","severity_hint":null,"reason":"off-topic"}\n'
    'Excerpt: "I don\'t enjoy anything anymore, I stopped doing things I used to love" '
    '-> {"status":"supports","severity_hint":3,"reason":"lost interest in activities"}\n\n'
    "Now judge this one. Reply with ONLY the JSON."
)


def llama2_prompt(system, user):
    """Manual LLaMA-2 chat format (MentaLLaMA has no chat_template)."""
    return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"


def user_msg_window(item_text, window_text):
    ex = window_text.replace("[TURN_SEP]", ";").strip()
    return f"PHQ-8 symptom: {item_text}\nExcerpt:\n{ex}"


def user_msg_set(item_text, window_texts):
    lines = []
    for i, w in enumerate(window_texts, 1):
        lines.append(f"Excerpt {i}: {w.replace('[TURN_SEP]', ';').strip()}")
    return f"PHQ-8 symptom: {item_text}\nEvidence:\n" + "\n".join(lines)


def _regex_severity(text):
    m = re.search(r'"?severity_hint"?\s*[:=]\s*"?(null|[0-3])', text, re.I)
    if m and m.group(1).lower() != "null":
        return int(m.group(1))
    return None


def parse_judgment(text):
    """Robust to truncated JSON: full parse first, else regex status + severity
    (both appear before the free-text reason, so a cut-off reason costs nothing)."""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            status = str(obj.get("status", "")).strip().lower()
            if status in STATUSES:
                hint = obj.get("severity_hint")
                hint = int(hint) if isinstance(hint, (int, float)) and 0 <= int(hint) <= 3 else None
                return status, hint, str(obj.get("reason", ""))[:120]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    m = re.search(r'"?status"?\s*[:=]\s*"?(supports|against|ambiguous|none)', text, re.I)
    if m:
        return m.group(1).lower(), _regex_severity(text), "[regex-parsed]"
    return None, None, "[unparseable]"


def vote(statuses, hints):
    valid = [s for s in statuses if s]
    if not valid:
        return None, None, {}
    counts = {s: valid.count(s) for s in set(valid)}
    top = max(counts.values())
    winners = sorted(s for s, c in counts.items() if c == top)
    status = winners[0] if len(winners) == 1 else "ambiguous"
    wh = [h for s, h in zip(statuses, hints) if s == status and h is not None]
    hint = int(statistics.median(wh)) if wh else None
    return status, hint, counts


def token_budget(item_tokens):
    """Evidence-token budget for the MentalBERT sentence-pair encoder.

    [CLS] item [SEP] evidence [SEP] : 3 special tokens + item tokens + safety.
    """
    reserved = 3 + item_tokens + SAFETY_MARGIN
    return MAX_LENGTH - reserved


def build_budget_prefix(ranked, item_tokens, sep_tokens):
    """Select ranked windows until the next whole window exceeds the budget.

    ranked: list of dicts with 'mbert_token_count'. Returns (selected, stats).
    No window is truncated; the first over-budget window is excluded and its
    length recorded.
    """
    budget = token_budget(item_tokens)
    selected, total, excluded_next = [], 0, 0
    for w in ranked:
        add = int(w["mbert_token_count"]) + (sep_tokens if selected else 0)
        if total + add > budget:
            excluded_next = int(w["mbert_token_count"])
            break
        selected.append(w); total += add
    stats = {"budget": int(budget), "n_windows": len(selected),
             "total_tokens": int(total), "unused_capacity": int(budget - total),
             "excluded_next_len": int(excluded_next)}
    return selected, stats


class Judge:
    def __init__(self, model_name, n_samples, temperature, max_new_tokens=128,
                 load_4bit=False, dry_run=False, batch_size=8, seqlen_cap=6000):
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.dry_run = dry_run
        self.model_name = model_name
        self.batch_size = batch_size
        self.SEQLEN_CAP = seqlen_cap
        if dry_run:
            self.tok = self.model = None
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"          # required for batched generation
        # Use the model's own chat template when it has one (Qwen); else fall back
        # to manual LLaMA-2 [INST] formatting (MentaLLaMA has no template).
        self.use_ct = bool(getattr(self.tok, "chat_template", None))
        self.add_special = not self.use_ct      # chat template already embeds specials
        if load_4bit:
            from transformers import BitsAndBytesConfig
            qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.bfloat16,
                                      bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=qcfg, device_map="auto")
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map=dev)
        self.model.eval()

    def _format(self, system, user):
        if self.use_ct:
            return self.tok.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)
        return llama2_prompt(system, user)

    def _finish(self, statuses, hints, reasons, raws):
        status, hint, counts = vote(statuses, hints)
        reason = next((r for s, r in zip(statuses, reasons) if s == status),
                      reasons[0] if reasons else "")
        return status, hint, counts, reason, raws

    def judge(self, system, user):
        """Single (kept for compatibility). Return (status,hint,counts,reason,raws)."""
        return self.judge_batch(system, [user])[0]

    # Peak generation memory scales with (sequences x padded_len). Cap the
    # product so a batch of long prompts shrinks automatically -> no OOM
    # regardless of prompt-length distribution. seq = batch x n_samples.
    SEQLEN_CAP = 6000

    def _dynamic_batches(self, prompts):
        """Length-sorted, token-aware batches. Returns list of index lists (into
        prompts), each safe under SEQLEN_CAP; also tokenizes once (reused)."""
        lens = [len(self.tok(p, add_special_tokens=self.add_special)["input_ids"]) for p in prompts]
        order = sorted(range(len(prompts)), key=lambda i: lens[i])
        batches, cur, cur_max = [], [], 0
        for i in order:
            nmax = max(cur_max, lens[i])
            if cur and ((len(cur) + 1) * self.n_samples * nmax > self.SEQLEN_CAP
                        or len(cur) >= self.batch_size):
                batches.append(cur); cur, cur_max = [], 0
            cur.append(i); cur_max = max(cur_max, lens[i])
        if cur:
            batches.append(cur)
        return batches

    def judge_batch(self, system, users):
        """Batched judge. Return list of (status, hint, counts, reason, raws)
        aligned to `users`. Uses length-sorted token-aware batching to bound
        peak memory; each prompt is expanded to n_samples chains."""
        if not users:
            return []
        if self.dry_run:
            return [("none", None, {"none": self.n_samples}, "[dry-run]", [])
                    for _ in users]
        prompts = [self._format(system, u) for u in users]
        results = [None] * len(users)
        for batch_idx in self._dynamic_batches(prompts):
            bp = [prompts[i] for i in batch_idx]
            enc = self.tok(bp, return_tensors="pt", padding=True,
                           add_special_tokens=self.add_special).to(self.model.device)
            in_len = enc["input_ids"].shape[1]
            with self.torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens,
                    do_sample=self.n_samples > 1, pad_token_id=self.tok.eos_token_id,
                    temperature=self.temperature if self.n_samples > 1 else None,
                    top_p=0.9 if self.n_samples > 1 else None,
                    num_return_sequences=self.n_samples)
            for j, orig_i in enumerate(batch_idx):
                seqs = out[j * self.n_samples:(j + 1) * self.n_samples]
                statuses, hints, reasons, raws = [], [], [], []
                for seq in seqs:
                    txt = self.tok.decode(seq[in_len:], skip_special_tokens=True)
                    s, h, r = parse_judgment(txt)
                    statuses.append(s); hints.append(h); reasons.append(r)
                    raws.append(txt.strip()[:400])
                results[orig_i] = self._finish(statuses, hints, reasons, raws)
        return results


def load_item_texts():
    base = pd.read_csv(BASE_DS, usecols=["item_id", "item_name", "item_text"]).drop_duplicates()
    return {int(r.item_id): (r.item_name, r.item_text) for r in base.itertuples()}


def main():
    ap = argparse.ArgumentParser(description="Stage D independent evidence judge")
    ap.add_argument("--mode", choices=["window", "set"], required=True)
    ap.add_argument("--rankings", default=str(RANKINGS))
    ap.add_argument("--model-name", default=DEFAULT_JUDGE)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--configs", nargs="+", default=["L0_CORE", "L1_CORE_LAY",
                                                     "L2_CORE_CLINICAL", "L3_FULL"])
    ap.add_argument("--retrievers", nargs="+", default=["bm25", "semantic"])
    ap.add_argument("--top-n", type=int, default=10, help="window mode: per-window depth")
    ap.add_argument("--prefixes", nargs="+", default=["top3", "top5", "budget"])
    ap.add_argument("--fold", type=int, default=0, help="0=all; else shard by fold")
    ap.add_argument("--limit", type=int, default=0, help="cap participants (smoke)")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seqlen-cap", type=int, default=6000,
                    help="max sequences*padded_len per batch (raise on 48GB GPUs)")
    ap.add_argument("--dry-run", action="store_true", help="no model; exercise plumbing")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.rankings)
    df["participant_id"] = df["participant_id"].astype(str)
    df = df[df.config.isin(args.configs) & df.retriever.isin(args.retrievers)]
    if args.fold:
        df = df[df.fold == args.fold]
    if args.limit:
        keep = list(dict.fromkeys(df.participant_id))[:args.limit]
        df = df[df.participant_id.isin(set(keep))]
    item_texts = load_item_texts()

    # MentalBERT tokenizer for the item-token count in the budget calc.
    if not args.dry_run:
        from transformers import AutoTokenizer
        mbert = AutoTokenizer.from_pretrained(MBERT)
        item_tok = {iid: len(mbert.encode(txt, add_special_tokens=False))
                    for iid, (_, txt) in item_texts.items()}
        sep_tokens = len(mbert.encode(WINDOW_SEP, add_special_tokens=False))
    else:
        item_tok = {iid: 12 for iid in item_texts}
        sep_tokens = 5

    judge = Judge(args.model_name, args.n_samples, args.temperature,
                  args.max_new_tokens, args.load_4bit, args.dry_run, args.batch_size,
                  args.seqlen_cap)
    tag = f"_{args.tag}" if args.tag else (f"_fold{args.fold}" if args.fold else "")

    if args.mode == "window":
        run_window(df, judge, item_texts, args.top_n, tag)
    else:
        run_set(df, judge, item_texts, item_tok, sep_tokens, args.prefixes, tag)


def run_window(df, judge, item_texts, top_n, tag, chunk=256):
    sub = df[df["rank"] <= top_n]
    uniq = sub.drop_duplicates(["participant_id", "item_id", "window_id"])[
        ["participant_id", "item_id", "item_name", "window_id", "window_text"]
    ].reset_index(drop=True)
    print(f"[window] judging {len(uniq)} unique (participant,item,window) "
          f"(Top-{top_n}, {judge.n_samples}x SC @ T={judge.temperature}, bs={judge.batch_size})")
    rows, raw_path = [], OUT_DIR / f"window_raw{tag}.jsonl"
    t0 = time.time()
    with open(raw_path, "w") as raw_f:
        for start in range(0, len(uniq), chunk):
            block = uniq.iloc[start:start + chunk]
            users = [user_msg_window(item_texts[int(r.item_id)][1], r.window_text)
                     for r in block.itertuples()]
            results = judge.judge_batch(SYSTEM_PROMPT, users)
            for r, (status, hint, counts, reason, raws) in zip(block.itertuples(), results):
                rows.append({"participant_id": r.participant_id, "item_id": int(r.item_id),
                             "item_name": r.item_name, "window_id": r.window_id,
                             "status": status, "severity_hint": hint,
                             "votes": json.dumps(counts), "reason": reason})
                raw_f.write(json.dumps({"participant_id": r.participant_id,
                                        "item_id": int(r.item_id), "window_id": r.window_id,
                                        "generations": raws}) + "\n")
            raw_f.flush()
            print(f"  [{min(start+chunk, len(uniq))}/{len(uniq)}] {time.time()-t0:.0f}s")
    out = OUT_DIR / f"window_judgments{tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    _report(rows, out, raw_path)


def run_set(df, judge, item_texts, item_tok, sep_tokens, prefixes, tag, chunk=256):
    groups = list(df.groupby(["config", "retriever", "participant_id", "item_id"]))
    # Build all (metadata, user-text|None) jobs first.
    jobs = []  # (meta_dict, user_text_or_None)
    for (config, retriever, pid, iid), g in groups:
        iid = int(iid)
        item_name, item_text = item_texts[iid]
        ranked = g.sort_values("rank").to_dict("records")
        for prefix in prefixes:
            if prefix == "top3":
                sel, stats = ranked[:3], {"n_windows": len(ranked[:3])}
            elif prefix == "top5":
                sel, stats = ranked[:5], {"n_windows": len(ranked[:5])}
            elif prefix == "budget":
                sel, stats = build_budget_prefix(ranked, item_tok[iid], sep_tokens)
            else:
                continue
            meta = {"config": config, "retriever": retriever, "participant_id": pid,
                    "item_id": iid, "item_name": item_name, "prefix": prefix,
                    **_stats_cols(stats)}
            user = user_msg_set(item_text, [w["window_text"] for w in sel]) if sel else None
            jobs.append((meta, user))
    print(f"[set] {len(groups)} groups x {len(prefixes)} prefixes = {len(jobs)} judgments "
          f"({judge.n_samples}x SC @ T={judge.temperature}, bs={judge.batch_size})")
    rows, raw_path = [], OUT_DIR / f"set_raw{tag}.jsonl"
    t0 = time.time()
    with open(raw_path, "w") as raw_f:
        for start in range(0, len(jobs), chunk):
            block = jobs[start:start + chunk]
            todo = [(k, m, u) for k, (m, u) in enumerate(block) if u is not None]
            results = judge.judge_batch(SYSTEM_PROMPT, [u for _, _, u in todo])
            res_by_k = {k: res for (k, _, _), res in zip(todo, results)}
            for k, (meta, user) in enumerate(block):
                if user is None:
                    rows.append({**meta, "status": "none", "severity_hint": None,
                                 "votes": json.dumps({}), "reason": "[empty]"})
                    continue
                status, hint, counts, reason, raws = res_by_k[k]
                rows.append({**meta, "status": status, "severity_hint": hint,
                             "votes": json.dumps(counts), "reason": reason})
                raw_f.write(json.dumps({"config": meta["config"], "retriever": meta["retriever"],
                                        "participant_id": meta["participant_id"],
                                        "item_id": meta["item_id"], "prefix": meta["prefix"],
                                        "generations": raws}) + "\n")
            raw_f.flush()
            print(f"  [{min(start+chunk, len(jobs))}/{len(jobs)}] {time.time()-t0:.0f}s")
    out = OUT_DIR / f"set_judgments{tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    _report(rows, out, raw_path)


def _stats_cols(stats):
    return {"n_windows": stats.get("n_windows"),
            "budget_total_tokens": stats.get("total_tokens"),
            "budget_unused": stats.get("unused_capacity"),
            "budget_excluded_next_len": stats.get("excluded_next_len"),
            "budget_cap": stats.get("budget")}


def _report(rows, out, raw_path):
    df = pd.DataFrame(rows)
    n = len(df)
    parsed = int(df["status"].notna().sum())
    print("-" * 60)
    print(f"rows={n} parse_rate={parsed/max(n,1):.3f}")
    if "status" in df:
        print(df["status"].value_counts(dropna=False).to_string())
    print(f"Saved: {out}\nRaw:   {raw_path}")


if __name__ == "__main__":
    main()
