"""
Stage C — build BM25-only and semantic-only Top-20 rankings for L0-L3.

For every lexicon config (L0_CORE, L1_CORE_LAY, L2_CORE_CLINICAL, L3_FULL) and
every (participant, item), produce two independent Top-20 rankings over the
frozen w3 context-window bank:

  * BM25-only   -- query = item_text + config lexicon expansion (r2_lexicons)
  * semantic-only -- max cosine over the config's active natural-language queries
                     (semantic_multiquery_retriever), per-query scores retained

NO hybrid fusion here, and NO predictor is trained. Window construction (w3) is
unchanged; diversity control (turn-overlap 0.5) matches production so the ranked
prefixes equal the evidence actually assembled downstream.

Each row records: participant_id, fold, item, config, retriever, window_id, rank,
window_text, turn_ids, center_turn_index, start_time, end_time, n_turns,
mbert_token_count, raw + normalized BM25 score, semantic per-query scores,
winning semantic query, final (max) semantic score.

    .venv/bin/python -m src.retrieval.build_r2_rankings --limit-participants 2   # smoke
    .venv/bin/python -m src.retrieval.build_r2_rankings                          # full
"""

from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd

from src.retrieval.prod_context.bm25_context_retriever import (
    tokenize, score_bm25_okapi, select_diverse_windows as bm25_diverse)
from src.retrieval.prod_context.semantic_multiquery_retriever import (
    encode_normalized, rank_multiquery)
from src.retrieval.r2_lexicons import CONFIGS, expanded_query
from src.retrieval.semantic_queries import active_queries, queries_for

PR = Path(__file__).resolve().parents[2]
BASE_DS = PR / "data" / "processed" / "phq8_item_dataset_context_windows_hybrid_w3.csv"
BANK = PR / "data" / "processed" / "context_window_bank_w3.json"
OOF = PR / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"
OUT_DIR = PR / "outputs" / "r2_systematic" / "retrieval"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MBERT = "mental/mental-bert-base-uncased"
TOPK = 20
OVERLAP, CMULT = 0.5, 5


def minmax(values):
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def bm25_topk(query_text, windows, win_tokens, k=TOPK):
    """BM25 Top-k reusing pretokenized windows; production diversity + tie-break."""
    q_tokens = tokenize(query_text)
    valid = [i for i, t in enumerate(win_tokens) if t]
    if not valid:
        return []
    scores = score_bm25_okapi(q_tokens, [win_tokens[i] for i in valid])
    ranked = sorted(range(len(valid)), key=lambda p: (-float(scores[p]), valid[p]))
    cand_count = min(len(ranked), max(k, k * CMULT))
    cands = []
    for p in ranked[:cand_count]:
        w = dict(windows[valid[p]])
        w["bm25_score"] = float(scores[p])
        cands.append(w)
    if cands and all(c["bm25_score"] == 0.0 for c in cands):
        return []                      # no lexical overlap at all -> empty ranking
    return bm25_diverse(candidates=cands, k=k, max_turn_overlap_ratio=OVERLAP)


def main():
    ap = argparse.ArgumentParser(description="Stage C: BM25-only + semantic-only Top-20")
    ap.add_argument("--limit-participants", type=int, default=0)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    base = pd.read_csv(BASE_DS, usecols=["participant_id", "item_id", "item_name", "item_text"])
    base["participant_id"] = base["participant_id"].astype(str)
    fold_map = (pd.read_csv(OOF, usecols=["participant_id", "fold"])
                .assign(participant_id=lambda d: d.participant_id.astype(str))
                .drop_duplicates("participant_id").set_index("participant_id")["fold"].to_dict())
    bank = json.load(open(BANK))
    bank = {str(p): w for p, w in bank.items()}

    model = SentenceTransformer(MINILM)
    mbert_tok = AutoTokenizer.from_pretrained(MBERT)

    # Precompute query embeddings once (queries are fixed across participants).
    q_index = {}           # (item, qkey) -> row in q_emb
    q_texts_all, q_keys_all = [], []
    for item in base["item_name"].unique():
        for q in queries_for(item):
            key = f"{q['id']}:{q['pole']}"
            q_index[(item, key)] = len(q_texts_all)
            q_texts_all.append(q["text"]); q_keys_all.append((item, key))
    q_emb_all = encode_normalized(model, q_texts_all)

    participants = list(dict.fromkeys(base["participant_id"]))
    if args.limit_participants:
        participants = participants[:args.limit_participants]

    items_by_pid = {p: g[["item_id", "item_name", "item_text"]].values.tolist()
                    for p, g in base[base.participant_id.isin(set(participants))].groupby("participant_id")}

    rows = []
    t0 = time.time()
    for pi, pid in enumerate(participants):
        windows = bank[pid]
        wtexts = [w["window_text"] for w in windows]
        wemb = encode_normalized(model, wtexts)
        wtok = [len(mbert_tok.encode(t, add_special_tokens=False)) for t in wtexts]
        wtok_map = {w["context_window_id"]: n for w, n in zip(windows, wtok)}
        win_tokens = [tokenize(t) for t in wtexts]
        fold = int(fold_map[pid])

        for item_id, item_name, item_text in items_by_pid[pid]:
            item_id = int(item_id)
            for config in CONFIGS:
                # ---- BM25-only ----
                bm_query = f"{item_text} {expanded_query(item_name, config)}".strip()
                bm_ranked = bm25_topk(bm_query, windows, win_tokens, k=TOPK)
                bm_norm = minmax([w["bm25_score"] for w in bm_ranked])
                for rank, (w, nrm) in enumerate(zip(bm_ranked, bm_norm), 1):
                    rows.append(_row(pid, fold, item_id, item_name, config, "bm25",
                                     rank, w, wtok_map,
                                     bm25_raw=w["bm25_score"], bm25_norm=nrm))
                # ---- semantic-only (multi-query max cosine) ----
                aq = active_queries(item_name, config)
                qkeys = [f"{q['id']}:{q['pole']}" for q in aq]
                qtexts = [q["text"] for q in aq]
                qemb = np.stack([q_emb_all[q_index[(item_name, k)]] for k in qkeys])
                sem_ranked = rank_multiquery(windows, wemb, qtexts, qkeys, qemb, k=TOPK,
                                             max_turn_overlap_ratio=OVERLAP,
                                             candidate_multiplier=CMULT)
                for rank, w in enumerate(sem_ranked, 1):
                    rows.append(_row(pid, fold, item_id, item_name, config, "semantic",
                                     rank, w, wtok_map,
                                     sem_score=w["semantic_score"],
                                     per_query=w["per_query"],
                                     win_q=w["winning_query_key"]))
        if (pi + 1) % 20 == 0:
            print(f"  [{pi+1}/{len(participants)}] {time.time()-t0:.0f}s, {len(rows)} rows")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.out_tag}" if args.out_tag else ""
    stem = OUT_DIR / f"retrieval_window_scores{tag}"
    try:
        df.to_parquet(f"{stem}.parquet", index=False)
        out = f"{stem}.parquet"
    except Exception as e:  # pyarrow/fastparquet missing
        out = f"{stem}.csv"
        df.to_csv(out, index=False)
        print(f"[note] parquet unavailable ({e}); wrote CSV")
    print("-" * 60)
    print(f"participants={len(participants)} rows={len(df)} "
          f"configs={len(CONFIGS)} retrievers=2 in {time.time()-t0:.0f}s")
    print(df.groupby(["config", "retriever"]).size().to_string())
    # empty-ranking diagnostics (per config/retriever, per item/participant)
    cov = (df.groupby(["config", "retriever", "participant_id", "item_id"]).size()
             .rename("n").reset_index())
    exp = len(participants) * 8 * len(CONFIGS) * 2
    print(f"\n(participant,item,config,retriever) groups with >=1 window: {len(cov)} / {exp}")
    print(f"Saved: {out}")


def _row(pid, fold, item_id, item_name, config, retriever, rank, w, wtok_map,
         bm25_raw=None, bm25_norm=None, sem_score=None, per_query=None, win_q=None):
    return {
        "participant_id": pid, "fold": fold, "item_id": item_id,
        "item_name": item_name, "config": config, "retriever": retriever,
        "rank": rank, "window_id": w["context_window_id"],
        "window_text": w["window_text"],
        "turn_ids": "|".join(w.get("turn_ids", [])),
        "center_turn_index": w.get("center_turn_index"),
        "start_time": w.get("start_time"), "end_time": w.get("end_time"),
        "n_turns": w.get("n_turns"),
        "mbert_token_count": wtok_map.get(w["context_window_id"]),
        "bm25_raw": bm25_raw, "bm25_norm": bm25_norm,
        "semantic_score": sem_score,
        "semantic_per_query": json.dumps(per_query) if per_query is not None else None,
        "winning_query": win_q,
    }


if __name__ == "__main__":
    main()
