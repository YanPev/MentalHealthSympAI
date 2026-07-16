"""
Stage G — build hybrid (BM25 + corrected-semantic) rankings for the shortlisted
lexicon, at alpha in {0.25, 0.50, 0.75}.

Reuses the PRODUCTION hybrid fusion + normalization
(hybrid_context_retriever.build_hybrid_records): per (participant, item) it
unions the BM25 and semantic candidates, 0-fills the missing modality, per-row
min-max normalizes each, and fuses hybrid = alpha*bm25_norm + (1-alpha)*sem_norm,
then applies the same turn-overlap diversity + Top-k. The only change vs the old
production path is the SEMANTIC score: here it is the corrected multi-query
max-cosine (semantic_queries.active_queries for the config), not a single
keyword-dump query.

Output rows use the SAME schema as build_r2_rankings (config, retriever in
{hybrid_a25,hybrid_a50,hybrid_a75}, rank, window_id, mbert_token_count, ...) so
the Stage D judge and the Stage E-H selection consume them unchanged.

    .venv/bin/python -m src.retrieval.build_r2_hybrid --config L0_CORE
"""

from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd

from src.retrieval.prod_context.bm25_context_retriever import tokenize, score_bm25_okapi
from src.retrieval.prod_context.hybrid_context_retriever import (
    build_hybrid_records, load_context_window_lookup)
from src.retrieval.prod_context.semantic_multiquery_retriever import encode_normalized
from src.retrieval.r2_lexicons import expanded_query
from src.retrieval.semantic_queries import active_queries, queries_for

PR = Path(__file__).resolve().parents[2]
BASE_DS = PR / "data" / "processed" / "phq8_item_dataset_context_windows_hybrid_w3.csv"
BANK = PR / "data" / "processed" / "context_window_bank_w3.json"
OOF = PR / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"
OUT = PR / "outputs" / "r2_systematic" / "retrieval" / "retrieval_window_scores_hybrid.parquet"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MBERT = "mental/mental-bert-base-uncased"
TOPK, OVERLAP, CMULT = 20, 0.5, 5
ALPHAS = {"hybrid_a25": 0.25, "hybrid_a50": 0.50, "hybrid_a75": 0.75}
CAND = TOPK * CMULT   # candidate pool per modality (100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="L0_CORE")
    ap.add_argument("--limit-participants", type=int, default=0)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    base = pd.read_csv(BASE_DS, usecols=["participant_id", "item_id", "item_name", "item_text"])
    base["participant_id"] = base["participant_id"].astype(str)
    fold_map = (pd.read_csv(OOF, usecols=["participant_id", "fold"])
                .assign(participant_id=lambda d: d.participant_id.astype(str))
                .drop_duplicates("participant_id").set_index("participant_id")["fold"].to_dict())
    bank = {str(p): w for p, w in json.load(open(BANK)).items()}
    lookup = load_context_window_lookup(BANK)
    model = SentenceTransformer(MINILM)
    mbert = AutoTokenizer.from_pretrained(MBERT)

    # query embeddings (per item, L0/config active queries)
    q_index, q_texts, q_owner = {}, [], []
    for item in base.item_name.unique():
        for q in queries_for(item):
            q_index[(item, f"{q['id']}:{q['pole']}")] = len(q_texts)
            q_texts.append(q["text"]); q_owner.append((item, q["kind"]))
    q_emb = encode_normalized(model, q_texts)

    participants = list(dict.fromkeys(base.participant_id))
    if args.limit_participants:
        participants = participants[:args.limit_participants]
    items_by_pid = {p: g[["item_id", "item_name", "item_text"]].values.tolist()
                    for p, g in base[base.participant_id.isin(set(participants))].groupby("participant_id")}

    # 1) build BM25 + semantic candidate records per (participant, item)
    bm_records, sem_records = [], []
    wtok_map = {}
    t0 = time.time()
    for pi, pid in enumerate(participants):
        windows = bank[pid]
        wids = [w["context_window_id"] for w in windows]
        wtexts = [w["window_text"] for w in windows]
        wemb = encode_normalized(model, wtexts)
        win_tokens = [tokenize(t) for t in wtexts]
        for wid, t in zip(wids, wtexts):
            wtok_map[wid] = len(mbert.encode(t, add_special_tokens=False))

        for item_id, item_name, item_text in items_by_pid[pid]:
            item_id = int(item_id)
            # BM25 over all windows with the config-expanded query
            bm_query = f"{item_text} {expanded_query(item_name, args.config)}".strip()
            bm_scores = score_bm25_okapi(tokenize(bm_query), win_tokens)
            bm_order = sorted(range(len(windows)), key=lambda i: (-bm_scores[i], i))
            bm_top = [i for i in bm_order[:CAND] if bm_scores[i] > 0]
            # semantic multi-query max-cosine
            aq = active_queries(item_name, args.config)
            qi = [q_index[(item_name, f"{q['id']}:{q['pole']}")] for q in aq]
            sims = wemb @ q_emb[qi].T
            sem_scores = sims.max(1) if sims.ndim == 2 else sims
            sem_order = sorted(range(len(windows)), key=lambda i: (-float(sem_scores[i]), i))
            sem_top = sem_order[:CAND]

            key_common = {"participant_id": pid, "item_id": item_id,
                          "row_index": len(bm_records)}
            bm_records.append({**key_common,
                               "retrieved_context_window_ids_bm25": [wids[i] for i in bm_top],
                               "retrieved_context_bm25_scores": [float(bm_scores[i]) for i in bm_top]})
            sem_records.append({**key_common,
                                "retrieved_context_window_ids_semantic": [wids[i] for i in sem_top],
                                "retrieved_context_semantic_scores": [float(sem_scores[i]) for i in sem_top]})
        if (pi + 1) % 40 == 0:
            print(f"  scored {pi+1}/{len(participants)} participants ({time.time()-t0:.0f}s)")

    # 2) fuse per alpha with the production hybrid retriever
    win_meta = {w["context_window_id"]: w for p in participants for w in bank[p]}
    name_by_id = {int(r.item_id): r.item_name for r in base.drop_duplicates("item_id").itertuples()}
    rows = []
    for retr, alpha in ALPHAS.items():
        hyb = build_hybrid_records(bm25_records=bm_records, semantic_records=sem_records,
                                   context_window_lookup=lookup, alpha=alpha, k=TOPK,
                                   max_turn_overlap_ratio=OVERLAP, candidate_multiplier=CMULT)
        for rec in hyb:
            pid = rec["participant_id"]; iid = int(rec["item_id"])
            for rank, (wid, hs, bn, sn) in enumerate(zip(
                    rec["retrieved_context_window_ids_hybrid"],
                    rec["retrieved_context_hybrid_scores"],
                    rec["retrieved_context_bm25_normalized"],
                    rec["retrieved_context_semantic_normalized"]), 1):
                w = win_meta[wid]
                rows.append({
                    "participant_id": pid, "fold": int(fold_map[pid]), "item_id": iid,
                    "item_name": name_by_id[iid], "config": args.config, "retriever": retr,
                    "rank": rank, "window_id": wid, "window_text": w["window_text"],
                    "turn_ids": "|".join(w.get("turn_ids", [])),
                    "center_turn_index": w.get("center_turn_index"),
                    "start_time": w.get("start_time"), "end_time": w.get("end_time"),
                    "n_turns": w.get("n_turns"), "mbert_token_count": wtok_map[wid],
                    "bm25_raw": None, "bm25_norm": float(bn),
                    "semantic_score": float(hs),   # store fused hybrid score for ranking
                    "semantic_per_query": None, "winning_query": f"alpha={alpha}"})
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print("-" * 60)
    print(f"hybrid rankings: {len(df)} rows for {args.config} x {list(ALPHAS)}")
    print(df.groupby(["config", "retriever"]).size().to_string())
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
