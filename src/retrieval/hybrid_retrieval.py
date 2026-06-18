"""
Item-aware HYBRID retrieval (semantic + BM25), the production-style retriever.

The W5-hybrid evidence the best CoT used was built off-branch. To chase a result
that beats it we re-implement hybrid retrieval here and run it with TWO queries:
  * bare      = the PHQ-8 item text (control)
  * item-aware = item text + the symptom's vocabulary (the lever)
Same fusion for both, so the bare-vs-item-aware delta isolates query expansion;
the existing W5-hybrid CoT (outputs/cot/folds_w5) is the production reference to beat.

Fusion: per participant, min-max normalise BM25 scores and semantic cosine
similarities over the candidate windows, then fuse  alpha*semantic + (1-alpha)*bm25,
take top-k. Embeddings via MiniLM mean-pooling (transformers, no extra dep);
runs CPU-only on the login node.

Writes a CoT-consumable dataset with both evidence columns. No GPU.

    python -m src.retrieval.hybrid_retrieval --alpha 0.5 --k 5
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from src.retrieval.bm25_retriever import tokenize
from src.retrieval.item_aware_retrieval import EXPAND
from src.evaluation.retrieval_window_analysis import LEXICONS, windows_text, hit_terms

PR = Path(__file__).resolve().parents[2]
BANK_W5 = PR / "data" / "processed" / "context_window_bank.json"          # n_turns~5
BASE_DS = PR / "data" / "processed" / "phq8_item_dataset_context_windows_hybrid_w5.csv"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT_CSV = PR / "data" / "processed" / "phq8_item_dataset_hybrid_itemaware_w5.csv"
OUT_JSON = PR / "outputs" / "cot" / "hybrid_itemaware_retrieval.json"


def minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def main():
    import torch
    from transformers import AutoTokenizer, AutoModel
    from rank_bm25 import BM25Okapi

    ap = argparse.ArgumentParser(description="Item-aware hybrid retrieval")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on semantic score")
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    bank = {str(k): v for k, v in json.loads(BANK_W5.read_text()).items()}
    base = pd.read_csv(BASE_DS)
    base["participant_id"] = base["participant_id"].astype(str)
    items = base[["item_id", "item_name", "item_text"]].drop_duplicates().sort_values("item_id")

    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    model = AutoModel.from_pretrained(EMB_MODEL).eval()

    @torch.no_grad()
    def embed(texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True,
                      max_length=128, return_tensors="pt")
            o = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (o * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, dim=1)
            out.append(emb.cpu().numpy())
        return np.concatenate(out, 0)

    # 16 unique queries (8 items x {bare, item-aware}) -- embed once
    q_bare = {int(r.item_id): r.item_text for _, r in items.iterrows()}
    q_item = {int(r.item_id): f"{r.item_text} {EXPAND.get(r.item_name, '')}" for _, r in items.iterrows()}
    all_q = list(q_bare.values()) + list(q_item.values())
    qemb = embed(all_q)
    qbare_emb = {iid: qemb[i] for i, iid in enumerate(q_bare)}
    qitem_emb = {iid: qemb[len(q_bare) + i] for i, iid in enumerate(q_item)}

    # cache per-participant window embeddings + BM25 (built once per participant)
    pwin, pemb, pbm = {}, {}, {}
    for pid, ws in bank.items():
        texts = [w["window_text"] for w in ws if str(w.get("window_text", "")).strip()]
        pwin[pid] = texts
        if texts:
            pemb[pid] = embed(texts)
            pbm[pid] = BM25Okapi([tokenize(t) for t in texts])
    print(f"[hybrid] embedded {sum(len(v) for v in pwin.values())} windows for {len(pwin)} participants")

    def retrieve(pid, iid, item_aware):
        texts = pwin.get(pid, [])
        if not texts:
            return []
        sem = pemb[pid] @ (qitem_emb[iid] if item_aware else qbare_emb[iid])
        qtok = tokenize(q_item[iid] if item_aware else q_bare[iid])
        bm = pbm[pid].get_scores(qtok)
        fused = a.alpha * minmax(sem) + (1 - a.alpha) * minmax(bm)
        top = np.argsort(-fused)[:a.k]
        return [texts[i] for i in top]

    rows = []
    for _, r in base.iterrows():
        pid, iid = r["participant_id"], int(r["item_id"])
        bare = retrieve(pid, iid, False)
        item = retrieve(pid, iid, True)
        rows.append({
            "participant_id": pid, "item_id": iid, "item_name": r["item_name"],
            "item_text": r["item_text"], "label": int(r["label"]),
            "split": r.get("split", ""),
            "transcript_text": r.get("transcript_text", r["item_text"]),
            "retrieved_context_windows_hybridbare_list": json.dumps(bare),
            "retrieved_context_windows_hybriditemaware_list": json.dumps(item),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    # offline sanity: keyword-hit rate per item (bare vs item-aware), partly circular
    def hr(col):
        d = {}
        for iid in range(1, 9):
            g = out[out.item_id == iid]; name = g.item_name.iloc[0]; lex = LEXICONS[name]
            d[name] = float(g[col].apply(lambda s: len(hit_terms(windows_text(json.loads(s)), lex)) > 0).mean())
        return d
    hb, hi = hr("retrieved_context_windows_hybridbare_list"), hr("retrieved_context_windows_hybriditemaware_list")
    OUT_JSON.write_text(json.dumps({"alpha": a.alpha, "k": a.k,
                                    "hit_rate_hybrid_bare": hb, "hit_rate_hybrid_itemaware": hi}, indent=2))
    print(f"[hybrid] keyword-hit rate (sanity): bare->item-aware")
    for n in sorted(hi, key=lambda x: hi[x] - hb[x], reverse=True):
        print(f"  {n:14s} {hb[n]:.2f} -> {hi[n]:.2f}  ({hi[n]-hb[n]:+.2f})")
    print(f"Saved dataset: {OUT_CSV}")


if __name__ == "__main__":
    main()
