"""
Item-aware expansion on the PRODUCTION hybrid retriever (next-step #3).

Runs the actual off-branch retrieval pipeline (semantic all-MiniLM + BM25 +
min-max hybrid fusion with diversity control), vendored unchanged into
src/retrieval/prod_context/, with TWO query variants:
  * bare      = the PHQ-8 item text (matched control)
  * item-aware = item text + the symptom's vocabulary (the lever)
Both read the query from the dataset's ``item_text`` column, so we just feed an
expanded item_text — the production code is untouched. Writes two CoT-consumable
datasets (clean item_text + the hybrid-retrieved windows as evidence).

    python -m src.retrieval.build_itemaware_prod_hybrid
"""

from pathlib import Path
import json

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.retrieval.prod_context.semantic_context_retriever import (
    build_semantic_retrieval_records, load_context_window_bank)
from src.retrieval.prod_context.bm25_context_retriever import build_retrieval_records as build_bm25_records
from src.retrieval.prod_context.hybrid_context_retriever import (
    build_hybrid_records, load_context_window_lookup)
from src.retrieval.item_aware_retrieval import EXPAND

PR = Path(__file__).resolve().parents[2]
DS = PR / "data" / "processed" / "phq8_item_dataset_context_windows_hybrid_w5.csv"
BANK = PR / "data" / "processed" / "context_window_bank.json"   # W5 bank
TMP = PR / "outputs" / "cot" / "_prodret_tmp"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ALPHA, K, OVERLAP, CMULT = 0.5, 5, 0.5, 5


def run_variant(name, query_df, base_df, bank, lookup, model):
    TMP.mkdir(parents=True, exist_ok=True)
    qcsv = TMP / f"query_{name}.csv"
    query_df.to_csv(qcsv, index=False)
    sem, _ = build_semantic_retrieval_records(
        dataset_path=qcsv, context_window_bank=bank, model=model,
        k=K, max_turn_overlap_ratio=OVERLAP, candidate_multiplier=CMULT)
    bm = build_bm25_records(
        dataset_path=qcsv, context_window_bank=bank,
        k=K, max_turn_overlap_ratio=OVERLAP, candidate_multiplier=CMULT)
    hyb = build_hybrid_records(
        bm25_records=bm, semantic_records=sem, context_window_lookup=lookup,
        alpha=ALPHA, k=K, max_turn_overlap_ratio=OVERLAP, candidate_multiplier=CMULT)

    # map retrieved windows back onto the clean dataset (clean item_text, labels)
    by_key = {(str(r["participant_id"]), int(r["item_id"])):
              r["retrieved_context_windows_hybrid"] for r in hyb}
    out = base_df.copy()
    out["retrieved_context_windows_prodhybrid_list"] = [
        json.dumps(by_key.get((str(p), int(i)), []))
        for p, i in zip(out["participant_id"], out["item_id"])]
    miss = sum(1 for v in out["retrieved_context_windows_prodhybrid_list"] if v == "[]")
    outpath = PR / "data" / "processed" / f"phq8_item_dataset_prodhybrid_{name}_w5.csv"
    out.to_csv(outpath, index=False)
    print(f"[{name}] wrote {outpath.name} (empty-evidence rows: {miss})")
    return outpath


def main():
    base = pd.read_csv(DS)
    base["participant_id"] = base["participant_id"].astype(str)
    keep = ["participant_id", "item_id", "item_name", "item_text", "label",
            "split", "transcript_text"]
    base = base[keep]
    bank = load_context_window_bank(BANK)
    lookup = load_context_window_lookup(BANK)
    model = SentenceTransformer(MODEL)

    # bare query = item_text; item-aware = item_text + symptom vocabulary
    bare_q = base[["participant_id", "item_id", "item_text"]].copy()
    item_q = base[["participant_id", "item_id", "item_name", "item_text"]].copy()
    item_q["item_text"] = [f"{t} {EXPAND.get(n, '')}".strip()
                           for t, n in zip(item_q["item_text"], item_q["item_name"])]
    item_q = item_q[["participant_id", "item_id", "item_text"]]

    print(f"Production hybrid retrieval (MiniLM + BM25, alpha={ALPHA}, k={K}) — two query variants")
    run_variant("bare", bare_q, base, bank, lookup, model)
    run_variant("itemaware", item_q, base, bank, lookup, model)
    print("Done. Feed either CSV to cot_probe with "
          "--evidence-column retrieved_context_windows_prodhybrid_list")


if __name__ == "__main__":
    main()
