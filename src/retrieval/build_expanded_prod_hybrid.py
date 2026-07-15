"""
Expanded-dictionary retrieval on the PRODUCTION hybrid retriever (w3).

Same trick as build_itemaware_prod_hybrid.py -- the expansion lives in the
query CSV's ``item_text``, the production retrieval code is untouched -- but
the query is expanded with the full tiered lexicon (symptom_lexicon:
core glossary + general synonyms/paraphrases + professional clinical
terminology, run together), and BM25 vs semantic can receive DIFFERENT
queries because they take separate dataset paths. Three variants:

  V1  expbm25       expanded query -> BM25 only (no fusion; diagnostic)
  V2  exphyb_bm25q  hybrid: expanded query -> BM25, bare item text -> semantic
  V3  exphyb_bothq  hybrid: expanded query -> both retrievers

Each variant writes a CoT/encoder/MIL-consumable dataset with the STANDARD
evidence column names (retrieved_context_windows_hybrid_pack / _list, ids,
scores) so every downstream consumer works unmodified; for V1 those columns
carry the BM25-only ranking (see the ``retrieval_variant`` column).

    python -m src.retrieval.build_expanded_prod_hybrid            # w3 default
"""

from pathlib import Path
import argparse
import json

import pandas as pd

from src.retrieval.prod_context.semantic_context_retriever import (
    build_semantic_retrieval_records, load_context_window_bank)
from src.retrieval.prod_context.bm25_context_retriever import (
    build_retrieval_records as build_bm25_records)
from src.retrieval.prod_context.hybrid_context_retriever import (
    build_hybrid_records, load_context_window_lookup)
from src.retrieval.symptom_lexicon import expanded_query

PR = Path(__file__).resolve().parents[2]
BASE_DS = {"w3": PR / "data" / "processed" / "phq8_item_dataset_context_windows_hybrid_w3.csv",
           "w5": PR / "data" / "processed" / "phq8_item_dataset_context_windows_hybrid_w5.csv"}
BANK = {"w3": PR / "data" / "processed" / "context_window_bank_w3.json",
        "w5": PR / "data" / "processed" / "context_window_bank.json"}
TMP = PR / "outputs" / "cot" / "_expret_tmp"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ALPHA, K, OVERLAP, CMULT = 0.5, 5, 0.5, 5   # production hybrid settings

KEEP = ["participant_id", "item_id", "item_name", "item_text", "label",
        "split", "transcript_text"]


def query_csv(name, query_df):
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / f"query_{name}.csv"
    query_df.to_csv(path, index=False)
    return path


def write_variant(name, window, base_df, ids, scores, texts):
    """Map retrieved windows back onto the clean dataset and write the CSV."""
    out = base_df.copy()
    out["retrieved_context_window_ids_hybrid"] = [json.dumps(ids[k]) for k in _keys(out)]
    out["retrieved_context_hybrid_scores"] = [json.dumps(scores[k]) for k in _keys(out)]
    out["retrieved_context_windows_hybrid_list"] = [json.dumps(texts[k]) for k in _keys(out)]
    out["retrieved_context_windows_hybrid_pack"] = [
        " [WINDOW_SEP] ".join(texts[k]) for k in _keys(out)]
    out["retrieval_variant"] = name
    n_empty = sum(1 for k in _keys(out) if not texts[k])
    outpath = PR / "data" / "processed" / f"phq8_item_dataset_{name}_{window}.csv"
    out.to_csv(outpath, index=False)
    print(f"[{name}] wrote {outpath.name} (rows: {len(out)}, empty-evidence rows: {n_empty})")
    return outpath


def _keys(df):
    return [(str(p), int(i)) for p, i in zip(df["participant_id"], df["item_id"])]


def records_by_key(records, ids_field, scores_field, texts_field):
    ids, scores, texts = {}, {}, {}
    for r in records:
        key = (str(r["participant_id"]), int(r["item_id"]))
        ids[key] = list(r[ids_field])
        scores[key] = [float(s) for s in r[scores_field]]
        texts[key] = list(r[texts_field])
    return ids, scores, texts


def main():
    ap = argparse.ArgumentParser(description="Expanded-lexicon prod retrieval variants")
    ap.add_argument("--window", choices=["w3", "w5"], default="w3")
    ap.add_argument("--variants", nargs="+",
                    default=["expbm25", "exphyb_bm25q", "exphyb_bothq"])
    args = ap.parse_args()

    base = pd.read_csv(BASE_DS[args.window])
    base["participant_id"] = base["participant_id"].astype(str)
    base = base[KEEP]
    bank = load_context_window_bank(BANK[args.window])
    lookup = load_context_window_lookup(BANK[args.window])
    needs_semantic = any(v.startswith("exphyb") for v in args.variants)
    if needs_semantic:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL)
    else:
        model = None

    # bare query = item_text; expanded = item_text + full tiered lexicon
    bare_q = base[["participant_id", "item_id", "item_text"]].copy()
    exp_q = base[["participant_id", "item_id", "item_name", "item_text"]].copy()
    exp_q["item_text"] = [f"{t} {expanded_query(n)}".strip()
                          for t, n in zip(exp_q["item_text"], exp_q["item_name"])]
    exp_q = exp_q[["participant_id", "item_id", "item_text"]]
    bare_csv = query_csv("bare", bare_q)
    exp_csv = query_csv("expanded", exp_q)

    print(f"Expanded prod retrieval ({args.window}, alpha={ALPHA}, k={K}) "
          f"— variants: {args.variants}")

    retr_kw = dict(k=K, max_turn_overlap_ratio=OVERLAP, candidate_multiplier=CMULT)
    bm_exp = build_bm25_records(dataset_path=exp_csv, context_window_bank=bank, **retr_kw)

    if "expbm25" in args.variants:
        ids, scores, texts = records_by_key(
            bm_exp, "retrieved_context_window_ids_bm25",
            "retrieved_context_bm25_scores", "retrieved_context_windows_bm25")
        write_variant("expbm25", args.window, base, ids, scores, texts)

    sem_cache = {}
    for name, sem_query in [("exphyb_bm25q", bare_csv), ("exphyb_bothq", exp_csv)]:
        if name not in args.variants:
            continue
        if sem_query not in sem_cache:
            sem_cache[sem_query], _ = build_semantic_retrieval_records(
                dataset_path=sem_query, context_window_bank=bank, model=model, **retr_kw)
        hyb = build_hybrid_records(
            bm25_records=bm_exp, semantic_records=sem_cache[sem_query],
            context_window_lookup=lookup, alpha=ALPHA, **retr_kw)
        ids, scores, texts = records_by_key(
            hyb, "retrieved_context_window_ids_hybrid",
            "retrieved_context_hybrid_scores", "retrieved_context_windows_hybrid")
        write_variant(name, args.window, base, ids, scores, texts)

    print("Done. Evidence columns use the standard hybrid names "
          "(retrieved_context_windows_hybrid_pack / _list).")


if __name__ == "__main__":
    main()
