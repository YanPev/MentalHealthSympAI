"""
Build the downstream dataset for a selected R2 retrieval configuration.

Given R2 = (lexicon config, retriever, evidence prefix), assemble the evidence
sets from the Stage C rankings into the STANDARD evidence columns
(retrieved_context_windows_hybrid_pack / _list, ids, scores) so the existing
MentalBERT+CORN (cross_validate.py) and Qwen staged-tolerant CoT (cot_joint.py)
consume the dataset unmodified -- exactly the trick build_expanded_prod_hybrid
used, but here the evidence is the label-free-selected R2 set.

The evidence prefix is deterministic from the ranking:
  top3   -> first 3 ranked windows
  top5   -> first 5 ranked windows
  budget -> MentalBERT token-budget prefix (judge_windows.build_budget_prefix)

    .venv/bin/python -m src.retrieval.build_r2_dataset \
        --config L1_CORE_LAY --retriever bm25 --prefix top5 --out-tag r2
"""

from pathlib import Path
import argparse
import json

import pandas as pd

from src.evaluation.judge_windows import build_budget_prefix, WINDOW_SEP

PR = Path(__file__).resolve().parents[2]
RANKINGS = PR / "outputs" / "r2_systematic" / "retrieval" / "retrieval_window_scores.parquet"
BASE_DS = PR / "data" / "processed" / "phq8_item_dataset_context_windows_hybrid_w3.csv"
SEL = PR / "outputs" / "r2_systematic" / "retrieval" / "r2_selection.json"
MBERT = "mental/mental-bert-base-uncased"
KEEP = ["participant_id", "item_id", "item_name", "item_text", "label",
        "split", "transcript_text"]


def prefix_windows(g, prefix, item_tokens, sep_tokens):
    ranked = g.sort_values("rank").to_dict("records")
    if prefix == "top3":
        return ranked[:3]
    if prefix == "top5":
        return ranked[:5]
    if prefix == "budget":
        sel, _ = build_budget_prefix(ranked, item_tokens, sep_tokens)
        return sel
    raise ValueError(prefix)


def main():
    ap = argparse.ArgumentParser(description="Build the R2 downstream dataset")
    ap.add_argument("--config", default=None)
    ap.add_argument("--retriever", default=None)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--rankings", default=str(RANKINGS))
    ap.add_argument("--out-tag", default="r2")
    args = ap.parse_args()

    # default to the selection file if flags omitted
    if not (args.config and args.retriever and args.prefix):
        sel = json.loads(Path(SEL).read_text())["selected_R2"]
        args.config = args.config or sel["config"]
        args.retriever = args.retriever or sel["retriever"]
        args.prefix = args.prefix or sel["prefix"]
    print(f"R2 = config={args.config} retriever={args.retriever} prefix={args.prefix}")

    from transformers import AutoTokenizer
    mbert = AutoTokenizer.from_pretrained(MBERT)
    sep_tokens = len(mbert.encode(WINDOW_SEP, add_special_tokens=False))

    base = pd.read_csv(BASE_DS)
    base["participant_id"] = base["participant_id"].astype(str)
    base = base[KEEP].copy()
    item_tok = {int(r.item_id): len(mbert.encode(r.item_text, add_special_tokens=False))
                for r in base[["item_id", "item_text"]].drop_duplicates().itertuples()}

    rk = pd.read_parquet(args.rankings)
    rk["participant_id"] = rk["participant_id"].astype(str)
    # hybrid retrievers live in a separate parquet -> concat it if present
    hyb = Path(args.rankings).parent / "retrieval_window_scores_hybrid.parquet"
    if hyb.exists():
        h = pd.read_parquet(hyb); h["participant_id"] = h["participant_id"].astype(str)
        rk = pd.concat([rk, h], ignore_index=True)
    rk = rk[(rk.config == args.config) & (rk.retriever == args.retriever)]
    if rk.empty:
        raise SystemExit(f"no rankings for config={args.config} retriever={args.retriever} "
                         "(hybrid must be built by build_r2_hybrid first)")

    ev = {}
    for (pid, iid), g in rk.groupby(["participant_id", "item_id"]):
        sel = prefix_windows(g, args.prefix, item_tok[int(iid)], sep_tokens)
        ev[(pid, int(iid))] = ([w["window_id"] for w in sel],
                               [float(w["semantic_score"] if args.retriever == "semantic"
                                      else w["bm25_norm"]) if pd.notna(
                                   w["semantic_score"] if args.retriever == "semantic"
                                   else w["bm25_norm"]) else 0.0 for w in sel],
                               [w["window_text"] for w in sel])

    keys = [(str(p), int(i)) for p, i in zip(base.participant_id, base.item_id)]
    out = base.copy()
    out["retrieved_context_window_ids_hybrid"] = [json.dumps(ev.get(k, ([], [], []))[0]) for k in keys]
    out["retrieved_context_hybrid_scores"] = [json.dumps(ev.get(k, ([], [], []))[1]) for k in keys]
    out["retrieved_context_windows_hybrid_list"] = [json.dumps(ev.get(k, ([], [], []))[2]) for k in keys]
    out["retrieved_context_windows_hybrid_pack"] = [
        WINDOW_SEP.join(ev.get(k, ([], [], []))[2]) for k in keys]
    out["retrieval_variant"] = f"r2_{args.config}_{args.retriever}_{args.prefix}"

    n_empty = sum(1 for k in keys if not ev.get(k, ([], [], []))[2])
    outpath = PR / "data" / "processed" / f"phq8_item_dataset_{args.out_tag}_w3.csv"
    out.to_csv(outpath, index=False)
    print(f"wrote {outpath.name} rows={len(out)} empty-evidence={n_empty}")
    print(f"evidence col = retrieved_context_windows_hybrid_pack / _list "
          f"(variant {out['retrieval_variant'].iloc[0]})")


if __name__ == "__main__":
    main()
