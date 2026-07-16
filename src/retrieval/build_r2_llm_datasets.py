"""
Stage J — build the LLM evidence-policy dataset variants from the R2 dataset.

The inference-only Qwen predictor conditions (brief §12) that need a modified
evidence set (never zero/mask/drop terminology -- this is filtering, not training):

  L2_R2_FILTER_NONE       remove windows the per-window judge marked `none`.
  L3_R2_INFORMATIVE_FIRST  order supports/against first, then ambiguous; exclude none.
  L4_R2_FALLBACK          keep R2 windows, BUT when the set-level judgment is
                          `none` or no window survives filtering, use the
                          long-context transcript for that (participant, item).

L1_R2_KEEP uses the R2 dataset unchanged; L5_LONG_CONTEXT uses cot_joint's
--full-transcript-items path (no new dataset). Per-window statuses come from the
Stage D window judge; set-level statuses from the set judge (R2 config/retriever/
budget). Evidence lists (ids/scores/texts) are filtered/reordered in parallel.

    .venv/bin/python -m src.retrieval.build_r2_llm_datasets
"""

from pathlib import Path
import argparse
import glob
import json

import pandas as pd

PR = Path(__file__).resolve().parents[2]
DP = PR / "data" / "processed"
JUD = PR / "outputs" / "r2_systematic" / "judge"
RET = PR / "outputs" / "r2_systematic" / "retrieval"
R2_DS = DP / "phq8_item_dataset_r2_w3.csv"
STATUS_PRIORITY = {"supports": 0, "against": 0, "ambiguous": 1, "none": 2}


def _load_window_status():
    files = sorted(glob.glob(str(JUD / "window_judgments_fold*.csv")))
    w = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    w["participant_id"] = w["participant_id"].astype(str)
    return {(r.participant_id, int(r.item_id), r.window_id): r.status for r in w.itertuples()}


def _load_set_status():
    sel = json.loads((RET / "r2_selection.json").read_text())["selected_R2"]
    files = sorted(glob.glob(str(JUD / "set_judgments_fold*.csv")))
    s = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    s["participant_id"] = s["participant_id"].astype(str)
    s = s[(s.config == sel["config"]) & (s.retriever == sel["retriever"]) & (s.prefix == sel["prefix"])]
    return {(r.participant_id, int(r.item_id)): r.status for r in s.itertuples()}


def _fit_transcript(txt, max_tokens=6000):
    """Deterministic head+tail truncation proxy (word-based) for the fallback
    transcript; the actual token truncation is done by cot_joint.fit_transcript
    at generation time. Here we just carry the full transcript text through."""
    return str(txt)


def build():
    df = pd.read_csv(R2_DS)
    df["participant_id"] = df["participant_id"].astype(str)
    win_status = _load_window_status()
    set_status = _load_set_status()

    ids_col, sc_col, tx_col = ("retrieved_context_window_ids_hybrid",
                               "retrieved_context_hybrid_scores",
                               "retrieved_context_windows_hybrid_list")
    variants = {"filter": [], "informative_first": [], "fallback": []}
    counts = {k: {"changed": 0, "emptied": 0, "fallback": 0} for k in variants}

    for _, r in df.iterrows():
        pid, iid = r.participant_id, int(r.item_id)
        ids = json.loads(r[ids_col]) if isinstance(r[ids_col], str) and r[ids_col].strip() else []
        texts = json.loads(r[tx_col]) if isinstance(r[tx_col], str) and r[tx_col].strip() else []
        scores = json.loads(r[sc_col]) if isinstance(r[sc_col], str) and r[sc_col].strip() else []
        stat = [win_status.get((pid, iid, wid), "none") for wid in ids]

        keep = [(i, t, s) for i, t, s, st in zip(ids, texts, scores, stat) if st != "none"]
        # FILTER: drop none
        f_ids, f_tx, f_sc = ([x[0] for x in keep], [x[1] for x in keep], [x[2] for x in keep])
        if len(f_ids) != len(ids):
            counts["filter"]["changed"] += 1
        if not f_ids:
            counts["filter"]["emptied"] += 1
        variants["filter"].append((f_ids, f_sc, f_tx))

        # INFORMATIVE_FIRST: order by status priority, exclude none
        ordered = sorted([(i, t, s, st) for i, t, s, st in zip(ids, texts, scores, stat) if st != "none"],
                         key=lambda x: STATUS_PRIORITY.get(x[3], 3))
        o_ids, o_tx, o_sc = ([x[0] for x in ordered], [x[1] for x in ordered], [x[2] for x in ordered])
        variants["informative_first"].append((o_ids, o_sc, o_tx))

        # FALLBACK: transcript when set none or filter emptied
        if set_status.get((pid, iid)) == "none" or not f_ids:
            tx = _fit_transcript(r.get("transcript_text", ""))
            variants["fallback"].append((["__transcript__"], [1.0], [tx]))
            counts["fallback"]["fallback"] += 1
        else:
            variants["fallback"].append((f_ids, f_sc, f_tx))

    for name, rows in variants.items():
        out = df.copy()
        out[ids_col] = [json.dumps(x[0]) for x in rows]
        out[sc_col] = [json.dumps(x[1]) for x in rows]
        out[tx_col] = [json.dumps(x[2]) for x in rows]
        out["retrieved_context_windows_hybrid_pack"] = [" [WINDOW_SEP] ".join(x[2]) for x in rows]
        out["retrieval_variant"] = f"r2_{name}"
        path = DP / f"phq8_item_dataset_r2_{name}_w3.csv"
        out.to_csv(path, index=False)
        print(f"[{name}] wrote {path.name}  ({counts[name]})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    build()
