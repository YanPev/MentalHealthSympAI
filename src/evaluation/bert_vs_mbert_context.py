"""
Head-to-head BERT vs MentalBERT across the full context-window matrix.

Pairs every BERT run (ctx_*) with its MentalBERT counterpart (ctxm_*) — same
evidence condition, loss, max_length, folds (seed 42) — and reports the
comprehensive metric suite plus per-cell deltas (MentalBERT - BERT).

    python -m src.evaluation.bert_vs_mbert_context
"""

from pathlib import Path
import json

import pandas as pd

from src.evaluation.model_comparison_eval import comprehensive

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "outputs" / "cv"
OUT = ROOT / "outputs"

CONDS = [("old", "Old BM25 utt"), ("hybw3", "Hybrid W3"), ("hybw5", "Hybrid W5"),
         ("bm25w3", "BM25 W3"), ("bm25w5", "BM25 W5")]
# (display, bert_prefix, mbert_prefix, suffix)
ARMS = [
    ("plain CE",    "ctx_ceplain", "ctxm_ceplain", ""),
    ("weighted CE", "ctx_ce",      "ctxm_ce",      ""),
    ("CORN",        "ctx_corn",    "ctxm_corn",    ""),
    ("CORN-512",    "ctx_corn",    "ctxm_corn",    "_512"),
]


def load(tag):
    f = CV / f"oof_predictions_{tag}.csv"
    return comprehensive(pd.read_csv(f)) if f.exists() else None


def flat(m):
    """Pull the headline numbers out of a comprehensive() result."""
    if m is None:
        return None
    return {
        "macro_f1": m["item"]["macro_f1"], "qwk": m["item"]["qwk"], "mae": m["item"]["mae"],
        "f1_c3": m["item"]["f1_per_class"]["3"],
        "total_mae": m["total"]["total_mae"], "total_qwk": m["total"]["total_qwk"],
        "bacc10": m["total_thresholds"][">=10"]["balanced_accuracy"],
    }


def main():
    rows = []
    print("=" * 116)
    print("BERT vs MentalBERT — full context-window matrix (5-fold CV, shared folds, OOF)")
    print("=" * 116)
    h = (f"{'arm':12}{'condition':14}|{'  macroF1 B/M':>16}{'  Δ':>7}|{'  QWK B/M':>14}|"
         f"{' totQWK B/M':>15}|{' bAcc10 B/M':>15}")
    print(h); print("-" * len(h))
    for disp, bp, mp, suf in ARMS:
        for ck, cl in CONDS:
            b = flat(load(f"{bp}_{ck}{suf}"))
            m = flat(load(f"{mp}_{ck}{suf}"))
            if b is None and m is None:
                continue
            if b is None or m is None:
                which = "BERT only" if m is None else "MentalBERT only"
                print(f"{disp:12}{cl:14}|  ({which}; counterpart pending)")
                continue
            d = m["macro_f1"] - b["macro_f1"]
            rows.append({"arm": disp, "condition": cl, "bert": b, "mbert": m,
                         "delta_macro_f1": round(d, 4)})
            print(f"{disp:12}{cl:14}|{b['macro_f1']:>8.3f}/{m['macro_f1']:<7.3f}{d:>+7.3f}|"
                  f"{b['qwk']:>6.3f}/{m['qwk']:<7.3f}|{b['total_qwk']:>6.3f}/{m['total_qwk']:<8.3f}|"
                  f"{b['bacc10']:>6.3f}/{m['bacc10']:<8.3f}")

    # summary: how often does MentalBERT win?
    if rows:
        wins = sum(1 for r in rows if r["delta_macro_f1"] > 0)
        mean_d = sum(r["delta_macro_f1"] for r in rows) / len(rows)
        print("-" * len(h))
        print(f"MentalBERT macro-F1 wins {wins}/{len(rows)} paired cells; mean Δ(M-B) = {mean_d:+.4f}")

    (OUT / "bert_vs_mbert_context.json").write_text(json.dumps(rows, indent=2))
    print(f"Wrote {OUT/'bert_vs_mbert_context.json'} ({len(rows)} paired cells)")


if __name__ == "__main__":
    main()
