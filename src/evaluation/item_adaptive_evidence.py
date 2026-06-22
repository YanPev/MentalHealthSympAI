"""
Item-adaptive evidence (no GPU): per PHQ-8 item, pick the better evidence source
— full transcript vs focused W5 retrieval — using ONLY the other folds, then
apply to the held-out fold (leakage-free). Tests the full-transcript finding that
the optimal evidence amount is item-dependent (full helps Appetite/NoInterest,
focused helps Depressed/Sleep).

Reuses existing predictions: outputs/cot/folds_w5 and folds_fulltranscript.
Writes folds_itemadaptive/ (cot_probe schema) so the ensemble can consume it.

    python -m src.evaluation.item_adaptive_evidence
"""

from pathlib import Path
import glob

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.evaluation.build_cot_report import metric_block, LABELS

PR = Path(__file__).resolve().parents[2]
ENC_OOF = PR / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"
OUTDIR = PR / "outputs" / "cot" / "folds_itemadaptive"
KEY = ["participant_id", "item_id"]
PROBS = [f"prob_{c}" for c in LABELS]


def load(d):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(PR / "outputs" / "cot" / d / "*.csv")))],
                   ignore_index=True)
    df["participant_id"] = df["participant_id"].astype(str)
    return df


def item_f1(df, iid):
    g = df[df.item_id == iid]
    return f1_score(g.label, g.prediction, average="macro", labels=LABELS, zero_division=0)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    w5 = load("folds_w5"); ft = load("folds_fulltranscript")
    fold = pd.read_csv(ENC_OOF)[["participant_id", "item_id", "fold"]]
    fold["participant_id"] = fold["participant_id"].astype(str)
    for d in (w5, ft):
        d.drop(columns=[c for c in ["fold"] if c in d], inplace=True, errors="ignore")
    w5 = w5.merge(fold, on=KEY); ft = ft.merge(fold, on=KEY)

    folds = sorted(w5.fold.unique())
    chosen = {}                      # (fold, item) -> 'full'/'w5'
    out_rows = {f: [] for f in folds}
    for f in folds:
        tr_w5, tr_ft = w5[w5.fold != f], ft[ft.fold != f]
        te_w5, te_ft = w5[w5.fold == f], ft[ft.fold == f]
        for iid in range(1, 9):
            src = "full" if item_f1(tr_ft, iid) > item_f1(tr_w5, iid) else "w5"
            chosen[(f, iid)] = src
            pick = (te_ft if src == "full" else te_w5)
            out_rows[f].append(pick[pick.item_id == iid])

    # assemble per-fold item-adaptive predictions (cot_probe schema) + pooled
    pooled = []
    for f in folds:
        fdf = pd.concat(out_rows[f], ignore_index=True)
        cols = KEY + ["item_name", "label", "prediction"] + \
            [c for c in PROBS if c in fdf] + (["reasoning"] if "reasoning" in fdf else [])
        fdf[cols].to_csv(OUTDIR / f"cot_probe_qwen_hybw3_fold{f}.csv", index=False)
        pooled.append(fdf)
    pooled = pd.concat(pooled, ignore_index=True)

    # ---- report ----
    enc = pd.read_csv(ENC_OOF)
    refs = {
        "encoder (CORN)": metric_block(enc.label.to_numpy(), enc.prediction.to_numpy()),
        "CoT W5 (focused)": metric_block(w5.label.to_numpy(), w5.prediction.to_numpy()),
        "CoT full-transcript": metric_block(ft.label.to_numpy(), ft.prediction.to_numpy()),
        "CoT item-adaptive": metric_block(pooled.label.to_numpy(), pooled.prediction.to_numpy()),
    }
    print("Item-adaptive evidence — pooled OOF (n=%d):" % len(pooled))
    print(f"{'method':22s} {'F1':>6} {'QWK':>6} {'MAE':>6} {'c2':>6} {'c3':>6}")
    for k, m in refs.items():
        print(f"{k:22s} {m['macro_f1']:6.3f} {m['qwk']:6.3f} {m['mae']:6.3f} "
              f"{m['f1_per_class'][2]:6.3f} {m['f1_per_class'][3]:6.3f}")

    # per-item source vote across folds (majority)
    print("\nper-item source chosen (across 5 folds):")
    names = {iid: w5[w5.item_id == iid].item_name.iloc[0] for iid in range(1, 9)}
    for iid in range(1, 9):
        votes = [chosen[(f, iid)] for f in folds]
        print(f"  {names[iid]:14s} {'/'.join(votes)}  -> {'FULL' if votes.count('full')>=3 else 'W5'}")
    print(f"\nWrote item-adaptive OOF to {OUTDIR}")


if __name__ == "__main__":
    main()
