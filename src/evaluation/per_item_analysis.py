"""
Per-item PHQ-8 analysis: which symptom items does the model predict best?

Pools the multi-seed weighted runs in ``outputs/seeds_weighted/`` (both models x
5 seeds = 10 runs) and computes, per PHQ-8 item, metrics averaged across runs
(mean +/- std) plus difficulty descriptors (label spread, majority-class share).
Items are ranked by macro-F1. Writes a tidy CSV for downstream use.

    python -m src.evaluation.per_item_analysis
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = ROOT / "outputs" / "seeds_weighted"
OUT_CSV = ROOT / "outputs" / "per_item_analysis.csv"
LABELS = [0, 1, 2, 3]


def run_metrics(g):
    y, p = g["true_label"].to_numpy(), g["prediction"].to_numpy()
    return (
        accuracy_score(y, p),
        f1_score(y, p, average="macro", labels=LABELS, zero_division=0),
        mean_absolute_error(y, p),
    )


def main():
    files = sorted(SEED_DIR.glob("*_ret_weighted_s*.csv"))
    if not files:
        print("No seed run CSVs found in", SEED_DIR)
        return

    # Per-run, per-item metrics -> aggregate across the 10 runs.
    records = []
    for f in files:
        df = pd.read_csv(f)
        for (iid, iname), g in df.groupby(["item_id", "item_name"]):
            acc, mf1, mae = run_metrics(g)
            records.append({"item_id": iid, "item_name": iname,
                            "accuracy": acc, "macro_f1": mf1, "mae": mae})
    runs = pd.DataFrame(records)

    agg = runs.groupby(["item_id", "item_name"]).agg(
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
    ).reset_index()

    # Difficulty descriptors from the (seed-invariant) true labels.
    truth = pd.read_csv(files[0])
    desc = []
    for iid, g in truth.groupby("item_id"):
        y = g["true_label"].to_numpy()
        vc = np.bincount(y, minlength=4) / len(y)
        desc.append({
            "item_id": iid,
            "support": len(y),
            "mean_severity": float(y.mean()),
            "majority_share": float(vc.max()),
            "pct_severe_2_3": float((y >= 2).mean()),
        })
    desc = pd.DataFrame(desc)

    out = agg.merge(desc, on="item_id").sort_values("f1_mean", ascending=False)
    out.to_csv(OUT_CSV, index=False)

    print("Per-item performance (pooled over 10 weighted runs: 2 models x 5 seeds)")
    print("Ranked by macro-F1 (best first). 33 test examples per item.")
    print("=" * 92)
    hdr = (f"{'item':14} {'macroF1':>14} {'acc':>14} {'MAE':>14} "
           f"{'maj%':>6} {'sev%':>6} {'meanSev':>8}")
    print(hdr)
    print("-" * 92)
    for _, r in out.iterrows():
        print(f"{r['item_name']:14} "
              f"{r['f1_mean']:.3f}±{r['f1_std']:.3f}  "
              f"{r['acc_mean']:.3f}±{r['acc_std']:.3f}  "
              f"{r['mae_mean']:.3f}±{r['mae_std']:.3f}  "
              f"{r['majority_share']*100:5.0f} {r['pct_severe_2_3']*100:5.0f} "
              f"{r['mean_severity']:8.2f}")
    print("-" * 92)
    best, worst = out.iloc[0], out.iloc[-1]
    print(f"\nBest item : {best['item_name']:14} macroF1 {best['f1_mean']:.3f}  MAE {best['mae_mean']:.3f}")
    print(f"Worst item: {worst['item_name']:14} macroF1 {worst['f1_mean']:.3f}  MAE {worst['mae_mean']:.3f}")
    corr = out["f1_mean"].corr(out["majority_share"])
    corr_s = out["f1_mean"].corr(out["pct_severe_2_3"])
    print(f"\nCorr(macroF1, majority_share) = {corr:+.2f}  "
          f"| Corr(macroF1, %severe) = {corr_s:+.2f}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
