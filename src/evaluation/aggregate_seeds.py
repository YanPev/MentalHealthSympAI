"""
Aggregate the multi-seed weighted runs to test whether MentalBERT's edge is real.

Reads per-seed prediction CSVs from ``outputs/seeds_weighted/`` for both models
(retrieval + class weights), computes per-seed metrics, and reports mean +/- std
plus a paired (per-seed) comparison of MentalBERT vs BERT.

    python -m src.evaluation.aggregate_seeds
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = ROOT / "outputs" / "seeds_weighted"
LABELS = [0, 1, 2, 3]


def metrics_for(csv):
    df = pd.read_csv(csv)
    y, p = df["true_label"].to_numpy(), df["prediction"].to_numpy()
    _, _, f1c, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    return {
        "acc": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "mae": mean_absolute_error(y, p),
        "f1_2": f1c[2], "f1_3": f1c[3],
    }


def collect(tag):
    rows = {}
    for csv in sorted(SEED_DIR.glob(f"{tag}_ret_weighted_s*.csv")):
        m = re.search(r"_s(\d+)\.csv$", csv.name)
        if not m:
            continue
        rows[int(m.group(1))] = metrics_for(csv)
    return rows


def fmt(vals):
    a = np.array(vals)
    return f"{a.mean():.3f} ± {a.std(ddof=1):.3f}"


def main():
    bert = collect("bert")
    mbert = collect("mentalbert")
    seeds = sorted(set(bert) & set(mbert))
    if not seeds:
        print("No matched seed runs found in", SEED_DIR)
        return

    keys = ["acc", "macro_f1", "mae", "f1_2", "f1_3"]
    names = {"acc": "Accuracy", "macro_f1": "Macro-F1", "mae": "MAE",
             "f1_2": "class-2 F1", "f1_3": "class-3 F1"}

    print(f"Seeds: {seeds}  (n={len(seeds)})")
    print("=" * 64)
    print(f"{'metric':12} {'BERT-base':>18} {'MentalBERT':>18}")
    print("-" * 64)
    for k in keys:
        print(f"{names[k]:12} {fmt([bert[s][k] for s in seeds]):>18} "
              f"{fmt([mbert[s][k] for s in seeds]):>18}")

    print("\nPaired comparison (MentalBERT - BERT, per seed):")
    print("-" * 64)
    for k in ["macro_f1", "f1_2", "f1_3"]:
        diffs = np.array([mbert[s][k] - bert[s][k] for s in seeds])
        wins = int((diffs > 0).sum())
        sig = ""
        if diffs.std(ddof=1) > 0:
            t = diffs.mean() / (diffs.std(ddof=1) / np.sqrt(len(diffs)))
            sig = f"  (paired t={t:+.2f})"
        print(f"{names[k]:12} mean Δ {diffs.mean():+.3f}  | MentalBERT wins {wins}/{len(seeds)} seeds{sig}")

    print("\nPer-seed macro-F1:")
    print(f"{'seed':>6} {'BERT':>8} {'MentalBERT':>12} {'Δ':>8}")
    for s in seeds:
        d = mbert[s]["macro_f1"] - bert[s]["macro_f1"]
        print(f"{s:>6} {bert[s]['macro_f1']:>8.3f} {mbert[s]['macro_f1']:>12.3f} {d:>+8.3f}")


if __name__ == "__main__":
    main()
