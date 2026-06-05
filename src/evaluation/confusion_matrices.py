"""
Print confusion matrices for PHQ-8 item-severity prediction files.

Each predictions CSV (written by ``train_transformer_classifier``) has
``true_label`` and ``prediction`` columns over the 4 severity classes (0..3).
This reports, per file, the 4x4 confusion matrix (rows = true, cols = predicted)
plus per-class precision/recall/F1 and the headline metrics.

Usage
-----
python -m src.evaluation.confusion_matrices outputs/predictions_*_test.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)

LABELS = [0, 1, 2, 3]


def report(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    y_true = df["true_label"].to_numpy()
    y_pred = df["prediction"].to_numpy()

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mae = mean_absolute_error(y_true, y_pred)

    print("=" * 60)
    print(csv_path.name)
    print(f"  n={len(df)}  acc={acc:.3f}  macroF1={macro_f1:.3f}  MAE={mae:.3f}")
    print()
    print("  Confusion matrix (rows=true, cols=pred)")
    header = "        " + "".join(f"pred{c:>4d}" for c in LABELS) + "   | total"
    print(header)
    for i, c in enumerate(LABELS):
        cells = "".join(f"{cm[i, j]:>8d}" for j in range(len(LABELS)))
        print(f"  true{c:<2d}{cells}   | {cm[i].sum():>5d}")
    totals = "".join(f"{cm[:, j].sum():>8d}" for j in range(len(LABELS)))
    print(f"  {'col tot':<6}{totals}   | {cm.sum():>5d}")
    print()
    print("  Per-class:")
    print(f"    {'class':>5} {'precision':>10} {'recall':>8} {'f1':>6} {'support':>8}")
    for i, c in enumerate(LABELS):
        print(f"    {c:>5} {prec[i]:>10.3f} {rec[i]:>8.3f} {f1[i]:>6.3f} {int(sup[i]):>8}")
    print()


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        print("usage: python -m src.evaluation.confusion_matrices <pred1.csv> [pred2.csv ...]")
        sys.exit(1)
    for p in paths:
        report(Path(p))


if __name__ == "__main__":
    main()
