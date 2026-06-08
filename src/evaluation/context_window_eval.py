"""
Person B evaluation of Person A's context-window evidence variants.

Reads the cross-validation out-of-fold (OOF) predictions for each evidence
condition x loss and produces the full comparison Person A requested:

  * item-level: accuracy, macro-F1, per-class F1 (esp. 2 & 3), MAE, QWK
  * near-miss: top-1 vs top-2 accuracy, off-by-one rate
  * per-item macro-F1 / MAE (+ Appetite spotlight)
  * reconstructed PHQ-8 TOTAL score (sum of 8 item predictions/participant):
      total MAE, total QWK, and clinical threshold (total >= 10) metrics
      (sensitivity, specificity, balanced accuracy, F1)

Everything is computed from OOF predictions, so every participant contributes
exactly once and the totals are well defined.

    python -m src.evaluation.context_window_eval
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
    mean_absolute_error, precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "outputs" / "cv"
OUT = ROOT / "outputs"
LABELS = [0, 1, 2, 3]
DEP_THRESHOLD = 10  # PHQ-8 >= 10 => clinically significant depression

# (key, label, CE tag, CORN tag)
CONDITIONS = [
    ("old",    "Old BM25 utterances", "ctx_ce_old",    "ctx_corn_old"),
    ("hybw3",  "Hybrid W3",           "ctx_ce_hybw3",  "ctx_corn_hybw3"),
    ("hybw5",  "Hybrid W5",           "ctx_ce_hybw5",  "ctx_corn_hybw5"),
    ("bm25w3", "BM25 W3",             "ctx_ce_bm25w3", "ctx_corn_bm25w3"),
    ("bm25w5", "BM25 W5",             "ctx_ce_bm25w5", "ctx_corn_bm25w5"),
]


def _threshold_metrics(true_tot, pred_tot, thr=DEP_THRESHOLD):
    yt = (true_tot >= thr).astype(int)
    yp = (pred_tot >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = f1_score(yt, yp, zero_division=0)
    return {
        "sensitivity": float(sens), "specificity": float(spec),
        "balanced_accuracy": float((sens + spec) / 2), "f1": float(f1),
        "n_pos": int(yt.sum()), "n_neg": int((1 - yt).sum()),
    }


def metrics_from_oof(df):
    y, p = df["label"].to_numpy(), df["prediction"].to_numpy()
    n = len(y)
    # near-miss from probs if present
    prob_cols = [f"prob_{c}" for c in LABELS]
    near = {}
    if all(c in df.columns for c in prob_cols):
        P = df[prob_cols].to_numpy()
        order = np.argsort(-P, axis=1)
        top1, top2 = order[:, 0], order[:, 1]
        in_top2 = (top1 == y) | (top2 == y)
        err = top1 != y
        near = {
            "top1_acc": float((top1 == y).mean()),
            "top2_acc": float(in_top2.mean()),
            "off_by_one_rate": float((err & (np.abs(top1 - y) == 1)).sum() / max(err.sum(), 1)),
        }
    _, _, f1c, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)

    # reconstructed PHQ-8 total per participant
    g = df.groupby("participant_id")
    sizes = g.size()
    pred_tot = g["prediction"].sum().to_numpy()
    true_tot = g["label"].sum().to_numpy()

    item = {
        "n": n,
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y, p)),
        "qwk": float(cohen_kappa_score(y, p, weights="quadratic", labels=LABELS)),
        "f1_per_class": [float(x) for x in f1c],
        **near,
    }
    total = {
        "n_participants": int(len(true_tot)),
        "all_8_items": bool((sizes == 8).all()),
        "total_mae": float(mean_absolute_error(true_tot, pred_tot)),
        "total_qwk": float(cohen_kappa_score(
            true_tot.round().astype(int), pred_tot.round().astype(int),
            weights="quadratic")),
        "total_pearson_r": float(np.corrcoef(true_tot, pred_tot)[0, 1]),
        **{f"thr_{k}": v for k, v in _threshold_metrics(true_tot, pred_tot).items()},
    }
    return item, total


def per_item(df):
    rows = []
    for (iid, iname), gdf in df.groupby(["item_id", "item_name"]):
        y, p = gdf["label"].to_numpy(), gdf["prediction"].to_numpy()
        rows.append({"item_id": iid, "item_name": iname,
                     "macro_f1": f1_score(y, p, average="macro", labels=LABELS, zero_division=0),
                     "mae": mean_absolute_error(y, p)})
    return pd.DataFrame(rows).sort_values("item_id")


def load(tag):
    f = CV / f"oof_predictions_{tag}.csv"
    return pd.read_csv(f) if f.exists() else None


def main():
    results = {}
    print("=" * 100)
    print("CONTEXT-WINDOW EVIDENCE COMPARISON (5-fold participant CV, bert-base, OOF predictions)")
    print("=" * 100)
    hdr = (f"{'condition':22} {'loss':5} {'macroF1':>8} {'acc':>6} {'MAE':>6} {'QWK':>6} "
           f"{'F1_c2':>6} {'F1_c3':>6} {'top2':>6} {'totMAE':>7} {'totQWK':>7} {'sens':>6} {'spec':>6} {'bAcc':>6}")
    print(hdr); print("-" * len(hdr))
    for key, label, ce_tag, corn_tag in CONDITIONS:
        for loss, tag in [("CE", ce_tag), ("CORN", corn_tag)]:
            df = load(tag)
            if df is None:
                print(f"{label:22} {loss:5}  (pending)")
                continue
            item, total = metrics_from_oof(df)
            results[tag] = {"condition": label, "loss": loss, "item": item, "total": total}
            t2 = item.get("top2_acc", float("nan"))
            print(f"{label:22} {loss:5} {item['macro_f1']:>8.3f} {item['accuracy']:>6.3f} "
                  f"{item['mae']:>6.3f} {item['qwk']:>6.3f} {item['f1_per_class'][2]:>6.3f} "
                  f"{item['f1_per_class'][3]:>6.3f} {t2:>6.3f} {total['total_mae']:>7.3f} "
                  f"{total['total_qwk']:>7.3f} {total['thr_sensitivity']:>6.3f} "
                  f"{total['thr_specificity']:>6.3f} {total['thr_balanced_accuracy']:>6.3f}")

    # Per-item Appetite spotlight across conditions (CORN)
    print("\n" + "=" * 60)
    print("APPETITE per-condition (CORN): macroF1 / MAE")
    print("=" * 60)
    for key, label, ce_tag, corn_tag in CONDITIONS:
        df = load(corn_tag)
        if df is None:
            continue
        pi = per_item(df)
        ap = pi[pi.item_name == "Appetite"]
        if len(ap):
            print(f"  {label:22} F1 {ap.macro_f1.iloc[0]:.3f}  MAE {ap.mae.iloc[0]:.3f}")

    out = OUT / "context_window_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}  ({len(results)} conditions evaluated)")


if __name__ == "__main__":
    main()
