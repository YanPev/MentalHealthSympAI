"""
Comprehensive evaluation for the BERT-vs-MentalBERT comparison on the unified
BM25 utterance baseline (data/processed/phq8_item_dataset_full_bm25.csv),
matching the context-window CV setup (participant-grouped 5-fold, shared folds).

For each OOF run it computes the full metric suite requested in the handoff:
  item level   : accuracy, macro-F1, MAE, QWK, per-class F1 (0-3), 4x4 confusion,
                 top-2 accuracy, off-by-one rate, item-binary (0-1 vs 2-3)
  per item     : accuracy/macro-F1/MAE/QWK for each PHQ-8 item
  PHQ-8 total  : reconstructed total (sum of 8 item preds/participant),
                 total MAE, total QWK, Pearson r, Spearman rho
  thresholds   : total >= {5,10,15,20} -> sensitivity, specificity, balanced acc,
                 precision, recall, F1, 2x2 confusion
  severity band: 0-4 / 5-9 / 10-14 / 15-19 / 20-24 -> band accuracy, band QWK, 5x5 confusion

Writes one comprehensive_metrics_<tag>.json per run and a comparison summary.

    python -m src.evaluation.model_comparison_eval
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
    mean_absolute_error, precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "outputs" / "cv"
OUT = ROOT / "outputs"
LABELS = [0, 1, 2, 3]
THRESHOLDS = [5, 10, 15, 20]
BAND_EDGES = [-1, 4, 9, 14, 19, 24]   # PHQ severity bands -> 0..4
BAND_NAMES = ["0-4", "5-9", "10-14", "15-19", "20-24"]

# run_id -> (label, model, loss)
RUNS = [
    ("bm25base_bert_ceplain",  "BERT-base · plain CE",     "bert-base-uncased",            "plain CE"),
    ("bm25base_bert_ce",       "BERT-base · weighted CE",  "bert-base-uncased",            "weighted CE"),
    ("bm25base_bert_corn",     "BERT-base · CORN",         "bert-base-uncased",            "CORN"),
    ("bm25base_mbert_ceplain", "MentalBERT · plain CE",    "mental/mental-bert-base-uncased", "plain CE"),
    ("bm25base_mbert_ce",      "MentalBERT · weighted CE", "mental/mental-bert-base-uncased", "weighted CE"),
    ("bm25base_mbert_corn",    "MentalBERT · CORN",        "mental/mental-bert-base-uncased", "CORN"),
]


def binary_metrics(y_true_bin, y_pred_bin):
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0           # recall / sensitivity
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return {
        "sensitivity": round(float(sens), 4), "specificity": round(float(spec), 4),
        "precision": round(float(prec), 4), "recall": round(float(sens), 4),
        "f1": round(float(f1), 4), "balanced_accuracy": round(float((sens + spec) / 2), 4),
        "confusion_2x2": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_pos": int((y_true_bin == 1).sum()), "n_neg": int((y_true_bin == 0).sum()),
    }


def comprehensive(df):
    y, p = df["label"].to_numpy(), df["prediction"].to_numpy()
    prob_cols = [f"prob_{c}" for c in LABELS]

    # ---- item level ----
    _, _, f1c, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    item = {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "macro_f1": round(float(f1_score(y, p, average="macro", labels=LABELS, zero_division=0)), 4),
        "mae": round(float(mean_absolute_error(y, p)), 4),
        "qwk": round(float(cohen_kappa_score(y, p, weights="quadratic", labels=LABELS)), 4),
        "f1_per_class": {str(c): round(float(f1c[c]), 4) for c in LABELS},
        "confusion_4x4": confusion_matrix(y, p, labels=LABELS).tolist(),
    }
    if all(c in df.columns for c in prob_cols):
        P = df[prob_cols].to_numpy()
        order = np.argsort(-P, axis=1)
        top1, top2 = order[:, 0], order[:, 1]
        err = top1 != y
        item["top1_accuracy"] = round(float((top1 == y).mean()), 4)
        item["top2_accuracy"] = round(float(((top1 == y) | (top2 == y)).mean()), 4)
        item["off_by_one_rate"] = round(float((err & (np.abs(top1 - y) == 1)).sum() / max(err.sum(), 1)), 4)

    # ---- item binary severity (0-1 vs 2-3) ----
    item_binary = binary_metrics((y >= 2).astype(int), (p >= 2).astype(int))

    # ---- per item ----
    per_item = []
    for (iid, iname), g in df.groupby(["item_id", "item_name"]):
        yy, pp = g["label"].to_numpy(), g["prediction"].to_numpy()
        per_item.append({
            "item_id": int(iid), "item_name": iname,
            "accuracy": round(float(accuracy_score(yy, pp)), 4),
            "macro_f1": round(float(f1_score(yy, pp, average="macro", labels=LABELS, zero_division=0)), 4),
            "mae": round(float(mean_absolute_error(yy, pp)), 4),
            "qwk": round(float(cohen_kappa_score(yy, pp, weights="quadratic", labels=LABELS)), 4),
        })

    # ---- PHQ-8 total reconstruction ----
    g = df.groupby("participant_id")
    assert (g.size() == 8).all(), "every participant must have all 8 items"
    true_tot = g["label"].sum().to_numpy()
    pred_tot = g["prediction"].sum().to_numpy()
    total = {
        "n_participants": int(len(true_tot)),
        "total_mae": round(float(mean_absolute_error(true_tot, pred_tot)), 4),
        "total_qwk": round(float(cohen_kappa_score(true_tot.astype(int), pred_tot.astype(int), weights="quadratic")), 4),
        "pearson_r": round(float(pearsonr(true_tot, pred_tot)[0]), 4),
        "spearman_rho": round(float(spearmanr(true_tot, pred_tot)[0]), 4),
        "true_mean": round(float(true_tot.mean()), 2), "pred_mean": round(float(pred_tot.mean()), 2),
    }

    # ---- thresholds ----
    thresholds = {}
    for thr in THRESHOLDS:
        thresholds[f">={thr}"] = binary_metrics((true_tot >= thr).astype(int), (pred_tot >= thr).astype(int))

    # ---- severity bands ----
    tb = np.digitize(true_tot, BAND_EDGES[1:-1])   # 0..4
    pb = np.digitize(pred_tot, BAND_EDGES[1:-1])
    bands = {
        "band_accuracy": round(float((tb == pb).mean()), 4),
        "band_qwk": round(float(cohen_kappa_score(tb, pb, weights="quadratic", labels=[0, 1, 2, 3, 4])), 4),
        "confusion_5x5": confusion_matrix(tb, pb, labels=[0, 1, 2, 3, 4]).tolist(),
        "band_names": BAND_NAMES,
        "true_band_counts": {BAND_NAMES[i]: int((tb == i).sum()) for i in range(5)},
    }

    return {"item": item, "item_binary_0v1_vs_2v3": item_binary,
            "per_item": per_item, "total": total,
            "total_thresholds": thresholds, "severity_bands": bands}


def _item_level(y, p):
    """Item-level accuracy / macro-F1 / MAE / QWK / per-class F1 on a subset
    (no total-reconstruction, so it is safe on an incomplete participant set)."""
    if len(y) == 0:
        return {"n": 0}
    _, _, f1c, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "macro_f1": round(float(f1_score(y, p, average="macro", labels=LABELS, zero_division=0)), 4),
        "mae": round(float(mean_absolute_error(y, p)), 4),
        "qwk": round(float(cohen_kappa_score(y, p, weights="quadratic", labels=LABELS)), 4),
        "f1_per_class": {str(c): round(float(f1c[c]), 4) for c in LABELS},
        "confusion_4x4": confusion_matrix(y, p, labels=LABELS).tolist(),
    }


def comprehensive_with_abstention(df):
    """Evaluation for an abstaining predictor (prediction may be NaN where the
    model declined to score, e.g. LLM-only with judge status == none).

    Leaves comprehensive() untouched. Reports:
      * coverage      : overall + per-item fraction of scored (non-NaN) rows.
      * covered_item  : item-level metrics on the covered subset only (the
                        model's quality where it chose to answer).
      * headline      : full comprehensive() with abstained predictions imputed
                        to 0 (the conservative "no evidence -> score 0" total).
      * prorated_total: sensitivity variant -- per participant, mean of the
                        observed (covered) item predictions x 8, vs the full
                        gold total.
    Gold labels are always complete, so `label` is never NaN.
    """
    df = df.copy()
    y_all = df["label"].to_numpy()
    p_raw = pd.to_numeric(df["prediction"], errors="coerce")
    covered = p_raw.notna().to_numpy()

    # ---- coverage ----
    per_item_cov = {}
    for (iid, iname), g in df.assign(_cov=covered).groupby(["item_id", "item_name"]):
        per_item_cov[iname] = round(float(g["_cov"].mean()), 4)
    coverage = {
        "coverage_rate": round(float(covered.mean()), 4),
        "n_covered": int(covered.sum()),
        "n_abstained": int((~covered).sum()),
        "n_total": int(len(covered)),
        "per_item_coverage": per_item_cov,
    }

    # ---- covered-subset item metrics ----
    covered_item = _item_level(y_all[covered], p_raw.to_numpy()[covered].astype(int))
    covered_per_item = []
    sub = df[covered].assign(_p=p_raw[covered].astype(int))
    for (iid, iname), g in sub.groupby(["item_id", "item_name"]):
        covered_per_item.append({
            "item_id": int(iid), "item_name": iname,
            **_item_level(g["label"].to_numpy(), g["_p"].to_numpy()),
        })

    # ---- headline: 0-impute abstentions, delegate to comprehensive() ----
    imputed = df.copy()
    imputed["prediction"] = p_raw.fillna(0).astype(int).to_numpy()
    headline = comprehensive(imputed)

    # ---- prorated total (sensitivity): mean of observed items x 8 ----
    g = df.assign(_p=p_raw, _cov=covered).groupby("participant_id")
    assert (g.size() == 8).all(), "every participant must have all 8 items"
    true_tot = g["label"].sum().to_numpy().astype(float)
    obs_mean = g[["_p", "_cov"]].apply(
        lambda x: x.loc[x["_cov"], "_p"].mean() if x["_cov"].any() else 0.0)
    pred_tot_prorated = (obs_mean.to_numpy() * 8.0)
    n_obs = g["_cov"].sum().to_numpy()
    prorated_total = {
        "n_participants": int(len(true_tot)),
        "mean_items_observed": round(float(n_obs.mean()), 2),
        "total_mae": round(float(mean_absolute_error(true_tot, pred_tot_prorated)), 4),
        "pearson_r": round(float(pearsonr(true_tot, pred_tot_prorated)[0]), 4)
                     if np.ptp(pred_tot_prorated) > 0 else None,
        "spearman_rho": round(float(spearmanr(true_tot, pred_tot_prorated)[0]), 4)
                        if np.ptp(pred_tot_prorated) > 0 else None,
        "true_mean": round(float(true_tot.mean()), 2),
        "pred_mean": round(float(pred_tot_prorated.mean()), 2),
    }

    return {
        "coverage": coverage,
        "covered_item": covered_item,
        "covered_per_item": covered_per_item,
        "headline_0imputed": headline,
        "prorated_total": prorated_total,
    }


def main():
    summary = []
    for tag, label, model, loss in RUNS:
        f = CV / f"oof_predictions_{tag}.csv"
        if not f.exists():
            print(f"  pending: {label} ({tag})")
            continue
        res = comprehensive(pd.read_csv(f))
        res["run_id"] = f"cv/{tag}"
        res["label"] = label
        res["model_name"] = model
        res["loss_type"] = loss
        (CV / f"comprehensive_metrics_{tag}.json").write_text(json.dumps(res, indent=2))
        summary.append((label, res))

    # comparison print
    print("=" * 110)
    print("UNIFIED BM25-utterance baseline — BERT vs MentalBERT (5-fold participant CV, shared folds, OOF)")
    print("=" * 110)
    h = (f"{'config':26}{'macroF1':>8}{'acc':>6}{'MAE':>6}{'QWK':>6}{'F1c2':>6}{'F1c3':>6}"
         f"{'top2':>6}{'sevF1':>6}{'totMAE':>7}{'totQWK':>7}{'b10Acc':>7}{'sens10':>7}")
    print(h); print("-" * len(h))
    for label, r in summary:
        i = r["item"]; ib = r["item_binary_0v1_vs_2v3"]; t = r["total"]; th = r["total_thresholds"][">=10"]
        print(f"{label:26}{i['macro_f1']:>8.3f}{i['accuracy']:>6.3f}{i['mae']:>6.3f}{i['qwk']:>6.3f}"
              f"{i['f1_per_class']['2']:>6.3f}{i['f1_per_class']['3']:>6.3f}{i.get('top2_accuracy',float('nan')):>6.3f}"
              f"{ib['f1']:>6.3f}{t['total_mae']:>7.2f}{t['total_qwk']:>7.3f}"
              f"{th['balanced_accuracy']:>7.3f}{th['sensitivity']:>7.3f}")

    # paired BERT vs MentalBERT (per loss) on macro-F1
    print("\nPaired BERT vs MentalBERT (same folds), macro-F1:")
    by = {r["loss_type"]: {} for _, r in summary}
    for _, r in summary:
        by[r["loss_type"]][r["model_name"]] = r["item"]["macro_f1"]
    for loss, d in by.items():
        b = d.get("bert-base-uncased"); m = d.get("mental/mental-bert-base-uncased")
        if b is not None and m is not None:
            print(f"  {loss:14} BERT {b:.3f}  vs  MentalBERT {m:.3f}   Δ(M-B) {m-b:+.3f}")

    (OUT / "model_comparison_results.json").write_text(
        json.dumps([{"label": l, **r} for l, r in summary], indent=2))
    print(f"\nWrote {OUT/'model_comparison_results.json'} ({len(summary)} runs)")


if __name__ == "__main__":
    main()
