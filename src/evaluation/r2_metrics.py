"""
Shared PHQ-8 metric + paired-bootstrap helpers for the R2 downstream stages
(encoder / LLM / cascade). Definitions match the frozen pipeline exactly
(sklearn; severe = class 3; far-off = |y-p|>=2; QWK = quadratic-weighted kappa;
participant-cluster bootstrap B=2000, seed=42, paired, percentile 95% CI).

Predictions may be NaN (abstention). Metrics computed on covered rows; coverage
reported separately by the caller.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             mean_absolute_error, precision_recall_fscore_support)

LABELS = [0, 1, 2, 3]
B_DEFAULT, SEED = 2000, 42


def item_metrics(y, p):
    """Full item-level metric dict. y, p: integer arrays (aligned, no NaN)."""
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=int)
    n = len(y)
    if n == 0:
        return {k: np.nan for k in
                ("n", "accuracy", "macro_f1", "qwk", "mae", "far_off_rate",
                 "within_one", "f1_class3", "severe_recall", "false_severe_rate")}
    _, _, f1c, _ = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    sev = y == 3
    nonsev = y < 3
    err = np.abs(y - p)
    return {
        "n": int(n),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y, p, weights="quadratic", labels=LABELS))
               if len(set(y.tolist() + p.tolist())) > 1 else 0.0,
        "mae": float(mean_absolute_error(y, p)),
        "far_off_rate": float((err >= 2).mean()),
        "within_one": float((err <= 1).mean()),
        "f1_class3": float(f1c[3]),
        "severe_recall": float((p[sev] == 3).mean()) if sev.any() else np.nan,
        "false_severe_rate": float((p[nonsev] == 3).mean()) if nonsev.any() else np.nan,
    }


def by_item(df, label_col="label", pred_col="prediction"):
    """Per-item metrics table for a prediction frame (covered rows only)."""
    rows = []
    d = df.dropna(subset=[pred_col])
    for (iid, name), g in d.groupby(["item_id", "item_name"]):
        m = item_metrics(g[label_col], g[pred_col])
        rows.append({"item_id": int(iid), "item_name": name, **m})
    return pd.DataFrame(rows).sort_values("item_id")


def overall(df, label_col="label", pred_col="prediction"):
    d = df.dropna(subset=[pred_col])
    m = item_metrics(d[label_col], d[pred_col])
    m["coverage"] = float(df[pred_col].notna().mean())
    m["n_total"] = int(len(df))
    return m


def paired_bootstrap(df, pred_a, pred_b, label_col="label", metrics=None,
                     B=B_DEFAULT, seed=SEED):
    """Paired participant-cluster bootstrap of metric(pred_a) - metric(pred_b).

    df must have participant_id, item_id, label_col, and the two prediction
    columns (aligned rows). Rows where either prediction is NaN are dropped
    pairwise. Returns {metric: {"delta": point, "ci95": [lo, hi], "excludes_0": bool}}.
    For mae/far_off_rate/false_severe_rate a NEGATIVE delta is an improvement.
    """
    metrics = metrics or ["macro_f1", "qwk", "mae", "far_off_rate",
                          "f1_class3", "severe_recall", "false_severe_rate"]
    d = df.dropna(subset=[pred_a, pred_b]).copy()
    y = d[label_col].to_numpy(int)
    pa = d[pred_a].to_numpy(int); pb = d[pred_b].to_numpy(int)
    pid = d["participant_id"].to_numpy()
    parts = np.unique(pid)
    rows_by = {q: np.where(pid == q)[0] for q in parts}
    rng = np.random.default_rng(seed)

    def scal(yy, pp, k):
        return item_metrics(yy, pp)[k]

    point = {k: scal(y, pa, k) - scal(y, pb, k) for k in metrics}
    boot = {k: [] for k in metrics}
    for _ in range(B):
        samp = rng.choice(parts, len(parts), replace=True)
        idx = np.concatenate([rows_by[q] for q in samp])
        yy, aa, bb = y[idx], pa[idx], pb[idx]
        for k in metrics:
            va, vb = scal(yy, aa, k), scal(yy, bb, k)
            boot[k].append((va - vb) if not (np.isnan(va) or np.isnan(vb)) else np.nan)
    out = {}
    for k in metrics:
        arr = np.array([v for v in boot[k] if not np.isnan(v)])
        lo, hi = (np.percentile(arr, 2.5), np.percentile(arr, 97.5)) if len(arr) else (np.nan, np.nan)
        out[k] = {"delta": round(float(point[k]), 4),
                  "ci95": [round(float(lo), 4), round(float(hi), 4)],
                  "excludes_0": bool((lo > 0 or hi < 0)) if len(arr) else False}
    return out


def align_oof(*frames, label_col="label"):
    """Inner-join prediction frames on (participant_id, item_id); keep one label.

    Returns a single frame with pred columns renamed pred_<i> and the shared
    label/fold. Verifies label agreement.
    """
    base = frames[0][["participant_id", "item_id", "item_name", label_col]].copy()
    base["participant_id"] = base["participant_id"].astype(str)
    if "fold" in frames[0].columns:
        base["fold"] = frames[0]["fold"].values
    for i, f in enumerate(frames):
        f = f.copy(); f["participant_id"] = f["participant_id"].astype(str)
        base = base.merge(f[["participant_id", "item_id", "prediction"]]
                          .rename(columns={"prediction": f"pred_{i}"}),
                          on=["participant_id", "item_id"], how="inner")
    return base
