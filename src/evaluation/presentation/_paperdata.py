"""Shared loaders + metrics for the paper figure set (fig_* figures).

Every config is a set of out-of-fold predictions over the same 1752 item-examples.
This module standardises loading (encoder OOF, CoT fold dirs, pooled SC, cascade)
and metric computation so the 9 data figures agree number-for-number.
"""

import glob

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             mean_absolute_error)

from src.evaluation.presentation import _style as S

CV = S.ROOT / "outputs" / "cv"
COT = S.ROOT / "outputs" / "cot"
KEY = ["participant_id", "item_id"]
LAB = [0, 1, 2, 3]
PROBS = [f"prob_{c}" for c in LAB]
ENC_W3 = CV / "oof_predictions_ctxm_corn_hybw3.csv"
ENC_W5 = CV / "oof_predictions_ctxm_corn_hybw5.csv"


def load_enc(path):
    d = pd.read_csv(path); d["participant_id"] = d["participant_id"].astype(str)
    return d[KEY + ["item_name", "label", "fold", "prediction"] + PROBS]


def load_dirs(*names):
    """Concat one-or-more CoT fold dirs; if several, POOL their prob vectors
    (averaged) and argmax -> a lower-variance vote. Returns one row per item."""
    frames = []
    for nm in names:
        files = sorted(glob.glob(str(COT / nm / "*.csv")))
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df["participant_id"] = df["participant_id"].astype(str)
        P = df[PROBS].to_numpy(float); P = P / P.sum(1, keepdims=True).clip(min=1e-9)
        df[PROBS] = P
        frames.append(df[KEY + ["item_name", "label"] + PROBS])
    base = frames[0][KEY + ["item_name", "label"]].copy()
    Psum = np.zeros((len(base), len(LAB)))
    for f in frames:
        g = base.merge(f, on=KEY)
        Psum += g[PROBS].to_numpy()
    Psum /= len(frames)
    for c in LAB:
        base[f"prob_{c}"] = Psum[:, c]
    base["prediction"] = Psum.argmax(1)
    return base


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    err = np.abs(y - p)
    fpc = f1_score(y, p, average=None, labels=LAB, zero_division=0)
    tot = len(y)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", labels=LAB, zero_division=0)),
        "qwk": float(cohen_kappa_score(y, p, weights="quadratic", labels=LAB)),
        "mae": float(mean_absolute_error(y, p)),
        "far_off": float((err >= 2).mean()),
        "within_1": float((err <= 1).mean()),
        "exact": float((err == 0).mean()),
        "f1_per_class": [float(x) for x in fpc],
        "over_call": float((p > y).mean()),
        "under_call": float((p < y).mean()),
        # severe-specific
        "severe_undercall": int(((y == 3) & (p < 3)).sum()),
        "severe_overcall": int(((p == 3) & (y < 3)).sum()),
        "n_true_severe": int((y == 3).sum()),
    }


def align(*dfs, names=None):
    """Inner-merge several prediction frames on KEY; returns merged df with
    label + per-config prediction columns named pred_<name>."""
    names = names or [f"m{i}" for i in range(len(dfs))]
    out = dfs[0][KEY + ["label", "item_name"]].copy()
    for nm, d in zip(names, dfs):
        out = out.merge(d[KEY + ["prediction"]].rename(columns={"prediction": f"pred_{nm}"}),
                        on=KEY)
    return out
