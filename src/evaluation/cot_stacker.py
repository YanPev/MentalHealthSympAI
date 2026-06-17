"""
Learned stacking: a nested-CV meta-classifier over encoder + CoT signals, to see
if a trained combiner beats the hand-built severity-routing rule (macro-F1 0.377,
QWK 0.379).

Features per item (paired OOF):
  encoder CORN probs (4) + CoT self-consistency soft probs (4)
  + W5-CoT one-hot prediction (4) + item_id one-hot (8)   = 20 dims
Target: severity 0-3.

Leakage-free: for each held-out fold we train the meta-classifier on the other
four folds only (the base models' OOF preds are themselves already out-of-fold).
No GPU.

    python -m src.evaluation.cot_stacker
"""

from pathlib import Path
import glob
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.evaluation.build_cot_report import metric_block, LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENC_OOF = PROJECT_ROOT / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"
SC_DIR = PROJECT_ROOT / "outputs" / "cot" / "folds_sc5"
W5_DIR = PROJECT_ROOT / "outputs" / "cot" / "folds_w5"
OUT_JSON = PROJECT_ROOT / "outputs" / "cot" / "cot_stacker_metrics.json"
KEY = ["participant_id", "item_id"]
PROBS = [f"prob_{c}" for c in LABELS]


def load_dir(d):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(Path(d) / "*.csv")))],
                   ignore_index=True)
    df["participant_id"] = df["participant_id"].astype(str)
    return df


def main():
    enc = pd.read_csv(ENC_OOF)
    enc["participant_id"] = enc["participant_id"].astype(str)
    df = enc[KEY + ["label", "fold", "prediction"]].rename(columns={"prediction": "enc_pred"})
    for c in LABELS:
        df[f"enc_{c}"] = enc[f"prob_{c}"]

    sc = load_dir(SC_DIR)
    for c in LABELS:
        df = df.merge(sc[KEY + [f"prob_{c}"]].rename(columns={f"prob_{c}": f"sc_{c}"}), on=KEY)
    w5 = load_dir(W5_DIR)[KEY + ["prediction"]].rename(columns={"prediction": "w5_pred"})
    df = df.merge(w5, on=KEY)

    # ---- feature matrix ----
    feat_cols = [f"enc_{c}" for c in LABELS] + [f"sc_{c}" for c in LABELS]
    X = df[feat_cols].to_numpy(dtype=float)
    w5_oh = np.eye(len(LABELS))[df["w5_pred"].to_numpy()]
    item_oh = pd.get_dummies(df["item_id"]).to_numpy(dtype=float)
    X = np.concatenate([X, w5_oh, item_oh], axis=1)
    y = df["label"].to_numpy()
    folds = df["fold"].to_numpy()

    # ---- nested-CV stacking ----
    oof = np.empty(len(df), dtype=int)
    for f in sorted(set(folds)):
        tr, te = folds != f, folds == f
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict(X[te])

    # ---- references ----
    enc_m = metric_block(y, df["enc_pred"].to_numpy())
    w5c = df["w5_pred"].to_numpy()
    routed = np.where(w5c >= 2, w5c, df["enc_pred"].to_numpy())  # the prior best
    refs = {"encoder (CORN)": metric_block(y, df["enc_pred"].to_numpy()),
            "severity-routed (W5) [prior best]": metric_block(y, routed),
            "stacker (LogReg, nested-CV)": metric_block(y, oof)}

    OUT_JSON.write_text(json.dumps({"n": int(len(df)), "methods": refs}, indent=2))
    print(f"Learned stacking (paired OOF n={len(df)}):")
    print(f"{'method':36s} {'F1':>6} {'QWK':>6} {'MAE':>6} {'acc':>6} {'F1c2':>6} {'F1c3':>6}")
    for k, m in refs.items():
        print(f"{k:36s} {m['macro_f1']:6.3f} {m['qwk']:6.3f} {m['mae']:6.3f} "
              f"{m['accuracy']:6.3f} {m['f1_per_class'][2]:6.3f} {m['f1_per_class'][3]:6.3f}")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
