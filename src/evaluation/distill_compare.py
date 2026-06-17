"""
Distillation Phase C: did the small student retain the 7B CoT's value?

Compares, on the paired 5-fold OOF (n=1752):
  parents   : encoder (CORN), 7B CoT (greedy), distilled 1.5B student
  ensembles : severity-routed(encoder, 7B)  -- the established best
              severity-routed(encoder, distilled)  -- the *deployable* version
              prob-avg(encoder, distilled)

Key question: does severity-routed(encoder, distilled) still beat the encoder on
macro-F1 + QWK the way severity-routed(encoder, 7B) did? If the student lost the
severe-class recovery, the deployable ensemble won't reproduce the win.

    python -m src.evaluation.distill_compare
"""

from pathlib import Path
import glob
import json

import numpy as np
import pandas as pd

from src.evaluation.build_cot_report import metric_block, LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENC_OOF = PROJECT_ROOT / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"
COT7B_DIR = PROJECT_ROOT / "outputs" / "cot" / "folds"
DISTILL_DIR = PROJECT_ROOT / "outputs" / "cot" / "folds_distill"
OUT_JSON = PROJECT_ROOT / "outputs" / "cot" / "distill_compare_metrics.json"
KEY = ["participant_id", "item_id"]


def load_pred(d, col):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(Path(d) / "*.csv")))],
                   ignore_index=True)
    df["participant_id"] = df["participant_id"].astype(str)
    return df[KEY + ["prediction"]].rename(columns={"prediction": col})


def main():
    enc = pd.read_csv(ENC_OOF)
    enc["participant_id"] = enc["participant_id"].astype(str)
    df = enc[KEY + ["label", "prediction"]].rename(columns={"prediction": "enc"})
    df = df.merge(load_pred(COT7B_DIR, "cot7b"), on=KEY) \
           .merge(load_pred(DISTILL_DIR, "distill"), on=KEY)
    y = df["label"].to_numpy()

    def routed(base_col, cot_col):
        c = df[cot_col].to_numpy()
        return np.where(c >= 2, c, df[base_col].to_numpy())

    methods = {
        "encoder (CORN)": df["enc"].to_numpy(),
        "7B CoT (greedy)": df["cot7b"].to_numpy(),
        "distilled 1.5B": df["distill"].to_numpy(),
        "routed(enc, 7B)": routed("enc", "cot7b"),
        "routed(enc, distilled)": routed("enc", "distill"),
    }
    metrics = {k: metric_block(y, v) for k, v in methods.items()}
    OUT_JSON.write_text(json.dumps({"n": int(len(df)), "methods": metrics}, indent=2))

    print(f"Distillation Phase C (paired OOF n={len(df)}):")
    print(f"{'method':26s} {'F1':>6} {'QWK':>6} {'MAE':>6} {'acc':>6} {'F1c2':>6} {'F1c3':>6}")
    for k, m in metrics.items():
        print(f"{k:26s} {m['macro_f1']:6.3f} {m['qwk']:6.3f} {m['mae']:6.3f} "
              f"{m['accuracy']:6.3f} {m['f1_per_class'][2]:6.3f} {m['f1_per_class'][3]:6.3f}")

    enc_m = metrics["encoder (CORN)"]
    r7 = metrics["routed(enc, 7B)"]
    rd = metrics["routed(enc, distilled)"]
    dist = metrics["distilled 1.5B"]
    cot7 = metrics["7B CoT (greedy)"]
    print("-" * 64)
    retained_f1 = (dist["macro_f1"] - enc_m["macro_f1"]) / max(cot7["macro_f1"] - enc_m["macro_f1"], 1e-9)
    print(f"student retained {retained_f1*100:.0f}% of the 7B's macro-F1 edge over the encoder")
    print(f"deployable routed(enc,distilled): macroF1 {rd['macro_f1']:.3f} vs enc {enc_m['macro_f1']:.3f} "
          f"({'BEATS' if rd['macro_f1']>enc_m['macro_f1'] else 'below'}), "
          f"QWK {rd['qwk']:.3f} vs enc {enc_m['qwk']:.3f} "
          f"({'BEATS' if rd['qwk']>enc_m['qwk'] else 'below'})")
    print(f"vs the 7B routed reference: macroF1 {rd['macro_f1']:.3f} vs {r7['macro_f1']:.3f}, "
          f"QWK {rd['qwk']:.3f} vs {r7['qwk']:.3f}")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
