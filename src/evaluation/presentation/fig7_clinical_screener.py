"""Fig 7 - The model as a depression screener (PHQ-8 total >= 10).

We reconstruct each participant's PHQ-8 total from the 8 item predictions and ask
the deployment question: how well does it flag clinically significant depression
(total >= 10)? Using the continuous expected-severity sum as a risk score lets us
move the decision threshold instead of taking the default argmax-sum, trading
specificity for the sensitivity a screener actually needs.

Left  : ROC of the continuous risk score (AUC), with default vs tuned operating points.
Right : sensitivity & specificity vs decision threshold, with the high-sensitivity
        operating point highlighted.

Encoder: MentalBERT+CORN+Hybrid-W3 OOF (outputs/cv/oof_predictions_ctxm_corn_hybw3.csv).

    python -m src.evaluation.presentation.fig7_clinical_screener
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from src.evaluation.presentation import _style as S

OOF = S.ROOT / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"
PROBS = [f"prob_{c}" for c in S.LABELS]
THR = 10           # PHQ-8 >= 10 => clinically significant depression
TARGET_SENS = 0.85  # screening priority: catch true cases


def main():
    S.apply_rc()
    df = pd.read_csv(OOF)
    P = df[PROBS].to_numpy(float)
    P = P / P.sum(1, keepdims=True).clip(min=1e-9)
    df["ev"] = P @ np.array(S.LABELS, float)   # expected severity per item

    g = df.groupby("participant_id")
    true_tot = g["label"].sum()
    pred_argmax_tot = g["prediction"].sum()
    risk = g["ev"].sum()                        # continuous risk score
    part = pd.DataFrame({"true_tot": true_tot, "pred_argmax": pred_argmax_tot,
                         "risk": risk}).reset_index()
    y = (part["true_tot"] >= THR).astype(int).to_numpy()
    s = part["risk"].to_numpy()

    auc = roc_auc_score(y, s)
    fpr, tpr, thr = roc_curve(y, s)

    def oppoint(yt, yp):
        tp = int(((yp == 1) & (yt == 1)).sum()); fn = int(((yp == 0) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        return sens, spec

    # default operating point: argmax-sum >= 10
    sens_def, spec_def = oppoint(y, (part["pred_argmax"] >= THR).astype(int).to_numpy())

    # tuned threshold on the continuous risk score for TARGET_SENS
    order = np.argsort(-s)
    cand = np.unique(s)
    best_t, best_spec = cand.min(), -1
    for t in cand:
        sens, spec = oppoint(y, (s >= t).astype(int))
        if sens >= TARGET_SENS and spec > best_spec:
            best_spec, best_t = spec, t
    sens_tuned, spec_tuned = oppoint(y, (s >= best_t).astype(int))

    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # ---- (a) ROC ------------------------------------------------------------
    axL.plot(fpr, tpr, color=S.ENS_COLOR, lw=2.4, label=f"risk score (AUC {auc:.3f})")
    axL.plot([0, 1], [0, 1], ls="--", color=S.MUTED, lw=1.2)
    axL.scatter([1 - spec_def], [sens_def], s=120, color=S.MUTED, zorder=5,
                edgecolors="white", lw=1.5,
                label=f"default argmax≥10 (sens {sens_def:.2f}, spec {spec_def:.2f})")
    axL.scatter([1 - spec_tuned], [sens_tuned], s=140, color=S.BAD, zorder=6,
                edgecolors="white", lw=1.5,
                label=f"tuned for screening (sens {sens_tuned:.2f}, spec {spec_tuned:.2f})")
    axL.set_xlabel("1 − specificity (false-positive rate)")
    axL.set_ylabel("sensitivity (true-positive rate)")
    axL.set_xlim(0, 1); axL.set_ylim(0, 1.02)
    axL.set_title("Screening for PHQ-8 ≥ 10: the threshold is a choice", color=S.ACCENT)
    axL.legend(fontsize=9, loc="lower right")

    # ---- (b) sensitivity / specificity vs threshold -------------------------
    ts = np.linspace(s.min(), s.max(), 200)
    sens_curve = [oppoint(y, (s >= t).astype(int))[0] for t in ts]
    spec_curve = [oppoint(y, (s >= t).astype(int))[1] for t in ts]
    axR.plot(ts, sens_curve, color=S.BAD, lw=2.2, label="sensitivity")
    axR.plot(ts, spec_curve, color=S.GOOD, lw=2.2, label="specificity")
    axR.axvline(best_t, ls="--", color=S.ACCENT, lw=1.4)
    axR.axhline(TARGET_SENS, ls=":", color=S.MUTED, lw=1)
    axR.text(best_t, 0.02, f"tuned\nthr={best_t:.1f}", ha="center", fontsize=9,
             color=S.ACCENT)
    axR.text(ts.min() + 0.3, TARGET_SENS + 0.02, f"target sensitivity {TARGET_SENS:.0%}",
             fontsize=8.5, color=S.MUTED)
    axR.set_xlabel("decision threshold on reconstructed PHQ-8 risk score")
    axR.set_ylabel("rate"); axR.set_ylim(0, 1.02)
    axR.set_title("Moving the threshold trades specificity for sensitivity", color=S.ACCENT)
    axR.legend(fontsize=9.5, loc="lower left")

    fig.suptitle("Reframed as a screener: tuning the decision threshold recovers clinically useful sensitivity",
                 fontsize=13.5, fontweight="bold", color=S.ACCENT, y=1.02)
    S.save(fig, "fig7_clinical_screener")

    out = {
        "n_participants": int(len(part)), "n_positive": int(y.sum()),
        "threshold_phq8": THR, "auc": float(auc),
        "default_argmax": {"sensitivity": sens_def, "specificity": spec_def},
        "tuned_for_screening": {"target_sensitivity": TARGET_SENS,
                                "risk_threshold": float(best_t),
                                "sensitivity": sens_tuned, "specificity": spec_tuned},
    }
    cp = S.companion_path("fig7_clinical_screener", "json")
    cp.write_text(json.dumps(out, indent=2))
    print(f"  wrote {cp}")
    print(f"  AUC={auc:.3f}  default sens/spec={sens_def:.2f}/{spec_def:.2f}  "
          f"tuned sens/spec={sens_tuned:.2f}/{spec_tuned:.2f} @thr={best_t:.1f}")


if __name__ == "__main__":
    main()
