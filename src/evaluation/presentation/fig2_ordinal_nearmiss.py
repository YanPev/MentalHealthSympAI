"""Fig 2 - Ordinal near-miss: "almost always almost right".

A 4-class exact-accuracy of ~0.42 undersells an ordinally coherent model: its
errors stay adjacent. Left panel = row-normalised confusion matrix (mass hugs
the diagonal; the minimal<->severe corners are near-empty). Right panel =
error-structure breakdown (exact / off-by-one / far-off), contrasting exact
accuracy with within-one accuracy.

Headline model: MentalBERT + CORN + Hybrid-W3 (outputs/cv/oof_predictions_ctxm_corn_hybw3.csv).

    python -m src.evaluation.presentation.fig2_ordinal_nearmiss
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.evaluation.build_cot_report import metric_block, LABELS
from src.evaluation.presentation import _style as S

OOF = S.ROOT / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"


def error_structure(y, p):
    err = np.abs(np.asarray(y) - np.asarray(p))
    n = len(err)
    return {
        "exact": float((err == 0).mean()),
        "adjacent": float((err == 1).mean()),
        "far": float((err >= 2).mean()),
        "within_one": float((err <= 1).mean()),
        "off_by_one_among_errors": float((err == 1).sum() / max((err >= 1).sum(), 1)),
        "n": int(n),
    }


def main():
    S.apply_rc()
    df = pd.read_csv(OOF)
    y, p = df["label"].to_numpy(), df["prediction"].to_numpy()

    cm = confusion_matrix(y, p, labels=LABELS)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    overall = error_structure(y, p)
    # per-fold for error bars on the structure bars
    fold_struct = [error_structure(g["label"], g["prediction"])
                   for _, g in df.groupby("fold")]
    struct_keys = ["exact", "adjacent", "far"]
    means = [overall[k] for k in struct_keys]
    stds = [float(np.std([fs[k] for fs in fold_struct])) for k in struct_keys]

    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.4),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- (a) row-normalised confusion matrix --------------------------------
    axL.grid(False)
    im = axL.imshow(cm_norm, cmap=S.SEQ_BLUES, vmin=0, vmax=1, aspect="auto")
    axL.set_xticks(range(4)); axL.set_yticks(range(4))
    axL.set_xticklabels(S.CLASS_SHORT, fontsize=10.5)
    axL.set_yticklabels(S.CLASS_SHORT, fontsize=10.5)
    axL.set_xlabel("Predicted severity"); axL.set_ylabel("True severity")
    axL.set_title("Errors hug the diagonal", color=S.ACCENT)
    for i in range(4):
        for j in range(4):
            frac, cnt = cm_norm[i, j], cm[i, j]
            txt = f"{frac*100:.0f}%\n({cnt})"
            color = "white" if frac > 0.5 else S.ACCENT
            weight = "bold" if i == j else "normal"
            axL.text(j, i, txt, ha="center", va="center",
                     fontsize=9.5, color=color, fontweight=weight)
    # catastrophic over-calls (minimal -> severe) are ~0; flag honestly. The
    # opposite corner (severe -> minimal) is a real weakness, flagged separately.
    axL.add_patch(plt.Rectangle((3 - 0.5, 0 - 0.5), 1, 1, fill=False,
                                edgecolor=S.GOOD, lw=2.2))
    axL.text(3, -0.72, f"over-calls ≈ 0\n(minimal→severe {cm_norm[0,3]*100:.0f}%)",
             ha="center", fontsize=8.3, color=S.GOOD)
    axL.add_patch(plt.Rectangle((0 - 0.5, 3 - 0.5), 1, 1, fill=False,
                                edgecolor=S.WARN, lw=2.2))
    axL.text(0, -0.72, f"severe under-call {cm_norm[3,0]*100:.0f}%\n(class imbalance)",
             ha="center", fontsize=8.3, color=S.WARN)
    cbar = fig.colorbar(im, ax=axL, fraction=0.046, pad=0.04)
    cbar.set_label("row-normalised rate", fontsize=9)

    # ---- (b) error-structure bars -------------------------------------------
    colors = [S.GOOD, S.WARN, S.BAD]
    labels = ["Exact\n(correct)", "Off-by-one\n(adjacent class)", "Far-off\n(≥2 classes)"]
    x = np.arange(3)
    axR.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="white",
            linewidth=1.2, error_kw=dict(ecolor=S.MUTED, lw=1.2))
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=11)
    axR.set_ylabel("share of all predictions")
    axR.set_ylim(0, 0.72)
    axR.set_title("Most misses are one class away", color=S.ACCENT)
    for xi, v, e in zip(x, means, stds):
        axR.text(xi, v + e + 0.015, f"{v*100:.1f}%", ha="center",
                 fontweight="bold", fontsize=11)

    # the reframe callout
    axR.text(0.5, 0.63,
             f"exact acc {overall['exact']*100:.0f}%   →   within-one {overall['within_one']*100:.0f}%",
             ha="center", fontsize=11.5, color=S.ACCENT, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f0fdf4", ec=S.GOOD))
    axR.text(1.0, 0.50,
             f"{overall['off_by_one_among_errors']*100:.0f}% of all errors\nare just ±1 class",
             ha="center", fontsize=10, color=S.MUTED)

    fig.suptitle("The model is almost always almost right — a near-miss view of ordinal severity",
                 fontsize=15, fontweight="bold", color=S.ACCENT, y=1.01)
    S.save(fig, "fig2_ordinal_nearmiss")

    out = {
        "model": "MentalBERT+CORN+Hybrid-W3 (OOF, 5-fold)",
        "confusion_matrix_counts": cm.tolist(),
        "confusion_matrix_row_normalised": cm_norm.round(4).tolist(),
        "error_structure": overall,
        "error_structure_fold_std": dict(zip(struct_keys, stds)),
    }
    cp = S.companion_path("fig2_ordinal_nearmiss", "json")
    cp.write_text(json.dumps(out, indent=2))
    print(f"  wrote {cp}")
    print(f"  exact={overall['exact']:.3f} within1={overall['within_one']:.3f} "
          f"off-by-one-among-errors={overall['off_by_one_among_errors']:.3f}")


if __name__ == "__main__":
    main()
