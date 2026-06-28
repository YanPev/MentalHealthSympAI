"""Fig 8 - Selective prediction: a "defer to clinician" mode.

A deployable system need not answer every item. If it abstains on its least
confident calls and routes them to a human, accuracy on the auto-decided items
should climb. We also check *which* items it defers: a trustworthy model should
defer disproportionately on the ambiguous moderate/severe cases — exactly the
ones a clinician should see.

Left  : risk-coverage curve (accuracy on retained items vs coverage), with the
        answer-everything baseline and example operating points.
Right  : at a 30% deferral budget, the true-class composition of deferred items.

Encoder: MentalBERT+CORN+Hybrid-W3 OOF; confidence = max class probability.

    python -m src.evaluation.presentation.fig8_selective_prediction
"""

import json

import numpy as np
import pandas as pd

from src.evaluation.presentation import _style as S

OOF = S.ROOT / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"
PROBS = [f"prob_{c}" for c in S.LABELS]
DEFER = 0.30  # example deferral budget for the composition panel


def main():
    S.apply_rc()
    df = pd.read_csv(OOF)
    P = df[PROBS].to_numpy(float)
    P = P / P.sum(1, keepdims=True).clip(min=1e-9)
    conf = P.max(1)
    correct = (df["prediction"].to_numpy() == df["label"].to_numpy())
    y = df["label"].to_numpy()

    order = np.argsort(-conf)          # most confident first
    correct_sorted = correct[order]
    coverages = np.linspace(0.1, 1.0, 91)
    acc_at = [correct_sorted[:max(1, int(c * len(df)))].mean() for c in coverages]
    base_acc = correct.mean()

    # operating points
    ops = {}
    for c in [1.0, 0.8, 0.6]:
        k = max(1, int(c * len(df)))
        ops[c] = float(correct_sorted[:k].mean())

    # deferred-set composition at DEFER budget
    n_def = int(DEFER * len(df))
    deferred_idx = order[-n_def:]
    retained_idx = order[:-n_def]
    def class_share(idx):
        yy = y[idx]
        return [float((yy == k).mean()) for k in S.LABELS]
    overall_share = [float((y == k).mean()) for k in S.LABELS]
    deferred_share = class_share(deferred_idx)
    acc_retained = float(correct[retained_idx].mean())

    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- (a) risk-coverage --------------------------------------------------
    axL.plot(coverages * 100, np.array(acc_at) * 100, color=S.ENS_COLOR, lw=2.4)
    axL.axhline(base_acc * 100, ls="--", color=S.MUTED, lw=1.2,
                label=f"answer everything ({base_acc*100:.1f}%)")
    for c, col in [(0.8, S.WARN), (0.6, S.BAD)]:
        axL.scatter([c * 100], [ops[c] * 100], s=120, color=col, zorder=5,
                    edgecolors="white", lw=1.5,
                    label=f"defer {round((1-c)*100)}% → acc {ops[c]*100:.1f}%")
    axL.set_xlabel("coverage — % of items auto-decided")
    axL.set_ylabel("accuracy on auto-decided items (%)")
    axL.set_title("Deferring the hardest calls lifts accuracy on the rest", color=S.ACCENT)
    axL.legend(fontsize=9.5, loc="lower left")
    axL.invert_xaxis()  # read left→right as "defer more"

    # ---- (b) deferred composition -------------------------------------------
    x = np.arange(4); w = 0.38
    axR.bar(x - w/2, np.array(overall_share) * 100, w, color=S.MUTED,
            label="all items")
    axR.bar(x + w/2, np.array(deferred_share) * 100, w, color=S.BAD,
            label=f"deferred {int(DEFER*100)}% (lowest conf.)")
    for xi, (a, b) in enumerate(zip(overall_share, deferred_share)):
        axR.text(xi - w/2, a*100 + 0.8, f"{a*100:.0f}%", ha="center", fontsize=8.5,
                 color=S.MUTED)
        axR.text(xi + w/2, b*100 + 0.8, f"{b*100:.0f}%", ha="center", fontsize=8.5,
                 color=S.BAD, fontweight="bold")
    axR.set_xticks(x); axR.set_xticklabels(S.CLASS_SHORT)
    axR.set_xlabel("true severity class")
    axR.set_ylabel("share of items (%)")
    axR.set_title("It defers the ambiguous moderate/severe cases", color=S.ACCENT)
    axR.legend(fontsize=9.5, loc="upper right")

    fig.suptitle("Knows what it doesn't know: abstaining on low-confidence items routes the right cases to a human",
                 fontsize=13.5, fontweight="bold", color=S.ACCENT, y=1.02)
    S.save(fig, "fig8_selective_prediction")

    out = {
        "n": int(len(df)), "base_accuracy": base_acc,
        "accuracy_at_coverage": {f"{c:.1f}": ops[c] for c in ops},
        "defer_budget": DEFER, "accuracy_retained_at_budget": acc_retained,
        "class_share_overall": dict(zip(map(str, S.LABELS), overall_share)),
        "class_share_deferred": dict(zip(map(str, S.LABELS), deferred_share)),
    }
    cp = S.companion_path("fig8_selective_prediction", "json")
    cp.write_text(json.dumps(out, indent=2))
    print(f"  wrote {cp}")
    print(f"  base acc={base_acc:.3f}; defer40%->{ops[0.6]:.3f}, defer20%->{ops[0.8]:.3f}")
    print(f"  severe share: overall={overall_share[3]:.3f} deferred={deferred_share[3]:.3f}")


if __name__ == "__main__":
    main()
