"""Fig 10 - Bucket-B experiments: different tools win different goals.

Four targeted interventions, each trained on the identical MentalBERT / hybrid-W3
/ CORN / 5-fold config (only the named factor changes), shown as deltas vs the
plain-CORN baseline. The point of the slide: no method dominates — pick by goal.

  * attention-MIL      : best aggregate (macro-F1, MAE) + per-window attribution
  * corn_focal         : best ordinal agreement (QWK), at no cost
  * corn_balanced      : lifts moderate/severe F1, costs MAE
  * rerank+CORN        : best severe-class recall, costs ordinal metrics

Reads the OOF predictions written by the cluster jobs.

    python -m src.evaluation.presentation.fig10_bucketb_summary
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score, mean_absolute_error

from src.evaluation.presentation import _style as S

CV = S.ROOT / "outputs" / "cv"
LAB = [0, 1, 2, 3]
# (display name, oof tag)
METHODS = [
    ("Attention-MIL", "mil_hybw3"),
    ("Focal CORN", "ctxm_cornfocal_hybw3"),
    ("Balanced CORN", "ctxm_cornbal_hybw3"),
    ("Rerank + CORN", "ctxm_corn_rerank_w3"),
]
BASE = "ctxm_corn_hybw3"
# (column label, metric key, higher_is_better)
COLS = [("macro-F1", "f1", True), ("QWK", "qwk", True),
        ("MAE", "mae", False), ("Moderate-F1", "c2", True), ("Severe-F1", "c3", True)]


def metrics(tag):
    d = pd.read_csv(CV / f"oof_predictions_{tag}.csv")
    y, p = d["label"], d["prediction"]
    pc = f1_score(y, p, average=None, labels=LAB, zero_division=0)
    return {"f1": f1_score(y, p, average="macro", labels=LAB, zero_division=0),
            "qwk": cohen_kappa_score(y, p, weights="quadratic", labels=LAB),
            "mae": mean_absolute_error(y, p), "c2": pc[2], "c3": pc[3]}


def main():
    S.apply_rc()
    base = metrics(BASE)
    M = {name: metrics(tag) for name, tag in METHODS}

    # signed improvement vs baseline (positive = better, accounting for MAE direction)
    delta = np.zeros((len(METHODS), len(COLS)))
    raw = np.zeros_like(delta)
    for i, (name, _) in enumerate(METHODS):
        for j, (_, key, hib) in enumerate(COLS):
            d = M[name][key] - base[key]
            raw[i, j] = M[name][key]
            delta[i, j] = d if hib else -d   # flip MAE so positive=better

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.grid(False)
    vmax = np.abs(delta).max() * 1.05
    im = ax.imshow(delta, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels([c[0] for c in COLS], fontsize=11)
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels([m[0] for m in METHODS], fontsize=11)
    ax.set_xlabel(f"metric (Δ vs plain-CORN baseline; green = better, MAE sign-flipped)")

    for i in range(len(METHODS)):
        for j, (_, key, hib) in enumerate(COLS):
            d = M[METHODS[i][0]][key] - base[key]
            sign = "+" if d > 0 else ""
            best = (delta[:, j].argmax() == i)
            ax.text(j, i, f"{raw[i, j]:.3f}\n({sign}{d:.3f})", ha="center", va="center",
                    fontsize=9, fontweight="bold" if best else "normal",
                    color=S.ACCENT)
            if best:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor=S.ACCENT, lw=2.2))

    # baseline reference row label
    base_str = "  |  ".join(f"{c[0]} {base[c[1]]:.3f}" for c in COLS)
    ax.set_title("Bucket-B: every intervention wins a different metric (boxed = best in column)",
                 color=S.ACCENT, fontsize=13.5)
    fig.text(0.5, -0.02, f"Plain-CORN baseline:  {base_str}",
             ha="center", fontsize=9.5, color=S.MUTED)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Δ (better →)", fontsize=9)

    S.save(fig, "fig10_bucketb_summary")

    out = {"baseline": base,
           "methods": {name: M[name] for name, _ in METHODS},
           "note": "Trained identical config (MentalBERT/hybrid-W3/CORN/5-fold seed42); only the named factor differs."}
    cp = S.companion_path("fig10_bucketb_summary", "json")
    cp.write_text(json.dumps(out, indent=2))
    print(f"  wrote {cp}")
    for name, _ in METHODS:
        m = M[name]
        print(f"  {name:16s} F1={m['f1']:.3f} QWK={m['qwk']:.3f} MAE={m['mae']:.3f} "
              f"c2={m['c2']:.3f} c3={m['c3']:.3f}")


if __name__ == "__main__":
    main()
