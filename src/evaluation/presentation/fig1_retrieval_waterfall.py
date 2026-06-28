"""Fig 1 - Retrieval contribution waterfall: "the evidence is the hero".

Story: holding the model and loss fixed (MentalBERT-class encoder + CORN), we
swap only the *evidence* fed to the classifier. Lexical BM25 context windows
barely move macro-F1 over isolated utterances; adding the *semantic* hybrid
retriever is what unlocks the gain -- and total-score QWK jumps 0.074 -> 0.485.

Reads the pooled headline numbers from outputs/context_window_results.json and
recomputes per-fold metrics from the matching OOF predictions for error bars.

    python -m src.evaluation.presentation.fig1_retrieval_waterfall
"""

import json

import numpy as np
import pandas as pd

from src.evaluation.context_window_eval import metrics_from_oof, CV
from src.evaluation.presentation import _style as S

# evidence stages (narrative order) -> (label, condition key, OOF tag, colour)
STAGES = [
    ("Isolated\nutterances", "ctx_corn_old", "ctx_corn_old", S.RETR_COLORS["Isolated"]),
    ("BM25 context\nwindows (W3)", "ctx_corn_bm25w3", "ctx_corn_bm25w3", S.RETR_COLORS["BM25"]),
    ("Hybrid context\nwindows (W3)", "ctx_corn_hybw3", "ctx_corn_hybw3", S.RETR_COLORS["Hybrid"]),
]
RESULTS = S.ROOT / "outputs" / "context_window_results.json"


def per_fold_stats(tag):
    """mean, std across the 5 CV folds for item macro-F1 and total QWK."""
    f = CV / f"oof_predictions_{tag}.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f)
    mf1, tqwk = [], []
    for _, g in df.groupby("fold"):
        item, total = metrics_from_oof(g)
        mf1.append(item["macro_f1"])
        tqwk.append(total["total_qwk"])
    return {
        "macro_f1": (float(np.mean(mf1)), float(np.std(mf1))),
        "total_qwk": (float(np.mean(tqwk)), float(np.std(tqwk))),
    }


def main():
    S.apply_rc()
    pooled = json.loads(RESULTS.read_text())

    labels, colors = [], []
    mf1_pool, qwk_pool = [], []
    mf1_err, qwk_err = [], []
    rows = []
    for label, key, tag, color in STAGES:
        labels.append(label)
        colors.append(color)
        mf1_pool.append(pooled[key]["item"]["macro_f1"])
        qwk_pool.append(pooled[key]["total"]["total_qwk"])
        stats = per_fold_stats(tag)
        mf1_err.append(stats.get("macro_f1", (None, 0.0))[1])
        qwk_err.append(stats.get("total_qwk", (None, 0.0))[1])
        rows.append({
            "stage": label.replace("\n", " "),
            "item_macro_f1_pooled": mf1_pool[-1],
            "item_macro_f1_fold_std": mf1_err[-1],
            "total_qwk_pooled": qwk_pool[-1],
            "total_qwk_fold_std": qwk_err[-1],
        })

    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5.2))
    x = np.arange(len(labels))

    def panel(ax, vals, errs, title, ylab, ymax):
        bars = ax.bar(x, vals, yerr=errs, capsize=5, color=colors,
                      edgecolor="white", linewidth=1.2,
                      error_kw=dict(ecolor=S.MUTED, lw=1.2))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(ylab)
        ax.set_title(title, color=S.ACCENT)
        ax.set_ylim(0, ymax)
        for xi, v, e in zip(x, vals, errs):
            ax.text(xi, v + (e or 0) + ymax * 0.02, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        return bars

    panel(axL, mf1_pool, mf1_err, "Item-level macro-F1", "macro-F1", 0.45)
    panel(axR, qwk_pool, qwk_err, "Reconstructed PHQ-8 total — QWK", "total-score QWK", 0.72)

    # annotate the two key jumps on the macro-F1 panel
    axL.annotate("", xy=(1, mf1_pool[1]), xytext=(0, mf1_pool[0]),
                 arrowprops=dict(arrowstyle="->", color=S.MUTED, lw=1.4))
    axL.text(0.5, max(mf1_pool[:2]) + 0.06,
             f"lexical adds\n+{mf1_pool[1]-mf1_pool[0]:+.3f}".replace("++", "+"),
             ha="center", fontsize=9.5, color=S.MUTED)
    axL.annotate("", xy=(2, mf1_pool[2]), xytext=(1, mf1_pool[1]),
                 arrowprops=dict(arrowstyle="->", color=S.BAD, lw=1.8))
    axL.text(1.5, mf1_pool[2] + 0.03,
             f"semantic adds\n+{mf1_pool[2]-mf1_pool[1]:.3f}",
             ha="center", fontsize=10, color=S.BAD, fontweight="bold")

    # headline QWK multiplier
    mult = qwk_pool[2] / qwk_pool[0] if qwk_pool[0] else float("nan")
    axR.text(1.0, 0.66, f"{qwk_pool[0]:.3f}  →  {qwk_pool[2]:.3f}\n({mult:.1f}× ordinal agreement)",
             ha="center", fontsize=11, color=S.ACCENT, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f0f9ff", ec=S.ENS_COLOR))

    fig.suptitle("The evidence is the hero: semantic retrieval, not context length, drives the gain",
                 fontsize=15, fontweight="bold", color=S.ACCENT, y=1.02)
    fig.text(0.5, -0.04,
             "Same encoder (MentalBERT) + same loss (CORN); only the retrieved evidence changes. "
             "Error bars = std across 5 participant-stratified CV folds.",
             ha="center", fontsize=10, color=S.MUTED)

    S.save(fig, "fig1_retrieval_waterfall")
    pd.DataFrame(rows).to_csv(S.companion_path("fig1_retrieval_waterfall", "csv"), index=False)
    print(f"  wrote {S.companion_path('fig1_retrieval_waterfall', 'csv')}")
    print(f"  headline: macro-F1 {mf1_pool[0]:.3f}->{mf1_pool[2]:.3f}, "
          f"total-QWK {qwk_pool[0]:.3f}->{qwk_pool[2]:.3f} ({mult:.1f}x)")


if __name__ == "__main__":
    main()
