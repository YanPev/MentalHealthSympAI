"""Fig 3 - Evidence-quality x predictability map: "why some symptoms are hard".

Per PHQ-8 item, plot the *quality of retrieved evidence* (keyword hit-rate of the
hybrid context windows) against *predictability* (per-item macro-F1). Bubble size
= class imbalance (majority-class share); bubble colour = raw accuracy.

The point: evidence presence != predictability. "Moving" looks easy by accuracy
(0.72) only because one class dominates (majority share 0.73) — its F1 is mediocre.
"Sleep" has abundant on-topic evidence yet stays hard. Good evidence is necessary,
not sufficient.

    python -m src.evaluation.presentation.fig3_evidence_quality_map
"""

import json

import numpy as np
import pandas as pd

from src.evaluation.presentation import _style as S

PER_ITEM = S.ROOT / "outputs" / "per_item_analysis.csv"
RETR = S.ROOT / "outputs" / "cot" / "retrieval_window_analysis.json"


def main():
    S.apply_rc()
    pi = pd.read_csv(PER_ITEM)
    retr = {r["item"]: r for r in json.loads(RETR.read_text())["per_item"]}
    pi["keyword_hit_rate"] = pi["item_name"].map(lambda n: retr[n]["keyword_hit_rate"])
    pi = pi.sort_values("item_id").reset_index(drop=True)

    corr = float(np.corrcoef(pi["keyword_hit_rate"], pi["f1_mean"])[0, 1])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6.6))

    sizes = (pi["majority_share"] ** 2) * 5200  # area ~ imbalance, emphasised
    sc = ax.scatter(pi["keyword_hit_rate"], pi["f1_mean"], s=sizes,
                    c=pi["acc_mean"], cmap="viridis", vmin=0.28, vmax=0.74,
                    edgecolors="white", linewidths=1.6, alpha=0.92, zorder=3)

    # median guide lines -> quadrants
    xmed, ymed = pi["keyword_hit_rate"].median(), pi["f1_mean"].median()
    ax.axvline(xmed, color=S.MUTED, ls=":", lw=1, zorder=1)
    ax.axhline(ymed, color=S.MUTED, ls=":", lw=1, zorder=1)

    # labels (nudge to avoid overlap)
    nudge = {
        "Moving": (0.012, -0.004), "Sleep": (0.012, 0.004),
        "Appetite": (0.012, -0.006), "Tired": (-0.012, -0.010),
        "Concentrating": (0.010, 0.004), "NoInterest": (-0.012, -0.014),
        "Depressed": (0.006, 0.012), "Failure": (0.018, -0.012),
    }
    for _, r in pi.iterrows():
        dx, dy = nudge.get(r["item_name"], (0.012, 0.0))
        ha = "left" if dx > 0 else "right"
        ax.annotate(r["item_name"], (r["keyword_hit_rate"], r["f1_mean"]),
                    (r["keyword_hit_rate"] + dx, r["f1_mean"] + dy),
                    fontsize=10.5, fontweight="bold", color=S.ACCENT, ha=ha, va="center")

    # call out the two instructive cases
    def spotlight(name, text, tx, ty):
        r = pi[pi.item_name == name].iloc[0]
        ax.annotate(text, (r["keyword_hit_rate"], r["f1_mean"]), (tx, ty),
                    fontsize=9, color=S.BAD, ha="center",
                    arrowprops=dict(arrowstyle="->", color=S.BAD, lw=1.3),
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec=S.BAD, lw=1))
    spotlight("Moving", "acc 0.72 but F1 0.31:\nsaturated by majority class\n(big bubble)",
              0.55, 0.30)
    spotlight("Sleep", "abundant on-topic evidence,\nstill hard to predict",
              0.93, 0.31)

    ax.set_xlabel("Evidence quality — keyword hit-rate of retrieved hybrid windows  →")
    ax.set_ylabel("Predictability — per-item macro-F1  →")
    ax.set_title("Good evidence is necessary, not sufficient: per-symptom difficulty map",
                 color=S.ACCENT)
    ax.set_xlim(0.30, 1.0)
    ax.set_ylim(0.18, 0.345)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("per-item accuracy")

    # legend for bubble size
    for share, lab in [(0.40, "balanced (0.40)"), (0.73, "imbalanced (0.73)")]:
        ax.scatter([], [], s=(share ** 2) * 5200, c="#cbd5e1",
                   edgecolors="white", linewidths=1.4, label=f"majority share {lab}")
    leg = ax.legend(title="bubble size = class imbalance", loc="lower left",
                    fontsize=9, title_fontsize=9, labelspacing=1.6, borderpad=1.0)
    leg.get_frame().set_alpha(0)

    ax.text(0.99, 0.185,
            f"Pearson r(evidence, F1) = {corr:+.2f}  →  weak link",
            ha="right", fontsize=9.5, color=S.MUTED, style="italic")

    S.save(fig, "fig3_evidence_quality_map")

    pi_out = pi[["item_id", "item_name", "keyword_hit_rate", "f1_mean", "acc_mean",
                 "mae_mean", "majority_share", "pct_severe_2_3"]].copy()
    pi_out["evidence_f1_pearson_r"] = corr
    cp = S.companion_path("fig3_evidence_quality_map", "csv")
    pi_out.to_csv(cp, index=False)
    print(f"  wrote {cp}")
    print(f"  Pearson r(keyword_hit_rate, f1) = {corr:+.3f}")


if __name__ == "__main__":
    main()
