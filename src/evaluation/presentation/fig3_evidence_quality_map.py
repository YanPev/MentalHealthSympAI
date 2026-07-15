"""Fig 3 - Evidence-quality x predictability map, with the cascade transition.

Per PHQ-8 item, plot the *quality of retrieved evidence* (keyword hit-rate of the
hybrid context windows) on x against *predictability* (per-item macro-F1) on y.

Each item now carries TWO dots joined by a vertical arrow, showing how F1 moves
when we swap the plain encoder for the winning cascade:

    hollow grey dot  = MentalBERT+CORN, Hybrid W3   (encoder alone)
    filled dot       = Attention-MIL merged-gate cascade
    arrow            = the F1 change (baseline -> cascade)

Evidence quality is a property of the item, not the model, so both dots share an
x and the arrow is purely vertical: "at the same evidence level, how much does the
cascade move predictability?".

The filled dot's COLOUR is the mean difficulty score the merged-gate cascade
assigned to that item (mean of the staged LLM's `difficult_frac`, the same signal
the gate routes on). Bubble size = class imbalance (majority-class share), as
before.

The point: the cascade lifts every item, but the items it can barely move
(Moving, Appetite, Sleep) are precisely the ones it rates most difficult and/or
that are saturated by a majority class -- good evidence is necessary, not
sufficient, and the residual ceiling is intrinsic to those symptoms.

    python -m src.evaluation.presentation.fig3_evidence_quality_map
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.evaluation.cot_cascade import (load_llm, load_encoder, align,
                                        gate_mask, apply_cascade)
from src.evaluation.presentation import _style as S

PER_ITEM = S.ROOT / "outputs" / "per_item_analysis.csv"
RETR = S.ROOT / "outputs" / "cot" / "retrieval_window_analysis.json"
CV = S.ROOT / "outputs" / "cv"
COT = S.ROOT / "outputs" / "cot"
LLM_GLOBS = [str(COT / "folds_tolerant_sc5" / "*.csv"),
             str(COT / "folds_tolerant_sc5_dv" / "*.csv")]
BASE_TAG = "ctxm_corn_hybw3"   # MentalBERT + CORN, Hybrid W3 (encoder alone)
MIL_TAG = "mil_hybw3"          # Attention-MIL, the cascade's winning encoder
LAB = S.LABELS


def per_item_f1(df, y_col, pred_col):
    """macro-F1 per item_id, returned as {item_id: f1} (pooled out-of-fold)."""
    return {int(i): f1_score(g[y_col], g[pred_col], average="macro",
                             labels=LAB, zero_division=0)
            for i, g in df.groupby("item_id")}


def main():
    S.apply_rc()

    # ---- pooled out-of-fold predictions for both configs (apples-to-apples) ----
    llm = load_llm(LLM_GLOBS)
    base = align(llm, load_encoder(str(CV / f"oof_predictions_{BASE_TAG}.csv")))
    mil = align(llm, load_encoder(str(CV / f"oof_predictions_{MIL_TAG}.csv")))
    mask = gate_mask(mil, "merged", 0.8, 0.5)
    mil["casc"] = apply_cascade(mil, mask)
    mil["routed"] = mask.astype(int)

    f1_base = per_item_f1(base, "label", "enc_pred")
    f1_casc = per_item_f1(mil, "label", "casc")
    mean_diff = mil.groupby("item_id")["difficult_frac"].mean().to_dict()
    routed = (100 * mil.groupby("item_id")["routed"].mean()).to_dict()

    # ---- per-item context: imbalance (from CV) + evidence quality (retrieval) ----
    pi = pd.read_csv(PER_ITEM).sort_values("item_id").reset_index(drop=True)
    retr = {r["item"]: r for r in json.loads(RETR.read_text())["per_item"]}
    pi["hit"] = pi["item_name"].map(lambda n: retr[n]["keyword_hit_rate"])
    pi["f1_base"] = pi["item_id"].map(f1_base)
    pi["f1_casc"] = pi["item_id"].map(f1_casc)
    pi["mean_difficult"] = pi["item_id"].map(mean_diff)
    pi["routed_pct"] = pi["item_id"].map(routed)
    pi["delta_f1"] = pi["f1_casc"] - pi["f1_base"]

    corr = float(np.corrcoef(pi["hit"], pi["f1_casc"])[0, 1])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11.5, 7.0))

    # bubble area ~ class imbalance, normalised to a modest, readable range so the
    # markers stay small and the transition arrows between them remain visible.
    SIZE_LO, SIZE_HI = 0.35, 0.75          # majority-share reference bounds
    S_MIN, S_MAX = 90, 340                  # marker-area range (points^2)

    def size_of(share):
        t = np.clip((share - SIZE_LO) / (SIZE_HI - SIZE_LO), 0, 1)
        return S_MIN + t * (S_MAX - S_MIN)

    sizes = size_of(pi["majority_share"].to_numpy())
    radius_pt = np.sqrt(sizes / np.pi)      # marker radius, to clear the arrows
    cmap = plt.cm.plasma
    dmax = 0.25  # difficult_frac ceiling for the colour scale

    # transition arrows: encoder alone -> cascade, one per item (vertical). Shrink
    # each end to its own dot's radius so the arrowhead lands in the gap, visible.
    for r_pt, (_, r) in zip(radius_pt, pi.iterrows()):
        ax.annotate("", (r["hit"], r["f1_casc"]), (r["hit"], r["f1_base"]),
                    arrowprops=dict(arrowstyle="-|>", color=S.GOOD, lw=2.2,
                                    mutation_scale=18, shrinkA=r_pt + 1.5,
                                    shrinkB=r_pt + 1.5, alpha=0.95), zorder=2)

    # baseline dots (encoder alone): hollow, low-emphasis
    ax.scatter(pi["hit"], pi["f1_base"], s=sizes, facecolors="white",
               edgecolors=S.MUTED, linewidths=1.5, zorder=3)
    # cascade dots: filled, coloured by mean difficulty score
    sc = ax.scatter(pi["hit"], pi["f1_casc"], s=sizes, c=pi["mean_difficult"],
                    cmap=cmap, vmin=0.0, vmax=dmax, edgecolors="white",
                    linewidths=1.4, alpha=0.95, zorder=4)

    # median evidence-quality guide line (self-labelled)
    xmed = pi["hit"].median()
    ax.axvline(xmed, color=S.MUTED, ls=":", lw=1, zorder=1)
    ax.text(xmed - 0.006, ax.get_ylim()[1], "median hit-rate", rotation=90,
            ha="right", va="top", fontsize=8, color=S.MUTED, style="italic")

    # per-item labels + the F1 gain, placed beside the cascade dot (nudged apart
    # for the three near-overlapping x-clusters: Conc/Tired, Depr/Fail, App/Moving)
    nudge = {
        "Sleep": (0.015, 0.006), "Failure": (0.015, 0.008),
        "Depressed": (-0.015, -0.002), "NoInterest": (-0.015, 0.008),
        "Tired": (0.013, 0.010), "Concentrating": (-0.010, 0.018),
        "Appetite": (-0.015, -0.008), "Moving": (0.020, -0.024),
    }
    for _, r in pi.iterrows():
        dx, dy = nudge.get(r["item_name"], (0.012, 0.0))
        ha = "left" if dx > 0 else "right"
        d = r["delta_f1"]
        dtxt = f"+{d:.3f}" if d < 0.02 else f"+{d:.2f}"
        ax.annotate(f"{r['item_name']}\n{dtxt} F1",
                    (r["hit"], r["f1_casc"]),
                    (r["hit"] + dx, r["f1_casc"] + dy),
                    fontsize=9.5, fontweight="bold", color=S.ACCENT,
                    ha=ha, va="center")

    # call out the two instructive extremes
    def spotlight(name, text, tx, ty):
        r = pi[pi.item_name == name].iloc[0]
        ax.annotate(text, (r["hit"], r["f1_casc"]), (tx, ty),
                    fontsize=9, color=S.BAD, ha="center",
                    arrowprops=dict(arrowstyle="->", color=S.BAD, lw=1.3),
                    bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2",
                              ec=S.BAD, lw=1))
    spotlight("Concentrating",
              "biggest cascade gain (+0.13 F1):\nweak evidence, but routing helps",
              0.44, 0.27)
    spotlight("Moving",
              "majority-saturated (share 0.73):\ncascade barely moves it (+0.004)",
              0.83, 0.265)
    spotlight("Sleep",
              "highest difficulty score (0.23):\nrouted most, gains only +0.04",
              0.66, 0.455)

    ax.set_xlabel("Evidence quality — keyword hit-rate of retrieved hybrid windows  →")
    ax.set_ylabel("Predictability — per-item macro-F1  →")
    ax.set_title("Encoder → cascade: the F1 lift, and where difficulty caps it",
                 color=S.ACCENT)
    ax.set_xlim(0.30, 1.0)
    ax.set_ylim(0.22, 0.49)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("mean difficulty score assigned by cascade\n(staged-LLM difficult_frac)")

    # legend: marker meaning (baseline vs cascade + arrow) and bubble size
    from matplotlib.lines import Line2D
    marker_handles = [
        Line2D([], [], marker="o", ls="", mfc="white", mec=S.MUTED, mew=1.6,
               ms=11, label="MentalBERT+CORN · Hybrid W3 (encoder alone)"),
        Line2D([], [], marker="o", ls="", mfc=cmap(0.5), mec="white", mew=1.4,
               ms=11, label="Attention-MIL merged-gate cascade"),
        Line2D([], [], marker=r"$\uparrow$", ls="", color=S.GOOD, ms=13,
               label="arrow = F1 change (baseline → cascade)"),
    ]
    leg1 = ax.legend(handles=marker_handles, loc="upper left", fontsize=9,
                     labelspacing=0.9, borderpad=0.9)
    leg1.get_frame().set_alpha(0.9)
    leg1.get_frame().set_edgecolor(S.MUTED)
    ax.add_artist(leg1)

    size_handles = [ax.scatter([], [], s=size_of(share), c="#cbd5e1",
                               edgecolors="white", linewidths=1.2,
                               label=f"majority share {lab}")
                    for share, lab in [(0.40, "0.40 (balanced)"),
                                       (0.73, "0.73 (imbalanced)")]]
    leg2 = ax.legend(handles=size_handles,
                     title="bubble size = class imbalance\n(majority-class share)",
                     loc="lower left", fontsize=9, title_fontsize=8.5,
                     labelspacing=1.4, borderpad=1.1, handletextpad=1.6)
    leg2.get_frame().set_alpha(0.85)
    leg2.get_frame().set_edgecolor(S.MUTED)

    ax.text(0.99, 0.225,
            f"Pearson r(evidence, cascade F1) = {corr:+.2f}  →  weak link",
            ha="right", fontsize=9.5, color=S.MUTED, style="italic")

    S.save(fig, "fig3_evidence_quality_map")

    pi_out = pi[["item_id", "item_name", "hit", "majority_share",
                 "f1_base", "f1_casc", "delta_f1", "mean_difficult",
                 "routed_pct"]].copy()
    pi_out["evidence_cascadeF1_pearson_r"] = corr
    cp = S.companion_path("fig3_evidence_quality_map", "csv")
    pi_out.to_csv(cp, index=False)
    print(f"  wrote {cp}")
    print(f"  Pearson r(hit_rate, cascade f1) = {corr:+.3f}")
    print(pi_out.to_string(index=False))


if __name__ == "__main__":
    main()
