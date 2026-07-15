"""Fig 17 - Best model (Cascade * merged-gate) vs MentalBERT+CORN+Hybrid-W5, per item.

Same head-to-head layout as fig16, swapping the baseline CoT for the project's
best system: the confidence-gated cascade (pooled 10-chain self-consistency LLM,
encoder breaks ties on the ~items it flags not-confident via the merged gate --
see cot_cascade.py and fig12_per_item_best.py, which uses the identical config).

Because the cascade already calls on the SAME encoder to break ties on a subset
of items, this is not two fully independent models -- it's "cascade" (LLM +
selective encoder tiebreak) vs "encoder alone". That's the honest comparison a
deployer cares about: does adding the LLM in front of the encoder ever help.

  * Cascade * merged-gate = pooled 10-chain SC LLM (folds_tolerant_sc5[_dv]),
                            W5 encoder breaks ties (gate="merged", tau=0.8, diff>=0.5)
  * MentalBERT+W5 (alone) = outputs/cv/oof_predictions_ctxm_corn_hybw5.csv

    python -m src.evaluation.presentation.fig17_cascade_vs_mbert_hybw5
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             cohen_kappa_score)

from src.evaluation.cot_cascade import load_llm, load_encoder, align, gate_mask, apply_cascade
from src.evaluation.presentation import _style as S

COT = S.ROOT / "outputs" / "cot"
CV = S.ROOT / "outputs" / "cv"
LLM_GLOBS = [str(COT / "folds_tolerant_sc5" / "*.csv"),
             str(COT / "folds_tolerant_sc5_dv" / "*.csv")]
ENC_W5 = CV / "oof_predictions_ctxm_corn_hybw5.csv"
LAB = S.LABELS

CASC_LABEL = "Cascade · merged-gate (pooled 10-chain LLM + W5 encoder tiebreak)"
ENC_LABEL = "MentalBERT+CORN+Hybrid-W5 (alone)"
CASC_COLOR = S.CALM_CASCADE


def metric_block(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", labels=LAB, zero_division=0)),
        "mae": float(mean_absolute_error(y, p)),
        "qwk": float(cohen_kappa_score(y, p, weights="quadratic", labels=LAB)),
        "f1_per_class": [float(x) for x in
                         f1_score(y, p, average=None, labels=LAB, zero_division=0)],
    }


def main():
    S.apply_rc()
    llm = load_llm(LLM_GLOBS)
    enc = load_encoder(str(ENC_W5))
    df = align(llm, enc)
    df["casc_merg"] = apply_cascade(df, gate_mask(df, "merged", 0.8, 0.5))
    df["item_name"] = df["item_id"].map(S.ITEM_NAMES)
    print(f"matched rows: {len(df)}  ({df.participant_id.nunique()} participants)")

    items = list(range(1, 9))
    item_names = [S.ITEM_NAMES[i] for i in items]

    per_item = {"casc": {}, "enc": {}}
    for iid in items:
        g = df[df.item_id == iid]
        per_item["casc"][iid] = metric_block(g.label, g.casc_merg)
        per_item["enc"][iid] = metric_block(g.label, g.enc_pred)
    overall = {"casc": metric_block(df.label, df.casc_merg),
               "enc": metric_block(df.label, df.enc_pred)}

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14.5, 10.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.42, wspace=0.32)

    # ---- top: per-item macro-F1, headline comparison ----
    ax0 = fig.add_subplot(gs[0, :])
    x = np.arange(len(items))
    w = 0.34
    casc_f1 = [per_item["casc"][i]["macro_f1"] for i in items]
    enc_f1 = [per_item["enc"][i]["macro_f1"] for i in items]
    ax0.bar(x - w / 2, casc_f1, w, label=CASC_LABEL, color=CASC_COLOR,
           edgecolor="white", linewidth=0.6)
    ax0.bar(x + w / 2, enc_f1, w, label=ENC_LABEL, color=S.CALM_ENC,
           edgecolor="white", linewidth=0.6)
    for xi, (cv, ev) in enumerate(zip(casc_f1, enc_f1)):
        winner_x = xi + (-w / 2 if cv >= ev else w / 2)
        ax0.text(winner_x, max(cv, ev) + 0.012, "★", ha="center", fontsize=11, color=S.ACCENT)
    ax0.set_xticks(x)
    ax0.set_xticklabels(item_names, fontsize=11)
    ax0.set_ylabel("macro-F1 (per item)")
    ax0.set_ylim(0, max(casc_f1 + enc_f1) * 1.22)
    ax0.set_title("Per-item macro-F1: Cascade·merged-gate (best system) vs MentalBERT+CORN+Hybrid-W5 alone  (★ = winner)",
                 color=S.ACCENT, fontsize=13.5)
    ax0.legend(ncol=2, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, 1.19))
    ax0.grid(axis="x", visible=False)

    # ---- bottom row: per-item F1 broken out by severity class 0..3 ----
    class_axes = [fig.add_subplot(gs[1, k]) for k in range(4)]
    for k, ax in enumerate(class_axes):
        casc_c = [per_item["casc"][i]["f1_per_class"][k] for i in items]
        enc_c = [per_item["enc"][i]["f1_per_class"][k] for i in items]
        ax.bar(x - w / 2, casc_c, w, color=CASC_COLOR, edgecolor="white", linewidth=0.5)
        ax.bar(x + w / 2, enc_c, w, color=S.CALM_ENC, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(item_names, fontsize=7.5, rotation=60, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"F1 — {S.CLASS_SHORT[k]} ({k})", fontsize=11, color=S.ACCENT)
        ax.grid(axis="x", visible=False)
        if k == 0:
            ax.set_ylabel("F1 (per item, per class)")

    # ---- footer: pooled accuracy / MAE / QWK ----
    foot = (f"Pooled (n={overall['casc']['n']}):   "
           f"accuracy  Cascade {overall['casc']['accuracy']:.3f}  vs  enc {overall['enc']['accuracy']:.3f}"
           f"     |     macro-F1  Cascade {overall['casc']['macro_f1']:.3f}  vs  enc {overall['enc']['macro_f1']:.3f}"
           f"     |     MAE (lower better)  Cascade {overall['casc']['mae']:.3f}  vs  enc {overall['enc']['mae']:.3f}"
           f"     |     QWK  Cascade {overall['casc']['qwk']:.3f}  vs  enc {overall['enc']['qwk']:.3f}")
    fig.text(0.5, -0.015, foot, ha="center", fontsize=10.5, color=S.MUTED)

    S.save(fig, "fig17_cascade_vs_mbert_hybw5")

    out_rows = []
    for iid in items:
        for tag in ["casc", "enc"]:
            m = per_item[tag][iid]
            out_rows.append({"item_id": iid, "item_name": S.ITEM_NAMES[iid], "model": tag,
                             "n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                             "mae": m["mae"], "qwk": m["qwk"],
                             "f1_class0": m["f1_per_class"][0], "f1_class1": m["f1_per_class"][1],
                             "f1_class2": m["f1_per_class"][2], "f1_class3": m["f1_per_class"][3]})
    for tag in ["casc", "enc"]:
        m = overall[tag]
        out_rows.append({"item_id": 0, "item_name": "OVERALL", "model": tag,
                         "n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                         "mae": m["mae"], "qwk": m["qwk"],
                         "f1_class0": m["f1_per_class"][0], "f1_class1": m["f1_per_class"][1],
                         "f1_class2": m["f1_per_class"][2], "f1_class3": m["f1_per_class"][3]})
    out_df = pd.DataFrame(out_rows)
    cp = S.companion_path("fig17_cascade_vs_mbert_hybw5", "csv")
    out_df.to_csv(cp, index=False)
    print(f"  wrote {cp}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
