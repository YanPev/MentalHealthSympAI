"""Fig 16b - Baseline CoT vs MentalBERT+CORN+Hybrid-W3, per PHQ-8 item.

Head-to-head between the two standalone models (no cascade/ensemble):
  * "baseline CoT"    = plain greedy few-shot CoT, Qwen2.5-7B, Hybrid-W3 evidence,
                        no self-consistency / item-adaptive tricks
                        (outputs/cot/folds/cot_probe_qwen_hybw3_fold*.csv)
  * "MentalBERT+W3"   = the trained encoder, MentalBERT+CORN+Hybrid-W3
                        (outputs/cv/oof_predictions_ctxm_corn_hybw3.csv)
Same pooled out-of-fold slice, matched on (participant_id, item_id). Unlike the
W5 variant (fig16_cot_vs_mbert_hybw5), here both models see the same Hybrid-W3
evidence, so retrieval is held constant.

Top panel: per-item macro-F1 (headline). Bottom row: per-item F1 broken out by
PHQ-8 severity class (0 Minimal .. 3 Severe) -- this is what shows the CoT's
severe-class recovery the aggregate macro-F1 hides. Footer: pooled accuracy /
MAE / QWK for both models.

    python -m src.evaluation.presentation.fig16_cot_vs_mbert_hybw3
"""

import glob

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             cohen_kappa_score)

from src.evaluation.presentation import _style as S

COT_DIR = S.ROOT / "outputs" / "cot" / "folds"
ENC_OOF = S.ROOT / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"
LAB = S.LABELS
KEY = ["participant_id", "item_id"]

COT_LABEL = "Baseline CoT (Qwen2.5-7B, greedy, Hybrid-W3)"
ENC_LABEL = "MentalBERT+CORN+Hybrid-W3"


def load_cot():
    files = sorted(glob.glob(str(COT_DIR / "*.csv")))
    if not files:
        raise SystemExit(f"no CoT fold files under {COT_DIR}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["participant_id"] = df["participant_id"].astype(str)
    return df[KEY + ["label", "prediction"]].rename(columns={"prediction": "cot_pred"})


def load_enc():
    df = pd.read_csv(ENC_OOF)
    df["participant_id"] = df["participant_id"].astype(str)
    return df[KEY + ["label", "prediction"]].rename(columns={"prediction": "enc_pred"})


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
    cot, enc = load_cot(), load_enc()
    df = cot.merge(enc, on=KEY, suffixes=("_cot", "_enc"))
    mismatched = (df["label_cot"] != df["label_enc"]).sum()
    assert mismatched == 0, f"{mismatched} rows have mismatched gold labels between the two OOF files"
    df["label"] = df["label_cot"]
    df["item_name"] = df["item_id"].map(S.ITEM_NAMES)
    print(f"matched rows: {len(df)}  ({df.participant_id.nunique()} participants)")

    items = list(range(1, 9))
    item_names = [S.ITEM_NAMES[i] for i in items]

    per_item = {"cot": {}, "enc": {}}
    for iid in items:
        g = df[df.item_id == iid]
        per_item["cot"][iid] = metric_block(g.label, g.cot_pred)
        per_item["enc"][iid] = metric_block(g.label, g.enc_pred)
    overall = {"cot": metric_block(df.label, df.cot_pred),
               "enc": metric_block(df.label, df.enc_pred)}

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14.5, 10.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.42, wspace=0.32)

    # ---- top: per-item macro-F1, headline comparison ----
    ax0 = fig.add_subplot(gs[0, :])
    x = np.arange(len(items))
    w = 0.34
    cot_f1 = [per_item["cot"][i]["macro_f1"] for i in items]
    enc_f1 = [per_item["enc"][i]["macro_f1"] for i in items]
    ax0.bar(x - w / 2, cot_f1, w, label=COT_LABEL, color=S.CALM_COT,
           edgecolor="white", linewidth=0.6)
    ax0.bar(x + w / 2, enc_f1, w, label=ENC_LABEL, color=S.CALM_ENC,
           edgecolor="white", linewidth=0.6)
    for xi, (cv, ev) in enumerate(zip(cot_f1, enc_f1)):
        winner_x = xi + (-w / 2 if cv >= ev else w / 2)
        ax0.text(winner_x, max(cv, ev) + 0.012, "★", ha="center", fontsize=11, color=S.ACCENT)
    ax0.set_xticks(x)
    ax0.set_xticklabels(item_names, fontsize=11)
    ax0.set_ylabel("macro-F1 (per item)")
    ax0.set_ylim(0, max(cot_f1 + enc_f1) * 1.22)
    ax0.set_title("Per-item macro-F1: baseline CoT vs MentalBERT+CORN+Hybrid-W3  (★ = winner)",
                 color=S.ACCENT, fontsize=14.5)
    ax0.legend(ncol=2, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.19))
    ax0.grid(axis="x", visible=False)

    # ---- bottom row: per-item F1 broken out by severity class 0..3 ----
    class_axes = [fig.add_subplot(gs[1, k]) for k in range(4)]
    for k, ax in enumerate(class_axes):
        cot_c = [per_item["cot"][i]["f1_per_class"][k] for i in items]
        enc_c = [per_item["enc"][i]["f1_per_class"][k] for i in items]
        ax.bar(x - w / 2, cot_c, w, color=S.CALM_COT, edgecolor="white", linewidth=0.5)
        ax.bar(x + w / 2, enc_c, w, color=S.CALM_ENC, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(item_names, fontsize=7.5, rotation=60, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"F1 — {S.CLASS_SHORT[k]} ({k})", fontsize=11, color=S.ACCENT)
        ax.grid(axis="x", visible=False)
        if k == 0:
            ax.set_ylabel("F1 (per item, per class)")

    # ---- footer: pooled accuracy / MAE / QWK ----
    foot = (f"Pooled (n={overall['cot']['n']}):   "
           f"accuracy  CoT {overall['cot']['accuracy']:.3f}  vs  enc {overall['enc']['accuracy']:.3f}"
           f"     |     macro-F1  CoT {overall['cot']['macro_f1']:.3f}  vs  enc {overall['enc']['macro_f1']:.3f}"
           f"     |     MAE (lower better)  CoT {overall['cot']['mae']:.3f}  vs  enc {overall['enc']['mae']:.3f}"
           f"     |     QWK  CoT {overall['cot']['qwk']:.3f}  vs  enc {overall['enc']['qwk']:.3f}")
    fig.text(0.5, -0.015, foot, ha="center", fontsize=10.5, color=S.MUTED)

    S.save(fig, "fig16_cot_vs_mbert_hybw3")

    out_rows = []
    for iid in items:
        for tag, key in [("cot", "cot"), ("enc", "enc")]:
            m = per_item[key][iid]
            out_rows.append({"item_id": iid, "item_name": S.ITEM_NAMES[iid], "model": tag,
                             "n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                             "mae": m["mae"], "qwk": m["qwk"],
                             "f1_class0": m["f1_per_class"][0], "f1_class1": m["f1_per_class"][1],
                             "f1_class2": m["f1_per_class"][2], "f1_class3": m["f1_per_class"][3]})
    for tag, key in [("cot", "cot"), ("enc", "enc")]:
        m = overall[key]
        out_rows.append({"item_id": 0, "item_name": "OVERALL", "model": tag,
                         "n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                         "mae": m["mae"], "qwk": m["qwk"],
                         "f1_class0": m["f1_per_class"][0], "f1_class1": m["f1_per_class"][1],
                         "f1_class2": m["f1_per_class"][2], "f1_class3": m["f1_per_class"][3]})
    out_df = pd.DataFrame(out_rows)
    cp = S.companion_path("fig16_cot_vs_mbert_hybw3", "csv")
    out_df.to_csv(cp, index=False)
    print(f"  wrote {cp}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
