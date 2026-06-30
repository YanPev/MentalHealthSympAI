"""Fig 12 - Per-item macro-F1 for the best model configurations.

Grouped bars: for each PHQ-8 symptom, the macro-F1 of the headline configs —
the trained encoder, the per-item CoT, the best single model (attention-MIL),
and the best overall system (severity-routed ensemble). Shows which method wins
which symptom (e.g. encoder still owns the dissociable Appetite/Moving items).

All from pooled out-of-fold predictions (n=1752); no GPU.

    python -m src.evaluation.presentation.fig12_per_item_best
"""

import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.evaluation.cot_cascade import load_llm, load_encoder, align, gate_mask, apply_cascade
from src.evaluation.presentation import _style as S

CV = S.ROOT / "outputs" / "cv"
COT = S.ROOT / "outputs" / "cot"
LLM_GLOBS = [str(COT / "folds_tolerant_sc5" / "*.csv"),
             str(COT / "folds_tolerant_sc5_dv" / "*.csv")]
ENC_W5 = CV / "oof_predictions_ctxm_corn_hybw5.csv"
KEY = ["participant_id", "item_id"]
LAB = S.LABELS


def main():
    S.apply_rc()
    # cascade family (pooled 10-chain LLM + W5 encoder) — reconstruct per-item preds
    llm = load_llm(LLM_GLOBS)
    enc = load_encoder(str(ENC_W5))
    df = align(llm, enc)
    df["llm"] = df[[f"llm_{c}" for c in LAB]].to_numpy().argmax(1)
    df["enc"] = df["enc_pred"]
    df["casc_diff"] = apply_cascade(df, gate_mask(df, "difficult", 0.8, 0.8))
    df["casc_merg"] = apply_cascade(df, gate_mask(df, "merged", 0.8, 0.5))
    df["item_name"] = df["item_id"].map(S.ITEM_NAMES)

    configs = [
        ("Encoder · MentalBERT+CORN+W5", "enc", S.ENC_COLOR),
        ("LLM-alone · pooled 10-chain", "llm", S.COT_COLOR),
        ("Cascade · merged-gate", "casc_merg", "#0d9488"),
        ("Cascade · difficult-gate", "casc_diff", S.ENS_COLOR),
    ]
    items = list(range(1, 9))
    item_names = [S.ITEM_NAMES[i] for i in items]

    # per-item macro-F1 for each config
    table = {}
    for label, col, _ in configs:
        table[label] = [
            f1_score(g["label"], g[col], average="macro", labels=LAB, zero_division=0)
            for _, g in df.groupby("item_id")]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    x = np.arange(len(items))
    nb = len(configs); w = 0.8 / nb
    for k, (label, col, color) in enumerate(configs):
        vals = table[label]
        bars = ax.bar(x + (k - (nb - 1) / 2) * w, vals, w, label=label,
                      color=color, edgecolor="white", linewidth=0.6)
        # mark the per-item winner
        for xi, v in zip(x, vals):
            if abs(v - max(table[c][xi] for c in table)) < 1e-9:
                ax.text(xi + (k - (nb - 1) / 2) * w, v + 0.008, "★",
                        ha="center", fontsize=10, color=S.ACCENT)

    ax.set_xticks(x)
    ax.set_xticklabels(item_names, fontsize=11)
    ax.set_ylabel("macro-F1 (per item)")
    ax.set_xlabel("PHQ-8 symptom")
    ax.set_ylim(0, 0.62)
    ax.set_title("Per-symptom macro-F1 of the best model configurations  (★ = best per item)",
                 color=S.ACCENT, fontsize=15)
    ax.legend(ncol=4, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.grid(axis="x", visible=False)

    fig.text(0.5, -0.03,
             "Pooled out-of-fold (n=1752). Cascade = pooled 10-chain SC LLM with the W5 encoder breaking ties on "
             "the ~6% it flags ‘difficult’. The LLM/cascade lead most symptoms; the encoder still keeps the dissociable items.",
             ha="center", fontsize=10, color=S.MUTED)

    S.save(fig, "fig12_per_item_best")

    out = {"items": item_names,
           "macro_f1_per_item": {label: [round(v, 4) for v in table[label]]
                                 for label, _, _ in configs}}
    cp = S.companion_path("fig12_per_item_best", "json")
    cp.write_text(json.dumps(out, indent=2))
    print(f"  wrote {cp}")
    print(f"  {'item':14s} " + " ".join(f"{c[0][:10]:>11}" for c in configs))
    for i, nm in enumerate(item_names):
        print(f"  {nm:14s} " + " ".join(f"{table[c[0]][i]:11.3f}" for c in configs))


if __name__ == "__main__":
    main()
