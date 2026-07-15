"""Fig 15 - Cascade family metrics, per PHQ-8 item.

Same four configs and four metrics as fig14 (macro-F1, QWK, MAE, err>=2), but
broken down by symptom instead of pooled — using the Attention-MIL encoder
(the winning retrieval-processing variant from fig14) for the encoder-dependent
configs. Shows WHERE the merged-gate cascade's aggregate win actually comes
from, and which symptoms still resist every configuration.

All from pooled out-of-fold predictions (n=1752, ~219/item); no GPU.

    python -m src.evaluation.presentation.fig15_cascade_metrics_per_item
"""

import json

import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, mean_absolute_error

from src.evaluation.cot_cascade import load_llm, load_encoder, align, gate_mask, apply_cascade
from src.evaluation.presentation import _style as S

CV = S.ROOT / "outputs" / "cv"
COT = S.ROOT / "outputs" / "cot"
LLM_GLOBS = [str(COT / "folds_tolerant_sc5" / "*.csv"),
             str(COT / "folds_tolerant_sc5_dv" / "*.csv")]
ENC_TAG = "mil_hybw3"   # winning retrieval-processing variant, per fig14
LAB = S.LABELS

CONFIGS = [
    ("LLM alone", "llm", S.CALM_COT),
    ("Encoder · Attention-MIL", "enc", S.CALM_ENC),
    ("Cascade · difficult>=0.8", "casc_diff", S.CALM_CASCADE_ALT),
    ("Cascade · merged(tau.8,>=.5)", "casc_merg", S.CALM_CASCADE),
]
METRICS = [("macro-F1", True), ("QWK", True), ("MAE", False), ("err>=2", False)]


def err_ge2(y, p):
    return float(np.mean(np.abs(y - p) >= 2))


def metrics(y, p):
    return {
        "macro-F1": f1_score(y, p, average="macro", labels=LAB, zero_division=0),
        "QWK": cohen_kappa_score(y, p, weights="quadratic", labels=LAB),
        "MAE": mean_absolute_error(y, p),
        "err>=2": err_ge2(y, p),
    }


def main():
    S.apply_rc()
    llm = load_llm(LLM_GLOBS)
    enc = load_encoder(str(CV / f"oof_predictions_{ENC_TAG}.csv"))
    df = align(llm, enc)
    df["llm"] = df[[f"llm_{c}" for c in LAB]].to_numpy().argmax(1)
    df["enc"] = df["enc_pred"]
    df["casc_diff"] = apply_cascade(df, gate_mask(df, "difficult", 0.8, 0.8))
    df["casc_merg"] = apply_cascade(df, gate_mask(df, "merged", 0.8, 0.5))
    df["item_name"] = df["item_id"].map(S.ITEM_NAMES)

    items = list(range(1, 9))
    item_names = [S.ITEM_NAMES[i] for i in items]

    # table[metric][config_label] = [8 per-item values]
    table = {m: {} for m, _ in METRICS}
    for label, col, _ in CONFIGS:
        per_item = [metrics(g["label"], g[col]) for _, g in df.groupby("item_id")]
        for m, _ in METRICS:
            table[m][label] = [d[m] for d in per_item]

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))
    x = np.arange(len(items))
    nb = len(CONFIGS)
    w = 0.8 / nb

    for ax, (metric, higher_better) in zip(axes.flat, METRICS):
        for k, (label, col, color) in enumerate(CONFIGS):
            vals = table[metric][label]
            ax.bar(x + (k - (nb - 1) / 2) * w, vals, w, label=label,
                   color=color, edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(item_names, fontsize=9.5, rotation=20)
        arrow = "higher better" if higher_better else "lower better"
        ax.set_title(f"{metric}  ({arrow})", fontsize=13, color=S.ACCENT)
        ax.grid(axis="x", visible=False)

    fig.subplots_adjust(top=0.82, hspace=0.4, wspace=0.15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=4, fontsize=10,
               loc="upper center", bbox_to_anchor=(0.5, 0.98), frameon=False)

    fig.suptitle("Per-symptom breakdown: LLM / Encoder(MIL) / Cascade metrics",
                 fontsize=15.5, color=S.ACCENT, y=1.06)
    fig.text(0.5, -0.03,
             "Pooled out-of-fold (n=1752, ~219/item). Encoder = Attention-MIL, the winning retrieval-processing "
             "variant from fig14. Appetite/Sleep/Moving stay hard for every configuration -- the residual ceiling "
             "is intrinsic to those symptoms, not the retrieval or cascade design.",
             ha="center", fontsize=9.5, color=S.MUTED)

    S.save(fig, "fig15_cascade_metrics_per_item", tight=False)

    cp = S.companion_path("fig15_cascade_metrics_per_item", "json")
    out = {"items": item_names, **{m: {k: [round(v, 4) for v in vals] for k, vals in table[m].items()}
                                    for m in table}}
    cp.write_text(json.dumps(out, indent=2))
    print(f"  wrote {cp}")
    for metric, _ in METRICS:
        print(f"\n  {metric}:")
        print(f"    {'item':14s} " + " ".join(f"{c[0][:14]:>15}" for c in CONFIGS))
        for i, nm in enumerate(item_names):
            print(f"    {nm:14s} " + " ".join(f"{table[metric][c[0]][i]:15.3f}" for c in CONFIGS))


if __name__ == "__main__":
    main()
