"""Fig 14 - Cascade x retrieval-processing: all four metrics, side by side.

For each of the four headline model configs (LLM alone / Encoder alone /
Cascade difficult-gate>=0.8 / Cascade merged-gate) and each of the five encoder
variants (plain CORN baseline + the four Bucket-B retrieval-processing
interventions: Attention-MIL, Focal CORN, Balanced CORN, Cross-encoder rerank),
plot macro-F1, QWK, MAE, and err>=2 in a 2x2 grid. LLM alone never touches the
encoder, so its bar is identical across variants (shown for reference).

All from pooled out-of-fold predictions (n=1752); no GPU.

    python -m src.evaluation.presentation.fig14_cascade_retrieval_metrics
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
LAB = S.LABELS

VARIANTS = [
    ("Baseline CORN", "ctxm_corn_hybw3", S.CALM_SLATE),
    ("Attention-MIL", "mil_hybw3", S.CALM_ENC),
    ("Focal CORN", "ctxm_cornfocal_hybw3", S.CALM_CASCADE_ALT),
    ("Balanced CORN", "ctxm_cornbal_hybw3", S.CALM_GOLD),
    ("Rerank + CORN", "ctxm_corn_rerank_w3", S.CALM_ROSE),
]
CONFIGS = ["LLM alone", "Encoder alone", "Cascade\ndifficult>=0.8", "Cascade\nmerged(tau.8,>=.5)"]
CONFIG_TICKS = ["LLM\nalone", "Encoder\nalone", "Cascade\ndifficult>=0.8", "Cascade\nmerged"]
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

    # {variant_name: {config_name: {metric: value}}}
    table = {}
    for vname, tag, _ in VARIANTS:
        enc = load_encoder(str(CV / f"oof_predictions_{tag}.csv"))
        df = align(llm, enc)
        y = df["label"].to_numpy()
        llm_pred = df[[f"llm_{c}" for c in LAB]].to_numpy().argmax(1)
        enc_pred = df["enc_pred"].to_numpy()
        casc_diff = apply_cascade(df, gate_mask(df, "difficult", 0.8, 0.8))
        casc_merg = apply_cascade(df, gate_mask(df, "merged", 0.8, 0.5))
        table[vname] = {
            "LLM alone": metrics(y, llm_pred),
            "Encoder alone": metrics(y, enc_pred),
            "Cascade\ndifficult>=0.8": metrics(y, casc_diff),
            "Cascade\nmerged(tau.8,>=.5)": metrics(y, casc_merg),
        }

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    x = np.arange(len(CONFIGS))
    nb = len(VARIANTS)
    w = 0.8 / nb

    for ax, (metric, higher_better) in zip(axes.flat, METRICS):
        for k, (vname, _, color) in enumerate(VARIANTS):
            vals = [table[vname][cfg][metric] for cfg in CONFIGS]
            ax.bar(x + (k - (nb - 1) / 2) * w, vals, w, label=vname,
                   color=color, edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(CONFIG_TICKS, fontsize=10)
        arrow = "higher better" if higher_better else "lower better"
        ax.set_title(f"{metric}  ({arrow})", fontsize=13, color=S.ACCENT)
        ax.grid(axis="x", visible=False)

    fig.subplots_adjust(top=0.82, hspace=0.35, wspace=0.15)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=5, fontsize=10,
               loc="upper center", bbox_to_anchor=(0.5, 0.98), frameon=False)

    fig.suptitle("Cascade family x retrieval-processing variant: macro-F1 / QWK / MAE / err>=2",
                 fontsize=15.5, color=S.ACCENT, y=1.06)
    fig.text(0.5, -0.02,
             "Pooled out-of-fold (n=1752). LLM alone never touches the encoder, so it is identical across variants "
             "(reference bar). Attention-MIL is the best encoder for the merged-gate cascade on every metric.",
             ha="center", fontsize=10, color=S.MUTED)

    S.save(fig, "fig14_cascade_retrieval_metrics", tight=False)

    cp = S.companion_path("fig14_cascade_retrieval_metrics", "json")
    cp.write_text(json.dumps(table, indent=2))
    print(f"  wrote {cp}")
    for vname, _, _ in VARIANTS:
        print(f"\n  {vname}:")
        for cfg in CONFIGS:
            m = table[vname][cfg]
            print(f"    {cfg.replace(chr(10), ' '):28s} "
                  f"F1={m['macro-F1']:.3f} QWK={m['QWK']:.3f} MAE={m['MAE']:.3f} err2={m['err>=2']:.3f}")


if __name__ == "__main__":
    main()
