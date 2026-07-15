"""Fig 16 - Current best models x encoder complementarity (fig4, updated cast).

fig4 compared the ORIGINAL CoT probe against the plain-CORN encoder. This
repeats that exact "who's right by true severity" comparison for the CURRENT
two best models, each against the encoder (Attention-MIL -- the winning
retrieval-processing variant, per fig14):

Left  : Encoder(MIL) vs LLM alone (pooled 10-chain self-consistency, tolerant
        prompt, no training).
Right : Encoder(MIL) vs Cascade merged-gate (tau=.8, diff>=.5) -- the best
        model overall.

Since the cascade already uses the encoder as its tiebreak on ~41.5% of items,
the right panel is expected to look different from the left: the cascade
should almost never lose to the encoder on items it deferred to the encoder
for, so any "encoder-only right" cases on the right come only from the
~58.5% the cascade did NOT route.

    python -m src.evaluation.presentation.fig16_vs_encoder_complementarity
"""

import json

import numpy as np

from src.evaluation.cot_cascade import load_llm, load_encoder, align, gate_mask, apply_cascade
from src.evaluation.presentation import _style as S

CV = S.ROOT / "outputs" / "cv"
COT = S.ROOT / "outputs" / "cot"
LLM_GLOBS = [str(COT / "folds_tolerant_sc5" / "*.csv"),
             str(COT / "folds_tolerant_sc5_dv" / "*.csv")]
ENC_TAG = "mil_hybw3"   # winning retrieval-processing variant, per fig14
LAB = S.LABELS


def complementarity(y, enc_pred, other_pred):
    agree = float((enc_pred == other_pred).mean())
    enc_ok, other_ok = (enc_pred == y), (other_pred == y)
    disagree = enc_pred != other_pred
    enc_only = disagree & enc_ok & ~other_ok
    other_only = disagree & other_ok & ~enc_ok
    by = lambda mask: [int((mask & (y == k)).sum()) for k in S.LABELS]
    return {
        "agree": agree,
        "disagree": float(disagree.mean()),
        "enc_only_by_class": by(enc_only),
        "other_only_by_class": by(other_only),
        "enc_only_total": int(enc_only.sum()),
        "other_only_total": int(other_only.sum()),
    }


def draw_panel(ax, res, other_label, other_color, title):
    x = np.arange(len(S.LABELS))
    w = 0.4
    ax.bar(x - w/2, res["enc_only_by_class"], w, color=S.CALM_ENC, edgecolor="white",
           label="Encoder right, other wrong")
    ax.bar(x + w/2, res["other_only_by_class"], w, color=other_color, edgecolor="white",
           label=f"{other_label} right, encoder wrong")
    for xi, (a, b) in enumerate(zip(res["enc_only_by_class"], res["other_only_by_class"])):
        ax.text(xi - w/2, a + 0.6, str(a), ha="center", fontsize=9, color=S.CALM_ENC)
        ax.text(xi + w/2, b + 0.6, str(b), ha="center", fontsize=9, color=other_color,
                fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(S.CLASS_SHORT)
    ax.set_xlabel("True severity class")
    ax.set_ylabel("# items where exactly one is right")
    ax.set_title(title, color=S.ACCENT, fontsize=12.6)
    ax.legend(fontsize=8.6, loc="upper right")


def main():
    S.apply_rc()
    llm = load_llm(LLM_GLOBS)
    enc = load_encoder(str(CV / f"oof_predictions_{ENC_TAG}.csv"))
    df = align(llm, enc)
    y = df["label"].to_numpy()
    llm_pred = df[[f"llm_{c}" for c in LAB]].to_numpy().argmax(1)
    enc_pred = df["enc_pred"].to_numpy()
    casc_merged = apply_cascade(df, gate_mask(df, "merged", 0.8, 0.5))

    res_llm = complementarity(y, enc_pred, llm_pred)
    res_casc = complementarity(y, enc_pred, casc_merged)

    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    draw_panel(axL, res_llm, "LLM alone", S.CALM_COT,
               f"Encoder(MIL) vs LLM alone -- agree {res_llm['agree']*100:.0f}%")
    draw_panel(axR, res_casc, "Cascade merged", S.CALM_CASCADE,
               f"Encoder(MIL) vs Cascade merged-gate -- agree {res_casc['agree']*100:.0f}%")
    ymax = max(max(res_llm["enc_only_by_class"] + res_llm["other_only_by_class"]),
              max(res_casc["enc_only_by_class"] + res_casc["other_only_by_class"])) * 1.15
    axL.set_ylim(0, ymax); axR.set_ylim(0, ymax)

    fig.subplots_adjust(top=0.80, wspace=0.28)
    fig.suptitle(
        "Both current-best models still disagree with the encoder on severity, "
        "same way the original CoT probe did (fig4)",
        fontsize=14, fontweight="bold", color=S.ACCENT, y=0.98)
    S.save(fig, "fig16_vs_encoder_complementarity", tight=False)

    out = {
        "n": int(len(df)),
        "encoder_vs_llm_alone": res_llm,
        "encoder_vs_cascade_merged": res_casc,
    }
    cp = S.companion_path("fig16_vs_encoder_complementarity", "json")
    cp.write_text(json.dumps(out, indent=2))
    print(f"  wrote {cp}")
    print(f"  Encoder vs LLM-alone:      agree={res_llm['agree']:.3f} "
          f"enc-only={res_llm['enc_only_total']} other-only={res_llm['other_only_total']}")
    print(f"  Encoder vs Cascade-merged: agree={res_casc['agree']:.3f} "
          f"enc-only={res_casc['enc_only_total']} other-only={res_casc['other_only_total']}")


if __name__ == "__main__":
    main()
