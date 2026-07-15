"""Fig 3b — retrieval trajectory: how expanded retrieval (R0 -> R1) moves each
PHQ-8 item in the evidence-quality x predictability plane.

Companion to fig3 (which is left untouched as the frozen baseline). Where fig3
shows the encoder->cascade *model* transition at a fixed retrieval, this shows
the *retrieval* transition (old Hybrid-W3 -> new expanded exphyb_bm25q) for a
fixed model configuration:

    open dot   = R0 (old retrieval): (LLM-judged informative rate, per-item F1)
    filled dot = R1 (expanded retrieval)
    arrow      = the 2-D move (evidence-quality change x predictability change)

Unlike fig3 the arrow is generally diagonal: expanded retrieval can shift BOTH
the evidence quality (x) and the resulting F1 (y). Arrow colour encodes the sign
of the F1 gain (green = up, red = down).

Reads outputs/cot/retrieval_effect_analysis.json (so the numbers match the
results tables exactly) and prefers the best available config: the MIL+merged
cascade if its R1 is ready, else the ctxm_corn encoder.

    python -m src.evaluation.presentation.fig3b_retrieval_trajectory
"""
import json

import numpy as np

from src.evaluation.presentation import _style as S

ANALYSIS = S.ROOT / "outputs" / "cot" / "retrieval_effect_analysis.json"
CONFIG_PREFERENCE = ["cascade_MIL_merged", "encoder_MIL", "encoder_ctxm_corn"]
CONFIG_LABELS = {"cascade_MIL_merged": "Attention-MIL merged-gate cascade",
                 "encoder_MIL": "Attention-MIL encoder",
                 "encoder_ctxm_corn": "MentalBERT+CORN encoder"}


def main():
    S.apply_rc()
    a = json.loads(ANALYSIS.read_text())
    cfg = next((c for c in CONFIG_PREFERENCE if c in a["configs"]), None)
    if cfg is None:
        raise SystemExit("no config available in retrieval_effect_analysis.json")
    evq = a["evidence_quality"]["per_item"]
    pi0 = a["configs"][cfg]["R0"]["per_item"]
    pi1 = a["configs"][cfg]["R1"]["per_item"]

    items = [n for n in evq if n in pi0 and n in pi1]
    x0 = np.array([evq[n]["R0"] for n in items])
    x1 = np.array([evq[n]["R1"] for n in items])
    y0 = np.array([pi0[n]["macro_f1"] for n in items])
    y1 = np.array([pi1[n]["macro_f1"] for n in items])
    df1 = y1 - y0

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11.0, 7.0))

    # per-item trajectory arrows (diagonal: evidence x AND F1 y can both move)
    for i, n in enumerate(items):
        col = S.GOOD if df1[i] >= 0 else S.BAD
        ax.annotate("", (x1[i], y1[i]), (x0[i], y0[i]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2,
                                    mutation_scale=16, alpha=0.9,
                                    shrinkA=6, shrinkB=6), zorder=2)

    ax.scatter(x0, y0, s=150, facecolors="white", edgecolors=S.MUTED,
               linewidths=1.6, zorder=3, label="R0 · Hybrid-W3 (old retrieval)")
    ax.scatter(x1, y1, s=150, c=[S.GOOD if d >= 0 else S.BAD for d in df1],
               edgecolors="white", linewidths=1.4, zorder=4,
               label="R1 · expanded (exphyb_bm25q)")

    nudges = {"Sleep": (0.008, 0.008), "Depressed": (0.008, -0.012),
              "Moving": (0.012, 0.004), "Appetite": (0.012, -0.006),
              "Concentrating": (-0.006, 0.014), "NoInterest": (-0.006, 0.012),
              "Tired": (0.010, 0.008), "Failure": (0.010, -0.010)}
    for i, n in enumerate(items):
        dx, dy = nudges.get(n, (0.01, 0.01))
        ax.annotate(f"{n}\n{df1[i]:+.3f} F1", (x1[i], y1[i]),
                    (x1[i] + dx, y1[i] + dy), fontsize=9, fontweight="bold",
                    color=S.ACCENT, ha="left" if dx >= 0 else "right", va="center")

    md = a["evidence_quality"]["mean_delta"]
    mf1 = float(np.mean(df1))
    ax.set_xlabel("Evidence quality — LLM-judged informative rate (P(supports ∪ against))  →")
    ax.set_ylabel("Predictability — per-item macro-F1  →")
    ax.set_title(f"Retrieval trajectory: Hybrid-W3 → expanded  ·  {CONFIG_LABELS[cfg]}",
                 color=S.ACCENT)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.text(0.01, 0.99,
            f"mean Δ evidence {md:+.3f}   ·   mean Δ F1 {mf1:+.3f}\n"
            f"green = F1 improved, red = declined",
            transform=ax.transAxes, va="top", fontsize=9.5, color=S.MUTED,
            style="italic")

    S.save(fig, "fig3b_retrieval_trajectory")

    # companion CSV: superset of fig3's columns for the moved config
    import pandas as pd
    rows = [{"item_name": n,
             "llm_informative_rate_old": round(float(x0[i]), 4),
             "llm_informative_rate_new": round(float(x1[i]), 4),
             "f1_old": round(float(y0[i]), 4), "f1_new": round(float(y1[i]), 4),
             "delta_f1": round(float(df1[i]), 4)} for i, n in enumerate(items)]
    out = pd.DataFrame(rows)
    out["config"] = cfg
    cp = S.companion_path("fig3b_retrieval_trajectory", "csv")
    out.to_csv(cp, index=False)
    print(f"  wrote {cp}  (config={cfg})")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
