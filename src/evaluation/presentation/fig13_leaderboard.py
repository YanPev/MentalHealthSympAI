"""Fig 13 - Corrected configuration leaderboard (macro-F1), W5 encoder baseline.

Includes the confidence-gated cascade family (the project's actual best), built
on the pooled 10-chain self-consistency tolerant LLM + the MentalBERT+CORN
hybrid-W5 encoder. Horizontal bars by macro-F1; QWK / MAE annotated; the W5
encoder is the baseline reference; the cascade difficult-gate is starred as best.

Reconstructs the cascade from cot_cascade (no GPU). Honest note: the cascade's
edge over LLM-alone is within the cluster-bootstrap CIs; only the gap vs the
encoder is clearly significant.

    python -m src.evaluation.presentation.fig13_leaderboard
"""

import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score, mean_absolute_error

from src.evaluation.cot_cascade import load_llm, load_encoder, align, gate_mask, apply_cascade
from src.evaluation.presentation import _style as S

CV = S.ROOT / "outputs" / "cv"
COT = S.ROOT / "outputs" / "cot"
LLM_GLOBS = [str(COT / "folds_tolerant_sc5" / "*.csv"),
             str(COT / "folds_tolerant_sc5_dv" / "*.csv")]
ENC_W5 = CV / "oof_predictions_ctxm_corn_hybw5.csv"
LAB = S.LABELS
KEY = ["participant_id", "item_id"]


def m(y, p):
    return (f1_score(y, p, average="macro", labels=LAB, zero_division=0),
            cohen_kappa_score(y, p, weights="quadratic", labels=LAB),
            mean_absolute_error(y, p))


def main():
    S.apply_rc()
    # --- cascade family (pooled 10-chain LLM + W5 encoder) ---
    llm = load_llm(LLM_GLOBS)
    enc = load_encoder(str(ENC_W5))
    df = align(llm, enc)
    y = df["label"].to_numpy()
    llm_pred = df[[f"llm_{c}" for c in LAB]].to_numpy().argmax(1)
    casc_diff = apply_cascade(df, gate_mask(df, "difficult", 0.8, 0.8))
    casc_merg = apply_cascade(df, gate_mask(df, "merged", 0.8, 0.5))
    enc_w5_pred = df["enc_pred"].to_numpy()

    # --- severity-routed (W5 encoder + W5 CoT gate), for context ---
    w5cot = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(COT / "folds_w5" / "*.csv")))],
                      ignore_index=True)
    w5cot["participant_id"] = w5cot["participant_id"].astype(str)
    enc_full = pd.read_csv(ENC_W5); enc_full["participant_id"] = enc_full["participant_id"].astype(str)
    rdf = enc_full[KEY + ["label", "prediction"]].rename(columns={"prediction": "enc"}).merge(
        w5cot[KEY + ["prediction"]].rename(columns={"prediction": "w5"}), on=KEY)
    routed = np.where(rdf["w5"].to_numpy() >= 2, rdf["w5"].to_numpy(), rdf["enc"].to_numpy())

    # (label, predictions, family-colour)
    rows = [
        ("Encoder · MentalBERT+CORN+W5  (baseline)", enc_w5_pred, y, S.ENC_COLOR),
        ("Severity-routed ensemble", routed, rdf["label"].to_numpy(), S.WARN),
        ("Cascade · merged-gate (τ.8, diff≥.5)", casc_merg, y, "#0d9488"),
        ("LLM-alone · pooled 10-chain SC", llm_pred, y, S.COT_COLOR),
        ("Cascade · difficult-gate (semantic)", casc_diff, y, S.ENS_COLOR),
    ]
    data = []
    for label, p, yy, color in rows:
        f1, qwk, mae = m(yy, p)
        data.append({"label": label, "F1": f1, "QWK": qwk, "MAE": mae, "color": color})
    data.sort(key=lambda d: d["F1"])
    best = max(range(len(data)), key=lambda i: data[i]["F1"])
    base_f1 = next(d["F1"] for d in data if "baseline" in d["label"])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    yps = np.arange(len(data))
    ax.barh(yps, [d["F1"] for d in data], color=[d["color"] for d in data],
            edgecolor="white", height=0.62, zorder=3)
    ax.axvline(base_f1, ls="--", color=S.ENC_COLOR, lw=1.3, zorder=2)
    ax.text(base_f1 - 0.004, 1.5, "W5 encoder baseline ", color=S.ENC_COLOR,
            fontsize=9, va="center", ha="right", rotation=90)
    for i, d in enumerate(data):
        star = "  ★" if i == best else ""
        ax.text(d["F1"] + 0.004, i, f"{d['F1']:.3f}{star}", va="center",
                fontsize=11, fontweight="bold" if i == best else "normal", color=S.ACCENT)
        ax.text(0.008, i, d["label"], va="center", ha="left", fontsize=10.5,
                color="white", fontweight="bold", zorder=4)
        ax.text(d["F1"] + 0.045, i, f"QWK {d['QWK']:.3f}   MAE {d['MAE']:.3f}",
                va="center", fontsize=8.8, color=S.MUTED)
    ax.set_yticks([]); ax.set_xlim(0, 0.47)
    ax.set_xlabel("macro-F1 (pooled out-of-fold, n=1752)")
    ax.set_title("Configuration leaderboard — the confidence-gated cascade is the project's best",
                 color=S.ACCENT, fontsize=14.5, pad=14)
    ax.grid(axis="y", visible=False)
    fig.text(0.5, -0.04,
             "★ best macro-F1. Cascade = pooled 10-chain SC LLM, handing the ~6% of items it flags as "
             "‘difficult’ to the W5 encoder. Honest caveat: cascade vs LLM-alone is within the bootstrap "
             "CIs; only the gap vs the encoder is clearly significant.",
             ha="center", fontsize=9.2, color=S.MUTED)
    S.save(fig, "fig13_leaderboard")

    cp = S.companion_path("fig13_leaderboard", "json")
    cp.write_text(json.dumps({"baseline": "ctxm_corn_hybw5",
                              "configs": [{k: d[k] for k in ("label", "F1", "QWK", "MAE")} for d in data]},
                             indent=2))
    print(f"  wrote {cp}")
    for d in reversed(data):
        print(f"  {d['label']:42s} F1={d['F1']:.3f} QWK={d['QWK']:.3f} MAE={d['MAE']:.3f}")


if __name__ == "__main__":
    main()
