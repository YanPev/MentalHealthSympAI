"""Fig 18 - Do the summed 8-item predictions land in the right PHQ-8 severity band?

Every figure so far scores the model per item (0-3, four classes). But the number
a clinician reads off PHQ-8 is the TOTAL: sum the 8 items (0-24) and map it to a
severity band. Item-level errors can cancel or compound when summed, so a model
that looks good per item can still misplace a patient's overall severity. This
figure closes that loop -- it sums each participant's 8 predicted items, sums the
8 true items, and asks whether the predicted total falls in the SAME PHQ-8 band.

PHQ-8 total -> severity band (the standard cut-points):
     0- 4  Minimal            10-14  Moderate           20-24  Severe
     5- 9  Mild               15-19  Moderately severe

Two models, side by side (rows):
  * MentalBERT+CORN+Hybrid-W3 (encoder alone)
        = outputs/cv/oof_predictions_ctxm_corn_hybw3.csv
  * Attention-MIL merged-gate cascade (project best)
        = pooled 10-chain SC LLM (folds_tolerant_sc5[_dv]), Attention-MIL encoder
          (oof_predictions_mil_hybw3.csv) breaks ties on merged-gate items
          (tau=0.8, diff>=0.5) -- the exact config fig14 flags as project best.

Left panel: predicted total vs true total scatter, band grid shaded; a point is
green when its predicted band matches the true band, red otherwise. Right panel:
5x5 band confusion matrix (row = true band).

    python -m src.evaluation.presentation.fig18_total_score_band_agreement
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

from src.evaluation.cot_cascade import (align, apply_cascade, gate_mask,
                                        load_encoder, load_llm)
from src.evaluation.presentation import _style as S

CV = S.ROOT / "outputs" / "cv"
COT = S.ROOT / "outputs" / "cot"

ENC_W3 = CV / "oof_predictions_ctxm_corn_hybw3.csv"          # MentalBERT+CORN+Hybrid-W3
MIL_ENC = CV / "oof_predictions_mil_hybw3.csv"               # Attention-MIL encoder
LLM_GLOBS = [str(COT / "folds_tolerant_sc5" / "*.csv"),
             str(COT / "folds_tolerant_sc5_dv" / "*.csv")]

# PHQ-8 severity bands: (low, high) inclusive on the 0-24 total.
BAND_EDGES = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24)]
BAND_NAMES = ["Minimal\n0-4", "Mild\n5-9", "Moderate\n10-14",
              "Mod. severe\n15-19", "Severe\n20-24"]
N_BANDS = len(BAND_EDGES)

MODELS = [
    ("MentalBERT+CORN+Hybrid-W3 (encoder alone)", S.CALM_ENC),
    ("Attention-MIL merged-gate cascade (project best)", S.CALM_CASCADE),
]


def to_band(total):
    """Map a 0-24 PHQ-8 total to a severity-band index 0..4."""
    return int(min(total // 5, N_BANDS - 1))


def participant_totals(df, pred_col):
    """Sum the 8 item scores per participant -> true & predicted totals + bands.

    Guards that every participant contributes exactly 8 items, so a partial join
    can't silently deflate a total into the wrong band."""
    g = df.groupby("participant_id")
    counts = g.size()
    if not (counts == 8).all():
        bad = counts[counts != 8]
        raise ValueError(f"{len(bad)} participants without 8 items: {bad.to_dict()}")
    out = pd.DataFrame({
        "participant_id": counts.index,
        "true_total": g["label"].sum().values,
        "pred_total": g[pred_col].sum().values,
    })
    out["true_band"] = out["true_total"].map(to_band)
    out["pred_band"] = out["pred_total"].map(to_band)
    return out


def load_mbert_w3():
    df = pd.read_csv(ENC_W3)
    df["participant_id"] = df["participant_id"].astype(str)
    return participant_totals(df, "prediction")


def load_mil_cascade():
    llm = load_llm(LLM_GLOBS)
    enc = load_encoder(str(MIL_ENC))
    df = align(llm, enc)
    df["casc"] = apply_cascade(df, gate_mask(df, "merged", 0.8, 0.5))
    return participant_totals(df.rename(columns={"casc": "prediction"}), "prediction")


def band_metrics(t):
    yt, yp = t["true_band"].to_numpy(), t["pred_band"].to_numpy()
    tt, tp = t["true_total"].to_numpy(), t["pred_total"].to_numpy()
    exact = float(np.mean(yt == yp))
    adjacent = float(np.mean(np.abs(yt - yp) <= 1))
    return {
        "n": int(len(t)),
        "band_exact": exact,
        "band_adjacent": adjacent,
        "band_qwk": float(cohen_kappa_score(yt, yp, weights="quadratic",
                                            labels=list(range(N_BANDS)))),
        "total_mae": float(mean_absolute_error(tt, tp)),
        "total_bias": float(np.mean(tp - tt)),
    }


def confusion(t):
    m = np.zeros((N_BANDS, N_BANDS), dtype=int)
    for a, b in zip(t["true_band"], t["pred_band"]):
        m[a, b] += 1
    return m


def main():
    S.apply_rc()
    totals = {
        MODELS[0][0]: load_mbert_w3(),
        MODELS[1][0]: load_mil_cascade(),
    }
    mets = {name: band_metrics(t) for name, t in totals.items()}
    for name, m in mets.items():
        print(f"{name}: {m}")

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=0.34, wspace=0.26)

    for r, (name, color) in enumerate(MODELS):
        t = totals[name]
        m = mets[name]

        # ---- left: predicted vs true total, band grid, hit/miss coloured ----
        ax = fig.add_subplot(gs[r, 0])
        hit = t["true_band"].to_numpy() == t["pred_band"].to_numpy()
        jx = rng.uniform(-0.18, 0.18, len(t))
        jy = rng.uniform(-0.18, 0.18, len(t))
        for edge in [4.5, 9.5, 14.5, 19.5]:
            ax.axhline(edge, color=S.MUTED, lw=0.7, ls=":", zorder=1)
            ax.axvline(edge, color=S.MUTED, lw=0.7, ls=":", zorder=1)
        ax.plot([0, 24], [0, 24], color=S.ACCENT, lw=1.0, ls="--", zorder=2,
                label="perfect (pred = true)")
        ax.scatter(t["true_total"] + jx, t["pred_total"] + jy,
                   c=np.where(hit, S.GOOD, S.BAD), s=26, alpha=0.72,
                   edgecolor="white", linewidth=0.3, zorder=3)
        ax.set_xlim(-0.8, 24.8)
        ax.set_ylim(-0.8, 24.8)
        ax.set_xlabel("True PHQ-8 total (sum of 8 items)")
        ax.set_ylabel("Predicted PHQ-8 total")
        ax.set_title(name, color=color, fontsize=13)
        # band tick labels on both axes
        centers = [(lo + hi) / 2 for lo, hi in BAND_EDGES]
        short = ["Min", "Mild", "Mod", "M.sev", "Sev"]
        ax.set_xticks(centers); ax.set_xticklabels(short, fontsize=9)
        ax.set_yticks(centers); ax.set_yticklabels(short, fontsize=9)
        txt = (f"n={m['n']}   band-exact {m['band_exact']*100:.1f}%   "
               f"within±1 band {m['band_adjacent']*100:.1f}%\n"
               f"band QWK {m['band_qwk']:.3f}   total MAE {m['total_mae']:.2f}   "
               f"bias {m['total_bias']:+.2f}")
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                fontsize=9.5, color=S.ACCENT,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=0.9))
        if r == 0:
            handles = [Patch(fc=S.GOOD, label="same band (correct)"),
                       Patch(fc=S.BAD, label="wrong band")]
            ax.legend(handles=handles, loc="lower right", fontsize=9)

        # ---- right: 5x5 band confusion matrix (row = true band) ----
        axc = fig.add_subplot(gs[r, 1])
        cm = confusion(t)
        row = cm.sum(1, keepdims=True)
        frac = np.divide(cm, row, out=np.zeros_like(cm, float), where=row > 0)
        im = axc.imshow(frac, cmap=S.SEQ_BLUES, vmin=0, vmax=1, aspect="auto")
        for i in range(N_BANDS):
            for j in range(N_BANDS):
                if cm[i, j] == 0:
                    continue
                axc.text(j, i, f"{cm[i, j]}\n{frac[i, j]*100:.0f}%",
                         ha="center", va="center", fontsize=8.5,
                         color="white" if frac[i, j] > 0.55 else S.ACCENT)
        # outline the diagonal (correct band)
        for i in range(N_BANDS):
            axc.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                        edgecolor=color, lw=2.0, zorder=5))
        axc.set_xticks(range(N_BANDS)); axc.set_yticks(range(N_BANDS))
        axc.set_xticklabels(BAND_NAMES, fontsize=7.5)
        axc.set_yticklabels(BAND_NAMES, fontsize=7.5)
        axc.set_xlabel("Predicted band")
        axc.set_ylabel("True band")
        axc.set_title("Severity-band confusion (row-normalised)", color=color,
                      fontsize=12)
        axc.grid(False)

    fig.suptitle("Fig 18 · Does the summed 8-item prediction land in the right PHQ-8 severity band?",
                 fontsize=16, color=S.ACCENT, y=0.975)
    foot = ("Per-item scores (0-3) are summed to a 0-24 total, then binned into the "
            "five standard PHQ-8 bands. Diagonal outline = correct band; off-diagonal "
            "= a severity misclassification a clinician would see. "
            "Within±1 band tolerates a neighbouring-band slip.")
    fig.text(0.5, 0.045, foot, ha="center", fontsize=10, color=S.MUTED, wrap=True)

    S.save(fig, "fig18_total_score_band_agreement", tight=False)

    # ---- companion data ----
    rows = []
    for name, t in totals.items():
        tt = t.copy()
        tt.insert(0, "model", name)
        rows.append(tt)
    per_part = pd.concat(rows, ignore_index=True)
    cp_csv = S.companion_path("fig18_total_score_band_agreement", "csv")
    per_part.to_csv(cp_csv, index=False)
    print(f"  wrote {cp_csv}")

    payload = {
        "band_edges": BAND_EDGES,
        "band_names": [b.replace("\n", " ") for b in BAND_NAMES],
        "models": {name: {"metrics": mets[name],
                          "confusion_true_x_pred": confusion(t).tolist()}
                   for name, t in totals.items()},
    }
    cp_json = S.companion_path("fig18_total_score_band_agreement", "json")
    cp_json.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {cp_json}")


if __name__ == "__main__":
    main()
