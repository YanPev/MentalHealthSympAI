"""Fig 9 - Are the headline gains real? Cluster-bootstrap confidence intervals.

With only 219 participants, point estimates need error bars. We resample
*participants* with replacement (cluster bootstrap — items within a participant
are correlated) and recompute each metric delta 2000 times, giving a 95% CI on
every headline improvement. A comparison is "significant" when the CI excludes 0.

Comparisons (each swaps one factor on a fixed base model):
  * Retrieval (BERT-base):  Hybrid-W3 − Isolated   on total-QWK and macro-F1
  * Fusion (MentalBERT):    CoT − Encoder           on macro-F1
                            Severity-routed − Encoder on macro-F1

    python -m src.evaluation.presentation.fig9_bootstrap_ci
"""

import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score

from src.evaluation.presentation import _style as S

CV = S.ROOT / "outputs" / "cv"
W5_DIR = S.ROOT / "outputs" / "cot" / "folds_w5"
SC_DIR = S.ROOT / "outputs" / "cot" / "folds_sc5"
KEY = ["participant_id", "item_id"]
PROBS = [f"prob_{c}" for c in S.LABELS]
N_BOOT = 2000
RNG = np.random.default_rng(42)


def load(tag):
    df = pd.read_csv(CV / f"oof_predictions_{tag}.csv")
    df["participant_id"] = df["participant_id"].astype(str)
    return df


def load_cot_dir(d, name):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(d / "*.csv")))],
                   ignore_index=True)
    df["participant_id"] = df["participant_id"].astype(str)
    return df[KEY + ["prediction"]].rename(columns={"prediction": name})


def macro_f1(y, p):
    return f1_score(y, p, average="macro", labels=S.LABELS, zero_division=0)


def total_qwk(part_tbl, col):
    return cohen_kappa_score(part_tbl["true_tot"].round().astype(int),
                             part_tbl[col].round().astype(int), weights="quadratic")


def bootstrap_item(merged, colA, colB):
    """Cluster bootstrap of macro-F1(B) − macro-F1(A) over participants."""
    by_p = {pid: g for pid, g in merged.groupby("participant_id")}
    pids = np.array(list(by_p.keys()))
    deltas = np.empty(N_BOOT)
    for b in range(N_BOOT):
        samp = RNG.choice(pids, size=len(pids), replace=True)
        rows = pd.concat([by_p[p] for p in samp], ignore_index=True)
        y = rows["label"].to_numpy()
        deltas[b] = macro_f1(y, rows[colB].to_numpy()) - macro_f1(y, rows[colA].to_numpy())
    return deltas


def bootstrap_total(part_tbl, colA, colB):
    """Cluster bootstrap of total-QWK(B) − total-QWK(A) over participants."""
    pids = part_tbl["participant_id"].to_numpy()
    deltas = np.empty(N_BOOT)
    for b in range(N_BOOT):
        samp = RNG.choice(np.arange(len(pids)), size=len(pids), replace=True)
        s = part_tbl.iloc[samp]
        deltas[b] = total_qwk(s, colB) - total_qwk(s, colA)
    return deltas


def summ(d):
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(d.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
            "p_gt0": float((d > 0).mean()), "sig": bool(lo > 0 or hi < 0)}


def main():
    S.apply_rc()
    rows = []  # (label, group, summary-dict)

    # ---- retrieval (BERT-base): Hybrid-W3 vs Isolated ----------------------
    old, hyb = load("ctx_corn_old"), load("ctx_corn_hybw3")
    m = old[KEY + ["label"]].merge(
        old[KEY + ["prediction"]].rename(columns={"prediction": "old"}), on=KEY).merge(
        hyb[KEY + ["prediction"]].rename(columns={"prediction": "hyb"}), on=KEY)
    rows.append(("Hybrid − Isolated  ·  macro-F1", "Retrieval (BERT-base)",
                 summ(bootstrap_item(m, "old", "hyb"))))
    # total-QWK
    pt = m.groupby("participant_id").agg(true_tot=("label", "sum"),
                                         old=("old", "sum"), hyb=("hyb", "sum")).reset_index()
    rows.append(("Hybrid − Isolated  ·  total-QWK", "Retrieval (BERT-base)",
                 summ(bootstrap_total(pt, "old", "hyb"))))

    # ---- fusion (MentalBERT): CoT & routing vs encoder ---------------------
    enc = load("ctxm_corn_hybw3")
    sc = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(SC_DIR / "*.csv")))],
                   ignore_index=True)
    sc["participant_id"] = sc["participant_id"].astype(str)
    sc_pred = sc[KEY].copy()
    Pn = sc[PROBS].to_numpy(float); Pn = Pn / Pn.sum(1, keepdims=True).clip(min=1e-9)
    sc_pred["cot"] = Pn.argmax(1)
    w5 = load_cot_dir(W5_DIR, "w5")
    mf = enc[KEY + ["label", "prediction"]].rename(columns={"prediction": "enc"}) \
        .merge(sc_pred, on=KEY).merge(w5, on=KEY)
    mf["routed"] = np.where(mf["w5"].to_numpy() >= 2, mf["w5"].to_numpy(), mf["enc"].to_numpy())
    rows.append(("CoT − Encoder  ·  macro-F1", "Fusion (MentalBERT)",
                 summ(bootstrap_item(mf, "enc", "cot"))))
    rows.append(("Severity-routed − Encoder  ·  macro-F1", "Fusion (MentalBERT)",
                 summ(bootstrap_item(mf, "enc", "routed"))))

    # ---- forest plot (split by metric scale) --------------------------------
    import matplotlib.pyplot as plt
    f1_rows = [r for r in rows if "macro-F1" in r[0]]
    qwk_rows = [r for r in rows if "total-QWK" in r[0]]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 4.6),
                                   gridspec_kw={"width_ratios": [1.7, 1]})

    def forest(ax, rws, xlim, title, pad):
        yps = np.arange(len(rws))[::-1]
        for yp, (lab, grp, s) in zip(yps, rws):
            col = S.GOOD if s["sig"] else S.MUTED
            ax.errorbar(s["delta"], yp,
                        xerr=[[s["delta"] - s["ci_lo"]], [s["ci_hi"] - s["delta"]]],
                        fmt="o", color=col, ecolor=col, elinewidth=2.2, capsize=5, ms=9)
            ax.text(s["delta"], yp + 0.18,
                    f"+{s['delta']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"
                    + ("  ✓" if s["sig"] else "  n.s."),
                    va="bottom", ha="center", fontsize=9,
                    color=S.ACCENT if s["sig"] else S.MUTED)
        ax.axvline(0, color=S.BAD, lw=1.4, ls="--")
        ax.set_yticks(yps)
        ax.set_yticklabels([r[0].split("·")[0].strip() for r in rws], fontsize=10)
        ax.set_xlim(*xlim); ax.set_ylim(-0.6, len(rws) - 0.2)
        ax.set_xlabel("Δ  with 95% cluster-bootstrap CI")
        ax.set_title(title, color=S.ACCENT, fontsize=12.5)

    forest(axL, f1_rows, (-0.04, 0.16), "macro-F1 improvements", 0.004)
    forest(axR, qwk_rows, (-0.05, 0.70), "total-QWK improvement", 0.02)
    axL.text(0.0, -0.55, "no improvement", color=S.BAD, fontsize=8.5, ha="center")

    fig.suptitle("Do the gains survive uncertainty? Cluster bootstrap over 219 participants (2000 resamples)",
                 fontsize=13.5, fontweight="bold", color=S.ACCENT, y=1.03)
    fig.text(0.5, -0.06,
             "Retrieval gains are large and unambiguous; the CoT macro-F1 edge alone is not significant, "
             "while severity-routing's small gain is.",
             ha="center", fontsize=9.5, color=S.MUTED)
    S.save(fig, "fig9_bootstrap_ci")
    out = {"n_boot": N_BOOT, "n_participants": int(pt.shape[0]),
           "comparisons": [{"label": lab, "group": grp, **s} for lab, grp, s in rows]}
    cp = S.companion_path("fig9_bootstrap_ci", "json")
    cp.write_text(json.dumps(out, indent=2))
    print(f"  wrote {cp}")
    for lab, grp, s in rows:
        print(f"  {lab:42s} Δ={s['delta']:+.3f} [{s['ci_lo']:+.3f},{s['ci_hi']:+.3f}] "
              f"{'sig' if s['sig'] else 'n.s.'}")


if __name__ == "__main__":
    main()
