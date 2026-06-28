"""Fig 6 - Calibrated fusion: why the naive blend failed, and what fixes it.

Direction #1 follow-up. The learned stacker (cot_stacker.py) already edges the
parents on severe-class F1 but loses MAE because (a) the base probabilities are
mis-calibrated and (b) argmax decoding ignores ordinality. Here we:

  1. temperature-scale the encoder + CoT probabilities (per-fold, leakage-free)
     and show the reliability curve + ECE before/after;
  2. fuse the *calibrated* probabilities with a nested-CV blend weight;
  3. decode ordinally (expected-value rounding) instead of argmax.

Everything is OOF over the 5 folds (n=1752); no GPU.

    python -m src.evaluation.presentation.fig6_calibrated_fusion
"""

import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.evaluation.build_cot_report import metric_block, LABELS
from src.evaluation.presentation import _style as S

ENC_OOF = S.ROOT / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"
SC_DIR = S.ROOT / "outputs" / "cot" / "folds_sc5"
W5_DIR = S.ROOT / "outputs" / "cot" / "folds_w5"
KEY = ["participant_id", "item_id"]
PROBS = [f"prob_{c}" for c in LABELS]
EPS = 1e-9


def load_dir(d):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(d / "*.csv")))],
                   ignore_index=True)
    df["participant_id"] = df["participant_id"].astype(str)
    return df


def softmax_T(logp, T):
    z = logp / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(P, y, grid=np.linspace(0.3, 5.0, 95)):
    """Scalar T minimising NLL on (P, y). P are probabilities -> use log P as logits."""
    logp = np.log(P + EPS)
    best_T, best_nll = 1.0, np.inf
    for T in grid:
        q = softmax_T(logp, T)
        nll = -np.log(q[np.arange(len(y)), y] + EPS).mean()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def ece(P, y, n_bins=10):
    conf = P.max(1)
    pred = P.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e, xs, accs, confs = 0.0, [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        a, c = acc[m].mean(), conf[m].mean()
        e += (m.mean()) * abs(a - c)
        xs.append((lo + hi) / 2); accs.append(a); confs.append(c)
    return float(e), np.array(xs), np.array(accs), np.array(confs)


def ordinal_decode(P):
    """Expected-value rounding: respects the ordinal scale (better MAE/QWK)."""
    ev = P @ np.array(LABELS, dtype=float)
    return np.clip(np.rint(ev), 0, 3).astype(int)


def main():
    S.apply_rc()
    enc = pd.read_csv(ENC_OOF)
    enc["participant_id"] = enc["participant_id"].astype(str)
    df = enc[KEY + ["label", "fold", "prediction"]].rename(columns={"prediction": "enc_pred"})
    enc_P = enc[PROBS].to_numpy(float)
    enc_P = enc_P / enc_P.sum(1, keepdims=True).clip(min=EPS)

    sc = load_dir(SC_DIR)
    df = df.merge(sc[KEY + PROBS], on=KEY)
    cot_P = df[PROBS].to_numpy(float)
    cot_P = cot_P / cot_P.sum(1, keepdims=True).clip(min=EPS)
    w5 = load_dir(W5_DIR)[KEY + ["prediction"]].rename(columns={"prediction": "w5_pred"})
    df = df.merge(w5, on=KEY)

    y = df["label"].to_numpy()
    folds = df["fold"].to_numpy()

    # ---- per-fold temperature scaling (leakage-free) + nested blend weight ----
    enc_cal = np.zeros_like(enc_P)
    cot_cal = np.zeros_like(cot_P)
    blend_pred = np.zeros(len(y), dtype=int)
    grid_w = np.round(np.linspace(0, 1, 21), 2)
    for f in sorted(set(folds)):
        tr, te = folds != f, folds == f
        Te = fit_temperature(enc_P[tr], y[tr])
        Tc = fit_temperature(cot_P[tr], y[tr])
        enc_cal[te] = softmax_T(np.log(enc_P[te] + EPS), Te)
        cot_cal[te] = softmax_T(np.log(cot_P[te] + EPS), Tc)
        enc_tr = softmax_T(np.log(enc_P[tr] + EPS), Te)
        cot_tr = softmax_T(np.log(cot_P[tr] + EPS), Tc)
        # choose blend weight maximising macro-F1 on training folds (ordinal decode)
        best_w, best = 0.5, -1
        for w in grid_w:
            p = ordinal_decode(w * cot_tr + (1 - w) * enc_tr)
            s = f1_score(y[tr], p, average="macro", labels=LABELS, zero_division=0)
            if s > best:
                best, best_w = s, w
        blend_pred[te] = ordinal_decode(best_w * cot_cal[te] + (1 - best_w) * enc_cal[te])

    # ---- ECE before / after (encoder) ----
    ece_raw, xr, accr, confr = ece(enc_P, y)
    ece_cal, xc, accc, confc = ece(enc_cal, y)

    # ---- method comparison ----
    w5c = df["w5_pred"].to_numpy()
    routed = np.where(w5c >= 2, w5c, df["enc_pred"].to_numpy())
    # learned stacker (conditions on item + probs) from cot_stacker.py
    stk_path = S.ROOT / "outputs" / "cot" / "cot_stacker_metrics.json"
    stk = json.loads(stk_path.read_text())["methods"]["stacker (LogReg, nested-CV)"]
    methods = {
        "Encoder\n(CORN)": metric_block(y, df["enc_pred"].to_numpy()),
        "CoT\n(self-consist.)": metric_block(y, cot_P.argmax(1)),
        "Calibrated\nscalar blend": metric_block(y, blend_pred),
        "Learned stacker\n(cond. on item)": stk,
        "Severity-routed\n(cond. on severity)": metric_block(y, routed),
    }

    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    # ---- (a) reliability diagram --------------------------------------------
    axL.plot([0, 1], [0, 1], ls="--", color=S.MUTED, lw=1.2, label="perfect calibration")
    axL.plot(confr, accr, "-o", color=S.BAD, ms=5, label=f"raw encoder (ECE {ece_raw:.3f})")
    axL.plot(confc, accc, "-o", color=S.GOOD, ms=5,
             label=f"temperature-scaled (ECE {ece_cal:.3f})")
    axL.set_xlabel("predicted confidence"); axL.set_ylabel("empirical accuracy")
    axL.set_xlim(0, 1); axL.set_ylim(0, 1)
    axL.set_title("Calibration: probabilities become trustworthy", color=S.ACCENT)
    axL.legend(fontsize=9, loc="upper left")
    axL.text(0.97, 0.05, f"ECE {ece_raw:.3f} → {ece_cal:.3f}\n({(1-ece_cal/ece_raw)*100:.0f}% lower)",
             ha="right", fontsize=10, color=S.ACCENT, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f0fdf4", ec=S.GOOD))

    # ---- (b) method comparison: F1 + QWK ------------------------------------
    names = list(methods.keys())
    x = np.arange(len(names)); w = 0.38
    f1s = [methods[n]["macro_f1"] for n in names]
    qwks = [methods[n]["qwk"] for n in names]
    cols = [S.ENC_COLOR, S.COT_COLOR, S.MUTED, S.ENS_COLOR, S.WARN]
    axR.bar(x - w/2, f1s, w, color=cols, edgecolor="white", label="macro-F1")
    axR.bar(x + w/2, qwks, w, color=cols, edgecolor="white", alpha=0.55, hatch="//",
            label="QWK")
    for xi, (a, b) in enumerate(zip(f1s, qwks)):
        axR.text(xi - w/2, a + 0.006, f"{a:.3f}", ha="center", fontsize=8.5, fontweight="bold")
        axR.text(xi + w/2, b + 0.006, f"{b:.3f}", ha="center", fontsize=8.5, color=S.MUTED)
    axR.axhline(f1s[0], ls=":", color=S.ENC_COLOR, lw=1, zorder=0)
    axR.set_xticks(x); axR.set_xticklabels(names, fontsize=8.8)
    axR.set_ylabel("score"); axR.set_ylim(0, 0.46)
    axR.set_title("A scalar blend can't capture it — conditional combiners can\n(solid=F1, hatched=QWK)",
                  color=S.ACCENT, fontsize=12.5)
    axR.legend(fontsize=9, loc="upper right")
    axR.annotate("global blend ≈ encoder\n(no gain)", (2, f1s[2]), (2, 0.40),
                 fontsize=8.5, color=S.MUTED, ha="center",
                 arrowprops=dict(arrowstyle="->", color=S.MUTED, lw=1.1))

    fig.suptitle("Calibrate to trust the probabilities — but the combiner must condition, not just blend",
                 fontsize=13.5, fontweight="bold", color=S.ACCENT, y=1.04)
    S.save(fig, "fig6_calibrated_fusion")

    out = {
        "n": int(len(y)),
        "ece_encoder_raw": ece_raw,
        "ece_encoder_calibrated": ece_cal,
        "methods": {n.replace("\n", " "): m for n, m in methods.items()},
    }
    cp = S.companion_path("fig6_calibrated_fusion", "json")
    cp.write_text(json.dumps(out, indent=2))
    print(f"  wrote {cp}")
    print(f"  ECE {ece_raw:.3f} -> {ece_cal:.3f}")
    for n, m in methods.items():
        print(f"  {n.replace(chr(10),' '):34s} F1={m['macro_f1']:.3f} QWK={m['qwk']:.3f} "
              f"MAE={m['mae']:.3f} c3={m['f1_per_class'][3]:.3f}")


if __name__ == "__main__":
    main()
