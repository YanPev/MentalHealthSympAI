"""Fig 4 - CoT x encoder complementarity (the flagship result).

A frozen LLM probe (Qwen-7B Chain-of-Thought, no task training) and a trained
encoder (MentalBERT+CORN) reach similar accuracy but DISAGREE on ~55% of items.
Among disagreements, who is right depends on severity: CoT disproportionately
rescues the severe classes the encoder under-calls. Because their errors differ,
blending them helps.

Left  : among disagreements, "encoder-only right" vs "CoT-only right" by true class.
Right : macro-F1 vs blend weight (from outputs/cot/cot_ensemble_metrics.json),
        with both parents and the severity-routed ensemble marked.

    python -m src.evaluation.presentation.fig4_cot_complementarity
"""

import glob
import json

import numpy as np
import pandas as pd

from src.evaluation.presentation import _style as S

ENC_OOF = S.ROOT / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"
COT_GREEDY_DIR = S.ROOT / "outputs" / "cot" / "folds"
COT_GLOB = "cot_probe_qwen_hybw3_fold*.csv"
ENS_JSON = S.ROOT / "outputs" / "cot" / "cot_ensemble_metrics.json"
KEY = ["participant_id", "item_id"]


def load_paired():
    enc = pd.read_csv(ENC_OOF)
    enc["participant_id"] = enc["participant_id"].astype(str)
    enc = enc[KEY + ["label", "prediction"]].rename(columns={"prediction": "enc_pred"})

    files = sorted(glob.glob(str(COT_GREEDY_DIR / COT_GLOB)))
    cot = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cot["participant_id"] = cot["participant_id"].astype(str)
    cot = cot[KEY + ["prediction"]].rename(columns={"prediction": "cot_pred"})

    df = enc.merge(cot, on=KEY)
    assert len(df) == len(enc), f"merge mismatch {len(df)} vs {len(enc)}"
    return df


def main():
    S.apply_rc()
    df = load_paired()
    y = df["label"].to_numpy()
    e, c = df["enc_pred"].to_numpy(), df["cot_pred"].to_numpy()

    agree = float((e == c).mean())
    enc_ok, cot_ok = (e == y), (c == y)
    disagree = e != c
    enc_only = disagree & enc_ok & ~cot_ok
    cot_only = disagree & cot_ok & ~enc_ok

    # by true severity class + overall
    classes = S.LABELS
    enc_only_by = [int((enc_only & (y == k)).sum()) for k in classes]
    cot_only_by = [int((cot_only & (y == k)).sum()) for k in classes]

    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.6),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- (a) complementary wins by severity ---------------------------------
    x = np.arange(len(classes))
    w = 0.4
    axL.bar(x - w/2, enc_only_by, w, color=S.ENC_COLOR, edgecolor="white",
            label="Encoder right, CoT wrong")
    axL.bar(x + w/2, cot_only_by, w, color=S.COT_COLOR, edgecolor="white",
            label="CoT right, Encoder wrong")
    for xi, (a, b) in enumerate(zip(enc_only_by, cot_only_by)):
        axL.text(xi - w/2, a + 1, str(a), ha="center", fontsize=9, color=S.ENC_COLOR)
        axL.text(xi + w/2, b + 1, str(b), ha="center", fontsize=9, color=S.COT_COLOR,
                 fontweight="bold")
    axL.set_xticks(x); axL.set_xticklabels(S.CLASS_SHORT)
    axL.set_xlabel("True severity class")
    axL.set_ylabel("# items where exactly one model is right")
    axL.set_title("Encoder hedges to the middle; CoT commits to the extremes",
                  color=S.ACCENT)
    axL.legend(fontsize=9.5, loc="upper right")
    # severe spotlight
    axL.annotate("on severe items, CoT-only\nwins far outnumber encoder-only",
                 (3, cot_only_by[3]), (2.1, cot_only_by[3] + 6),
                 fontsize=9, color=S.COT_COLOR, ha="center",
                 arrowprops=dict(arrowstyle="->", color=S.COT_COLOR, lw=1.3))
    # encoder-wins-the-middle spotlight
    axL.annotate("encoder wins\nthe Mild middle", (1, enc_only_by[1]),
                 (1.05, enc_only_by[1] + 35), fontsize=9, color=S.ENC_COLOR,
                 ha="center", arrowprops=dict(arrowstyle="->", color=S.ENC_COLOR, lw=1.3))

    # ---- (b) blend-weight sweep ---------------------------------------------
    ens = json.loads(ENS_JSON.read_text())
    sweep = ens["weight_sweep_macro_f1"]
    ws = [p[0] for p in sweep]; fs = [p[1] for p in sweep]
    m = ens["methods"]
    enc_f1, cot_f1 = m["encoder (CORN)"]["macro_f1"], m["CoT greedy"]["macro_f1"]
    route_f1 = m["severity-routed"]["macro_f1"]
    peak_i = int(np.argmax(fs))

    axR.plot(ws, fs, "-o", color=S.ENS_COLOR, lw=2, ms=4, label="50/50-style prob blend")
    axR.axhline(enc_f1, ls="--", color=S.ENC_COLOR, lw=1.4,
                label=f"Encoder alone ({enc_f1:.3f})")
    axR.axhline(cot_f1, ls="--", color=S.COT_COLOR, lw=1.4,
                label=f"CoT alone ({cot_f1:.3f})")
    axR.scatter([ws[peak_i]], [fs[peak_i]], s=120, color=S.ENS_COLOR, zorder=5,
                edgecolors="white", lw=1.5)
    axR.annotate(f"blend peak {fs[peak_i]:.3f}\n(w={ws[peak_i]:.2f})",
                 (ws[peak_i], fs[peak_i]), (ws[peak_i] - 0.02, fs[peak_i] + 0.004),
                 fontsize=9, ha="center", color=S.ACCENT)
    axR.axhline(route_f1, ls=":", color=S.BAD, lw=1.8,
                label=f"Severity-routed ({route_f1:.3f}) — beats both")
    axR.text(0.5, route_f1 + 0.0015, "severity-routing captures both strengths",
             ha="center", fontsize=8.6, color=S.BAD)
    axR.set_xlabel("blend weight w  (0 = all encoder → 1 = all CoT)")
    axR.set_ylabel("macro-F1")
    axR.set_title("Combining them captures both strengths", color=S.ACCENT)
    axR.set_ylim(min(fs) - 0.01, max(route_f1, max(fs)) + 0.012)
    axR.legend(fontsize=8.8, loc="lower center", ncol=1)

    fig.suptitle(
        f"Two minds, different routes: CoT and the encoder agree only {agree*100:.0f}% of the time — and blend productively",
        fontsize=14.5, fontweight="bold", color=S.ACCENT, y=1.02)
    S.save(fig, "fig4_cot_complementarity")

    out = {
        "n": int(len(df)),
        "agreement_rate": agree,
        "disagreement_rate": float(disagree.mean()),
        "encoder_only_correct_by_class": dict(zip(map(str, classes), enc_only_by)),
        "cot_only_correct_by_class": dict(zip(map(str, classes), cot_only_by)),
        "encoder_only_total": int(enc_only.sum()),
        "cot_only_total": int(cot_only.sum()),
        "parents_and_blend_macro_f1": {
            "encoder": enc_f1, "cot": cot_f1,
            "blend_peak": fs[peak_i], "blend_peak_w": ws[peak_i],
            "severity_routed": route_f1,
        },
        "severe_class_f1": {
            "encoder": m["encoder (CORN)"]["f1_per_class"][3],
            "cot": m["CoT greedy"]["f1_per_class"][3],
        },
    }
    cp = S.companion_path("fig4_cot_complementarity", "json")
    cp.write_text(json.dumps(out, indent=2))
    print(f"  wrote {cp}")
    print(f"  agreement={agree:.3f}  enc-only={int(enc_only.sum())} cot-only={int(cot_only.sum())}")
    print(f"  severe F1: enc={out['severe_class_f1']['encoder']:.3f} cot={out['severe_class_f1']['cot']:.3f}")


if __name__ == "__main__":
    main()
