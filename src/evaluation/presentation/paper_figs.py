"""Paper figure set (data-driven figures 1-9).

Builds, from existing OOF predictions (no GPU):
  fig_context_strategy_comparison, fig_peritem_class_error_profile,
  fig_joint_vs_staged, fig_whole_picture_progression, fig_severe_safety_frontier,
  fig_sc_run_variance, fig_final_complementarity, fig_final_class_profile,
  fig_final_bootstrap_ci

    python -m src.evaluation.presentation.paper_figs
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.presentation import _style as S
from src.evaluation.presentation import _paperdata as D
from src.evaluation.cot_cascade import (load_llm, load_encoder, align as Calign,
                                        gate_mask, apply_cascade)

LAB = D.LAB
CN = S.CLASS_SHORT                      # ['Minimal','Mild','Moderate','Severe']
ITEMS = [S.ITEM_NAMES[i] for i in range(1, 9)]
LLM_GLOBS = [str(D.COT / "folds_tolerant_sc5" / "*.csv"),
             str(D.COT / "folds_tolerant_sc5_dv" / "*.csv")]


def _save(fig, name):
    S.save(fig, name)


# ----------------------------------------------------------------- fig 1
def fig_context_strategy_comparison():
    S.apply_rc()
    strat = [("Utterances W3", D.load_dirs("folds"), "#94a3b8"),
             ("Windows W5", D.load_dirs("folds_w5"), S.ENS_COLOR),
             ("Full transcript", D.load_dirs("folds_fulltranscript"), S.COT_COLOR),
             ("Item-adaptive", D.load_dirs("folds_itemadaptive"), "#0d9488")]
    M = {nm: D.metrics(d.label, d.prediction) for nm, d, _ in strat}
    cols = [c for _, _, c in strat]; names = [nm for nm, _, _ in strat]
    panels = [("macro-F1", lambda m: m["macro_f1"], False),
              ("QWK", lambda m: m["qwk"], False),
              ("MAE", lambda m: m["mae"], True),
              ("Far-off rate", lambda m: m["far_off"], True),
              ("Class-3 F1", lambda m: m["f1_per_class"][3], False)]
    fig, axes = plt.subplots(1, 5, figsize=(15, 4.4))
    x = np.arange(len(names))
    for ax, (title, fn, lower) in zip(axes, panels):
        vals = [fn(M[nm]) for nm in names]
        ax.bar(x, vals, color=cols, edgecolor="white")
        best = np.argmin(vals) if lower else np.argmax(vals)
        for xi, v in enumerate(vals):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold" if xi == best else "normal")
        ax.set_title(title + ("  ↓" if lower else "  ↑"), fontsize=12, color=S.ACCENT)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.suptitle("Context-retrieval strategy comparison (CoT, Qwen-7B)  —  ↑ higher better, ↓ lower better",
                 fontsize=14, fontweight="bold", color=S.ACCENT, y=1.04)
    _save(fig, "fig_context_strategy_comparison")


# ----------------------------------------------------------------- fig 2
def fig_peritem_class_error_profile():
    S.apply_rc()
    enc = D.load_enc(D.ENC_W3); cot = D.load_dirs("folds")
    me, mc = D.metrics(enc.label, enc.prediction), D.metrics(cot.label, cot.prediction)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.4, 1]})
    # per-class F1
    x = np.arange(4); w = 0.38
    axL.bar(x - w/2, me["f1_per_class"], w, color=S.ENC_COLOR, label="Encoder-W3")
    axL.bar(x + w/2, mc["f1_per_class"], w, color=S.COT_COLOR, label="CoT-W3")
    for xi in x:
        axL.text(xi - w/2, me["f1_per_class"][xi] + .006, f"{me['f1_per_class'][xi]:.2f}", ha="center", fontsize=8.5)
        axL.text(xi + w/2, mc["f1_per_class"][xi] + .006, f"{mc['f1_per_class'][xi]:.2f}", ha="center", fontsize=8.5)
    axL.set_xticks(x); axL.set_xticklabels(CN); axL.set_ylabel("F1"); axL.set_ylim(0, 0.7)
    axL.set_title("Per-class F1 — Encoder vs CoT (W3)", color=S.ACCENT)
    axL.legend(fontsize=10)
    axL.text(0.02, 0.95, f"Encoder  MAE {me['mae']:.2f} · far-off {me['far_off']:.0%}\n"
             f"CoT      MAE {mc['mae']:.2f} · far-off {mc['far_off']:.0%}",
             transform=axL.transAxes, fontsize=9.5, va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="#eef6f6", ec="none"))
    # severe under/over-call
    cats = ["Severe\nUNDER-called\n(missed)", "Severe\nOVER-called\n(false)"]
    enc_v = [me["severe_undercall"], me["severe_overcall"]]
    cot_v = [mc["severe_undercall"], mc["severe_overcall"]]
    xx = np.arange(2)
    axR.bar(xx - w/2, enc_v, w, color=S.ENC_COLOR, label="Encoder-W3")
    axR.bar(xx + w/2, cot_v, w, color=S.COT_COLOR, label="CoT-W3")
    for xi in xx:
        axR.text(xi - w/2, enc_v[xi] + 1, str(enc_v[xi]), ha="center", fontsize=9)
        axR.text(xi + w/2, cot_v[xi] + 1, str(cot_v[xi]), ha="center", fontsize=9, fontweight="bold")
    axR.set_xticks(xx); axR.set_xticklabels(cats, fontsize=9.5)
    axR.set_ylabel("# items"); axR.set_title("Severe-class errors", color=S.ACCENT)
    axR.text(0.5, 0.93, f"of {me['n_true_severe']} true-severe items", transform=axR.transAxes,
             ha="center", fontsize=8.5, color=S.MUTED)
    axR.legend(fontsize=9.5)
    fig.suptitle("Where the encoder and CoT differ: the severe class",
                 fontsize=14.5, fontweight="bold", color=S.ACCENT, y=1.0)
    _save(fig, "fig_peritem_class_error_profile")


# ----------------------------------------------------------------- fig 3
def fig_joint_vs_staged():
    S.apply_rc()
    joint = D.load_dirs("folds_joint"); staged = D.load_dirs("folds_staged")
    mj, ms = D.metrics(joint.label, joint.prediction), D.metrics(staged.label, staged.prediction)
    panels = [("Accuracy", "accuracy", False), ("Macro-F1", "macro_f1", False),
              ("QWK", "qwk", False), ("MAE", "mae", True),
              ("Class-3 F1", None, False), ("Over-call rate", "over_call", True)]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4))
    for ax, (title, key, lower) in zip(axes.ravel(), panels):
        vj = mj["f1_per_class"][3] if key is None else mj[key]
        vs = ms["f1_per_class"][3] if key is None else ms[key]
        ax.bar([0, 1], [vj, vs], color=[S.ENS_COLOR, "#0d9488"], width=0.6, edgecolor="white")
        for xi, v in zip([0, 1], [vj, vs]):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Joint", "Staged"])
        ax.set_title(title + ("  ↓" if lower else "  ↑"), fontsize=12, color=S.ACCENT)
        ax.set_ylim(0, max(vj, vs) * 1.22)
    fig.suptitle("Reasoning structure matters: joint vs staged CoT (same LLM, same evidence)",
                 fontsize=14, fontweight="bold", color=S.ACCENT, y=1.02)
    _save(fig, "fig_joint_vs_staged")


# ----------------------------------------------------------------- fig 4
def fig_whole_picture_progression():
    S.apply_rc()
    stages = [("Joint", D.load_dirs("folds_joint")),
              ("Staged", D.load_dirs("folds_staged")),
              ("Staged-v2", D.load_dirs("folds_staged_v2")),
              ("Tolerant", D.load_dirs("folds_tolerant")),
              ("SC×5", D.load_dirs("folds_tolerant_sc5")),
              ("Pooled SC×10", D.load_dirs("folds_tolerant_sc5", "folds_tolerant_sc5_dv"))]
    M = [D.metrics(d.label, d.prediction) for _, d in stages]
    names = [n for n, _ in stages]; x = np.arange(len(names))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    # performance
    axL.plot(x, [m["macro_f1"] for m in M], "-o", color=S.COT_COLOR, lw=2.2, ms=7, label="macro-F1")
    axL.plot(x, [m["qwk"] for m in M], "-s", color=S.ENS_COLOR, lw=2.2, ms=6, label="QWK")
    for xi, m in zip(x, M):
        axL.text(xi, m["macro_f1"] + .006, f"{m['macro_f1']:.3f}", ha="center", fontsize=8.5, color=S.COT_COLOR)
    axL.set_xticks(x); axL.set_xticklabels(names, rotation=25, ha="right", fontsize=9.5)
    axL.set_ylabel("score"); axL.set_title("Performance", color=S.ACCENT); axL.legend(fontsize=10)
    # error profile (twin axis: MAE left, far-off right)
    axR.plot(x, [m["mae"] for m in M], "-o", color=S.BAD, lw=2.2, ms=7, label="MAE (↓)")
    axR.set_ylabel("MAE", color=S.BAD); axR.tick_params(axis="y", colors=S.BAD)
    ax2 = axR.twinx()
    ax2.plot(x, [m["far_off"] for m in M], "-^", color=S.WARN, lw=2.2, ms=7, label="far-off (↓)")
    ax2.set_ylabel("far-off rate", color=S.WARN); ax2.tick_params(axis="y", colors=S.WARN)
    axR.set_xticks(x); axR.set_xticklabels(names, rotation=25, ha="right", fontsize=9.5)
    axR.set_title("Error profile", color=S.ACCENT)
    l1, la1 = axR.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    axR.legend(l1 + l2, la1 + la2, fontsize=9.5, loc="upper right")
    fig.suptitle("The whole-picture reasoning family: from joint to pooled self-consistency",
                 fontsize=14, fontweight="bold", color=S.ACCENT, y=1.02)
    _save(fig, "fig_whole_picture_progression")


# ----------------------------------------------------------------- fig 5
def fig_severe_safety_frontier():
    S.apply_rc()
    pts = [("Encoder (W5)", D.load_enc(D.ENC_W5), S.ENC_COLOR),
           ("Staged", D.load_dirs("folds_staged"), "#0d9488"),
           ("Staged-v2", D.load_dirs("folds_staged_v2"), S.WARN),
           ("Tolerant", D.load_dirs("folds_tolerant"), S.COT_COLOR),
           ("SC (pooled×10)", D.load_dirs("folds_tolerant_sc5", "folds_tolerant_sc5_dv"), S.ENS_COLOR)]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    for nm, d, col in pts:
        m = D.metrics(d.label, d.prediction)
        x, y, f1 = m["f1_per_class"][3], m["far_off"], m["macro_f1"]
        ax.scatter(x, y, s=(f1 ** 2) * 9000, c=col, edgecolors="white", linewidths=1.8,
                   alpha=0.92, zorder=3)
        ax.annotate(f"{nm}\nmacro-F1 {f1:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, -34), ha="center", va="top", fontsize=9.5,
                    fontweight="bold", color=col, zorder=4)
    ax.set_xlabel("Class-3 (severe) F1  →  better severe detection")
    ax.set_ylabel("Far-off (≥2) error rate  →  less safe")
    ax.invert_yaxis()  # safer (low far-off) at top
    ax.set_title("The severe-safety frontier  (bubble size = macro-F1)", color=S.ACCENT)
    ax.text(0.02, 0.04, "↑ safer (fewer catastrophic misses)\n→ better at catching severe cases",
            transform=ax.transAxes, fontsize=9.5, color=S.MUTED, style="italic", va="bottom")
    _save(fig, "fig_severe_safety_frontier")


# ----------------------------------------------------------------- fig 6
def fig_sc_run_variance():
    S.apply_rc()
    runs = [("SC×5 run 1", D.load_dirs("folds_tolerant_sc5"), "#a78bfa"),
            ("SC×5 run 2", D.load_dirs("folds_tolerant_sc5_dv"), "#c4b5fd"),
            ("Pooled SC×10", D.load_dirs("folds_tolerant_sc5", "folds_tolerant_sc5_dv"), S.COT_COLOR)]
    M = {nm: D.metrics(d.label, d.prediction) for nm, d, _ in runs}
    names = [n for n, _, _ in runs]; cols = [c for _, _, c in runs]
    metrics3 = [("macro-F1", "macro_f1"), ("QWK", "qwk"), ("MAE", "mae")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, (title, key) in zip(axes, metrics3):
        vals = [M[n][key] for n in names]
        ax.bar(np.arange(3), vals, color=cols, edgecolor="white", width=0.62)
        for xi, v in enumerate(vals):
            ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(np.arange(3)); ax.set_xticklabels(names, rotation=18, ha="right", fontsize=9)
        ax.set_title(title, fontsize=12.5, color=S.ACCENT)
        ax.set_ylim(0, max(vals) * 1.2)
    fig.suptitle("Self-consistency run-to-run variance — pooling two SC×5 runs (→10 chains) stabilises the vote",
                 fontsize=13, fontweight="bold", color=S.ACCENT, y=1.03)
    _save(fig, "fig_sc_run_variance")


# ------------------------------------------------ cascade reconstruction
def _final_frame():
    llm = load_llm(LLM_GLOBS)
    enc = load_encoder(str(D.ENC_W5))
    df = Calign(llm, enc)
    df["llm"] = df[[f"llm_{c}" for c in LAB]].to_numpy().argmax(1)
    df["enc"] = df["enc_pred"]
    df["casc_diff"] = apply_cascade(df, gate_mask(df, "difficult", 0.8, 0.8))
    df["casc_merg"] = apply_cascade(df, gate_mask(df, "merged", 0.8, 0.5))
    df["item_name"] = df["item_id"].map(S.ITEM_NAMES)
    return df


# ----------------------------------------------------------------- fig 7
def fig_final_complementarity():
    S.apply_rc()
    df = _final_frame()
    y = df["label"].to_numpy(); e = df["enc"].to_numpy(); c = df["llm"].to_numpy()
    agree = (e == c).mean()
    dis = e != c; eo = dis & (e == y) & (c != y); lo = dis & (c == y) & (e != y)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), gridspec_kw={"width_ratios": [1, 1.25, 0.8]})
    # by class
    x = np.arange(4); w = 0.38
    eo_c = [int((eo & (y == k)).sum()) for k in LAB]; lo_c = [int((lo & (y == k)).sum()) for k in LAB]
    axes[0].bar(x - w/2, eo_c, w, color=S.ENC_COLOR, label="Encoder-only right")
    axes[0].bar(x + w/2, lo_c, w, color=S.COT_COLOR, label="LLM-only right")
    axes[0].set_xticks(x); axes[0].set_xticklabels(CN, fontsize=9.5)
    axes[0].set_title("Disagreement wins · by class", color=S.ACCENT); axes[0].legend(fontsize=9)
    axes[0].set_ylabel("# items where exactly one is right")
    # by item
    xi = np.arange(8)
    eo_i = [int((eo & (df["item_id"].to_numpy() == it)).sum()) for it in range(1, 9)]
    lo_i = [int((lo & (df["item_id"].to_numpy() == it)).sum()) for it in range(1, 9)]
    axes[1].bar(xi - w/2, eo_i, w, color=S.ENC_COLOR); axes[1].bar(xi + w/2, lo_i, w, color=S.COT_COLOR)
    axes[1].set_xticks(xi); axes[1].set_xticklabels(ITEMS, rotation=35, ha="right", fontsize=8.5)
    axes[1].set_title("Disagreement wins · by symptom", color=S.ACCENT)
    # routed subset (difficult_frac>=0.8)
    routed = df["difficult_frac"].to_numpy() >= 0.8
    acc_e = (e[routed] == y[routed]).mean(); acc_l = (c[routed] == y[routed]).mean()
    axes[2].bar([0, 1], [acc_l, acc_e], color=[S.COT_COLOR, S.ENC_COLOR], width=0.6, edgecolor="white")
    for k, v in zip([0, 1], [acc_l, acc_e]):
        axes[2].text(k, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["LLM", "Encoder"])
    axes[2].set_ylabel("accuracy"); axes[2].set_ylim(0, max(acc_e, acc_l) * 1.25)
    axes[2].set_title(f"Routed subset\n(difficult, n={int(routed.sum())})", color=S.ACCENT)
    fig.suptitle(f"Final-system complementarity: W5 encoder vs pooled SC×10 LLM — agree {agree*100:.0f}% of the time",
                 fontsize=14, fontweight="bold", color=S.ACCENT, y=1.03)
    _save(fig, "fig_final_complementarity")


# ----------------------------------------------------------------- fig 8
def fig_final_class_profile():
    S.apply_rc()
    df = _final_frame(); y = df["label"].to_numpy()
    cfgs = [("Encoder W5", "enc", S.ENC_COLOR), ("Pooled LLM", "llm", S.COT_COLOR),
            ("Difficult cascade", "casc_diff", S.ENS_COLOR), ("Merged cascade", "casc_merg", "#0d9488")]
    M = {nm: D.metrics(y, df[col].to_numpy()) for nm, col, _ in cfgs}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.7, 1]})
    x = np.arange(4); nb = len(cfgs); w = 0.8 / nb
    for k, (nm, col, color) in enumerate(cfgs):
        axL.bar(x + (k - (nb-1)/2)*w, M[nm]["f1_per_class"], w, color=color, label=nm, edgecolor="white")
    axL.set_xticks(x); axL.set_xticklabels(CN); axL.set_ylabel("F1"); axL.set_ylim(0, 0.72)
    axL.set_title("Per-class F1 — does the cascade keep severe recovery?", color=S.ACCENT)
    axL.legend(fontsize=9, ncol=2)
    # far-off + within-1
    names = [c[0] for c in cfgs]; cols = [c[2] for c in cfgs]
    xx = np.arange(len(cfgs)); w2 = 0.38
    axR.bar(xx - w2/2, [M[n]["far_off"] for n in names], w2, color=cols, edgecolor="white")
    axR.bar(xx + w2/2, [M[n]["within_1"] for n in names], w2, color=cols, alpha=0.5, hatch="//", edgecolor="white")
    axR.set_xticks(xx); axR.set_xticklabels(names, rotation=30, ha="right", fontsize=8.5)
    axR.set_title("Far-off (solid) & within-±1 (hatched)", color=S.ACCENT)
    axR.set_ylim(0, 1.0)
    fig.suptitle("Class-wise behaviour of the final systems",
                 fontsize=14.5, fontweight="bold", color=S.ACCENT, y=1.0)
    _save(fig, "fig_final_class_profile")


# ----------------------------------------------------------------- fig 9
def fig_final_bootstrap_ci(n_boot=2000):
    S.apply_rc()
    df = _final_frame(); rng = np.random.default_rng(42)
    y = df["label"].to_numpy()
    cols = {"enc": df["enc"].to_numpy(), "llm": df["llm"].to_numpy(),
            "diff": df["casc_diff"].to_numpy(), "merg": df["casc_merg"].to_numpy()}
    pid = df["participant_id"].to_numpy()
    by_p = {p: np.where(pid == p)[0] for p in np.unique(pid)}
    pids = np.array(list(by_p.keys()))
    from sklearn.metrics import f1_score
    def mf1(idx, col): return f1_score(y[idx], cols[col][idx], average="macro", labels=LAB, zero_division=0)
    comps = [("LLM − Encoder", "llm", "enc"), ("Difficult cascade − Encoder", "diff", "enc"),
             ("Merged cascade − Encoder", "merg", "enc"), ("Difficult cascade − LLM", "diff", "llm"),
             ("Merged cascade − LLM", "merg", "llm")]
    res = []
    for lab, a, b in comps:
        deltas = np.empty(n_boot)
        for i in range(n_boot):
            samp = rng.choice(pids, size=len(pids), replace=True)
            idx = np.concatenate([by_p[p] for p in samp])
            deltas[i] = mf1(idx, a) - mf1(idx, b)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        res.append((lab, float(deltas.mean()), float(lo), float(hi), bool(lo > 0 or hi < 0)))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    yps = np.arange(len(res))[::-1]
    for yp, (lab, d, lo, hi, sig) in zip(yps, res):
        col = S.GOOD if sig else S.MUTED
        ax.errorbar(d, yp, xerr=[[d-lo], [hi-d]], fmt="o", color=col, ecolor=col,
                    elinewidth=2.2, capsize=5, ms=9)
        ax.text(hi + 0.002, yp, f"+{d:.3f} [{lo:.3f}, {hi:.3f}]" + ("  ✓" if sig else "  n.s."),
                va="center", fontsize=9.5, color=S.ACCENT if sig else S.MUTED)
    ax.axvline(0, color=S.BAD, ls="--", lw=1.4)
    ax.set_yticks(yps); ax.set_yticklabels([r[0] for r in res], fontsize=10.5)
    ax.set_xlim(-0.03, 0.09); ax.set_xlabel("Δ macro-F1 with 95% cluster-bootstrap CI")
    ax.set_title("Final comparisons — cluster-bootstrap CIs vs the W5 encoder (219 participants)", color=S.ACCENT)
    fig.text(0.5, -0.04, "Against the strong W5 encoder, every CI crosses zero: point estimates favour the LLM/cascade, "
             "but none is significant at the participant-cluster level — and cascade − LLM is essentially nil.",
             ha="center", fontsize=9.2, color=S.MUTED)
    _save(fig, "fig_final_bootstrap_ci")


ALL = [fig_context_strategy_comparison, fig_peritem_class_error_profile, fig_joint_vs_staged,
       fig_whole_picture_progression, fig_severe_safety_frontier, fig_sc_run_variance,
       fig_final_complementarity, fig_final_class_profile, fig_final_bootstrap_ci]

if __name__ == "__main__":
    import sys
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    for fn in ALL:
        if only and not any(o in fn.__name__ for o in only):
            continue
        try:
            fn()
        except Exception as e:
            print(f"  !! {fn.__name__} FAILED: {e}")
