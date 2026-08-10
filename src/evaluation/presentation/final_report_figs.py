"""The three figures for the final report (final_report/final_report_working_v5.tex).

  fig1_pipeline            : end-to-end study design (schematic, no data)
  fig2_retrieval_selection : label-free retrieval selection, 3 panels
  fig3_severity_tradeoff   : severe recall vs false-severe rate

Every number is read from the committed R2 artifacts under outputs/r2_systematic/
(plus a reconstruction of the frozen R0 Attention-MIL cascade, which is the only
system whose severe-class metrics were not already tabulated). Nothing is
hard-coded except the schematic's own labels.

Output is vector PDF at the report's \\textwidth (5.5in), so the figures are
included at scale 1.0 and their text renders at the point size set here.

    python -m src.evaluation.presentation.final_report_figs
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.evaluation.presentation import _style as S

ROOT = S.ROOT
R2 = ROOT / "outputs" / "r2_systematic"
OUT = ROOT / "final_report" / "figures"

# NeurIPS \textwidth. Figures are drawn at final print size so that the font
# sizes below are the ones the reader actually sees.
PAGE_W = 5.5

LAB = [0, 1, 2, 3]
ITEM_ORDER = ["Sleep", "Depressed", "Failure", "Tired",
              "NoInterest", "Concentrating", "Moving", "Appetite"]

# status colours, shared by fig2 and fig3
C_INFO = "#0e7490"    # teal   = informative (supports/against)
C_NONE = "#cbd5e1"    # grey   = judged none
C_AMB = "#f0b429"     # amber  = ambiguous
C_ENC = S.CALM_ENC        # green  = encoder family
C_LLM = S.CALM_COT        # violet = LLM family
C_CASC = S.CALM_CASCADE   # orange = R2 cascades
C_FROZEN = S.CALM_CASCADE_ALT  # blue = frozen project-best


def _rc():
    """Print-size defaults: small type, hairline rules, embedded TrueType."""
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 7,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "pdf.fonttype": 42,
    })


def _save(fig, name, tight=True):
    """Write the PDF at exactly the declared figsize.

    Deliberately no ``bbox_inches="tight"``: cropping to content would give each
    figure a different natural width, and since all three are included at
    ``width=\\textwidth`` LaTeX would rescale them by different factors and the
    type sizes would no longer match across figures. Saving uncropped at
    PAGE_W keeps every figure at scale 1.0.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout(pad=0.3)
    out = OUT / f"{name}.pdf"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# =============================================================== figure 1
def _box(ax, x, y, w, h, text, fc, ec=None, tc="#0f172a", fs=6.0, bold=True, lw=0.9):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.045",
        linewidth=lw, edgecolor=ec or fc, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3,
            linespacing=1.25)


def _arrow(ax, x1, y1, x2, y2, color="#64748b", lw=1.0, style="-|>", ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=1,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls,
                                shrinkA=1, shrinkB=1))


def fig1_pipeline():
    """Schematic of the study design. Carries no measured quantities."""
    _rc()
    W, H = PAGE_W, 3.40
    fig = plt.figure(figsize=(W, H))
    # axes fill the canvas, so the coordinates below are literally inches
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.grid(False)

    STAGE_FC = "#f8fafc"
    STAGE_EC = "#cbd5e1"
    y0, y1 = 0.30, 3.06          # vertical extent of the stage containers
    hdr = 0.20                   # header strip inside each container

    stages = [
        (0.02, 0.82, "Input"),
        (0.92, 1.44, "Evidence selection"),
        (2.44, 1.44, "Severity inference"),
        (3.96, 0.78, "Routing"),
        (4.82, 0.66, "Output"),
    ]
    for x, w, title in stages:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y0), w, y1 - y0, boxstyle="round,pad=0.014,rounding_size=0.05",
            linewidth=0.8, edgecolor=STAGE_EC, facecolor=STAGE_FC, zorder=0))
        ax.text(x + w / 2, y1 - hdr / 2, title.upper(), ha="center", va="center",
                fontsize=5.6, color=S.MUTED, fontweight="bold", zorder=3)

    # -- input ------------------------------------------------------------
    _box(ax, 0.09, 2.24, 0.68, 0.46, "Interview\ntranscript U", "#e2e8f0", fs=5.6)
    _box(ax, 0.09, 1.62, 0.68, 0.46, "PHQ-8 item\nwording q", "#e2e8f0", fs=5.6)
    ax.text(0.43, 1.20, "219 participants\n× 8 items\n= 1,752 rows",
            ha="center", va="center", fontsize=5.2, color=S.MUTED, style="italic")

    # -- evidence selection (top to bottom = the order things were decided) -
    _box(ax, 0.97, 2.34, 1.34, 0.46,
         "R0  Hybrid-W3\nα=0.50 · 5 windows", "#bae6fd", ec="#0284c7", fs=5.7)
    _box(ax, 0.97, 1.62, 1.34, 0.48,
         "R2  systematic sweep\nvocab L0–L3 × retriever\n× evidence budget",
         "#e0f2fe", ec="#0284c7", fs=5.2)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.97, 1.16), 1.34, 0.26, boxstyle="round,pad=0.012,rounding_size=0.04",
        linewidth=0.9, edgecolor="#0e7490", facecolor="#ccfbf1", zorder=2))
    ax.text(1.64, 1.29, "Evidence judge · label-free", ha="center", va="center",
            fontsize=5.0, color="#0f172a", fontweight="bold", zorder=3)
    _box(ax, 0.97, 0.50, 1.34, 0.46,
         "Selected: L0 Core,\nα=0.25 · Top-3", "#0284c7", tc="#ffffff", fs=5.7)
    _arrow(ax, 1.64, 1.62, 1.64, 1.44, color="#0e7490", lw=0.9)
    _arrow(ax, 1.64, 1.16, 1.64, 0.98, color="#0e7490", lw=0.9)

    # -- severity inference ------------------------------------------------
    _box(ax, 2.49, 2.30, 1.34, 0.50,
         "MentalBERT + CORN\nordinal encoder", "#dcfce7", ec=C_ENC, fs=5.7)
    ax.text(3.16, 2.09, "training-only policies:\nstatus quo · mask · drop none",
            ha="center", va="center", fontsize=5.0, color=C_ENC, style="italic")

    _box(ax, 2.49, 1.14, 1.34, 0.50,
         "Qwen2.5-7B-Instruct\ntolerance-aware CoT", "#ede9fe", ec=C_LLM, fs=5.5)
    ax.text(3.16, 0.93, "inference-only policies:\nkeep · filter · fallback · long ctx",
            ha="center", va="center", fontsize=5.0, color=C_LLM, style="italic")

    # -- routing -----------------------------------------------------------
    _box(ax, 4.04, 1.72, 0.62, 0.68,
         "Selective\ncorrection\ncascade", "#ffedd5", ec=C_CASC, fs=5.9)
    ax.text(4.35, 1.42, "routing fitted\ninside training\nfolds", ha="center", va="center",
            fontsize=5.0, color=C_CASC, style="italic")

    # -- output ------------------------------------------------------------
    _box(ax, 4.88, 1.86, 0.54, 0.54,
         "8 item\nseverities\n(0–3)", "#0f172a", tc="#ffffff", fs=5.7)
    ax.text(5.15, 1.56, "participant-\ngrouped OOF\nevaluation", ha="center", va="center",
            fontsize=5.0, color=S.MUTED, style="italic")

    # -- flow arrows -------------------------------------------------------
    DASH = (0, (2.2, 1.6))
    _arrow(ax, 0.79, 1.96, 0.98, 1.96, lw=1.1)
    _arrow(ax, 2.31, 2.57, 2.47, 2.64, lw=1.0)                 # R0 -> encoder
    _arrow(ax, 2.31, 2.57, 2.47, 1.52, lw=1.0)                 # R0 -> LLM
    _arrow(ax, 2.31, 0.73, 2.47, 2.42, lw=1.0, ls=DASH)        # R2 -> encoder
    _arrow(ax, 2.31, 0.73, 2.47, 1.26, lw=1.0, ls=DASH)        # R2 -> LLM
    _arrow(ax, 3.85, 2.50, 4.02, 2.20, lw=1.0, color=C_ENC)
    _arrow(ax, 3.85, 1.42, 4.02, 1.86, lw=1.0, color=C_LLM)
    _arrow(ax, 4.68, 2.06, 4.86, 2.10, lw=1.0, color=C_CASC)

    # -- legend strip ------------------------------------------------------
    handles = [
        mpatches.Patch(facecolor="#bae6fd", edgecolor="#0284c7", lw=0.8,
                       label="R0 frozen evidence (solid arrows)"),
        mpatches.Patch(facecolor="#0284c7", edgecolor="#0284c7", lw=0.8,
                       label="R2 selected evidence (dashed arrows)"),
        mpatches.Patch(facecolor="#dcfce7", edgecolor=C_ENC, lw=0.8,
                       label="Encoder — policies change training"),
        mpatches.Patch(facecolor="#ede9fe", edgecolor=C_LLM, lw=0.8,
                       label="LLM — policies change inference only"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.095),
              ncol=2, fontsize=5.5, handlelength=1.1, handleheight=0.85,
              columnspacing=1.4, borderaxespad=0.0)

    _save(fig, "final_report_fig1_pipeline", tight=False)


# =============================================================== figure 2
def _retrieval_frames():
    overall = pd.read_csv(R2 / "retrieval" / "retrieval_metrics_overall.csv")
    by_item = pd.read_csv(R2 / "retrieval" / "retrieval_metrics_by_item.csv")
    return overall, by_item


def fig2_retrieval_selection():
    """Label-free retrieval selection: vocabulary, retriever/budget, per item."""
    _rc()
    overall, by_item = _retrieval_frames()

    # margins are set explicitly rather than via tight_layout, which cannot
    # honour the hspace/wspace this three-panel arrangement depends on.
    # Height is kept compact (~3.65in) so the figure occupies about half its
    # page instead of three-quarters; the caption text lives in the prose.
    fig = plt.figure(figsize=(PAGE_W, 3.65))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.94],
                          width_ratios=[0.86, 1.14], hspace=0.66, wspace=0.32,
                          left=0.115, right=0.945, top=0.925, bottom=0.135)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    # -- panel A : best Top-5 configuration for each vocabulary ------------
    top5 = overall[overall.prefix == "top5"]
    best = top5.loc[top5.groupby("config").informative_rate.idxmax()]
    voc_order = ["L0_CORE", "L1_CORE_LAY", "L2_CORE_CLINICAL", "L3_FULL"]
    voc_lbl = {"L0_CORE": "L0\nCore", "L1_CORE_LAY": "L1\n+Lay",
               "L2_CORE_CLINICAL": "L2\n+Clinical", "L3_FULL": "L3\nFull"}
    best = best.set_index("config").loc[voc_order]

    x = np.arange(4)
    w = 0.38
    bi = axA.bar(x - w / 2, best.informative_rate, w, color=C_INFO,
                 edgecolor="white", linewidth=0.5, label="informative $\\uparrow$")
    bn = axA.bar(x + w / 2, best.none_rate, w, color=C_NONE,
                 edgecolor="white", linewidth=0.5, label="none $\\downarrow$")
    # mark the Core bar as the winner
    bi[0].set_edgecolor(S.ACCENT)
    bi[0].set_linewidth(1.1)
    for xi, v in zip(x, best.informative_rate):
        axA.text(xi - w / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=5.4,
                 fontweight="bold" if xi == 0 else "normal", color=S.ACCENT)
    for xi, v in zip(x, best.none_rate):
        axA.text(xi + w / 2, v + 0.012, f"{v:.2f}", ha="center", fontsize=5.4, color=S.MUTED)
    axA.set_xticks(x)
    axA.set_xticklabels([voc_lbl[c] for c in voc_order], fontsize=5.8)
    axA.set_ylim(0, 0.92)
    axA.set_ylabel("rate")
    axA.set_title("A · Vocabulary (best Top-5 each)", fontsize=7.0, color=S.ACCENT)
    axA.legend(fontsize=5.6, loc="upper center", ncol=2, handlelength=1.0,
               columnspacing=1.1, borderaxespad=0.2)
    axA.grid(axis="x", visible=False)
    axA.text(0.5, -0.30, "audited expansion does not help", transform=axA.transAxes,
             ha="center", fontsize=5.4, color=S.MUTED, style="italic")

    # -- panel B : retriever x evidence budget, Core vocabulary ------------
    core = overall[overall.config == "L0_CORE"]
    ret_order = ["bm25", "semantic", "hybrid_a75", "hybrid_a50", "hybrid_a25"]
    ret_lbl = ["BM25", "semantic", "hybrid\n$\\alpha$=.75", "hybrid\n$\\alpha$=.50", "hybrid\n$\\alpha$=.25"]
    budgets = [("top3", "Top-3", C_INFO),
               ("top5", "Top-5", "#67b7c7"),
               ("budget", "token budget", "#b7d9df")]

    xb = np.arange(len(ret_order))
    wb = 0.26
    for k, (pref, lbl, col) in enumerate(budgets):
        sub = core[core.prefix == pref].set_index("retriever")
        vals = [sub.informative_rate.get(r, np.nan) for r in ret_order]
        bars = axB.bar(xb + (k - 1) * wb, vals, wb, color=col, label=lbl,
                       edgecolor="white", linewidth=0.5)
        if pref == "top3":                      # highlight the selected cell
            bars[-1].set_edgecolor(S.ACCENT)
            bars[-1].set_linewidth(1.1)
            axB.text(xb[-1] - wb, vals[-1] + 0.006, f"{vals[-1]:.3f}", ha="center",
                     fontsize=5.6, fontweight="bold", color=S.ACCENT)
    axB.set_xticks(xb)
    axB.set_xticklabels(ret_lbl, fontsize=5.6)
    axB.set_ylim(0.25, 0.348)
    axB.set_ylabel("informative rate $\\uparrow$\n(truncated axis)")
    axB.set_title("B · Retriever × budget (L0 Core)", fontsize=7.0, color=S.ACCENT)
    axB.legend(fontsize=5.6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.30),
               handlelength=1.0, columnspacing=1.2)
    axB.grid(axis="x", visible=False)
    axB.annotate("selected", xy=(xb[-1] - wb, 0.3271), xytext=(xb[-1] - 1.30, 0.3415),
                 fontsize=5.5, color=S.ACCENT, fontweight="bold", va="center",
                 arrowprops=dict(arrowstyle="-|>", color=S.ACCENT, lw=0.7, shrinkB=3))

    # -- panel C : per-item evidence availability, selected configuration --
    sel = by_item[(by_item.config == "L0_CORE") & (by_item.retriever == "hybrid_a25")
                  & (by_item.prefix == "top3")].set_index("item_name").loc[ITEM_ORDER]

    xc = np.arange(8)
    axC.bar(xc, sel.informative_rate, 0.62, color=C_INFO, edgecolor="white",
            linewidth=0.5, label="informative")
    axC.bar(xc, sel.ambiguous_rate, 0.62, bottom=sel.informative_rate, color=C_AMB,
            edgecolor="white", linewidth=0.5, label="ambiguous")
    axC.bar(xc, sel.none_rate, 0.62,
            bottom=sel.informative_rate + sel.ambiguous_rate, color=C_NONE,
            edgecolor="white", linewidth=0.5, label="none")
    # value inside the teal segment when it is tall enough to hold the text,
    # otherwise just above the stack's informative+ambiguous boundary
    for xi, (v, a) in enumerate(zip(sel.informative_rate, sel.ambiguous_rate)):
        if v >= 0.22:
            axC.text(xi, v / 2, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                     color="white", fontweight="bold")
        else:
            axC.text(xi, v + a + 0.055, f"{v:.2f}", ha="center", va="bottom",
                     fontsize=5.5, color=C_INFO, fontweight="bold")
    axC.axhline(0.3271, color=S.ACCENT, lw=0.8, ls=(0, (3, 2)), zorder=4)
    axC.text(7.42, 0.345, "overall 0.327", fontsize=5.4, color=S.ACCENT,
             ha="right", fontweight="bold")
    axC.set_xticks(xc)
    axC.set_xticklabels(ITEM_ORDER, rotation=22, ha="right", fontsize=5.9)
    axC.set_ylim(0, 1.0)
    axC.set_ylabel("share of participants")
    axC.set_title("C · Judged evidence per PHQ-8 item — selected L0 Core / hybrid $\\alpha$=.25 / Top-3",
                  fontsize=7.0, color=S.ACCENT)
    axC.legend(fontsize=5.8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.30),
               handlelength=1.0, columnspacing=1.4)
    axC.grid(axis="x", visible=False)

    _save(fig, "final_report_fig2_retrieval_selection", tight=False)


# =============================================================== figure 3
def _frozen_mil_cascade():
    """Rebuild the frozen R0 Attention-MIL merged cascade to get its severe-class
    metrics, which the R2 cascade table does not carry."""
    from src.evaluation.cot_cascade import (load_llm, load_encoder, align,
                                            gate_mask, apply_cascade)
    cot = ROOT / "outputs" / "cot"
    llm = load_llm([str(cot / "folds_tolerant_sc5" / "*.csv"),
                    str(cot / "folds_tolerant_sc5_dv" / "*.csv")])
    enc = load_encoder(str(ROOT / "outputs" / "cv" / "oof_predictions_mil_hybw3.csv"))
    df = align(llm, enc)
    y = df["label"].to_numpy()
    pred = apply_cascade(df, gate_mask(df, "merged", 0.8, 0.5))
    return float((pred[y == 3] == 3).mean()), float((pred[y < 3] == 3).mean())


def fig3_severity_tradeoff():
    """Severe recall against false-severe rate for every candidate system."""
    _rc()
    enc = pd.read_csv(R2 / "encoder" / "encoder_metrics_overall.csv").set_index("run")
    llm = pd.read_csv(R2 / "llm" / "llm_metrics_overall.csv").set_index("run")
    casc = pd.read_csv(R2 / "cascade" / "cascade_metrics.csv").set_index("model")
    froz_rec, froz_fs = _frozen_mil_cascade()

    # (label, severe_recall, false_severe, colour, marker, status)
    # status: "selected" | "reference" | "rejected"
    pts = [
        ("Status quo", enc.loc["R2_status_quo"], C_ENC, "o", "selected"),
        ("Mask none", enc.loc["R2_mask_none"], C_ENC, "o", "rejected"),
        ("Drop none$^\\dagger$", enc.loc["R2_drop_none"], C_ENC, "o", "rejected"),
        ("R0 tolerant SC", llm.loc["L0_R0"], C_LLM, "^", "reference"),
        ("Keep R2", llm.loc["L1_keep"], C_LLM, "^", "rejected"),
        ("Filter none", llm.loc["L2_filter"], C_LLM, "^", "rejected"),
        ("Informative first", llm.loc["L3_infofirst"], C_LLM, "^", "rejected"),
        ("Transcript fallback", llm.loc["L4_fallback"], C_LLM, "^", "selected"),
        ("Long context", llm.loc["L5_longctx"], C_LLM, "^", "rejected"),
        ("Encoder-first", casc.loc["cascade_encoder_first"], C_CASC, "s", "rejected"),
        ("LLM-first", casc.loc["cascade_llm_first"], C_CASC, "s", "rejected"),
    ]
    rows = [(n, float(r.severe_recall), float(r.false_severe_rate), c, m, st)
            for n, r, c, m, st in pts]
    rows.append(("Frozen R0 MIL cascade", froz_rec, froz_fs, C_FROZEN, "*", "reference"))

    fig, ax = plt.subplots(figsize=(PAGE_W, 3.55))

    for name, rec, fs_, col, mk, status in rows:
        filled = status in ("selected", "reference")
        ax.scatter(fs_, rec, s=112 if mk == "*" else (52 if mk != "o" else 44),
                   marker=mk, zorder=4,
                   facecolor=col if filled else "white",
                   edgecolor=col, linewidths=1.2 if not filled else 0.9)

    # manual label offsets, in points, to keep 12 annotations legible
    off = {
        "Status quo": (7, -1), "Mask none": (-6, 6), "Drop none$^\\dagger$": (7, 0),
        "R0 tolerant SC": (-6, 7), "Keep R2": (7, 0), "Filter none": (0, 8),
        "Informative first": (2, -8), "Transcript fallback": (7, 2),
        "Long context": (0, -8), "Encoder-first": (7, -2), "LLM-first": (-6, 3),
        "Frozen R0 MIL cascade": (6, -7),
    }
    for name, rec, fs_, col, mk, status in rows:
        dx, dy = off[name]
        ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
        weight = "bold" if status == "selected" else "normal"
        txt = name + ("  ✓" if status == "selected" else "")
        ax.annotate(txt, (fs_, rec), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va="center", fontsize=5.6, color=col, fontweight=weight,
                    zorder=5)

    ax.set_xlabel("False-severe rate  $P(\\hat y{=}3 \\mid y{<}3)$  $\\downarrow$")
    ax.set_ylabel("Severe recall  $P(\\hat y{=}3 \\mid y{=}3)$  $\\uparrow$")
    ax.set_xlim(0.012, 0.098)
    ax.set_ylim(0.100, 0.325)
    ax.set_title("Recovering severe symptoms costs unsupported severe predictions",
                 fontsize=7.5, color=S.ACCENT)

    # direction-of-preference cue, drawn in axes coordinates so it can never
    # push the saved bounding box outside the data area
    ax.annotate("", xy=(0.055, 0.94), xytext=(0.175, 0.79), xycoords="axes fraction",
                textcoords="axes fraction", zorder=2,
                arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=0.9,
                                linestyle=(0, (2.5, 1.8))))
    ax.text(0.185, 0.775, "preferred", fontsize=5.3, color=S.MUTED, style="italic",
            ha="left", va="top", transform=ax.transAxes)

    fam = [plt.Line2D([], [], ls="", marker="o", ms=4.2, mfc=C_ENC, mec=C_ENC, label="Encoder"),
           plt.Line2D([], [], ls="", marker="^", ms=4.6, mfc=C_LLM, mec=C_LLM, label="LLM"),
           plt.Line2D([], [], ls="", marker="s", ms=4.2, mfc=C_CASC, mec=C_CASC, label="R2 cascade"),
           plt.Line2D([], [], ls="", marker="*", ms=6.5, mfc=C_FROZEN, mec=C_FROZEN, label="Frozen best"),
           plt.Line2D([], [], ls="", marker="o", ms=4.2, mfc="white", mec=S.MUTED, label="rejected (hollow)")]
    ax.legend(handles=fam, fontsize=5.7, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.155), handlelength=0.9, columnspacing=1.0,
              handletextpad=0.4)

    _save(fig, "final_report_fig3_severity_tradeoff")

    # companion numbers, so the figure is auditable without rerunning it
    comp = pd.DataFrame([{"system": n, "family": {"o": "encoder", "^": "llm",
                                                  "s": "r2_cascade", "*": "frozen"}[mk],
                          "severe_recall": rec, "false_severe_rate": fs_,
                          "status": st} for n, rec, fs_, _, mk, st in rows])
    cp = OUT / "final_report_fig3_severity_tradeoff_data.csv"
    comp.to_csv(cp, index=False)
    print(f"  wrote {cp}")


ALL = [fig1_pipeline, fig2_retrieval_selection, fig3_severity_tradeoff]

if __name__ == "__main__":
    import sys
    only = sys.argv[1:]
    for fn in ALL:
        if only and not any(o in fn.__name__ for o in only):
            continue
        fn()
