"""Architecture diagrams for the paper (figures 10-12).

  fig_arch_whole_picture : Joint vs Staged reasoning pipelines
  fig_arch_uncertainty   : 5 chains -> majority vote + difficult_frac
  fig_arch_final_cascade : the final confidence-gated cascade

    python -m src.evaluation.presentation.paper_arch
"""

import matplotlib.patches as mpatches

from src.evaluation.presentation import _style as S
from src.evaluation.presentation.fig_architectures import box, arrow, new_ax


def diamond(ax, cx, cy, w, h, text, fc, ec, fs=9.3):
    pts = [(cx, cy + h/2), (cx + w/2, cy), (cx, cy - h/2), (cx - w/2, cy)]
    ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, lw=1.6, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color="#0F172A",
            fontweight="bold", zorder=3, linespacing=1.15)


# ------------------------------------------------ fig 10 : joint vs staged
def fig_arch_whole_picture():
    S.apply_rc()
    W, H = 13.5, 6.4
    fig, ax = new_ax(W, H)
    ev = "#bae6fd"

    # Joint (top)
    ax.text(0.25, H - 0.45, "JOINT  (one holistic pass)", fontsize=11.5, color=S.ENS_COLOR, fontweight="bold")
    yj = H - 2.0
    box(ax, 0.3, yj, 2.4, 1.2, "All 8 items'\nretrieved evidence", ev, fs=10)
    arrow(ax, 2.7, yj+0.6, 3.2, yj+0.6, color=S.ENS_COLOR)
    box(ax, 3.2, yj, 2.5, 1.2, "Frozen LLM\nsingle pass\n(overall impression)", "#e0f2fe", ec=S.ENS_COLOR, fs=10)
    arrow(ax, 5.7, yj+0.6, 6.2, yj+0.6, color=S.ENS_COLOR)
    box(ax, 6.2, yj, 2.4, 1.2, "Score all 8\nitems at once", "#e0f2fe", ec=S.ENS_COLOR, fs=10)
    arrow(ax, 8.6, yj+0.6, 9.1, yj+0.6, color=S.ENS_COLOR)
    box(ax, 9.1, yj+0.1, 1.7, 1.0, "8 scores", S.ENS_COLOR, tc="FFFFFF", fs=11)
    ax.text(11.2, yj+0.6, "risk: dilutes per-item\nfocus, loses severe\ncommitment", fontsize=8.5,
            color=S.MUTED, va="center", style="italic")

    # divider
    ax.plot([0.25, W-0.25], [H/2 - 0.35, H/2 - 0.35], color="#e2e8f0", lw=1)

    # Staged (bottom)
    ax.text(0.25, H/2 - 0.7, "STAGED  (evidence → confident → context → difficult → final)",
            fontsize=11.5, color="#0d9488", fontweight="bold")
    ys = 0.7
    steps = [("All 8 items'\nevidence", ev, "#0d9488"),
             ("1 · Score the\nCONFIDENT\nitems", "#ccfbf1", "#0d9488"),
             ("2 · Build the\nCLINICAL\nCONTEXT", "#ccfbf1", "#0d9488"),
             ("3 · Resolve the\nDIFFICULT items\n(context + evidence)", "#ccfbf1", "#0d9488"),
             ("4 · Final\n8 scores", "#0d9488", "#0d9488")]
    bw, gap = 2.3, 0.32; x = 0.3
    for i, (txt, fc, ec) in enumerate(steps):
        last = i == len(steps)-1
        box(ax, x, ys, bw, 1.35, txt, ec if last else fc, tc="FFFFFF" if last else "0F172A",
            ec=ec, fs=9.3, bold=last)
        if i < len(steps)-1:
            arrow(ax, x+bw, ys+0.67, x+bw+gap, ys+0.67, color="#0d9488", lw=1.6)
        x += bw + gap

    fig.suptitle("Whole-picture reasoning: Joint vs Staged pipelines",
                 fontsize=15, fontweight="bold", color=S.ACCENT, y=1.0)
    S.save(fig, "fig_arch_whole_picture", tight=False)


# ------------------------------------------------ fig 11 : uncertainty
def fig_arch_uncertainty():
    S.apply_rc()
    W, H = 13.5, 6.2
    fig, ax = new_ax(W, H)
    V = "#ede9fe"
    box(ax, 0.3, H/2-0.7, 2.3, 1.4, "Item +\nretrieved\nevidence", "#bae6fd", fs=10)
    # 5 chains
    cx = 3.5
    for i in range(5):
        cy = H - 0.95 - i*1.0
        box(ax, cx, cy-0.35, 2.7, 0.72, f"Chain {i+1}  (sample T=0.7)  → label", V, ec=S.COT_COLOR, fs=9)
        arrow(ax, 2.6, H/2, cx, cy, color=S.COT_COLOR, lw=1.2)
    ax.text(cx+1.35, H-0.35, "5 independent reasoning chains", ha="center", fontsize=9.5,
            color=S.COT_COLOR, fontweight="bold")
    # majority vote + difficult_frac
    box(ax, 7.2, H/2+0.15, 2.7, 1.1, "Majority vote\n→ final label\n+ vote fractions", S.COT_COLOR, tc="FFFFFF", fs=9.5)
    box(ax, 7.2, H/2-1.35, 2.7, 1.1, "difficult_frac =\n# chains flagging\n'difficult' ÷ 5", "#fef3c7", ec=S.WARN, fs=9.5)
    for ty in (H/2+0.7, H/2-0.8):
        arrow(ax, 6.2, H/2, 7.2, ty, color="#475569", lw=1.4)
    arrow(ax, 9.9, H/2+0.7, 10.5, H/2+0.7, color="#475569")
    box(ax, 10.5, H/2+0.2, 2.6, 1.0, "Severity 0–3\n(with confidence)", S.ENS_COLOR, tc="FFFFFF", fs=10)
    arrow(ax, 9.9, H/2-0.8, 10.5, H/2-0.8, color=S.WARN)
    box(ax, 10.5, H/2-1.3, 2.6, 1.0, "Uncertainty signal\n→ the cascade gate", "#fde68a", ec=S.WARN, fs=9.5)
    fig.suptitle("Uncertainty from self-consistency: vote + difficult_frac",
                 fontsize=15, fontweight="bold", color=S.ACCENT, y=1.0)
    S.save(fig, "fig_arch_uncertainty", tight=False)


# ------------------------------------------------ fig 12 : final cascade
def fig_arch_final_cascade():
    S.apply_rc()
    W, H = 13.5, 5.8
    fig, ax = new_ax(W, H)
    yc = H/2
    box(ax, 0.2, yc-0.55, 1.7, 1.1, "Item +\nevidence", "#bae6fd", fs=9.5)
    arrow(ax, 1.9, yc, 2.25, yc, color=S.COT_COLOR)
    box(ax, 2.25, yc-0.65, 2.4, 1.3, "Pooled 10-chain\nSC-tolerant LLM\n(2 × SC×5)", "#ede9fe", ec=S.COT_COLOR, fs=9.5)
    arrow(ax, 4.65, yc, 5.0, yc, color=S.COT_COLOR)
    box(ax, 5.0, yc-0.65, 2.2, 1.3, "Per item:\nlabel + top-2\n+ difficult_frac", "#ede9fe", ec=S.COT_COLOR, fs=9)
    # gate diamond
    arrow(ax, 7.2, yc, 7.55, yc, color="#475569")
    diamond(ax, 8.5, yc, 1.8, 2.0, "difficult?\nfrac ≥ 0.8\n(or vote < τ)", "#fef3c7", S.WARN)
    # no (up) -> keep LLM
    arrow(ax, 8.5, yc+1.0, 8.5, yc+1.45, color=S.GOOD)
    ax.text(8.65, yc+1.18, "no  (~94%)", fontsize=8.5, color=S.GOOD)
    box(ax, 7.5, yc+1.45, 2.1, 0.8, "keep LLM label", "#dcfce7", ec=S.GOOD, fs=9.5)
    # yes (down) -> encoder tie-break
    arrow(ax, 8.5, yc-1.0, 8.5, yc-1.45, color=S.BAD)
    ax.text(8.65, yc-1.25, "yes  (~6%)", fontsize=8.5, color=S.BAD)
    box(ax, 7.4, yc-2.35, 2.3, 0.9, "W5 encoder picks\nbetter of LLM's top-2", "#dcfce7", ec=S.ENC_COLOR, fs=8.8)
    # converge to final (right)
    box(ax, 11.0, yc-0.5, 2.3, 1.0, "Final\nseverity 0–3", S.ENS_COLOR, tc="FFFFFF", fs=11)
    arrow(ax, 9.6, yc+1.85, 11.0, yc+0.25, color="#475569", lw=1.4)   # keep-LLM -> final
    arrow(ax, 9.7, yc-1.9, 11.0, yc-0.25, color="#475569", lw=1.4)    # encoder  -> final
    fig.suptitle("The final confidence-gated cascade (difficult-gate)",
                 fontsize=15, fontweight="bold", color=S.ACCENT, y=1.0)
    S.save(fig, "fig_arch_final_cascade", tight=False)


ALL = [fig_arch_whole_picture, fig_arch_uncertainty, fig_arch_final_cascade]

if __name__ == "__main__":
    for fn in ALL:
        try:
            fn()
        except Exception as e:
            print(f"  !! {fn.__name__} FAILED: {e}")
