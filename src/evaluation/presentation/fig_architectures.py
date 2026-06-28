"""Architecture diagrams for the modelling strategies (deck support).

Two figures:
  arch_core     : the shared retrieval front-end + the two core strategies
                  (trained encoder vs frozen-LLM Chain-of-Thought).
  arch_variants : the three refinements (attention-MIL, cross-encoder rerank,
                  calibrated fusion / routing).

    python -m src.evaluation.presentation.fig_architectures
"""

import matplotlib.patches as mpatches

from src.evaluation.presentation import _style as S


def _c(c):
    return c if c.startswith("#") else "#" + c


def box(ax, x, y, w, h, text, fc, tc="0F172A", fs=11, bold=True, ec=None):
    fc, tc, ec = _c(fc), _c(tc), _c(ec) if ec else None
    p = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=1.4, edgecolor=ec or fc, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3, linespacing=1.15)


def arrow(ax, x1, y1, x2, y2, color="#475569", lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=2, shrinkB=2), zorder=1)


def new_ax(w, h):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    ax.grid(False)
    return fig, ax


# ---------------------------------------------------------------- core
def core():
    S.apply_rc()
    W, H = 13.5, 5.9
    fig, ax = new_ax(W, H)
    GREY = "#cbd5e1"; tint = lambda c: c

    # shared front-end (vertically centred)
    yc = H / 2 - 0.5
    box(ax, 0.25, yc, 1.7, 1.0, "Interview\ntranscript", "#e2e8f0")
    arrow(ax, 1.95, yc + 0.5, 2.35, yc + 0.5)
    box(ax, 2.35, yc - 0.1, 2.05, 1.2, "Item-aware\nhybrid retrieval\n(BM25 + dense)", S.RETR_COLORS["Hybrid"], tc="FFFFFF", fs=10.5)
    arrow(ax, 4.4, yc + 0.5, 4.8, yc + 0.5)
    box(ax, 4.8, yc - 0.1, 1.95, 1.2, "Top-k context\nwindows\n(evidence)", "#bae6fd", fs=10.5)
    ax.text(3.5, H - 0.35, "SHARED EVIDENCE FRONT-END", ha="center", fontsize=10,
            color=S.MUTED, fontweight="bold")

    # split
    ex = 6.75
    arrow(ax, ex, yc + 0.7, 7.15, H - 1.55, color=S.ENC_COLOR)
    arrow(ax, ex, yc + 0.3, 7.15, 1.55, color=S.COT_COLOR)

    # encoder branch (top)
    yE = H - 1.95
    ax.text(7.15, H - 0.35, "STRATEGY A · ENCODER  (trained)", ha="left", fontsize=10.5,
            color=S.ENC_COLOR, fontweight="bold")
    box(ax, 7.15, yE, 2.25, 1.25, "MentalBERT\nencoder\n([CLS] item ⊕ evidence)", "#dcfce7", ec=S.ENC_COLOR, fs=10)
    arrow(ax, 9.4, yE + 0.62, 9.75, yE + 0.62, color=S.ENC_COLOR)
    box(ax, 9.75, yE, 1.9, 1.25, "CORN\nordinal head\n(3 rank logits)", "#dcfce7", ec=S.ENC_COLOR, fs=10)
    arrow(ax, 11.65, yE + 0.62, 12.0, yE + 0.62, color=S.ENC_COLOR)
    box(ax, 12.0, yE + 0.12, 1.25, 1.0, "Severity\n0–3", S.ENC_COLOR, tc="FFFFFF", fs=11)

    # CoT branch (bottom)
    yC = 0.95
    ax.text(7.15, 0.3, "STRATEGY B · CoT PROBE  (frozen, no training)", ha="left",
            fontsize=10.5, color=S.COT_COLOR, fontweight="bold")
    box(ax, 7.15, yC, 2.25, 1.25, "Frozen LLM\n(Qwen-7B) +\nfew-shot prompt", "#ede9fe", ec=S.COT_COLOR, fs=10)
    arrow(ax, 9.4, yC + 0.62, 9.75, yC + 0.62, color=S.COT_COLOR)
    box(ax, 9.75, yC, 1.9, 1.25, "Chain-of-thought\nreasoning\n→ label", "#ede9fe", ec=S.COT_COLOR, fs=10)
    arrow(ax, 11.65, yC + 0.62, 12.0, yC + 0.62, color=S.COT_COLOR)
    box(ax, 12.0, yC + 0.12, 1.25, 1.0, "Self-consist.\nvote → 0–3", S.COT_COLOR, tc="FFFFFF", fs=9.5)

    S.save(fig, "fig_arch_core", tight=False)


# ------------------------------------------------------------ variants
def variants():
    S.apply_rc()
    W, H = 13.5, 6.6
    fig, ax = new_ax(W, H)

    rows = [
        ("REFINEMENT 1 · ATTENTION-MIL", S.ENS_COLOR, "#e0f2fe",
         ["k retrieved\nwindows",
          "MentalBERT\n(shared, encodes\neach window)",
          "Gated attention\npooling\n(weights per window)",
          "CORN head → 0–3\n+ attention =\nwhich window"]),
        ("REFINEMENT 2 · CROSS-ENCODER RERANK", S.WARN, "#fef3c7",
         ["All transcript\nwindows (~90/pt)",
          "Cross-encoder\nscores vs item\nquestion",
          "Keep top-k most\ndiagnostic\nwindows",
          "→ feeds the\nencoder pipeline\n(Strategy A)"]),
        ("REFINEMENT 3 · CALIBRATED FUSION", "#0d9488", "#ccfbf1",
         ["Encoder probs\n+ CoT probs",
          "Temperature\ncalibration\n(ECE 0.40→0.08)",
          "Condition on\nitem / severity\n(route or stack)",
          "Final\nseverity 0–3"]),
    ]
    bw, bh, gap = 2.45, 1.35, 0.35
    x0 = 1.95
    for i, (label, ec, fc) in enumerate([(r[0], r[1], r[2]) for r in rows]):
        yrow = H - 1.7 - i * 2.05
        ax.text(0.25, yrow + bh / 2, label.split(" · ")[0] + "\n" + label.split(" · ")[1],
                ha="left", va="center", fontsize=9.2, color=ec, fontweight="bold", linespacing=1.3)
        boxes = rows[i][3]
        for j, txt in enumerate(boxes):
            x = x0 + j * (bw + gap)
            last = j == len(boxes) - 1
            box(ax, x, yrow, bw, bh, txt, ec if last else fc,
                tc="FFFFFF" if last else "0F172A", ec=ec, fs=9.3, bold=last)
            if j < len(boxes) - 1:
                arrow(ax, x + bw, yrow + bh / 2, x + bw + gap, yrow + bh / 2, color=ec, lw=1.6)

    S.save(fig, "fig_arch_variants", tight=False)


if __name__ == "__main__":
    core()
    variants()
    print("done")
