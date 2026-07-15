"""Fig 19 - Headline 4-model comparison table WITH 95% CIs (deck-styled).

Renders the numbers from cascade_variant_table_ci.py (point estimate + 95%
participant-level cluster-bootstrap CI) as a slide-ready table PNG. Best-in-
column point estimates are bold; the project-best row (Attention-MIL merged
cascade) is shaded. MAE and Error>=2 are lower-is-better (arrow in the header).

Run cascade_variant_table_ci.py first (writes the JSON this reads):
    python -m src.evaluation.cascade_variant_table_ci
    python -m src.evaluation.presentation.fig19_cascade_variant_ci_table
"""
import json

import numpy as np

from src.evaluation.presentation import _style as S

CI_JSON = S.ROOT / "outputs" / "cot" / "cascade_variant_table_ci.json"

# column key -> (header label, lower_is_better)
COLS = [
    ("exact",   "Exact",     False),
    ("macroF1", "Macro-F1",  False),
    ("QWK",     "QWK",       False),
    ("MAE",     "MAE ↓",     True),
    ("err>=2",  "Error ≥2 ↓", True),
]
# model order + accent tag colour (consistent with fig14 retrieval-variant hues)
MODELS = [
    ("Midterm CORN",                 S.CALM_SLATE),
    ("Pooled LLM",                   S.CALM_COT),
    ("Attention-MIL merged cascade", S.CALM_CASCADE),
    ("Rerank merged cascade",        S.CALM_ROSE),
]
BEST_ROW = "Attention-MIL merged cascade"


def main():
    S.apply_rc()
    meta = json.loads(CI_JSON.read_text())
    data = meta["metrics"]

    # best (bold) point per column
    best = {}
    for key, _, lower in COLS:
        vals = {m: data[m][key]["point"] for m, _ in MODELS}
        best[key] = (min if lower else max)(vals, key=vals.get)

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    ncol = len(COLS) + 1
    nrow = len(MODELS)
    # column x-centres: model column wider on the left
    x_model = 0.02
    x_metrics = np.linspace(0.40, 0.965, len(COLS))
    row_h = 1.0 / (nrow + 1.4)
    y_header = 1 - row_h * 0.75

    fig, ax = plt.subplots(figsize=(12.6, 4.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # header
    ax.text(x_model, y_header, "Model", fontsize=13, fontweight="bold",
            color=S.ACCENT, va="center", ha="left")
    for (key, label, _), xc in zip(COLS, x_metrics):
        ax.text(xc, y_header, label, fontsize=13, fontweight="bold",
                color=S.ACCENT, va="center", ha="center")
    ax.plot([0.01, 0.99], [y_header - row_h * 0.55] * 2, color=S.ACCENT, lw=1.4)

    for r, (model, tag) in enumerate(MODELS):
        y = y_header - row_h * (r + 1.15)
        # shade the project-best row
        if model == BEST_ROW:
            ax.add_patch(FancyBboxPatch(
                (0.008, y - row_h * 0.44), 0.984, row_h * 0.9,
                boxstyle="round,pad=0.004", mutation_aspect=0.35,
                fc=S.CALM_CASCADE, ec="none", alpha=0.13, zorder=0))
        # model name + colour tag
        ax.add_patch(plt.Rectangle((x_model, y - row_h * 0.16), 0.012, row_h * 0.32,
                                   fc=tag, ec="none", transform=ax.transAxes))
        weight = "bold" if model == BEST_ROW else "normal"
        ax.text(x_model + 0.022, y, model, fontsize=11.5, fontweight=weight,
                color=S.ACCENT, va="center", ha="left")
        # metric cells: point (top) + CI (below, muted)
        for (key, _, _), xc in zip(COLS, x_metrics):
            pt = data[model][key]["point"]
            lo, hi = data[model][key]["ci95"]
            is_best = model == best[key]
            ax.text(xc, y + row_h * 0.16, f"{pt:.3f}", fontsize=12.5,
                    fontweight="bold" if is_best else "normal",
                    color=tag if is_best else S.ACCENT, va="center", ha="center")
            ax.text(xc, y - row_h * 0.20, f"[{lo:.3f}, {hi:.3f}]", fontsize=8.5,
                    color=S.MUTED, va="center", ha="center")

    ax.set_title("Fig 19 · Headline models with 95% confidence intervals",
                 fontsize=15.5, color=S.ACCENT, pad=16, loc="left", x=0.008)
    foot = (f"Point estimate [95% CI].  {meta['method']}, B={meta['B']} resamples, "
            f"n={meta['n_participants']} participants / {meta['n_items']} items.  "
            "Bold = best in column (↓ = lower is better).  Cascades = pooled 10-chain "
            "SC LLM + encoder tiebreak on merged-gate items (τ=0.8, diff≥0.5).")
    fig.text(0.012, 0.015, foot, ha="left", fontsize=8.8, color=S.MUTED, wrap=True)

    S.save(fig, "fig19_cascade_variant_ci_table", tight=False)


if __name__ == "__main__":
    main()
