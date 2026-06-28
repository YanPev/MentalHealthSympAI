"""Shared style + small helpers for the presentation figures.

Keeps every figure visually consistent (palette, fonts, 300-dpi export) and
centralises the project's domain vocabulary (PHQ-8 item names, severity-class
names) so the five scripts stay short and agree with each other.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "outputs" / "figures"

# ----------------------------------------------------------------------------
# Domain vocabulary (kept identical to src/evaluation house style)
# ----------------------------------------------------------------------------
LABELS = [0, 1, 2, 3]
CLASS_NAMES = ["Minimal\n(0)", "Mild\n(1)", "Moderate\n(2)", "Severe\n(3)"]
CLASS_SHORT = ["Minimal", "Mild", "Moderate", "Severe"]

# item_id -> PHQ-8 short symptom name (matches outputs/per_item_analysis.csv)
ITEM_NAMES = {
    1: "NoInterest", 2: "Depressed", 3: "Sleep", 4: "Tired",
    5: "Appetite", 6: "Failure", 7: "Concentrating", 8: "Moving",
}

# ----------------------------------------------------------------------------
# Palette (consistent with the existing HTML reports)
# ----------------------------------------------------------------------------
ENC_COLOR = "#16a34a"   # green  = established encoder (MentalBERT+CORN)
COT_COLOR = "#7c3aed"   # violet = the CoT / LLM probe
ENS_COLOR = "#0ea5e9"   # sky    = ensemble / blend
ACCENT = "#0f172a"      # near-black for emphasis
MUTED = "#64748b"       # slate for secondary text/grid
GOOD = "#16a34a"
WARN = "#f59e0b"
BAD = "#dc2626"

# retrieval-condition colours (used by fig1 / fig3 narratives)
RETR_COLORS = {
    "Isolated": "#94a3b8",   # grey  = no/lexical-only evidence
    "BM25": "#f59e0b",       # amber = lexical context windows
    "Hybrid": "#0ea5e9",     # sky   = semantic + lexical (the win)
}

SEQ_BLUES = plt.cm.Blues  # for confusion-matrix heatmaps


def apply_rc():
    """Global matplotlib defaults for a clean, slide-ready look."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 12,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
    })


def save(fig, name, tight=True):
    """Write ``<name>.png`` to outputs/figures/ at 300 dpi and report the path."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def companion_path(name, ext):
    """Path for a figure's companion data file (the numbers behind the chart)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR / f"{name}_data.{ext}"
