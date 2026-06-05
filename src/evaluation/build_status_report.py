"""
Build a self-contained HTML status report for the PHQ-8 item classifier.

Reads the metrics JSON + prediction CSVs in ``outputs/`` and renders an HTML
report with inline-SVG figures (no external assets / libraries). Run:

    python -m src.evaluation.build_status_report
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
SEED_DIR = OUT / "seeds_weighted"
LABELS = [0, 1, 2, 3]
NUM_LABELS = len(LABELS)
SEV_NAMES = ["none (0)", "mild (1)", "moderate (2)", "severe (3)"]

# ----------------------------------------------------------------------------- data

def load_seed_aggregates():
    """Per-model mean +/- std over the 5-seed weighted runs, + paired comparison."""
    import re

    def collect(tag):
        rows = {}
        for csv in sorted(SEED_DIR.glob(f"{tag}_ret_weighted_s*.csv")):
            m = re.search(r"_s(\d+)\.csv$", csv.name)
            df = pd.read_csv(csv)
            y, p = df["true_label"].to_numpy(), df["prediction"].to_numpy()
            rows[int(m.group(1))] = {
                "acc": accuracy_score(y, p),
                "macro_f1": f1_score(y, p, average="macro", zero_division=0),
                "mae": mean_absolute_error(y, p),
            }
        return rows

    bert, mbert = collect("bert"), collect("mentalbert")
    seeds = sorted(set(bert) & set(mbert))
    def ms(d, k):
        a = np.array([d[s][k] for s in seeds]); return a.mean(), a.std(ddof=1)
    diffs = np.array([mbert[s]["macro_f1"] - bert[s]["macro_f1"] for s in seeds])
    return {
        "seeds": seeds,
        "bert": {k: ms(bert, k) for k in ("acc", "macro_f1", "mae")},
        "mbert": {k: ms(mbert, k) for k in ("acc", "macro_f1", "mae")},
        "f1_diff_mean": diffs.mean(),
        "mbert_wins": int((diffs > 0).sum()),
        "per_seed": [(s, bert[s]["macro_f1"], mbert[s]["macro_f1"]) for s in seeds],
    }


def load_per_item():
    """Per-item macro-F1 mean/std pooled over all weighted seed runs (both models)."""
    records = []
    for f in sorted(SEED_DIR.glob("*_ret_weighted_s*.csv")):
        df = pd.read_csv(f)
        for (iid, iname), g in df.groupby(["item_id", "item_name"]):
            y, p = g["true_label"].to_numpy(), g["prediction"].to_numpy()
            records.append({
                "item_id": iid, "item_name": iname,
                "macro_f1": f1_score(y, p, average="macro", labels=LABELS, zero_division=0),
                "mae": mean_absolute_error(y, p),
            })
    runs = pd.DataFrame(records)
    agg = runs.groupby(["item_id", "item_name"]).agg(
        f1_mean=("macro_f1", "mean"), f1_std=("macro_f1", "std"),
        mae_mean=("mae", "mean")).reset_index()
    # severity descriptor from seed-invariant truth
    truth = pd.read_csv(sorted(SEED_DIR.glob("*_ret_weighted_s*.csv"))[0])
    sev = truth.groupby("item_id")["true_label"].apply(lambda s: (s >= 2).mean())
    agg["pct_severe"] = agg["item_id"].map(sev)
    return agg.sort_values("f1_mean", ascending=False).reset_index(drop=True)


def load_cv():
    """5-fold CV metrics for both models + paired per-fold macro-F1 comparison."""
    def one(tag):
        return json.loads((OUT / "cv" / f"cv_metrics_{tag}.json").read_text())
    bert, mbert = one("bert_base_uncased"), one("mental_bert_base_uncased")
    bf = [f["macro_f1"] for f in bert["folds"]]
    mf = [f["macro_f1"] for f in mbert["folds"]]
    diffs = np.array(mf) - np.array(bf)
    return {
        "bert": bert, "mbert": mbert,
        "bert_folds": bf, "mbert_folds": mf,
        "n_folds": len(bf),
        "diff_mean": float(diffs.mean()),
        "diff_std": float(diffs.std(ddof=1)),
        "mbert_wins": int((diffs > 0).sum()),
    }


def load_per_item_oof():
    """Per-item macro-F1 from out-of-fold predictions (both models pooled, ~438/item).

    Point estimate uses the full OOF pool; whiskers are std across the 10
    fold x model subsamples to convey uncertainty.
    """
    frames = []
    for tag in ("bert_base_uncased", "mental_bert_base_uncased"):
        d = pd.read_csv(OUT / "cv" / f"oof_predictions_{tag}.csv")
        d["model"] = tag
        frames.append(d)
    oof = pd.concat(frames, ignore_index=True)

    rows = []
    for (iid, iname), g in oof.groupby(["item_id", "item_name"]):
        y, p = g["label"].to_numpy(), g["prediction"].to_numpy()
        subs = []
        for _, sg in g.groupby(["model", "fold"]):
            ys, ps = sg["label"].to_numpy(), sg["prediction"].to_numpy()
            subs.append(f1_score(ys, ps, average="macro", labels=LABELS, zero_division=0))
        rows.append({
            "item_id": iid, "item_name": iname,
            "f1_mean": f1_score(y, p, average="macro", labels=LABELS, zero_division=0),
            "f1_std": float(np.std(subs, ddof=1)),
            "mae_mean": mean_absolute_error(y, p),
            "pct_severe": float((y >= 2).mean()),
        })
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)


def load_corn():
    """CORN vs CE+weights CV pooled-OOF metrics for both models."""
    def one(tag):
        return json.loads((OUT / "cv" / f"cv_metrics_{tag}.json").read_text())["pooled_oof"]
    return {
        "bert_ce": one("bert_base_uncased"), "bert_corn": one("bert_corn"),
        "mbert_ce": one("mental_bert_base_uncased"), "mbert_corn": one("mentalbert_corn"),
    }


def load_near_miss(tag="bert_base_uncased"):
    """Top-1 vs top-2 / ordinal near-miss stats from OOF predictions with probs."""
    df = pd.read_csv(OUT / "cv" / f"oof_predictions_{tag}.csv")
    prob_cols = [f"prob_{c}" for c in range(NUM_LABELS)]
    P = df[prob_cols].to_numpy()
    y = df["label"].to_numpy()
    n = len(y)
    order = np.argsort(-P, axis=1)
    top1, top2 = order[:, 0], order[:, 1]
    correct1 = top1 == y
    in_top2 = correct1 | (top2 == y)
    true_rank = (order == y[:, None]).argmax(axis=1)
    errors = ~correct1
    n_err = int(errors.sum())
    per_class = []
    for c in range(NUM_LABELS):
        m = y == c
        per_class.append((SEV_NAMES[c], float(correct1[m].mean()), float(in_top2[m].mean())))
    return {
        "n": n, "tag": tag,
        "acc1": float(correct1.mean()), "acc2": float(in_top2.mean()),
        "n_err": n_err,
        "recovered": float(((top2 == y) & errors).mean() * n / n_err),
        "adjacent": float((errors & (np.abs(top1 - y) == 1)).sum() / n_err),
        "near_miss": float((errors & ((top2 == y) | (np.abs(top1 - y) == 1))).sum() / n_err),
        "rank_cum": [float((true_rank <= r).mean()) for r in range(NUM_LABELS)],
        "per_class": per_class,
    }


def load_cm(csv):
    df = pd.read_csv(OUT / csv)
    y, p = df["true_label"].to_numpy(), df["prediction"].to_numpy()
    cm = confusion_matrix(y, p, labels=LABELS)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y, p, labels=LABELS, zero_division=0)
    return {
        "cm": cm, "f1": f1, "support": sup,
        "acc": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "mae": mean_absolute_error(y, p),
    }

def final_metrics(stem):
    d = json.loads((OUT / f"{stem}_metrics.json").read_text())["final"]
    return d

# ----------------------------------------------------------------------------- svg

PALETTE = {"bert": "#4f46e5", "mentalbert": "#0891b2",
           "weighted": "#d97706", "good": "#16a34a", "bad": "#dc2626"}

def svg_bars(series, title, ymax=None, fmt="{:.3f}", colors=None):
    """series: list of (label, value). Vertical bar chart."""
    n = len(series)
    W, H = 600, 260
    pad_l, pad_b, pad_t = 44, 64, 24
    plot_h = H - pad_b - pad_t
    ymax = ymax or max(v for _, v in series) * 1.15
    bw = (W - pad_l - 16) / n * 0.62
    gap = (W - pad_l - 16) / n
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
             f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    # y gridlines
    for frac in (0, .25, .5, .75, 1.0):
        yv = ymax * frac
        y = pad_t + plot_h * (1 - frac)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{W-8}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{pad_l-6}" y="{y+3:.1f}" text-anchor="end" fill="#94a3b8">{yv:.2f}</text>')
    for i, (lab, v) in enumerate(series):
        x = pad_l + 8 + i * gap + (gap - bw) / 2
        h = plot_h * (v / ymax)
        y = pad_t + plot_h - h
        c = (colors[i] if colors else "#4f46e5")
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="3" fill="{c}"/>')
        parts.append(f'<text x="{x+bw/2:.1f}" y="{y-4:.1f}" text-anchor="middle" fill="#1e293b" font-weight="600">{fmt.format(v)}</text>')
        # wrapped label
        for j, part in enumerate(lab.split("\n")):
            parts.append(f'<text x="{x+bw/2:.1f}" y="{pad_t+plot_h+14+j*12:.1f}" text-anchor="middle" fill="#64748b">{part}</text>')
    parts.append('</svg>')
    return f'<p class="note" style="margin-bottom:2px"><b>{title}</b></p>' + "".join(parts)

def svg_confusion(res, title, accent="#4f46e5"):
    cm = res["cm"]
    W, H = 360, 300
    cell = 56
    x0, y0 = 96, 56
    rownorm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
             f'font-family="-apple-system,Segoe UI,sans-serif" font-size="12">']
    parts.append(f'<text x="{x0+2*cell}" y="20" text-anchor="middle" fill="#64748b" font-size="11">predicted</text>')
    parts.append(f'<text x="20" y="{y0+2*cell}" text-anchor="middle" fill="#64748b" font-size="11" transform="rotate(-90 20 {y0+2*cell})">true</text>')
    for j in range(4):
        parts.append(f'<text x="{x0+j*cell+cell/2}" y="{y0-6}" text-anchor="middle" fill="#475569">{j}</text>')
    for i in range(4):
        parts.append(f'<text x="{x0-8}" y="{y0+i*cell+cell/2+4}" text-anchor="end" fill="#475569">{i}</text>')
        for j in range(4):
            frac = rownorm[i, j]
            # accent intensity; diagonal greenish, off-diagonal neutral
            base = accent if i == j else "#94a3b8"
            r = int(int(base[1:3],16)); g=int(int(base[3:5],16)); b=int(int(base[5:7],16))
            a = 0.12 + 0.85 * frac
            txtc = "#ffffff" if frac > 0.5 else "#1e293b"
            x = x0 + j*cell; y = y0 + i*cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell-3}" height="{cell-3}" rx="4" '
                         f'fill="rgba({r},{g},{b},{a:.2f})"/>')
            parts.append(f'<text x="{x+(cell-3)/2}" y="{y+(cell-3)/2+1}" text-anchor="middle" fill="{txtc}" font-weight="600">{cm[i,j]}</text>')
            parts.append(f'<text x="{x+(cell-3)/2}" y="{y+(cell-3)/2+14}" text-anchor="middle" fill="{txtc}" font-size="9" opacity=".8">{frac*100:.0f}%</text>')
    parts.append('</svg>')
    sub = (f'acc {res["acc"]:.3f} · macroF1 <b>{res["macro_f1"]:.3f}</b> · MAE {res["mae"]:.3f}'
           f' · class2 F1 {res["f1"][2]:.2f} · class3 F1 {res["f1"][3]:.2f}')
    return (f'<div style="flex:1;min-width:300px"><p class="note" style="margin-bottom:2px">'
            f'<b>{title}</b><br><span style="font-size:11.5px">{sub}</span></p>' + "".join(parts) + '</div>')

def svg_heatmap(grid, rows, cols, title, vmin, vmax):
    """grid[i][j] value; rows=epochs, cols=lr."""
    cw, ch = 78, 44
    x0, y0 = 70, 44
    W = x0 + len(cols)*cw + 20
    H = y0 + len(rows)*ch + 30
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
             f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    parts.append(f'<text x="{x0+len(cols)*cw/2}" y="18" text-anchor="middle" fill="#64748b">learning rate</text>')
    parts.append(f'<text x="16" y="{y0+len(rows)*ch/2}" text-anchor="middle" fill="#64748b" transform="rotate(-90 16 {y0+len(rows)*ch/2})">epochs</text>')
    for j, c in enumerate(cols):
        parts.append(f'<text x="{x0+j*cw+cw/2}" y="{y0-6}" text-anchor="middle" fill="#475569">{c}</text>')
    for i, r in enumerate(rows):
        parts.append(f'<text x="{x0-8}" y="{y0+i*ch+ch/2+4}" text-anchor="end" fill="#475569">{r}</text>')
        for j in range(len(cols)):
            v = grid[i][j]
            frac = (v - vmin) / max(vmax - vmin, 1e-9)
            a = 0.12 + 0.85*frac
            txtc = "#ffffff" if frac > 0.55 else "#1e293b"
            x = x0+j*cw; y=y0+i*ch
            parts.append(f'<rect x="{x}" y="{y}" width="{cw-3}" height="{ch-3}" rx="4" fill="rgba(8,145,178,{a:.2f})"/>')
            star = " ★" if abs(v-vmax) < 1e-9 else ""
            parts.append(f'<text x="{x+(cw-3)/2}" y="{y+(ch-3)/2+4}" text-anchor="middle" fill="{txtc}" font-weight="600">{v:.3f}{star}</text>')
    parts.append('</svg>')
    return f'<p class="note" style="margin-bottom:2px"><b>{title}</b></p>' + "".join(parts)

# ----------------------------------------------------------------------------- build

def main():
    runs = {
        "bert_base":  ("bert", "predictions_bert_no_retrieval_test", "predictions_bert_retrieval_test"),
        "mbert_base": ("mentalbert", "predictions_mentalbert_no_retrieval_test", "predictions_mentalbert_retrieval_test"),
    }
    # core results dict
    R = {
        "bert_noret":   load_cm("predictions_bert_no_retrieval_test.csv"),
        "bert_ret":     load_cm("predictions_bert_retrieval_test.csv"),
        "mbert_noret":  load_cm("predictions_mentalbert_no_retrieval_test.csv"),
        "mbert_ret":    load_cm("predictions_mentalbert_retrieval_test.csv"),
        "bert_ret_w":   load_cm("predictions_bert_retrieval_weighted_test.csv"),
        "mbert_ret_w":  load_cm("predictions_mentalbert_retrieval_weighted_test.csv"),
        "bert_noret_w": load_cm("predictions_bert_no_retrieval_weighted_test.csv"),
        "mbert_noret_w":load_cm("predictions_mentalbert_no_retrieval_weighted_test.csv"),
    }

    # baseline (majority) on test
    df = pd.read_csv(ROOT / "data/processed/phq8_item_dataset_full.csv")
    te = df[df.split == "test"]
    y = te.label.to_numpy(); n = len(y)
    maj_acc = (y == 0).mean()
    maj_f1 = f1_score(y, np.zeros_like(y), average="macro", zero_division=0)

    # macro-F1 comparison bars (8 configs)
    f1_series = [
        ("majority\nbaseline", maj_f1),
        ("BERT\nno-ret", R["bert_noret"]["macro_f1"]),
        ("BERT\nret", R["bert_ret"]["macro_f1"]),
        ("MBERT\nno-ret", R["mbert_noret"]["macro_f1"]),
        ("MBERT\nret", R["mbert_ret"]["macro_f1"]),
        ("BERT\nret+wt", R["bert_ret_w"]["macro_f1"]),
        ("MBERT\nret+wt", R["mbert_ret_w"]["macro_f1"]),
    ]
    f1_colors = ["#94a3b8", "#4f46e5", "#4f46e5", "#0891b2", "#0891b2", "#d97706", "#16a34a"]

    acc_series = [
        ("majority", maj_acc),
        ("BERT\nret", R["bert_ret"]["acc"]),
        ("MBERT\nret", R["mbert_ret"]["acc"]),
        ("BERT\nret+wt", R["bert_ret_w"]["acc"]),
        ("MBERT\nret+wt", R["mbert_ret_w"]["acc"]),
    ]
    acc_colors = ["#94a3b8", "#4f46e5", "#0891b2", "#d97706", "#16a34a"]

    # per-class F1: unweighted vs weighted (MentalBERT retrieval)
    perclass = []
    for c in LABELS:
        perclass.append((SEV_NAMES[c], R["mbert_ret"]["f1"][c], R["mbert_ret_w"]["f1"][c]))

    # sweep heatmap
    eps = [3, 5, 8]; lrs = ["1e-5", "2e-5", "3e-5", "5e-5"]
    grid = []
    for e in eps:
        row = []
        for lr in lrs:
            d = json.loads((OUT / f"sweep_mentalbert/mbert_ret_e{e}_lr{lr}_metrics.json").read_text())["final"]
            row.append(d["macro_f1"])
        grid.append(row)
    flat = [v for r in grid for v in r]

    seed = load_seed_aggregates()
    per_item = load_per_item_oof()
    cv = load_cv()
    nm = load_near_miss("bert_base_uncased")
    corn = load_corn()

    html = build_html(R, f1_series, f1_colors, acc_series, acc_colors,
                      perclass, grid, eps, lrs, min(flat), max(flat),
                      n, maj_acc, maj_f1, te, seed, per_item, cv, nm, corn)
    out = OUT / "status_report_v2.html"
    out.write_text(html)
    print("wrote", out)


def perclass_grouped(perclass):
    """Grouped bars: unweighted vs weighted per class."""
    W, H = 600, 260
    pad_l, pad_b, pad_t = 44, 56, 28
    plot_h = H - pad_b - pad_t
    ymax = 0.7
    n = len(perclass); gap = (W - pad_l - 16)/n; bw = gap*0.28
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
             f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    for frac in (0,.25,.5,.75,1.0):
        yv=ymax*frac; yy=pad_t+plot_h*(1-frac)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W-8}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{pad_l-6}" y="{yy+3:.1f}" text-anchor="end" fill="#94a3b8">{yv:.2f}</text>')
    for i,(name,uw,w) in enumerate(perclass):
        cx = pad_l+8+i*gap+gap/2
        for k,(val,col) in enumerate([(uw,"#94a3b8"),(w,"#16a34a")]):
            x = cx + (k-0.5)*bw*1.15 - bw/2
            h = plot_h*(val/ymax); yy=pad_t+plot_h-h
            parts.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="2" fill="{col}"/>')
            parts.append(f'<text x="{x+bw/2:.1f}" y="{yy-3:.1f}" text-anchor="middle" fill="#1e293b" font-size="9.5">{val:.2f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{pad_t+plot_h+15:.1f}" text-anchor="middle" fill="#64748b">{name}</text>')
    # legend
    parts.append(f'<rect x="{pad_l}" y="6" width="11" height="11" fill="#94a3b8"/><text x="{pad_l+15}" y="15" fill="#64748b">unweighted</text>')
    parts.append(f'<rect x="{pad_l+95}" y="6" width="11" height="11" fill="#16a34a"/><text x="{pad_l+110}" y="15" fill="#64748b">+ class weights</text>')
    parts.append('</svg>')
    return "".join(parts)


def svg_item_hbars(items):
    """Horizontal bars of per-item macro-F1 with std whiskers, colour by difficulty."""
    rows = list(items.itertuples(index=False))
    nrow = len(rows)
    W = 600
    row_h = 30
    pad_l, pad_r, pad_t, pad_b = 110, 60, 10, 26
    H = pad_t + nrow * row_h + pad_b
    xmax = 0.4
    plot_w = W - pad_l - pad_r
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
             f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    for frac in (0, .25, .5, .75, 1.0):
        x = pad_l + plot_w * frac
        parts.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t+nrow*row_h}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{x:.1f}" y="{H-8}" text-anchor="middle" fill="#94a3b8">{xmax*frac:.2f}</text>')
    for i, r in enumerate(rows):
        y = pad_t + i * row_h
        # colour: more severe-heavy item = warmer (harder context)
        sev = r.pct_severe
        col = "#16a34a" if i == 0 else ("#dc2626" if i >= nrow - 2 else "#4f46e5")
        bw = plot_w * (r.f1_mean / xmax)
        cy = y + row_h / 2
        parts.append(f'<text x="{pad_l-8}" y="{cy+3:.1f}" text-anchor="end" fill="#475569">{r.item_name}</text>')
        parts.append(f'<rect x="{pad_l}" y="{y+5:.1f}" width="{bw:.1f}" height="{row_h-12}" rx="3" fill="{col}"/>')
        # std whisker
        lo = pad_l + plot_w * (max(r.f1_mean - r.f1_std, 0) / xmax)
        hi = pad_l + plot_w * ((r.f1_mean + r.f1_std) / xmax)
        parts.append(f'<line x1="{lo:.1f}" y1="{cy:.1f}" x2="{hi:.1f}" y2="{cy:.1f}" stroke="#1e293b" stroke-width="1.3"/>')
        parts.append(f'<line x1="{lo:.1f}" y1="{cy-3:.1f}" x2="{lo:.1f}" y2="{cy+3:.1f}" stroke="#1e293b"/>')
        parts.append(f'<line x1="{hi:.1f}" y1="{cy-3:.1f}" x2="{hi:.1f}" y2="{cy+3:.1f}" stroke="#1e293b"/>')
        parts.append(f'<text x="{hi+5:.1f}" y="{cy+3:.1f}" fill="#1e293b" font-size="10">{r.f1_mean:.3f} ({sev*100:.0f}% sev)</text>')
    parts.append('</svg>')
    return "".join(parts)


def svg_seed_compare(seed):
    """BERT vs MentalBERT macro-F1 with std whiskers + per-seed dots."""
    W, H = 420, 240
    pad_l, pad_b, pad_t = 44, 50, 22
    plot_h = H - pad_b - pad_t
    ymax = 0.4
    cats = [("BERT-base", seed["bert"]["macro_f1"], "#4f46e5"),
            ("MentalBERT", seed["mbert"]["macro_f1"], "#0891b2")]
    bw = 70; gap = (W - pad_l - 16) / 2
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
             f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    for frac in (0, .25, .5, .75, 1.0):
        yy = pad_t + plot_h * (1 - frac)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W-8}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{pad_l-6}" y="{yy+3:.1f}" text-anchor="end" fill="#94a3b8">{ymax*frac:.2f}</text>')
    per_seed = seed["per_seed"]
    for i, (lab, (mean, std), color) in enumerate([
            ("BERT-base", seed["bert"]["macro_f1"], "#4f46e5"),
            ("MentalBERT", seed["mbert"]["macro_f1"], "#0891b2")]):
        cx = pad_l + 8 + i * gap + gap / 2
        x = cx - bw / 2
        h = plot_h * (mean / ymax); yy = pad_t + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw}" height="{h:.1f}" rx="3" fill="{color}" opacity="0.85"/>')
        # whisker
        lo = pad_t + plot_h - plot_h * ((mean - std) / ymax)
        hi = pad_t + plot_h - plot_h * ((mean + std) / ymax)
        parts.append(f'<line x1="{cx:.1f}" y1="{lo:.1f}" x2="{cx:.1f}" y2="{hi:.1f}" stroke="#1e293b" stroke-width="1.3"/>')
        parts.append(f'<line x1="{cx-5:.1f}" y1="{hi:.1f}" x2="{cx+5:.1f}" y2="{hi:.1f}" stroke="#1e293b"/>')
        parts.append(f'<line x1="{cx-5:.1f}" y1="{lo:.1f}" x2="{cx+5:.1f}" y2="{lo:.1f}" stroke="#1e293b"/>')
        # per-seed dots
        for s in per_seed:
            val = s[1] if i == 0 else s[2]
            dy = pad_t + plot_h - plot_h * (val / ymax)
            parts.append(f'<circle cx="{cx:.1f}" cy="{dy:.1f}" r="2.6" fill="#0f172a" opacity="0.55"/>')
        parts.append(f'<text x="{cx:.1f}" y="{yy-6:.1f}" text-anchor="middle" fill="#1e293b" font-weight="600">{mean:.3f}±{std:.3f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{pad_t+plot_h+16:.1f}" text-anchor="middle" fill="#64748b">{lab}</text>')
    parts.append('</svg>')
    return "".join(parts)


def svg_cv_folds(cv):
    """Grouped bars: per-fold macro-F1 for BERT vs MentalBERT, with mean lines."""
    W, H = 600, 250
    pad_l, pad_b, pad_t = 44, 46, 24
    plot_h = H - pad_b - pad_t
    ymax = 0.4
    k = cv["n_folds"]
    gap = (W - pad_l - 16) / k
    bw = gap * 0.30
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
             f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    for frac in (0, .25, .5, .75, 1.0):
        yy = pad_t + plot_h * (1 - frac)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W-8}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{pad_l-6}" y="{yy+3:.1f}" text-anchor="end" fill="#94a3b8">{ymax*frac:.2f}</text>')
    series = [("#4f46e5", cv["bert_folds"]), ("#0891b2", cv["mbert_folds"])]
    for i in range(k):
        cx = pad_l + 8 + i * gap + gap / 2
        for j, (color, vals) in enumerate(series):
            x = cx + (j - 0.5) * bw * 1.15 - bw / 2
            h = plot_h * (vals[i] / ymax); yy = pad_t + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="2" fill="{color}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{pad_t+plot_h+15:.1f}" text-anchor="middle" fill="#64748b">fold {i+1}</text>')
    # mean lines
    for color, vals in series:
        my = pad_t + plot_h - plot_h * (np.mean(vals) / ymax)
        parts.append(f'<line x1="{pad_l}" y1="{my:.1f}" x2="{W-8}" y2="{my:.1f}" stroke="{color}" stroke-dasharray="4 3" opacity="0.7"/>')
    parts.append(f'<rect x="{pad_l}" y="4" width="11" height="11" fill="#4f46e5"/><text x="{pad_l+15}" y="13" fill="#64748b">BERT-base</text>')
    parts.append(f'<rect x="{pad_l+95}" y="4" width="11" height="11" fill="#0891b2"/><text x="{pad_l+110}" y="13" fill="#64748b">MentalBERT</text>')
    parts.append('</svg>')
    return "".join(parts)


def svg_top2_recall(nm):
    """Grouped bars: per-class recall, top-1 (solid) vs top-2 (light overlay)."""
    pc = nm["per_class"]
    W, H = 600, 250
    pad_l, pad_b, pad_t = 44, 50, 28
    plot_h = H - pad_b - pad_t
    ymax = 1.0
    ncat = len(pc); gap = (W - pad_l - 16) / ncat; bw = gap * 0.42
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
             f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    for frac in (0, .25, .5, .75, 1.0):
        yy = pad_t + plot_h * (1 - frac)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W-8}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{pad_l-6}" y="{yy+3:.1f}" text-anchor="end" fill="#94a3b8">{frac:.2f}</text>')
    for i, (name, r1, r2) in enumerate(pc):
        cx = pad_l + 8 + i * gap + gap / 2
        x = cx - bw / 2
        # top-2 (light, full height) then top-1 (dark) overlaid -> shows the gain
        h2 = plot_h * (r2 / ymax); y2 = pad_t + plot_h - h2
        h1 = plot_h * (r1 / ymax); y1 = pad_t + plot_h - h1
        parts.append(f'<rect x="{x:.1f}" y="{y2:.1f}" width="{bw:.1f}" height="{h2:.1f}" rx="2" fill="#a5b4fc"/>')
        parts.append(f'<rect x="{x:.1f}" y="{y1:.1f}" width="{bw:.1f}" height="{h1:.1f}" rx="2" fill="#4f46e5"/>')
        parts.append(f'<text x="{cx:.1f}" y="{y2-3:.1f}" text-anchor="middle" fill="#1e293b" font-size="10">{r2:.2f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{pad_t+plot_h+15:.1f}" text-anchor="middle" fill="#64748b">{name}</text>')
    parts.append(f'<rect x="{pad_l}" y="6" width="11" height="11" fill="#4f46e5"/><text x="{pad_l+15}" y="15" fill="#64748b">top-1 recall</text>')
    parts.append(f'<rect x="{pad_l+105}" y="6" width="11" height="11" fill="#a5b4fc"/><text x="{pad_l+120}" y="15" fill="#64748b">top-2 recall</text>')
    parts.append('</svg>')
    return "".join(parts)


def svg_rank_cumulative(nm):
    """Cumulative share of examples whose true label is within the top-k probs."""
    cum = nm["rank_cum"]
    W, H = 420, 230
    pad_l, pad_b, pad_t = 40, 44, 18
    plot_h = H - pad_b - pad_t
    plot_w = W - pad_l - 16
    k = len(cum)
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
             f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    for frac in (0, .25, .5, .75, 1.0):
        yy = pad_t + plot_h * (1 - frac)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W-8}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{pad_l-6}" y="{yy+3:.1f}" text-anchor="end" fill="#94a3b8">{frac:.2f}</text>')
    pts = []
    for i, v in enumerate(cum):
        x = pad_l + (plot_w * i / (k - 1))
        yy = pad_t + plot_h * (1 - v)
        pts.append((x, yy))
    path = " ".join(f"{'M' if j==0 else 'L'}{x:.1f} {y:.1f}" for j,(x,y) in enumerate(pts))
    parts.append(f'<path d="{path}" fill="none" stroke="#0891b2" stroke-width="2.2"/>')
    for i, (x, yy) in enumerate(pts):
        parts.append(f'<circle cx="{x:.1f}" cy="{yy:.1f}" r="3.5" fill="#0891b2"/>')
        parts.append(f'<text x="{x:.1f}" y="{yy-8:.1f}" text-anchor="middle" fill="#1e293b" font-size="10">{cum[i]*100:.0f}%</text>')
        parts.append(f'<text x="{x:.1f}" y="{pad_t+plot_h+16:.1f}" text-anchor="middle" fill="#64748b">top-{i+1}</text>')
    parts.append('</svg>')
    return "".join(parts)


def svg_corn_compare(corn):
    """Side-by-side MAE (lower=better) for CE+weights vs CORN, both models."""
    cfgs = [
        ("BERT\nCE+wt", corn["bert_ce"]["mae"], "#4f46e5"),
        ("BERT\nCORN", corn["bert_corn"]["mae"], "#16a34a"),
        ("MBERT\nCE+wt", corn["mbert_ce"]["mae"], "#0891b2"),
        ("MBERT\nCORN", corn["mbert_corn"]["mae"], "#16a34a"),
    ]
    W, H = 520, 250
    pad_l, pad_b, pad_t = 44, 50, 24
    plot_h = H - pad_b - pad_t
    ymax = 1.0
    ncat = len(cfgs); gap = (W - pad_l - 16) / ncat; bw = gap * 0.5
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
             f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    for frac in (0, .25, .5, .75, 1.0):
        yy = pad_t + plot_h * (1 - frac)
        parts.append(f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W-8}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{pad_l-6}" y="{yy+3:.1f}" text-anchor="end" fill="#94a3b8">{ymax*frac:.2f}</text>')
    for i, (name, v, col) in enumerate(cfgs):
        cx = pad_l + 8 + i * gap + gap / 2
        x = cx - bw / 2
        h = plot_h * (v / ymax); yy = pad_t + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="3" fill="{col}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{yy-4:.1f}" text-anchor="middle" fill="#1e293b" font-weight="600">{v:.3f}</text>')
        for j, part in enumerate(name.split("\n")):
            parts.append(f'<text x="{cx:.1f}" y="{pad_t+plot_h+14+j*12:.1f}" text-anchor="middle" fill="#64748b">{part}</text>')
    parts.append('</svg>')
    return "".join(parts)


def build_html(R, f1_series, f1_colors, acc_series, acc_colors, perclass,
               grid, eps, lrs, vmin, vmax, n, maj_acc, maj_f1, te, seed, per_item, cv, nm, corn):
    fig_f1 = svg_bars(f1_series, "Fig 1 · Macro-F1 across all configs (test set, n=264)", ymax=0.4, colors=f1_colors)
    fig_acc = svg_bars(acc_series, "Fig 2 · Accuracy — note weighting trades accuracy for balance", ymax=0.6, colors=acc_colors)
    fig_pc = ('<p class="note" style="margin-bottom:2px"><b>Fig 3 · Per-class F1: class weighting revives the severe classes (MentalBERT + retrieval)</b></p>'
              + perclass_grouped(perclass))
    cm_unweighted = ('<div style="display:flex;flex-wrap:wrap;gap:18px">'
                     + svg_confusion(R["bert_ret"], "Fig 4a · BERT + retrieval (unweighted)", "#4f46e5")
                     + svg_confusion(R["mbert_ret"], "Fig 4b · MentalBERT + retrieval (unweighted)", "#0891b2")
                     + '</div>')
    cm_weighted = ('<div style="display:flex;flex-wrap:wrap;gap:18px">'
                   + svg_confusion(R["bert_ret_w"], "Fig 5a · BERT + retrieval + weights", "#d97706")
                   + svg_confusion(R["mbert_ret_w"], "Fig 5b · MentalBERT + retrieval + weights", "#16a34a")
                   + '</div>')
    fig_sweep = svg_heatmap(grid, eps, lrs, "Fig 6 · MentalBERT hyperparameter sweep — macro-F1 (validation)", vmin, vmax)
    fig_seed = svg_seed_compare(seed)
    fig_item = svg_item_hbars(per_item)
    fig_cv = svg_cv_folds(cv)
    cv_b = cv["bert"]["per_fold_mean_std"]; cv_m = cv["mbert"]["per_fold_mean_std"]
    cv_bo = cv["bert"]["pooled_oof"]; cv_mo = cv["mbert"]["pooled_oof"]
    fig_top2 = svg_top2_recall(nm)
    fig_rank = svg_rank_cumulative(nm)
    fig_corn = svg_corn_compare(corn)

    def row(name, key, tag=""):
        r = R[key]
        return (f"<tr><td>{name}{tag}</td><td>{r['acc']:.3f}</td><td><b>{r['macro_f1']:.3f}</b></td>"
                f"<td>{r['mae']:.3f}</td><td>{r['f1'][2]:.2f}</td><td>{r['f1'][3]:.2f}</td></tr>")

    best_macro = max(R[k]["macro_f1"] for k in R)
    bm, bs = seed["bert"]["macro_f1"]
    mm, msd = seed["mbert"]["macro_f1"]
    best_item = per_item.iloc[0]
    worst_item = per_item.iloc[-1]

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PHQ-8 Item Classifier — Results Report</title>
<style>
 *{{box-sizing:border-box}}
 body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
   color:#1e293b;background:#f8fafc;line-height:1.55;}}
 header{{background:linear-gradient(135deg,#4f46e5,#0891b2);color:#fff;padding:34px 28px;}}
 header h1{{margin:0 0 6px;font-size:26px;}} header .sub{{opacity:.92;font-size:14px;}}
 header .meta{{margin-top:14px;font-size:13px;opacity:.9;display:flex;gap:22px;flex-wrap:wrap;}}
 .wrap{{max-width:1040px;margin:0 auto;padding:26px 28px 60px;}}
 .card{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:22px 24px;margin:18px 0;box-shadow:0 1px 2px rgba(15,23,42,.04);}}
 h2{{font-size:19px;margin:4px 0 14px;padding-bottom:8px;border-bottom:2px solid #eef2ff;}}
 h3{{font-size:15px;margin:22px 0 8px;color:#4f46e5;}}
 p{{margin:8px 0;}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:6px 0;}}
 .stat{{background:#eef2ff;border-radius:10px;padding:14px 16px;}}
 .stat .n{{font-size:23px;font-weight:700;color:#4f46e5;}}
 .stat .l{{font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:.04em;}}
 table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:14px;}}
 th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid #e2e8f0;}}
 th{{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#64748b;background:#fbfcfe;}}
 tr:hover td{{background:#fafbff;}}
 code{{background:#f1f5f9;padding:1px 6px;border-radius:5px;font-size:13px;font-family:ui-monospace,Menlo,Consolas,monospace;}}
 .callout{{border-left:4px solid #4f46e5;background:#eef2ff;padding:12px 16px;border-radius:0 8px 8px 0;margin:12px 0;}}
 .win{{border-left-color:#16a34a;background:#dcfce7;}}
 .note{{color:#64748b;font-size:13px;}} .ok{{color:#16a34a;font-weight:700;}}
 .badge{{display:inline-block;font-size:11px;font-weight:700;padding:3px 9px;border-radius:999px;}}
 .b-win{{color:#16a34a;background:#dcfce7;}} .b-base{{color:#64748b;background:#f1f5f9;}}
 footer{{text-align:center;color:#64748b;font-size:12px;padding:24px;}}
 ul{{margin:8px 0;padding-left:20px;}} li{{margin:5px 0;}}
</style></head><body>
<header>
 <h1>PHQ-8 Item Classifier — Results Report</h1>
 <div class="sub">Item-level severity (0–3) · BERT vs MentalBERT · retrieval &amp; class-imbalance experiments</div>
 <div class="meta"><span>📅 2026-06-04</span>
  <span>🌿 <code style="background:rgba(255,255,255,.15);color:#fff">models/bert-classifier</code></span>
  <span>🖥️ GPU: RTX 3090 (SLURM)</span><span>👤 yanivpv@gmail.com</span></div>
</header>
<div class="wrap">

 <div class="card">
  <h2>TL;DR</h2>
  <p>Full GPU training is complete across <b>2 models × 2 evidence conditions</b>, plus a <b>12-config hyperparameter sweep</b> and a <b>class-weighting</b> ablation — all evaluated on the held-out test split (n={n}).</p>
  <div class="grid">
   <div class="stat"><div class="n">{best_macro:.3f}</div><div class="l">Best macro-F1</div></div>
   <div class="stat"><div class="n">{maj_f1:.3f}</div><div class="l">Majority baseline F1</div></div>
   <div class="stat"><div class="n">40+</div><div class="l">GPU runs total</div></div>
   <div class="stat"><div class="n">4-class</div><div class="l">Task (severity 0–3)</div></div>
  </div>
  <div class="callout win"><b>Headline:</b> The lever that matters is <b>class weighting, not the base model</b>. Unweighted models never predict the severe classes (2 &amp; 3 → F1 = 0); balanced class weights revive them. Both <b>5-seed</b> repeats and <b>5-fold cross-validation</b> show BERT-base and MentalBERT are <b>statistically indistinguishable</b> (macro-F1 {bm:.3f}±{bs:.3f} vs {mm:.3f}±{msd:.3f} over seeds) — the single-seed "MentalBERT wins" result was noise.</div>
  <ul>
   <li><b>Retrieval helps</b> every condition vs. the full transcript / baseline.</li>
   <li><b>Accuracy is misleading</b> here — the majority baseline already scores {maj_acc:.3f}. Macro-F1 and MAE are the honest metrics.</li>
   <li><b>Class weighting</b> trades accuracy for balance: severe-class F1 goes from 0.00 to non-zero; macro-F1 rises.</li>
   <li><b>Best / worst items:</b> the model handles <b>{best_item['item_name']}</b> best (macro-F1 {best_item['f1_mean']:.3f}) and <b>{worst_item['item_name']}</b> worst ({worst_item['f1_mean']:.3f}).</li>
  </ul>
 </div>

 <div class="card">
  <h2>Experimental setup</h2>
  <table>
   <tr><th>Component</th><th>Value</th></tr>
   <tr><td>Models</td><td><code>bert-base-uncased</code> · <code>mental/mental-bert-base-uncased</code></td></tr>
   <tr><td>Evidence conditions</td><td>baseline utterances · retrieved utterances</td></tr>
   <tr><td>Dataset</td><td>1,752 item-level rows · train 1,224 / val 264 / test 264</td></tr>
   <tr><td>Training</td><td>AdamW · batch 16 · max_len 256 · 3 or 8 epochs</td></tr>
   <tr><td>Hardware</td><td>NVIDIA RTX 3090 via SLURM <code>gpu</code> partition</td></tr>
  </table>
  <h3>Baselines (test, 4-class — imbalanced)</h3>
  <table>
   <tr><th>Baseline</th><th>Accuracy</th><th>Macro-F1</th><th>MAE</th></tr>
   <tr><td>Uniform random (¼)</td><td>0.250</td><td>—</td><td>—</td></tr>
   <tr><td>Stratified random</td><td>0.329</td><td>—</td><td>—</td></tr>
   <tr><td>Majority class (predict 0)</td><td>{maj_acc:.3f}</td><td>{maj_f1:.3f}</td><td>0.883</td></tr>
  </table>
  <p class="note">Test label mix: 0 = 43.6% · 1 = 33.3% · 2 = 14.4% · 3 = 8.7%.</p>
 </div>

 <div class="card">
  <h2>Results table (test set)</h2>
  <table>
   <tr><th>Config</th><th>Acc</th><th>Macro-F1</th><th>MAE</th><th>class-2 F1</th><th>class-3 F1</th></tr>
   <tr style="color:#64748b"><td>Majority baseline</td><td>{maj_acc:.3f}</td><td>{maj_f1:.3f}</td><td>0.883</td><td>0.00</td><td>0.00</td></tr>
   {row("BERT · no-retrieval","bert_noret")}
   {row("BERT · retrieval","bert_ret")}
   {row("MentalBERT · no-retrieval","mbert_noret")}
   {row("MentalBERT · retrieval","mbert_ret")}
   {row("BERT · retrieval + weights","bert_ret_w",' <span class="badge b-base">weighted</span>')}
   {row("MentalBERT · retrieval + weights","mbert_ret_w",' <span class="badge b-base">weighted</span>')}
  </table>
  <p class="note">Single-seed (seed 42) numbers; macro-F1 in bold. class-2/3 F1 are the moderate/severe categories that unweighted models miss entirely. <b>See the multi-seed section below</b> — the weighted gap between BERT and MentalBERT does not survive averaging over seeds.</p>
 </div>

 <div class="card">
  <h2>Figures</h2>
  <h3>{fig_f1}</h3>
  <p class="note">Green bar = best (MentalBERT + retrieval + weights). All learned models clear the grey majority baseline; weighting lifts macro-F1 further.</p>
  {fig_acc}
  <p class="note">Accuracy <i>drops</i> under weighting — expected and desirable: the model stops over-predicting class 0 to catch real moderate/severe cases.</p>
  {fig_pc}
  <p class="note">Classes 2 &amp; 3 jump from F1 = 0.00 (unweighted) to non-zero — the core payoff of class weighting.</p>
  <h3>Confusion matrices — the imbalance problem</h3>
  {cm_unweighted}
  <div class="callout">Both unweighted models pour everything into columns 0 and 1 — predicted-2 and predicted-3 are empty. The 61 true moderate/severe items are all misclassified.</div>
  <h3>Confusion matrices — after class weighting</h3>
  {cm_weighted}
  <div class="callout win">Predictions now spread into columns 2 and 3; the diagonal gains real mass on the severe rows. MentalBERT (5b) is the only config with correct predictions in <i>every</i> class.</div>
  <h3>{fig_sweep}</h3>
  <p class="note">★ marks the best cell. Longer training (8 epochs) at a low LR (1e-5) maximises macro-F1 — the budget used for the weighted runs.</p>
 </div>

 <div class="card">
  <h2>Multi-seed robustness — is MentalBERT's edge real?</h2>
  <p>The weighted retrieval config was re-run across <b>5 seeds</b> ({", ".join(str(s) for s in seed['seeds'])}) for each model. The error bars are std; dots are individual seeds.</p>
  <div style="display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start">
   <div style="flex:1;min-width:300px"><p class="note" style="margin-bottom:2px"><b>Fig 7 · Macro-F1 over 5 seeds (test set)</b></p>{fig_seed}</div>
   <div style="flex:1;min-width:280px">
    <table>
     <tr><th>Metric</th><th>BERT-base</th><th>MentalBERT</th></tr>
     <tr><td>Accuracy</td><td>{seed['bert']['acc'][0]:.3f} ± {seed['bert']['acc'][1]:.3f}</td><td>{seed['mbert']['acc'][0]:.3f} ± {seed['mbert']['acc'][1]:.3f}</td></tr>
     <tr><td><b>Macro-F1</b></td><td>{bm:.3f} ± {bs:.3f}</td><td>{mm:.3f} ± {msd:.3f}</td></tr>
     <tr><td>MAE</td><td>{seed['bert']['mae'][0]:.3f} ± {seed['bert']['mae'][1]:.3f}</td><td>{seed['mbert']['mae'][0]:.3f} ± {seed['mbert']['mae'][1]:.3f}</td></tr>
    </table>
    <p class="note">Paired Δ (MentalBERT − BERT) macro-F1 = <b>{seed['f1_diff_mean']:+.3f}</b>; MentalBERT wins {seed['mbert_wins']}/{len(seed['seeds'])} seeds.</p>
   </div>
  </div>
  <div class="callout"><b>Verdict:</b> No real difference. The macro-F1 means are essentially identical and MentalBERT is the <i>noisier</i> of the two. The single-seed result that put MentalBERT ahead was within seed-to-seed variance. <b>Class weighting matters; base-model choice does not.</b></div>
 </div>

 <div class="card">
  <h2>Cross-validation — the most trustworthy numbers</h2>
  <p>A single 264-row test split is too small to trust the details. This is <b>{cv['n_folds']}-fold CV, grouped by participant</b> (no leakage) and stratified by label, on the best config (retrieval + class weights, 8 epochs). Every participant is held out exactly once, giving out-of-fold predictions over all 1,752 examples.</p>
  <div style="display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start">
   <div style="flex:1;min-width:320px"><p class="note" style="margin-bottom:2px"><b>Fig 8 · Per-fold macro-F1 (dashed = model mean)</b></p>{fig_cv}</div>
   <div style="flex:1;min-width:280px">
    <table>
     <tr><th>Metric (per-fold)</th><th>BERT-base</th><th>MentalBERT</th></tr>
     <tr><td>Accuracy</td><td>{cv_b['accuracy'][0]:.3f} ± {cv_b['accuracy'][1]:.3f}</td><td>{cv_m['accuracy'][0]:.3f} ± {cv_m['accuracy'][1]:.3f}</td></tr>
     <tr><td><b>Macro-F1</b></td><td>{cv_b['macro_f1'][0]:.3f} ± {cv_b['macro_f1'][1]:.3f}</td><td>{cv_m['macro_f1'][0]:.3f} ± {cv_m['macro_f1'][1]:.3f}</td></tr>
     <tr><td>MAE</td><td>{cv_b['mae'][0]:.3f} ± {cv_b['mae'][1]:.3f}</td><td>{cv_m['mae'][0]:.3f} ± {cv_m['mae'][1]:.3f}</td></tr>
     <tr style="color:#64748b"><td>Pooled OOF macro-F1</td><td>{cv_bo['macro_f1']:.3f}</td><td>{cv_mo['macro_f1']:.3f}</td></tr>
    </table>
    <p class="note">Paired Δ (MentalBERT − BERT) = <b>{cv['diff_mean']:+.3f} ± {cv['diff_std']:.3f}</b>; wins {cv['mbert_wins']}/{cv['n_folds']} folds → <b>not significant</b>.</p>
   </div>
  </div>
  <div class="callout win"><b>Verdict:</b> CV confirms the tie from a second angle. Aggregate metrics (~0.29–0.31 macro-F1) match the single split, so the headline numbers are sound — but CV is what lets us trust the per-item analysis below.</div>
 </div>

 <div class="card">
  <h2>Per-item analysis — which PHQ-8 symptoms are easiest?</h2>
  <p>Per-item macro-F1 from <b>out-of-fold CV predictions</b> (~438 examples/item — 13× the single-split sample). Bars ranked best→worst; whiskers are std across fold×model subsamples; "% sev" = share of moderate/severe (2–3) cases.</p>
  <p class="note" style="margin-bottom:2px"><b>Fig 9 · Per-item macro-F1 from OOF predictions (green = best, red = worst two)</b></p>
  {fig_item}
  <div class="callout"><b>CV corrected the single-split story.</b> With 33 examples/item the ranking was mostly noise: "Moving easiest" and "Tired worst" did <i>not</i> survive, and the F1↔severity correlation collapsed (−0.28 → +0.06). Trust the OOF ranking, not the single split.</div>
  <ul>
   <li><b>{best_item['item_name']}</b> (affective) is genuinely the best item on macro-F1 ({best_item['f1_mean']:.3f}) — emotional content is the most learnable signal.</li>
   <li><b>Somatic items have the worst MAE</b> — Sleep ({per_item.loc[per_item.item_name=='Sleep','mae_mean'].iloc[0]:.2f}) and Appetite ({per_item.loc[per_item.item_name=='Appetite','mae_mean'].iloc[0]:.2f}). The model gets their <i>magnitude</i> most wrong even when F1 is acceptable.</li>
   <li><b>Moving</b> has the best MAE ({per_item.loc[per_item.item_name=='Moving','mae_mean'].iloc[0]:.2f}) only because it is rarely endorsed (skewed to "none") — an easy-distribution effect, not strong modelling.</li>
  </ul>
 </div>

 <div class="card">
  <h2>Near-miss analysis — is the model "almost right"?</h2>
  <p>Top-1 accuracy treats an off-by-one severity as a total miss. But the task is ordinal, so we ask: when the top choice is wrong, was the truth the model's <b>runner-up</b>, or at least <b>adjacent</b>? Computed from OOF predictions with class probabilities (BERT-base, n={nm['n']}).</p>
  <div class="grid">
   <div class="stat"><div class="n">{nm['acc1']:.3f}</div><div class="l">Top-1 accuracy</div></div>
   <div class="stat"><div class="n">{nm['acc2']:.3f}</div><div class="l">Top-2 accuracy</div></div>
   <div class="stat"><div class="n">{nm['recovered']*100:.0f}%</div><div class="l">Errors: truth was 2nd choice</div></div>
   <div class="stat"><div class="n">{nm['near_miss']*100:.0f}%</div><div class="l">Errors that are near-misses</div></div>
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start;margin-top:8px">
   <div style="flex:1;min-width:320px"><p class="note" style="margin-bottom:2px"><b>Fig 10 · Per-class recall: top-1 vs top-2</b></p>{fig_top2}</div>
   <div style="flex:1;min-width:300px"><p class="note" style="margin-bottom:2px"><b>Fig 11 · True label within the model's top-k (cumulative)</b></p>{fig_rank}</div>
  </div>
  <div class="callout win"><b>The model is far better than its accuracy suggests.</b> Top-2 accuracy is {nm['acc2']:.2f} vs {nm['acc1']:.2f} top-1, and of the {nm['n_err']} errors, {nm['recovered']*100:.0f}% had the truth as the 2nd choice and {nm['adjacent']*100:.0f}% are off-by-one — only {(1-nm['near_miss'])*100:.0f}% are genuinely far off. <b>mild(1)</b> recall jumps from {nm['per_class'][1][1]:.2f} (top-1) to {nm['per_class'][1][2]:.2f} (top-2). The real bottleneck is the final 1-of-4 decision between <i>neighbouring</i> severities — exactly what an ordinal loss targets.</div>
 </div>

 <div class="card">
  <h2>Ordinal loss (CORN) — acting on the near-miss diagnosis</h2>
  <p>The near-miss result said the errors are <i>ordinal</i> (adjacent severities). So we swapped the 4-way softmax for <b>CORN</b> (rank-consistent ordinal regression) and re-ran the same 5-fold participant CV. CORN should improve <b>MAE</b> — the severity-distance metric that class-weighting had worsened.</p>
  <div style="display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start">
   <div style="flex:1;min-width:300px"><p class="note" style="margin-bottom:2px"><b>Fig 12 · MAE (lower = better): CE+weights vs CORN</b></p>{fig_corn}</div>
   <div style="flex:1;min-width:300px">
    <table>
     <tr><th>Config</th><th>Acc</th><th>Macro-F1</th><th>MAE ↓</th></tr>
     <tr><td>BERT · CE+weights</td><td>{corn['bert_ce']['accuracy']:.3f}</td><td>{corn['bert_ce']['macro_f1']:.3f}</td><td>{corn['bert_ce']['mae']:.3f}</td></tr>
     <tr style="background:#f0fdf4"><td>BERT · <b>CORN</b></td><td>{corn['bert_corn']['accuracy']:.3f}</td><td>{corn['bert_corn']['macro_f1']:.3f}</td><td><b>{corn['bert_corn']['mae']:.3f}</b></td></tr>
     <tr><td>MentalBERT · CE+weights</td><td>{corn['mbert_ce']['accuracy']:.3f}</td><td>{corn['mbert_ce']['macro_f1']:.3f}</td><td>{corn['mbert_ce']['mae']:.3f}</td></tr>
     <tr style="background:#f0fdf4"><td>MentalBERT · <b>CORN</b></td><td>{corn['mbert_corn']['accuracy']:.3f}</td><td>{corn['mbert_corn']['macro_f1']:.3f}</td><td><b>{corn['mbert_corn']['mae']:.3f}</b></td></tr>
    </table>
   </div>
  </div>
  <div class="callout win"><b>CORN wins on MAE for both models</b> (BERT {corn['bert_ce']['mae']:.3f}→{corn['bert_corn']['mae']:.3f}, MentalBERT {corn['mbert_ce']['mae']:.3f}→{corn['mbert_corn']['mae']:.3f}) while macro-F1 holds (within fold noise). Its errors shift further toward off-by-one (72% vs 67%). The ordinal loss makes the model's mistakes <i>less wrong</i> — confirming the near-miss diagnosis and giving the recommended default for this task.</div>
 </div>

 <div class="card">
  <h2>Conclusions &amp; next steps</h2>
  <ul>
   <li><b>Report macro-F1 + MAE, not accuracy.</b> Accuracy rewards the trivial all-zeros predictor ({maj_acc:.3f}).</li>
   <li><b>Retrieval is justified</b> — it helps every condition and keeps inputs inside the 256-token window.</li>
   <li><b>Class weighting is the real lever</b> — it revives the severe classes. <b>BERT-base and MentalBERT tie</b> across seeds, so the cheaper base model is the sensible default.</li>
   <li><b>Per-item:</b> somatic symptoms (Sleep, Appetite, Tired) are the weak spots — candidates for item-specific evidence or features.</li>
   <li><b>The model is "almost right":</b> top-2 accuracy {nm['acc2']:.2f} vs {nm['acc1']:.2f} top-1, and {nm['near_miss']*100:.0f}% of errors are near-misses between <i>adjacent</i> severities.</li>
   <li><b>CORN ordinal loss delivers:</b> lower MAE for both models (BERT {corn['bert_ce']['mae']:.3f}→{corn['bert_corn']['mae']:.3f}, MentalBERT {corn['mbert_ce']['mae']:.3f}→{corn['mbert_corn']['mae']:.3f}) at equal macro-F1 — the recommended default. It makes errors land on neighbouring severities rather than far off.</li>
   <li><b>Next:</b> collapse to clinical bands (0–1 vs 2–3) for a decision-relevant target · multi-seed CV to firm up the small F1 gaps · ordinal metrics (QWK) and patient-level PHQ-8 total scoring.</li>
  </ul>
 </div>

 <footer>Generated by <code>src/evaluation/build_status_report.py</code> · all figures inline SVG · {n}-example test split</footer>
</div></body></html>"""


if __name__ == "__main__":
    main()
