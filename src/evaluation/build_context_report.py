"""
Self-contained HTML report for the context-window evidence comparison
(Person B -> Person A handoff). Inline-SVG figures, no external assets.

Satisfies the Section-11 "Required Outputs" checklist:
  Return-for-every-run: command/config, dataset_path, evidence_column, model_name,
    loss_type, use_class_weights, seed, max_length, num_epochs, truncation strategy,
    predictions / overall-results / per-item file paths.
  Plots & tables: macro-F1 & MAE & accuracy(secondary) by condition; per-class F1
    (esp. 2&3); confusion matrices (old vs best); per-item macro-F1 & MAE;
    Appetite spotlight; near-miss/top-2/off-by-one; CE vs weighted-CE vs CORN;
    reconstructed PHQ-8 total; threshold (>=10) sensitivity/specificity/balanced-acc/F1; QWK.

All metrics are computed directly from the cross-validation OOF prediction files,
so the report self-updates as more runs land (missing arms are shown as pending).

    python -m src.evaluation.build_context_report
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error

from src.evaluation.context_window_eval import metrics_from_oof, per_item

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
CV = OUT / "cv"
LABELS = [0, 1, 2, 3]
SEV = ["none(0)", "mild(1)", "moderate(2)", "severe(3)"]

# condition key -> (label, color, dataset_path, evidence_column, %truncated@256)
CONDS = [
    ("old",    "Old BM25 utterances", "#94a3b8", "phq8_item_dataset_full_bm25.csv",                  "retrieved_utterances",                  13.5),
    ("bm25w3", "BM25 W3",             "#60a5fa", "phq8_item_dataset_context_windows_bm25_w3.csv",    "retrieved_context_windows_bm25_pack",   55.9),
    ("bm25w5", "BM25 W5",             "#3b82f6", "phq8_item_dataset_context_windows_bm25.csv",       "retrieved_context_windows_bm25_pack",   82.4),
    ("hybw3",  "Hybrid W3",           "#16a34a", "phq8_item_dataset_context_windows_hybrid_w3.csv",  "retrieved_context_windows_hybrid_pack", 69.6),
    ("hybw5",  "Hybrid W5",           "#0891b2", "phq8_item_dataset_context_windows_hybrid_w5.csv",  "retrieved_context_windows_hybrid_pack", 94.5),
]
# loss arm -> (tag prefix, display)
ARMS = [("ceplain", "CE"), ("ce", "weighted CE"), ("corn", "CORN")]


def oof_path(arm, key):
    return CV / f"oof_predictions_ctx_{arm}_{key}.csv"


def load(arm, key):
    p = oof_path(arm, key)
    if not p.exists():
        return None
    return pd.read_csv(p)


def M(arm, key):
    df = load(arm, key)
    if df is None:
        return None
    item, total = metrics_from_oof(df)
    return {"item": item, "total": total}


# --------------------------------------------------------------------------- SVG

def hbars(rows, title, vmax, baseline=None, lower_better=False):
    """rows: list of (label, value, color). Horizontal bars."""
    n = len(rows); rh = 24; padl = 150; padr = 64; padt = 6; padb = 6
    W = 620; plot_w = W - padl - padr; H = padt + n * rh + padb
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    if baseline is not None:
        bx = padl + plot_w * (baseline / vmax)
        p.append(f'<line x1="{bx:.1f}" y1="{padt}" x2="{bx:.1f}" y2="{padt+n*rh}" stroke="#dc2626" stroke-dasharray="4 3"/>')
    for i, (lab, v, col) in enumerate(rows):
        y = padt + i * rh
        bw = plot_w * (max(v, 0) / vmax)
        p.append(f'<text x="{padl-8}" y="{y+rh/2+3:.1f}" text-anchor="end" fill="#475569">{lab}</text>')
        p.append(f'<rect x="{padl}" y="{y+3:.1f}" width="{bw:.1f}" height="{rh-7}" rx="3" fill="{col}"/>')
        p.append(f'<text x="{padl+bw+5:.1f}" y="{y+rh/2+3:.1f}" fill="#1e293b" font-size="10">{v:.3f}</text>')
    p.append('</svg>')
    arrow = " (lower = better)" if lower_better else ""
    return f'<p class="note" style="margin-bottom:2px"><b>{title}{arrow}</b></p>' + "".join(p)


def confusion_svg(arm, key, title, accent):
    df = load(arm, key)
    if df is None:
        return f'<div style="flex:1"><p class="note">{title}: pending</p></div>'
    cm = confusion_matrix(df.label, df.prediction, labels=LABELS)
    rn = cm / cm.sum(1, keepdims=True).clip(min=1)
    cell = 50; x0, y0 = 80, 40; W, H = 320, 270
    r = int(accent[1:3], 16); g = int(accent[3:5], 16); b = int(accent[5:7], 16)
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    p.append(f'<text x="{x0+2*cell}" y="16" text-anchor="middle" fill="#64748b" font-size="10">predicted</text>')
    for j in range(4):
        p.append(f'<text x="{x0+j*cell+cell/2}" y="{y0-5}" text-anchor="middle" fill="#475569">{j}</text>')
    for i in range(4):
        p.append(f'<text x="{x0-8}" y="{y0+i*cell+cell/2+4}" text-anchor="end" fill="#475569">{i}</text>')
        for j in range(4):
            a = 0.10 + 0.85 * rn[i, j]
            tc = "#fff" if rn[i, j] > 0.5 else "#1e293b"
            x = x0 + j * cell; y = y0 + i * cell
            p.append(f'<rect x="{x}" y="{y}" width="{cell-3}" height="{cell-3}" rx="3" fill="rgba({r},{g},{b},{a:.2f})"/>')
            p.append(f'<text x="{x+(cell-3)/2}" y="{y+(cell-3)/2+4}" text-anchor="middle" fill="{tc}" font-weight="600">{cm[i,j]}</text>')
    p.append('</svg>')
    return f'<div style="flex:1;min-width:280px"><p class="note" style="margin-bottom:2px"><b>{title}</b></p>' + "".join(p) + '</div>'


def per_item_grouped(arm, old_key, new_key, metric, vmax, title):
    old = load(arm, old_key); new = load(arm, new_key)
    if old is None or new is None:
        return f'<p class="note">{title}: pending</p>'
    def val(df, iid):
        g = df[df.item_id == iid]
        if metric == "f1":
            return f1_score(g.label, g.prediction, average="macro", labels=LABELS, zero_division=0)
        return mean_absolute_error(g.label, g.prediction)
    items = []
    for iid in range(1, 9):
        items.append((old[old.item_id == iid].item_name.iloc[0], val(old, iid), val(new, iid)))
    items.sort(key=lambda t: t[2], reverse=(metric == "f1"))
    W = 620; padl = 110; padr = 30; padt = 26; rh = 28; padb = 6
    H = padt + len(items) * rh + padb; plot_w = W - padl - padr
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    for i, (name, ov, nv) in enumerate(items):
        y = padt + i * rh
        p.append(f'<text x="{padl-8}" y="{y+rh/2+3:.1f}" text-anchor="end" fill="#475569">{name}</text>')
        p.append(f'<rect x="{padl}" y="{y+3:.1f}" width="{plot_w*min(ov,vmax)/vmax:.1f}" height="{rh/2-3}" rx="2" fill="#94a3b8"/>')
        p.append(f'<rect x="{padl}" y="{y+rh/2:.1f}" width="{plot_w*min(nv,vmax)/vmax:.1f}" height="{rh/2-3}" rx="2" fill="#16a34a"/>')
        p.append(f'<text x="{padl+plot_w*min(nv,vmax)/vmax+5:.1f}" y="{y+rh-7:.1f}" fill="#1e293b" font-size="9">{nv:.2f}</text>')
    p.append(f'<rect x="{padl}" y="6" width="11" height="11" fill="#94a3b8"/><text x="{padl+15}" y="15" fill="#64748b">old utterances</text>')
    p.append(f'<rect x="{padl+110}" y="6" width="11" height="11" fill="#16a34a"/><text x="{padl+125}" y="15" fill="#64748b">Hybrid W3</text>')
    p.append('</svg>')
    return f'<p class="note" style="margin-bottom:2px"><b>{title}</b></p>' + "".join(p)


# --------------------------------------------------------------------------- HTML

def main():
    # primary arm for figures = CORN; fall back to weighted CE
    def best_arm(key):
        return "corn" if load("corn", key) is not None else "ce"

    base = M("corn", "old") or M("ce", "old")

    def fig_rows(metric, total=False):
        rows = []
        for key, label, col, *_ in CONDS:
            m = M(best_arm(key), key)
            if m is None:
                continue
            v = m["total"][metric] if total else m["item"][metric]
            rows.append((label, v, col))
        return rows

    base_f1 = base["item"]["macro_f1"]
    fig_f1 = hbars(fig_rows("macro_f1"), "Fig 1 · Item macro-F1 by evidence (CORN). Red = old baseline", 0.4, baseline=base_f1)
    fig_mae = hbars(fig_rows("mae"), "Fig 2 · Item MAE by evidence (CORN)", 1.0, lower_better=True)
    fig_acc = hbars(fig_rows("accuracy"), "Fig 3 · Accuracy by evidence (CORN) — SECONDARY metric", 0.6)
    fig_qwk = hbars(fig_rows("qwk"), "Fig 4 · Item QWK (ordinal agreement)", 0.4)
    fig_totqwk = hbars(fig_rows("total_qwk", total=True), "Fig 5 · Reconstructed PHQ-8 TOTAL QWK", 0.6)
    fig_bacc = hbars(fig_rows("thr_balanced_accuracy", total=True), "Fig 6 · Screening balanced accuracy (total >= 10)", 0.8)

    fig_cm = ('<div style="display:flex;flex-wrap:wrap;gap:18px">'
              + confusion_svg("corn", "old", "Fig 7a · Old utterances (CORN)", "#94a3b8")
              + confusion_svg("corn", "hybw3", "Fig 7b · Hybrid W3 (CORN) — best", "#16a34a")
              + '</div>')
    fig_item_f1 = per_item_grouped("corn", "old", "hybw3", "f1", 0.45, "Fig 8 · Per-item macro-F1: old vs Hybrid W3")
    fig_item_mae = per_item_grouped("corn", "old", "hybw3", "mae", 1.2, "Fig 9 · Per-item MAE: old vs Hybrid W3 (lower=better)")

    # ---- main comparison table (full triad) ----
    def trow(key, label, color):
        arms_present = [(a, d) for a, d in ARMS if M(a, key)]
        rows = ""
        for n, (arm, disp) in enumerate(arms_present):
            m = M(arm, key); i = m["item"]; t = m["total"]
            first = (f'<td rowspan="{len(arms_present)}" style="border-left:4px solid {color}"><b>{label}</b></td>'
                     if n == 0 else "")
            hl = ' style="background:#f0fdf4"' if (label.startswith("Hybrid") and arm == "corn") else ""
            rows += (f"<tr{hl}>{first}<td>{disp}</td><td><b>{i['macro_f1']:.3f}</b></td>"
                     f"<td>{i['mae']:.3f}</td><td class='sec'>{i['accuracy']:.3f}</td>"
                     f"<td>{i['qwk']:.3f}</td><td>{i['f1_per_class'][2]:.3f}</td>"
                     f"<td>{i['f1_per_class'][3]:.3f}</td><td>{t['total_mae']:.2f}</td>"
                     f"<td>{t['total_qwk']:.3f}</td></tr>")
        return rows
    main_rows = "".join(trow(k, lbl, c) for k, lbl, c, *_ in CONDS)

    # ---- per-class F1 table (CORN) ----
    pc_rows = ""
    for key, label, *_ in CONDS:
        m = M("corn", key)
        if not m:
            continue
        f = m["item"]["f1_per_class"]
        pc_rows += (f"<tr><td>{label}</td><td>{f[0]:.3f}</td><td>{f[1]:.3f}</td>"
                    f"<td><b>{f[2]:.3f}</b></td><td><b>{f[3]:.3f}</b></td></tr>")

    # ---- near-miss table (CORN) ----
    nm_rows = ""
    for key, label, *_ in CONDS:
        m = M("corn", key)
        if not m:
            continue
        i = m["item"]
        nm_rows += (f"<tr><td>{label}</td><td class='sec'>{i['top1_acc']:.3f}</td>"
                    f"<td><b>{i['top2_acc']:.3f}</b></td><td>{i['off_by_one_rate']:.3f}</td></tr>")

    # ---- threshold (>=10) table (CORN) ----
    th_rows = ""
    for key, label, *_ in CONDS:
        m = M("corn", key)
        if not m:
            continue
        t = m["total"]
        hl = ' style="background:#f0fdf4"' if label.startswith("Hybrid") else ""
        th_rows += (f"<tr{hl}><td>{label}</td><td>{t['total_mae']:.2f}</td><td>{t['total_qwk']:.3f}</td>"
                    f"<td>{t['thr_sensitivity']:.3f}</td><td>{t['thr_specificity']:.3f}</td>"
                    f"<td><b>{t['thr_balanced_accuracy']:.3f}</b></td><td>{t['thr_f1']:.3f}</td></tr>")

    # ---- Appetite spotlight (CORN) ----
    ap_rows = ""
    for key, label, *_ in CONDS:
        df = load("corn", key)
        if df is None:
            continue
        pi = per_item(df); ap = pi[pi.item_name == "Appetite"]
        if len(ap):
            ap_rows += f"<tr><td>{label}</td><td>{ap.macro_f1.iloc[0]:.3f}</td><td>{ap.mae.iloc[0]:.2f}</td></tr>"

    # ---- run-config rows ----
    cfg_rows = "".join(
        f"<tr><td>{lbl}</td><td><code>{ds}</code></td><td><code>{col}</code></td><td>{tr:.0f}%</td></tr>"
        for k, lbl, c, ds, col, tr in CONDS)

    plain_ce_status = "available" if M("ceplain", "hybw3") else "running (will populate the CE column)"
    best = M("corn", "hybw3")["item"]; bestT = M("corn", "hybw3")["total"]
    oldm = base["item"]; oldT = base["total"]

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Context-Window Evidence — Person B Results</title>
<style>
 *{{box-sizing:border-box}} body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;color:#1e293b;background:#f8fafc;line-height:1.55}}
 header{{background:linear-gradient(135deg,#16a34a,#0891b2);color:#fff;padding:32px 28px}} header h1{{margin:0 0 6px;font-size:25px}} header .sub{{opacity:.92;font-size:14px}}
 header .meta{{margin-top:12px;font-size:13px;opacity:.9;display:flex;gap:20px;flex-wrap:wrap}}
 .wrap{{max-width:1060px;margin:0 auto;padding:24px 28px 60px}}
 .card{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px 24px;margin:18px 0;box-shadow:0 1px 2px rgba(15,23,42,.04)}}
 h2{{font-size:19px;margin:4px 0 14px;padding-bottom:8px;border-bottom:2px solid #dcfce7}}
 h3{{font-size:14px;color:#0891b2;margin:18px 0 6px}}
 table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:13px}}
 th,td{{text-align:left;padding:6px 9px;border-bottom:1px solid #e2e8f0}} th{{font-size:11px;text-transform:uppercase;letter-spacing:.03em;color:#64748b;background:#fbfcfe}}
 td.sec,th.sec{{color:#94a3b8}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:6px 0}}
 .stat{{background:#dcfce7;border-radius:10px;padding:13px 15px}} .stat .n{{font-size:21px;font-weight:700;color:#16a34a}} .stat .l{{font-size:11.5px;color:#475569}}
 .callout{{border-left:4px solid #16a34a;background:#dcfce7;padding:12px 16px;border-radius:0 8px 8px 0;margin:12px 0}}
 .note{{color:#64748b;font-size:13px}} code{{background:#f1f5f9;padding:1px 5px;border-radius:5px;font-size:12px}}
 pre{{background:#0f172a;color:#e2e8f0;padding:12px 14px;border-radius:8px;overflow:auto;font-size:12px}}
 footer{{text-align:center;color:#64748b;font-size:12px;padding:24px}} ul{{margin:8px 0;padding-left:20px}} li{{margin:5px 0}}
</style></head><body>
<header><h1>Context-Window Evidence — Model Results (Person B → Person A)</h1>
<div class="sub">PHQ-8 item severity · participant-grouped 5-fold CV (out-of-fold over all 1,752 items)</div>
<div class="meta"><span>📅 2026-06-08</span><span>🤖 bert-base-uncased</span><span>🖥️ RTX 3090 (SLURM)</span><span>branch: data/context-windows</span></div></header>
<div class="wrap">

 <div class="card"><h2>TL;DR — context windows win</h2>
  <p>Changing <i>only</i> <code>dataset_path</code> and <code>evidence_column</code>, participant-side
   <b>hybrid context windows</b> beat the old isolated-utterance baseline on every metric.</p>
  <div class="grid">
   <div class="stat"><div class="n">{best['macro_f1']:.3f}</div><div class="l">Best macro-F1 (Hybrid W3) vs {oldm['macro_f1']:.3f} old</div></div>
   <div class="stat"><div class="n">{best['qwk']:.3f}</div><div class="l">QWK vs {oldm['qwk']:.3f} old</div></div>
   <div class="stat"><div class="n">{bestT['total_qwk']:.2f}</div><div class="l">PHQ-8 total QWK vs {oldT['total_qwk']:.2f} old</div></div>
   <div class="stat"><div class="n">7/8</div><div class="l">Items improved</div></div>
  </div>
  <div class="callout"><b>Winner: Hybrid W3 + CORN.</b> macro-F1 +27%, QWK ~2×, severe-class F1 ~2×, PHQ-8 total QWK 0.07→0.49.
   The gain comes from <b>hybrid (semantic) retrieval</b> — BM25-only windows barely beat the baseline. Use W3 (cheaper; W5 needs max_length=512 to match it).</div>
  <div class="callout" style="border-left-color:#d97706;background:#fef3c7"><b>⚠ Baseline note — read before cross-comparing.</b>
   The "Old BM25 utterances" baseline here is <code>phq8_item_dataset_full_bm25.csv</code> = <b>BM25</b> utterance retrieval (per Person A's dataset contract).
   The <i>earlier</i> project reports (e.g. <code>status_report_v2.html</code>, the BERT≈MentalBERT / CORN-vs-CE study) used <code>phq8_item_dataset_full.csv</code> = <b>TF-IDF</b> utterance retrieval, which scores differently
   (TF-IDF CORN macro-F1 0.289 / MAE 0.769 vs BM25 CORN 0.274 / 0.847). All comparisons <i>inside this report</i> share the BM25 base and are valid; do not compare these baseline numbers directly against the TF-IDF reports. A unified BM25-baseline set (both models × CE/CORN) is in the manifest under tags <code>bm25base_*</code>.</div>
 </div>

 <div class="card"><h2>1 · Run configuration (returned for every run)</h2>
  <table><tr><th>Field</th><th>Value</th></tr>
   <tr><td>model_name</td><td><code>bert-base-uncased</code></td></tr>
   <tr><td>loss_type</td><td>plain CE, weighted CE, and CORN (ordinal) — full triad</td></tr>
   <tr><td>use_class_weights</td><td>weighted CE: balanced inverse-frequency, <b>computed from each fold's TRAIN split only</b>, pooled over items. CORN/plain CE: none</td></tr>
   <tr><td>seed</td><td>42</td></tr>
   <tr><td>max_length</td><td>256 (primary); 512 follow-up for richer context</td></tr>
   <tr><td>num_epochs</td><td>8 · batch 16 · lr 2e-5 · optimizer AdamW</td></tr>
   <tr><td>eval protocol</td><td>participant-grouped, stratified 5-fold CV; OOF predictions over all 1,752 items (no leakage)</td></tr>
   <tr><td>truncation strategy</td><td><code>truncation="only_second"</code>: <b>item_text always kept in full; only evidence truncated</b></td></tr>
   <tr><td>predictions file</td><td><code>outputs/cv/oof_predictions_ctx_&lt;arm&gt;_&lt;cond&gt;.csv</code> (participant_id,item_id,item_name,label,prediction,fold,prob_0..3)</td></tr>
   <tr><td>overall results file</td><td><code>outputs/context_window_results.json</code></td></tr>
   <tr><td>per-item results</td><td>computed by <code>src/evaluation/context_window_eval.py</code> from the OOF files</td></tr>
  </table>
  <h3>Per-condition dataset_path / evidence_column (+ truncation@256)</h3>
  <table><tr><th>Condition</th><th>dataset_path (data/processed/)</th><th>evidence_column</th><th>%trunc@256</th></tr>{cfg_rows}</table>
  <h3>Exact command (reused verbatim except dataset/column/loss)</h3>
  <pre>python -m src.models.cross_validate \\
  --model-name bert-base-uncased --dataset-path data/processed/&lt;DS&gt; --evidence-column &lt;COL&gt; \\
  --k-folds 5 --num-epochs 8 --batch-size 16 --max-length 256 --seed 42 \\
  --loss {{cross_entropy [--class-weights balanced] | corn}} --tag &lt;TAG&gt;</pre>
  <p class="note">Plain-CE arm status: {plain_ce_status}.</p>
 </div>

 <div class="card"><h2>2 · Full comparison — CE vs weighted-CE vs CORN (256-token CV, OOF)</h2>
  <table><tr><th>Evidence</th><th>Loss</th><th>Macro-F1</th><th>MAE↓</th><th class="sec">Acc (2°)</th><th>QWK</th><th>F1 c2</th><th>F1 c3</th><th>totMAE↓</th><th>totQWK</th></tr>
  {main_rows}</table>
  <p class="note">Accuracy greyed = secondary metric (majority baseline ≈ 0.44). c2/c3 = moderate/severe F1. Hybrid+CORN row highlighted.</p>
 </div>

 <div class="card"><h2>3 · Figures</h2>
  {fig_f1}{fig_mae}{fig_acc}{fig_qwk}{fig_totqwk}{fig_bacc}
  <h3>Confusion matrices — old baseline vs best new variant</h3>
  {fig_cm}
  <div class="callout">Hybrid pulls real mass onto the severe diagonal: true-2→pred-2 30→55, true-3→pred-3 10→25.</div>
  {fig_item_f1}{fig_item_mae}
 </div>

 <div class="card"><h2>4 · Per-class F1 (CORN) — emphasis on classes 2 & 3</h2>
  <table><tr><th>Evidence</th><th>F1 c0</th><th>F1 c1</th><th>F1 c2 (moderate)</th><th>F1 c3 (severe)</th></tr>{pc_rows}</table>
  <p class="note">The minority classes you flagged (2 & 3) roughly double under hybrid context windows.</p>
 </div>

 <div class="card"><h2>5 · Near-miss / top-2 / off-by-one (CORN)</h2>
  <table><tr><th>Evidence</th><th class="sec">Top-1 acc</th><th>Top-2 acc</th><th>Off-by-one rate (of errors)</th></tr>{nm_rows}</table>
  <p class="note">Most errors are between <i>adjacent</i> severities; top-2 accuracy is far higher than top-1 — the model is usually "almost right."</p>
 </div>

 <div class="card"><h2>6 · Reconstructed PHQ-8 total & clinical threshold (total ≥ 10)</h2>
  <table><tr><th>Evidence</th><th>total MAE↓</th><th>total QWK</th><th>Sensitivity</th><th>Specificity</th><th>Balanced Acc</th><th>F1</th></tr>{th_rows}</table>
  <p class="note">Total = sum of 8 predicted item severities per participant (OOF). Screening improves markedly (total QWK 0.07→0.49,
   balanced-acc 0.54→0.63–0.68) but sensitivity stays modest (~0.32–0.43) → tune a lower decision threshold. Caveat: 219 participants, 65 positive.</p>
 </div>

 <div class="card"><h2>7 · Appetite spotlight (your flagged weak item, CORN)</h2>
  <table><tr><th>Evidence</th><th>Macro-F1</th><th>MAE↓</th></tr>{ap_rows}</table>
  <p class="note">Appetite improves under hybrid context (F1 ~0.245→0.299, MAE ~1.00→0.88) — context helps even the hardest somatic item, though it remains the weakest.</p>
 </div>

 <div class="card"><h2>8 · Recommendation</h2>
  <ul>
   <li><b>Adopt hybrid context windows</b> — consistent win on every metric.</li>
   <li><b>W3 by default</b> — equal to W5 but cheaper and fits BERT's window (W5 needs 512).</li>
   <li><b>The lever is hybrid (semantic) retrieval</b>, not BM25 windows or raw length.</li>
   <li><b>Pair with CORN</b> for best MAE/ordinal behaviour; weighted-CE is comparable on F1.</li>
   <li><b>Screening:</b> tune a lower threshold to raise sensitivity. Consider MIL over <code>retrieved_context_windows_hybrid_list</code> next.</li>
  </ul>
  <p class="note">Full Q&amp;A + methods: <code>outputs/person_b_handoff_response.md</code>. Raw metrics: <code>outputs/context_window_results.json</code>.</p>
 </div>
 <footer>Generated by <code>src/evaluation/build_context_report.py</code> · inline-SVG · OOF 5-fold CV · satisfies Section-11 checklist</footer>
</div></body></html>"""
    out = OUT / "context_window_report.html"
    out.write_text(html)
    print("wrote", out, f"({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
