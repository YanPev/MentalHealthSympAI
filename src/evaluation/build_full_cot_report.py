"""
One comprehensive, self-contained HTML report telling the entire CoT investigation
step by step: feasibility -> validation -> self-consistency -> ensemble ->
distillation -> better evidence -> error analysis -> stacking, with inline-SVG
figures and the verdict.

Pulls numbers from the metric JSONs already written by the per-step scripts; recomputes
the error-analysis stats and the confusion matrices from the prediction CSVs.

    python -m src.evaluation.build_full_cot_report
"""

from pathlib import Path
import glob
import json

import numpy as np
import pandas as pd

from src.evaluation.build_cot_report import (
    metric_block, confusion_svg, LABELS, COT_COLOR, ENC_COLOR)
from src.llm.cot_probe import format_evidence

PR = Path(__file__).resolve().parents[2]
COT = PR / "outputs" / "cot"
ENC_OOF = PR / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"
OUT = COT / "cot_full_report.html"

ENS_COLOR = "#0ea5e9"
DIS_COLOR = "#f59e0b"
STK_COLOR = "#64748b"
NEU = "#94a3b8"


def J(name):
    return json.loads((COT / f"{name}.json").read_text())


def load_dir(d):
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(COT / d / "*.csv")))],
                   ignore_index=True)
    df["participant_id"] = df["participant_id"].astype(str)
    return df


# --------------------------------------------------------------------------- SVG
def bars(rows, title, vmax, lower_better=False, refs=None, note=""):
    """rows: list of (label, value, color). refs: list of (value, color, dash-label)."""
    n = len(rows); rh = 26; padl = 200; padr = 70; padt = 8; padb = 6
    W = 680; pw = W - padl - padr; H = padt + n * rh + padb
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
         f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    for ref in (refs or []):
        rv, rc = ref[0], ref[1]
        rx = padl + pw * (rv / vmax)
        p.append(f'<line x1="{rx:.1f}" y1="{padt}" x2="{rx:.1f}" y2="{padt+n*rh}" '
                 f'stroke="{rc}" stroke-dasharray="4 3" opacity="0.6"/>')
    for i, (lab, v, col) in enumerate(rows):
        y = padt + i * rh
        bw = pw * (max(v, 0) / vmax)
        p.append(f'<text x="{padl-8}" y="{y+rh/2+3:.1f}" text-anchor="end" fill="#475569">{lab}</text>')
        p.append(f'<rect x="{padl}" y="{y+3:.1f}" width="{bw:.1f}" height="{rh-7}" rx="3" fill="{col}"/>')
        p.append(f'<text x="{padl+bw+5:.1f}" y="{y+rh/2+3:.1f}" fill="#1e293b" font-size="10">{v:.3f}</text>')
    p.append('</svg>')
    arrow = " (lower = better)" if lower_better else ""
    sub = f'<br><span style="font-size:10px;color:#94a3b8">{note}</span>' if note else ""
    return f'<p class="note" style="margin:10px 0 2px"><b>{title}{arrow}</b>{sub}</p>' + "".join(p)


def grouped(rows, title, vmax, lower_better=False, legend=("A", "B"), cols=(COT_COLOR, ENC_COLOR)):
    """rows: (label, vA, vB)."""
    n = len(rows); rh = 34; padl = 200; padr = 64; padt = 26; padb = 6
    W = 680; pw = W - padl - padr; H = padt + n * rh + padb
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
         f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    p.append(f'<rect x="{padl}" y="6" width="11" height="11" fill="{cols[0]}"/>'
             f'<text x="{padl+15}" y="15" fill="#64748b">{legend[0]}</text>')
    p.append(f'<rect x="{padl+170}" y="6" width="11" height="11" fill="{cols[1]}"/>'
             f'<text x="{padl+185}" y="15" fill="#64748b">{legend[1]}</text>')
    for i, (lab, va, vb) in enumerate(rows):
        y = padt + i * rh
        p.append(f'<text x="{padl-8}" y="{y+rh/2+3:.1f}" text-anchor="end" fill="#475569">{lab}</text>')
        for k, (v, c) in enumerate([(va, cols[0]), (vb, cols[1])]):
            yy = y + 3 + k * (rh / 2 - 2)
            bw = pw * (max(v, 0) / vmax)
            p.append(f'<rect x="{padl}" y="{yy:.1f}" width="{bw:.1f}" height="{rh/2-4}" rx="2" fill="{c}"/>')
            p.append(f'<text x="{padl+bw+5:.1f}" y="{yy+rh/2-6:.1f}" fill="#1e293b" font-size="10">{v:.3f}</text>')
    p.append('</svg>')
    a = " (lower = better)" if lower_better else ""
    return f'<p class="note" style="margin:10px 0 2px"><b>{title}{a}</b></p>' + "".join(p)


def mtable(rows, headers):
    h = "".join(f"<th>{x}</th>" for x in headers)
    body = ""
    for r in rows:
        hl = ' style="background:#f0f9ff"' if r.get("hl") else ""
        cells = "".join(f"<td>{c}</td>" for c in r["cells"])
        body += f"<tr{hl}>{cells}</tr>"
    return f'<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def fmt(m, keys=("macro_f1", "qwk", "mae", "accuracy")):
    return [f"{m[k]:.3f}" for k in keys]


# --------------------------------------------------------------------------- build
def main():
    feas = J("cot_feasibility_metrics")          # W3 greedy
    sc = J("cot_feasibility_metrics_sc5")
    w5 = J("cot_feasibility_metrics_w5")
    ens = J("cot_ensemble_metrics")              # W3 ensemble
    ensw5 = J("cot_ensemble_metrics_w5")
    dis = J("distill_compare_metrics")
    stk = J("cot_stacker_metrics")

    enc = feas["encoder"]
    cot_w3 = feas["cot"]

    # ---- error-analysis recompute (W5 CoT) ----
    cotw5 = load_dir("folds_w5")
    ds = pd.read_csv(PR / "data" / "processed" / "phq8_item_dataset_context_windows_hybrid_w5.csv")
    ds["participant_id"] = ds["participant_id"].astype(str)
    m = cotw5.merge(ds[["participant_id", "item_id"]].assign(
        participant_id=ds["participant_id"].astype(str)), on=["participant_id", "item_id"], how="left")
    m["err"] = (m["prediction"] - m["label"]).abs()
    fo = m[m["err"] >= 2]
    n_fo, n_under, n_over = len(fo), int((fo.prediction < fo.label).sum()), int((fo.prediction > fo.label).sum())
    by_item = fo.item_name.value_counts().to_dict()
    by_gold = fo.label.value_counts().sort_index().to_dict()

    # confusion data
    enc_df = pd.read_csv(ENC_OOF)[["participant_id", "item_id", "label", "prediction"]]
    enc_df["participant_id"] = enc_df["participant_id"].astype(str)
    routed_df = cotw5.merge(enc_df.rename(columns={"prediction": "enc_pred"}),
                            on=["participant_id", "item_id", "label"])
    routed_df["prediction"] = np.where(routed_df["prediction"] >= 2,
                                       routed_df["prediction"], routed_df["enc_pred"])

    best = ensw5["methods"]["severity-routed"]   # the headline best config

    # ===================== figures =====================
    # journey overview
    journey = bars([
        ("Encoder (MentalBERT+CORN)", enc["macro_f1"], ENC_COLOR),
        ("CoT 7B (W3 evidence)", cot_w3["macro_f1"], COT_COLOR),
        ("CoT 7B (W5 evidence)", w5["cot"]["macro_f1"], COT_COLOR),
        ("Distilled 1.5B student", dis["methods"]["distilled 1.5B"]["macro_f1"], DIS_COLOR),
        ("Learned stacker", stk["methods"]["stacker (LogReg, nested-CV)"]["macro_f1"], STK_COLOR),
        ("★ Severity-routed ensemble", best["macro_f1"], ENS_COLOR),
    ], "Macro-F1 across every approach we tried (pooled 5-fold OOF, n=1752)", 0.42,
        refs=[(enc["macro_f1"], ENC_COLOR)],
        note="dashed line = encoder baseline. Higher is better.")

    # step 1
    pf = feas["per_fold"]
    s1_fold = grouped([(f"fold {p['fold']}", p["cot"]["macro_f1"], p["enc"]["macro_f1"]) for p in pf],
                      "Step 1 · per-fold macro-F1 — CoT vs encoder", 0.45,
                      legend=("CoT 7B", "encoder"))
    s1_pc = grouped([(f"class {c}", cot_w3["f1_per_class"][c], enc["f1_per_class"][c]) for c in LABELS],
                    "Step 1 · per-class F1 (class 3 = most severe)", 0.7, legend=("CoT 7B", "encoder"))

    # step 2
    s2 = grouped([("macro-F1", sc["cot"]["macro_f1"], cot_w3["macro_f1"]),
                  ("QWK", sc["cot"]["qwk"], cot_w3["qwk"]),
                  ("MAE", sc["cot"]["mae"], cot_w3["mae"]),
                  ("far-off (≥2)", sc["cot"]["far_off_rate"], cot_w3["far_off_rate"])],
                 "Step 2 · self-consistency (×5) vs greedy CoT", 0.8,
                 legend=("self-consistency", "greedy"), cols=("#a78bfa", COT_COLOR))

    # step 3
    em = ens["methods"]
    order3 = ["encoder (CORN)", "CoT greedy", "prob-avg 50/50", "severity-routed", "nested-CV blend"]
    cmap3 = {"encoder (CORN)": ENC_COLOR, "CoT greedy": COT_COLOR}
    s3_f1 = bars([(k, em[k]["macro_f1"], cmap3.get(k, ENS_COLOR)) for k in order3],
                 "Step 3 · macro-F1 by ensemble method", 0.42,
                 refs=[(enc["macro_f1"], ENC_COLOR), (em["CoT greedy"]["macro_f1"], COT_COLOR)],
                 note="dashed = the two parent models")
    s3_qwk = bars([(k, em[k]["qwk"], cmap3.get(k, ENS_COLOR)) for k in order3],
                  "Step 3 · QWK by ensemble method", 0.42,
                  refs=[(enc["qwk"], ENC_COLOR), (em["CoT greedy"]["qwk"], COT_COLOR)])
    s3_cm = ('<div style="display:flex;flex-wrap:wrap;gap:18px">'
             + confusion_svg(routed_df, "Severity-routed ensemble", ENS_COLOR)
             + confusion_svg(enc_df, "Encoder alone", ENC_COLOR) + '</div>')

    # step 4
    dm = dis["methods"]
    s4 = grouped([("macro-F1", dm["distilled 1.5B"]["macro_f1"], dm["7B CoT (greedy)"]["macro_f1"]),
                  ("QWK", dm["distilled 1.5B"]["qwk"], dm["7B CoT (greedy)"]["qwk"]),
                  ("class-3 F1", dm["distilled 1.5B"]["f1_per_class"][3], dm["7B CoT (greedy)"]["f1_per_class"][3])],
                 "Step 4 · distilled 1.5B student vs its 7B teacher", 0.5,
                 legend=("distilled 1.5B", "7B teacher"), cols=(DIS_COLOR, COT_COLOR))

    # step 5
    s5 = grouped([("macro-F1", w5["cot"]["macro_f1"], cot_w3["macro_f1"]),
                  ("QWK", w5["cot"]["qwk"], cot_w3["qwk"]),
                  ("MAE", w5["cot"]["mae"], cot_w3["mae"]),
                  ("accuracy", w5["cot"]["accuracy"], cot_w3["accuracy"])],
                 "Step 5 · CoT with W5 evidence vs W3 evidence", 0.8,
                 legend=("W5 excerpts", "W3 blob"), cols=("#22d3ee", COT_COLOR))
    ew5 = ensw5["methods"]
    s5_ens = bars([("encoder", enc["macro_f1"], ENC_COLOR),
                   ("★ severity-routed (W5)", ew5["severity-routed"]["macro_f1"], ENS_COLOR),
                   ("severity-routed (W3)", em["severity-routed"]["macro_f1"], NEU)],
                  "Step 5 · best ensemble: W5 vs W3 evidence (macro-F1)", 0.42)

    # step 6
    s6_item = bars([(k, v, NEU) for k, v in sorted(by_item.items(), key=lambda t: -t[1])],
                   f"Step 6 · far-off errors by item (total {n_fo})", max(by_item.values()) * 1.1,
                   note="somatic/internal items dominate")
    sm = stk["methods"]
    s6_stk = bars([("encoder", sm["encoder (CORN)"]["macro_f1"], ENC_COLOR),
                   ("★ severity-routed", sm["severity-routed (W5) [prior best]"]["macro_f1"], ENS_COLOR),
                   ("learned stacker", sm["stacker (LogReg, nested-CV)"]["macro_f1"], STK_COLOR)],
                  "Step 6 · learned stacker vs the hand routing rule (macro-F1)", 0.42)

    # ===================== tables =====================
    final_tbl = mtable([
        {"cells": ["Encoder (MentalBERT+CORN)", *fmt(enc)]},
        {"cells": ["7B CoT — W3 evidence", *fmt(cot_w3)]},
        {"cells": ["7B CoT — W5 evidence", *fmt(w5["cot"])]},
        {"cells": ["Self-consistency ×5 (W3)", *fmt(sc["cot"])]},
        {"cells": ["Distilled 1.5B student", *fmt(dm["distilled 1.5B"])]},
        {"cells": ["Learned stacker (nested-CV)", *fmt(sm["stacker (LogReg, nested-CV)"])]},
        {"cells": ["★ Severity-routed ensemble (W5)", *fmt(best)], "hl": True},
    ], ["Configuration", "macro-F1", "QWK", "MAE", "accuracy"])

    def step(num, title, tag, tagcolor, body):
        return f"""<div class="card"><div class="steptag" style="background:{tagcolor}">{tag}</div>
        <h2>Step {num} · {title}</h2>{body}</div>"""

    css = """
 body{margin:0;background:#f1f5f9;color:#0f172a;font-family:-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.55}
 header{background:linear-gradient(135deg,#0f172a,#1e293b);color:#fff;padding:30px 30px}
 header h1{margin:0;font-size:24px} .sub{color:#cbd5e1;font-size:14px;margin-top:6px;max-width:780px}
 .meta{margin-top:10px;font-size:12px;color:#94a3b8;display:flex;gap:16px;flex-wrap:wrap}
 .wrap{max-width:960px;margin:22px auto;padding:0 18px}
 .card{background:#fff;border-radius:14px;padding:20px 24px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.08);position:relative}
 .card h2{margin:0 0 10px;font-size:17px} .card h3{font-size:14px;margin:16px 0 4px;color:#334155}
 .note{font-size:12px;color:#64748b} p{font-size:14px}
 table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
 th,td{padding:7px 9px;border-bottom:1px solid #e2e8f0;text-align:right}
 th:first-child,td:first-child{text-align:left} thead th{background:#f8fafc;color:#475569;font-weight:600}
 .steptag{position:absolute;top:18px;right:20px;color:#fff;font-size:11px;font-weight:700;padding:3px 10px;border-radius:999px}
 .verdict{background:#ecfdf5;border-left:4px solid #16a34a;padding:6px 14px;border-radius:6px;font-size:13px;margin-top:8px}
 .neg{background:#fef2f2;border-left:4px solid #dc2626;padding:6px 14px;border-radius:6px;font-size:13px;margin-top:8px}
 .caveat{background:#fffbeb;border-left:4px solid #f59e0b;padding:8px 14px;border-radius:6px;font-size:13px;margin-top:8px}
 .quote{background:#f8fafc;border-left:3px solid #cbd5e1;padding:6px 12px;margin:6px 0;font-size:12.5px;color:#475569}
 .toc a{color:#0ea5e9;text-decoration:none} .toc li{margin:3px 0;font-size:13px}
"""
    GO, NEG, EV = "#16a34a", "#dc2626", "#0ea5e9"

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chain-of-Thought for PHQ-8 — full investigation</title><style>{css}</style></head><body>
<header><h1>Adding Chain-of-Thought reasoning to the PHQ-8 pipeline</h1>
<div class="sub">A complete, step-by-step investigation: can an LLM's chain-of-thought reasoning improve per-item depression-severity prediction over the tuned MentalBERT+CORN encoder? Feasibility → validation → self-consistency → ensemble → distillation → better evidence → error analysis → stacking.</div>
<div class="meta"><span>🤖 Qwen2.5-7B-Instruct (teacher) · 1.5B (student)</span><span>🆚 MentalBERT + CORN</span><span>🖥️ RTX 3090 / SLURM (offline)</span><span>📊 participant-grouped 5-fold OOF, n=1752</span></div></header>
<div class="wrap">

<div class="card"><h2>Executive summary</h2>
<p>An off-the-shelf LLM doing few-shot chain-of-thought (no training) is <b>genuinely complementary</b> to the trained encoder: it recovers the severe classes the encoder collapses on, while the encoder keeps errors ordinally closer. Neither dominates alone — but a simple <b>severity-routing ensemble</b> beats both parents on the headline metrics. Self-consistency, distillation to a small model, richer evidence, and a learned stacker were all tried; only better evidence nudged the best config further, and an error analysis shows the <b>residual errors are largely a self-report-vs-interview-content ceiling</b>, not a modelling gap.</p>
<div class="verdict"><b>Best result:</b> severity-routed(encoder, 7B-CoT-W5) — <b>macro-F1 {best['macro_f1']:.3f}, QWK {best['qwk']:.3f}</b> vs encoder {enc['macro_f1']:.3f}/{enc['qwk']:.3f} and CoT-alone {cot_w3['macro_f1']:.3f}/{cot_w3['qwk']:.3f}.</div>
{journey}
{final_tbl}
<p class="note" style="margin-top:8px">★ = best config. All numbers are pooled out-of-fold over participant-grouped 5-fold CV (the project's standard), so they are directly comparable to the encoder's existing results. Accuracy is a weak metric on this 4-class imbalanced task; macro-F1, QWK (ordinal agreement) and MAE are the headline metrics.</p>
</div>

<div class="card"><h2>Setup</h2>
<p><b>Task.</b> Predict each of the 8 PHQ-8 item severities (0–3) for a participant from retrieved snippets of their clinical interview (DAIC-WOZ). <b>Baseline.</b> The project's best encoder — MentalBERT fine-tuned with the CORN ordinal loss on Hybrid-W3 retrieved context windows (pooled OOF macro-F1 0.348). <b>CoT idea.</b> Give a capable open LLM the same evidence and have it reason about the PHQ-8 frequency anchors (0 not-at-all … 3 nearly-every-day) before answering. Everything runs offline on the cluster's 3090s; the comparison is <b>paired</b> on identical rows/folds.</p>
<ul class="toc">
<li><a href="#s1">Step 1 — Feasibility &amp; 5-fold validation</a></li>
<li><a href="#s2">Step 2 — Self-consistency</a></li>
<li><a href="#s3">Step 3 — The ensemble (the win)</a></li>
<li><a href="#s4">Step 4 — Distillation to a small model</a></li>
<li><a href="#s5">Step 5 — Better evidence</a></li>
<li><a href="#s6">Step 6 — Error analysis &amp; learned stacking</a></li>
</ul></div>

<a name="s1"></a>{step(1, "Feasibility & 5-fold validation", "GO", GO, f'''
<p>A few-shot CoT probe (4 crafted exemplars mapping evidence → anchor) was run on all 5 folds and scored on the identical rows as the encoder. <b>Untrained CoT beats the trained encoder on macro-F1</b> — validated across folds, not a single-fold fluke.</p>
{s1_fold}
<p>Pooled: CoT macro-F1 <b>{cot_w3['macro_f1']:.3f}</b> ± per-fold spread vs encoder {enc['macro_f1']:.3f} (CoT wins 4/5 folds). The decisive detail is per-class:</p>
{s1_pc}
<div class="verdict">CoT recovers the <b>most-severe class</b> (class-3 F1 {cot_w3['f1_per_class'][3]:.3f} vs encoder {enc['f1_per_class'][3]:.3f}). But it loses the ordinal metrics — QWK {cot_w3['qwk']:.3f} vs {enc['qwk']:.3f}, MAE {cot_w3['mae']:.3f} vs {enc['mae']:.3f} — because when wrong it is wrong by more. The two models are <b>complementary</b>, agreeing on only {feas['agreement_rate']*100:.0f}% of items.</div>''')}

<a name="s2"></a>{step(2, "Self-consistency", "NO GAIN", NEG, f'''
<p>CoT's weakness was far-off errors, so we sampled 5 reasoning chains and majority-voted (which also yields calibrated probabilities). It mechanically cut far-off errors and improved MAE a little — but <b>eroded the macro-F1 edge</b> (voting regresses toward the safe/modal label, killing the bold severe-class calls).</p>
{s2}
<div class="neg">Self-consistency is <b>not the lever</b>: macro-F1 {cot_w3['macro_f1']:.3f}→{sc['cot']['macro_f1']:.3f}, and it still trails the encoder on QWK/MAE. The ordinal gap is structural, not variance. Greedy decoding stays the better CoT config.</div>''')}

<a name="s3"></a>{step(3, "The ensemble — capturing both strengths", "WIN", EV, f'''
<p>Given the complementarity, we combined the two models (no extra training). A prior-motivated <b>severity-routing</b> rule — trust the CoT label when it predicts a severe class (≥2), else the encoder — beats <i>both</i> parents on the headline metrics simultaneously.</p>
{s3_f1}
{s3_qwk}
{s3_cm}
<div class="verdict"><b>Severity-routing wins on macro-F1 AND QWK at once</b> ({em['severity-routed']['macro_f1']:.3f} / {em['severity-routed']['qwk']:.3f}), plus best severe-class F1. Cost: worst MAE — trusting CoT on a wrong severe call makes a larger miss. A 50/50 probability average is the MAE-preserving alternative (recovers the encoder's MAE and the highest accuracy while still beating it on macro-F1).</div>''')}

<a name="s4"></a>{step(4, "Distillation to a cheap small model", "FAILED", NEG, f'''
<p>The only thing holding the CoT component back is that it is a 7B model. We distilled "step-by-step": the 7B teacher wrote {1752} label-conditioned rationales, then a 1.5B student was LoRA-fine-tuned per fold (leakage-safe) to reproduce reasoning+answer.</p>
{s4}
<div class="neg">The 1.5B <b>lost the severe-class recovery</b> (class-3 F1 {dm['7B CoT (greedy)']['f1_per_class'][3]:.3f}→{dm['distilled 1.5B']['f1_per_class'][3]:.3f}) that was the 7B's entire value, scoring even below the encoder on macro-F1 ({dm['distilled 1.5B']['macro_f1']:.3f}). Routing to its now-unreliable severe calls actively hurts. The value lives in the 7B's reasoning capacity; shrinking it naively throws it away.</div>''')}

<a name="s5"></a>{step(5, "Better evidence (Hybrid-W5 + per-excerpt formatting)", "SMALL GAIN", EV, f'''
<p>The CoT had been reasoning over a shuffled keyword blob. We fed it wider <b>W5</b> windows rendered as separate coherent excerpts. This did <i>not</i> raise CoT's standalone macro-F1 — but it improved the <b>best ensemble on its weak axis</b>.</p>
{s5}
{s5_ens}
<div class="verdict">New best: severity-routed(encoder, W5-CoT) — macro-F1 <b>{ew5['severity-routed']['macro_f1']:.3f}</b>, QWK <b>{ew5['severity-routed']['qwk']:.3f}</b> (the best ordinal agreement of any CoT method; MAE also improved). Better evidence helps the <i>combination's</i> ordinal metrics, not CoT's standalone macro-F1 — a hint we were nearing a ceiling.</div>''')}

<a name="s6"></a>{step(6, "Error analysis & learned stacking", "CEILING", DIS_COLOR, f'''
<p>We read the {n_fo} far-off (≥2) cases together with their evidence and rationale. <b>{n_under} of {n_fo} are under-predictions</b> (model says 0/1, self-report says 2/3), concentrated on somatic/internal items.</p>
{s6_item}
<p>In both directions the CoT reasoning is <b>faithful to the evidence</b> — it is the gold label that diverges from what the interview content supports:</p>
<div class="quote"><b>Under-call (Appetite, gold=2→0):</b> evidence is all grief and hopelessness, no mention of eating. The model correctly notes there is no appetite signal.</div>
<div class="quote"><b>Under-call (Sleep, gold=2→0):</b> transcript says <i>"most of the time it's pretty easy"</i> to sleep — directly contradicting the self-reported score.</div>
<div class="quote"><b>Over-call (Tired, gold=1→3):</b> <i>"always tired and lethargic, not excited about things"</i> — the interview sounds severe, but self-report is mild.</div>
<div class="caveat"><b>This is a self-report ↔ interview-content ceiling.</b> PHQ-8 self-report is only loosely coupled to what people verbalize (especially somatic items), so no amount of prompt/evidence engineering on the transcript can recover a label the transcript does not express. Consistent with the known top-2 (0.72) vs top-1 (0.43) accuracy gap. Only ~25% of misses are fixable (temporal reasoning: counting past/resolved symptoms as current) — and since under- and over-calls are opposite errors, nudging one worsens the other.</div>
<h3>Learned stacking (last attempt at a free gain)</h3>
<p>A nested-CV meta-classifier over the encoder + CoT probabilities can sit at either end of the F1↔MAE trade-off but <b>does not beat the simple hand rule</b>.</p>
{s6_stk}
<div class="neg">Learned stacking ≈ but does not dominate severity-routing. The hand rule, motivated directly by the error structure, remains best.</div>''')}

<div class="card"><h2>Conclusion &amp; recommendation</h2>
<p>The investigation establishes a clean, defensible result: <b>an LLM chain-of-thought component genuinely complements the encoder</b>, and a severity-routing ensemble of the two is the best configuration found — <b>macro-F1 {best['macro_f1']:.3f}, QWK {best['qwk']:.3f}</b>, beating both the tuned encoder and the CoT alone. Self-consistency, naive distillation, and learned stacking did not add value; better evidence gave a small ordinal-metric gain.</p>
<div class="caveat"><b>Stop tuning the CoT/evidence/combiner frontier.</b> The error analysis shows the remaining errors are mostly a label/self-report ceiling, not a modelling gap. Further gains require <i>different labels/data</i> — clinician severity ratings, or reframing the target toward observable interview behaviour — rather than more LLM engineering. Honesty notes: single seed; participant-grouped 5-fold OOF; DAIC-WOZ is public so the LLM may have seen related data; severe-class counts are small so per-class numbers are high-variance.</div>
<p class="note" style="margin-top:10px">Code: <code>src/llm/</code> (cot_probe, generate_rationales, distill_student), <code>src/evaluation/</code> (build_cot_report, cot_ensemble, distill_compare, cot_stacker, build_full_cot_report). Run scripts: <code>run_cot_probe*.sbatch</code>, <code>run_distill_*.sbatch</code>. Metrics &amp; per-step reports in <code>outputs/cot/</code>.</p>
</div>

</div></body></html>"""
    OUT.write_text(html)
    print(f"Saved comprehensive report: {OUT}  ({len(html)//1024} KB, {html.count('<svg')} figures)")


if __name__ == "__main__":
    main()
