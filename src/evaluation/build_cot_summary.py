"""
One-page executive summary of the CoT investigation: headline result, a compact
results-at-a-glance table across every approach, conclusions, and prioritized
next steps. Distinct from the detailed step-by-step cot_full_report.html.

Data-driven from the per-step metric JSONs + the pooled fold predictions.

    python -m src.evaluation.build_cot_summary
"""

from pathlib import Path
import glob
import json

import pandas as pd

from src.evaluation.build_cot_report import metric_block, LABELS

PR = Path(__file__).resolve().parents[2]
COT = PR / "outputs" / "cot"
ENC_OOF = PR / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"
OUT = COT / "cot_summary.html"


def J(name):
    return json.loads((COT / f"{name}.json").read_text())


def pooled(d):
    fs = sorted(glob.glob(str(COT / d / "*.csv")))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return metric_block(df.label.to_numpy(), df.prediction.to_numpy())


def bar(rows, vmax=0.42, best_label=None):
    """rows: (label, value, color). Horizontal bars, value labels, best starred."""
    n = len(rows); rh = 27; padl = 230; padr = 70; padt = 6; W = 720
    pw = W - padl - padr; H = padt + n * rh + 6
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
         f'font-family="-apple-system,Segoe UI,sans-serif" font-size="12">']
    for i, (lab, v, col) in enumerate(rows):
        y = padt + i * rh; bw = pw * max(v, 0) / vmax
        star = "★ " if lab == best_label else ""
        p.append(f'<text x="{padl-8}" y="{y+rh/2+4:.1f}" text-anchor="end" fill="#475569">{star}{lab}</text>')
        p.append(f'<rect x="{padl}" y="{y+3:.1f}" width="{bw:.1f}" height="{rh-8}" rx="3" fill="{col}"/>')
        p.append(f'<text x="{padl+bw+6:.1f}" y="{y+rh/2+4:.1f}" fill="#1e293b" font-size="11">{v:.3f}</text>')
    p.append('</svg>')
    return "".join(p)


def main():
    enc = pd.read_csv(ENC_OOF)
    encm = metric_block(enc.label.to_numpy(), enc.prediction.to_numpy())
    w5 = J("cot_feasibility_metrics_w5")["cot"]
    w3 = J("cot_feasibility_metrics")["cot"]
    sc = J("cot_feasibility_metrics_sc5")["cot"]
    dist = J("distill_compare_metrics")["methods"]["distilled 1.5B"]
    iaw = J("item_aware_downstream_metrics")["itemaware"]
    best = J("cot_ensemble_metrics_w5")["methods"]["severity-routed"]
    ft = pooled("folds_fulltranscript")
    ia = pooled("folds_itemadaptive")

    ENC, COTc, ENS, NEG, ADAPT, FT, NEU = ("#16a34a", "#7c3aed", "#0ea5e9",
                                           "#f59e0b", "#16a34a", "#e11d48", "#94a3b8")

    # results table (sorted by macro-F1 desc), best highlighted
    rows = [
        ("Severity-routed ensemble (encoder + CoT-W5)", best, True, "the best configuration found"),
        ("Encoder alone (MentalBERT + CORN)", encm, False, "the prior baseline"),
        ("CoT 7B — W5 evidence", w5, False, "best standalone CoT (macro-F1)"),
        ("CoT 7B — item-adaptive evidence", ia, False, "best standalone CoT (QWK)"),
        ("CoT 7B — W3 evidence", w3, False, "first feasibility probe"),
        ("CoT 7B — full transcript", ft, False, "recovers severe misses, over-attributes"),
        ("CoT 7B — self-consistency ×5", sc, False, "no gain over greedy"),
        ("CoT 7B — item-aware BM25", iaw, False, "query expansion helps weak retrieval"),
        ("Distilled 1.5B student", dist, False, "lost the severe-class recovery"),
    ]
    rows.sort(key=lambda r: r[1]["macro_f1"], reverse=True)
    trows = ""
    for lab, m, hl, note in rows:
        style = ' style="background:#ecfdf5;font-weight:600"' if hl else ""
        trows += (f"<tr{style}><td>{'★ ' if hl else ''}{lab}</td><td>{m['macro_f1']:.3f}</td>"
                  f"<td>{m['qwk']:.3f}</td><td>{m['mae']:.3f}</td>"
                  f"<td class='sec'>{m['f1_per_class'][3]:.3f}</td><td class='note'>{note}</td></tr>")

    overview = bar([
        ("Severity-routed ensemble", best["macro_f1"], ENS),
        ("CoT 7B (W5)", w5["macro_f1"], COTc),
        ("CoT item-adaptive", ia["macro_f1"], ADAPT),
        ("Encoder (CORN)", encm["macro_f1"], ENC),
        ("CoT full-transcript", ft["macro_f1"], FT),
        ("CoT item-aware BM25", iaw["macro_f1"], "#0891b2"),
        ("Distilled 1.5B", dist["macro_f1"], NEG),
    ], best_label="Severity-routed ensemble")

    d_f1 = best["macro_f1"] - encm["macro_f1"]
    d_qwk = best["qwk"] - encm["qwk"]

    css = """
 body{margin:0;background:#f1f5f9;color:#0f172a;font-family:-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.6}
 header{background:linear-gradient(135deg,#0f172a,#1e293b);color:#fff;padding:32px 32px}
 header h1{margin:0;font-size:25px} .sub{color:#cbd5e1;font-size:14px;margin-top:8px;max-width:820px}
 .wrap{max-width:920px;margin:24px auto;padding:0 20px}
 .card{background:#fff;border-radius:14px;padding:22px 26px;margin-bottom:20px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
 .card h2{margin:0 0 12px;font-size:18px} h3{font-size:15px;margin:18px 0 6px}
 .hero{background:#ecfdf5;border:1px solid #a7f3d0;border-radius:12px;padding:16px 20px;font-size:15px}
 .hero b{font-size:20px;color:#065f46}
 table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
 th,td{padding:8px 10px;border-bottom:1px solid #e2e8f0;text-align:right}
 th:first-child,td:first-child{text-align:left} td.note{text-align:left;color:#94a3b8;font-size:12px}
 thead th{background:#f8fafc;color:#475569;font-weight:600} .sec{color:#94a3b8}
 .note{font-size:12px;color:#64748b}
 ul{margin:6px 0 6px 0;padding-left:20px} li{margin:6px 0;font-size:14px}
 .tag{display:inline-block;font-size:11px;font-weight:700;color:#fff;padding:2px 9px;border-radius:999px;margin-right:6px}
 .win{background:#16a34a} .neg{background:#dc2626} .ins{background:#0ea5e9}
 .pri{display:inline-block;min-width:20px;font-weight:700;color:#0ea5e9}
 .caveat{background:#fffbeb;border-left:4px solid #f59e0b;padding:10px 14px;border-radius:6px;font-size:13px}
"""
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CoT for PHQ-8 — summary, conclusions & next steps</title><style>{css}</style></head><body>
<header><h1>Chain-of-Thought for PHQ-8 severity — executive summary</h1>
<div class="sub">Can an LLM's chain-of-thought reasoning improve per-item depression-severity prediction over the tuned MentalBERT+CORN encoder? Twelve experiments, participant-grouped 5-fold out-of-fold (n=1752). This is the summary; the full step-by-step report is <code>cot_full_report.html</code>.</div></header>
<div class="wrap">

<div class="card"><h2>Headline result</h2>
<div class="hero">A simple <b>severity-routing ensemble</b> of the encoder and a few-shot CoT LLM (Qwen-7B) is the best configuration found:
<b>macro-F1 {best['macro_f1']:.3f}</b> &nbsp;·&nbsp; <b>QWK {best['qwk']:.3f}</b><br>
<span style="font-size:13px;color:#047857">vs encoder alone {encm['macro_f1']:.3f} / {encm['qwk']:.3f} &nbsp;(Δ +{d_f1:.3f} macro-F1, +{d_qwk:.3f} QWK) — and it beats the CoT alone too.</span></div>
<p style="margin-top:12px">The two models are <b>complementary</b>: the LLM recovers the severe classes the encoder collapses on (class-3 F1 {w5['f1_per_class'][3]:.3f} vs {encm['f1_per_class'][3]:.3f}); the encoder keeps its errors ordinally closer. Routing — trust the LLM when it predicts a severe class (≥2), else the encoder — captures both strengths.</p>
</div>

<div class="card"><h2>Results at a glance</h2>
<p class="note">Pooled out-of-fold (n=1752), directly comparable to the encoder. macro-F1 &amp; QWK (ordinal agreement) are the headline metrics; accuracy is weak on this imbalanced 4-class task.</p>
{overview}
<table><thead><tr><th>Configuration</th><th>macro-F1</th><th>QWK</th><th>MAE</th><th>c3 F1</th><th></th></tr></thead>
<tbody>{trows}</tbody></table>
</div>

<div class="card"><h2>What worked, what didn't</h2>
<h3><span class="tag win">WORKS</span>Genuine value</h3>
<ul>
<li><b>The ensemble</b> — severity-routing beats both parents on macro-F1 <i>and</i> QWK simultaneously. The core positive result.</li>
<li><b>Better evidence (W5 windows)</b> — improved the ensemble's ordinal metrics; set the best config.</li>
<li><b>Item-adaptive evidence</b> — choosing evidence per item (full transcript for evidence-starved items like Appetite/NoInterest, focused retrieval for well-covered ones) lifts standalone CoT QWK to {ia['qwk']:.3f} (from {w5['qwk']:.3f}) with no macro-F1 loss. Principled, leakage-free — but doesn't beat the headline ensemble.</li>
</ul>
<h3><span class="tag ins">INSIGHT</span>Changed our understanding</h3>
<ul>
<li><b>Full transcript halved the under-called-severe cases (215→105)</b> — proving the severe misses were <i>part retrieval, part label</i>, not a pure label ceiling. But more context causes over-attribution, so the net is item-dependent.</li>
<li><b>Retrieval was an unnecessary crutch for the LLM</b> — inherited from the encoder's 512-token limit; the full interview (~1.3k tokens) fits the LLM's context many times over.</li>
</ul>
<h3><span class="tag neg">NO GAIN</span>Tested and bounded</h3>
<ul>
<li><b>Self-consistency</b> — eroded the macro-F1 edge (voting regresses to the safe label).</li>
<li><b>Distillation to 1.5B</b> — lost the severe-class recovery; the value is in the 7B's reasoning capacity, not transferable cheaply.</li>
<li><b>Domain model (MentaLLaMA 13B/7B)</b> — older LLaMA-2 base reasons about severity worse than a modern 7B; over-predicts, poor format adherence.</li>
<li><b>Item-aware retrieval</b> — helps weak BM25 (+0.034 macro-F1) but is subsumed by good semantic/hybrid retrieval.</li>
<li><b>Learned stacking</b> — does not beat the hand-built routing rule.</li>
</ul>
</div>

<div class="card"><h2>Conclusions</h2>
<ul>
<li><b>CoT earns its place as an ensemble component, not a replacement.</b> An untrained few-shot LLM genuinely complements a tuned encoder on this task.</li>
<li><b>Scaling/model levers don't help here</b> — bigger context, more samples, distillation, and a domain model all failed. The useful signal is in the 7B's reasoning, and it's already captured.</li>
<li><b>The remaining ceiling is mostly in the labels.</b> Three independent lines of evidence converge: error analysis (CoT reasoning is faithful; the gold label diverges from interview content), the retrieval diagnostic (some symptoms are barely verbalised), and the full-transcript test (only ~half the severe misses were recoverable with more context). PHQ-8 self-report is only loosely coupled to what people say in the interview — especially somatic items.</li>
</ul>
</div>

<div class="card"><h2>Recommended next steps</h2>
<p class="note">Ordered by expected value. The first is the only one likely to lift the headline ceiling; the rest are refinements or rigor.</p>
<ul>
<li><span class="pri">1.</span> <b>Get better labels (the real lever).</b> The ceiling is a self-report-vs-content gap, so the highest-value move is data, not modelling: clinician-rated item severity, or annotations of symptoms actually expressed in the interview. Alternatively, reframe the target toward the clinical <b>screening decision</b> (PHQ-8 total ≥10) and evaluate there — the ensemble's severe-class recovery may matter most for sensitivity.</li>
<li><span class="pri">2.</span> <b>Tame full-transcript over-attribution, then re-ensemble.</b> Item-adaptive evidence already lifts standalone QWK; pairing it with a calibration/abstention prompt (let the LLM defer to the encoder when evidence is thin) could let it beat 0.377 in the ensemble rather than just match it.</li>
<li><span class="pri">3.</span> <b>Item-aware expansion on the <i>production</i> hybrid retriever.</b> Our test used a weaker MiniLM reimplementation; applying query expansion to the real off-branch hybrid is the one retrieval experiment that could still fairly beat the pipeline (most promising on Sleep/Failure/Appetite).</li>
<li><span class="pri">4.</span> <b>Multi-seed confirmation.</b> All headline numbers are single-seed; confirm the ensemble gain and the small item-adaptive QWK improvement across seeds before publishing.</li>
<li><span class="pri">5.</span> <b>Per-item ensemble combiner.</b> The routing rule is global; a per-item route/blend (informed by the item-adaptive selection) is a cheap, no-GPU refinement.</li>
</ul>
</div>

<div class="card"><div class="caveat"><b>Honesty notes.</b> Single seed; participant-grouped 5-fold OOF; DAIC-WOZ is public so the LLM may have seen related data; severe-class counts are small so per-class numbers are high-variance; the item-aware-hybrid test used a MiniLM reimplementation weaker than the production retriever; the domain model was judged on canary folds. All gains beyond the ensemble are modest and should be multi-seed-confirmed.</div>
<p class="note" style="margin-top:10px">Full step-by-step report with all figures: <code>outputs/cot/cot_full_report.html</code>. Code in <code>src/llm/</code>, <code>src/retrieval/</code>, <code>src/evaluation/</code>.</p>
</div>

</div></body></html>"""
    OUT.write_text(html)
    print(f"Saved summary: {OUT}  ({len(html)//1024} KB)")
    print(f"Best: severity-routed {best['macro_f1']:.3f}/{best['qwk']:.3f} | encoder {encm['macro_f1']:.3f}/{encm['qwk']:.3f}")


if __name__ == "__main__":
    main()
