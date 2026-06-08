"""
Self-contained HTML report: BERT vs MentalBERT on the unified BM25 utterance
baseline (data/processed/phq8_item_dataset_full_bm25.csv), participant-grouped
5-fold CV with shared folds. Reads outputs/cv/comprehensive_metrics_bm25base_*.json.

    python -m src.evaluation.build_model_comparison_report
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "outputs" / "cv"
OUT = ROOT / "outputs"
LOSSES = [("plain CE", "ceplain"), ("weighted CE", "ce"), ("CORN", "corn")]
BERT_C, MBERT_C = "#4f46e5", "#0891b2"


def load(model, loss):
    f = CV / f"comprehensive_metrics_bm25base_{model}_{loss}.json"
    return json.loads(f.read_text()) if f.exists() else None


def grouped_bars(title, getter, vmax, fmt="{:.3f}", note=""):
    """One group per loss, two bars (BERT, MentalBERT)."""
    W, H = 600, 250
    padl, padb, padt = 44, 52, 24
    plot_h = H - padb - padt
    ng = len(LOSSES); gap = (W - padl - 16) / ng; bw = gap * 0.28
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    for fr in (0, .25, .5, .75, 1.0):
        yy = padt + plot_h * (1 - fr)
        p.append(f'<line x1="{padl}" y1="{yy:.1f}" x2="{W-8}" y2="{yy:.1f}" stroke="#e2e8f0"/>')
        p.append(f'<text x="{padl-6}" y="{yy+3:.1f}" text-anchor="end" fill="#94a3b8">{vmax*fr:.2f}</text>')
    for gi, (lname, lkey) in enumerate(LOSSES):
        cx = padl + 8 + gi * gap + gap / 2
        for si, (model, col) in enumerate([("bert", BERT_C), ("mbert", MBERT_C)]):
            m = load(model, lkey)
            if m is None:
                continue
            v = getter(m)
            x = cx + (si - 0.5) * bw * 1.2 - bw / 2
            h = plot_h * (max(v, 0) / vmax); yy = padt + plot_h - h
            p.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{h:.1f}" rx="2" fill="{col}"/>')
            p.append(f'<text x="{x+bw/2:.1f}" y="{yy-3:.1f}" text-anchor="middle" fill="#1e293b" font-size="9">{fmt.format(v)}</text>')
        p.append(f'<text x="{cx:.1f}" y="{padt+plot_h+16:.1f}" text-anchor="middle" fill="#64748b">{lname}</text>')
    p.append(f'<rect x="{padl}" y="6" width="11" height="11" fill="{BERT_C}"/><text x="{padl+15}" y="15" fill="#64748b">BERT-base</text>')
    p.append(f'<rect x="{padl+95}" y="6" width="11" height="11" fill="{MBERT_C}"/><text x="{padl+110}" y="15" fill="#64748b">MentalBERT</text>')
    p.append('</svg>')
    return f'<p class="note" style="margin-bottom:2px"><b>{title}</b></p>' + "".join(p) + (f'<p class="note">{note}</p>' if note else "")


def confusion_svg(model, loss, title, accent):
    m = load(model, loss)
    cm = m["item"]["confusion_4x4"]
    rs = [sum(r) or 1 for r in cm]
    cell = 50; x0, y0 = 80, 40; W, H = 320, 270
    r = int(accent[1:3], 16); g = int(accent[3:5], 16); b = int(accent[5:7], 16)
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    p.append(f'<text x="{x0+2*cell}" y="16" text-anchor="middle" fill="#64748b" font-size="10">predicted</text>')
    for j in range(4):
        p.append(f'<text x="{x0+j*cell+cell/2}" y="{y0-5}" text-anchor="middle" fill="#475569">{j}</text>')
    for i in range(4):
        p.append(f'<text x="{x0-8}" y="{y0+i*cell+cell/2+4}" text-anchor="end" fill="#475569">{i}</text>')
        for j in range(4):
            frac = cm[i][j] / rs[i]
            a = 0.10 + 0.85 * frac
            tc = "#fff" if frac > 0.5 else "#1e293b"
            x = x0 + j * cell; y = y0 + i * cell
            p.append(f'<rect x="{x}" y="{y}" width="{cell-3}" height="{cell-3}" rx="3" fill="rgba({r},{g},{b},{a:.2f})"/>')
            p.append(f'<text x="{x+(cell-3)/2}" y="{y+(cell-3)/2+4}" text-anchor="middle" fill="{tc}" font-weight="600">{cm[i][j]}</text>')
    p.append('</svg>')
    return f'<div style="flex:1;min-width:280px"><p class="note" style="margin-bottom:2px"><b>{title}</b></p>' + "".join(p) + '</div>'


def per_item_grouped(loss, title):
    bm = load("bert", loss); mm = load("mbert", loss)
    bi = {r["item_id"]: r for r in bm["per_item"]}
    mi = {r["item_id"]: r for r in mm["per_item"]}
    items = sorted(bi, key=lambda k: mi[k]["macro_f1"], reverse=True)
    W = 600; padl = 110; padr = 30; padt = 26; rh = 28; padb = 6
    H = padt + len(items) * rh + padb; plot_w = W - padl - padr; vmax = 0.45
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    for i, iid in enumerate(items):
        y = padt + i * rh; bv = bi[iid]["macro_f1"]; mv = mi[iid]["macro_f1"]
        p.append(f'<text x="{padl-8}" y="{y+rh/2+3:.1f}" text-anchor="end" fill="#475569">{bi[iid]["item_name"]}</text>')
        p.append(f'<rect x="{padl}" y="{y+3:.1f}" width="{plot_w*bv/vmax:.1f}" height="{rh/2-3}" rx="2" fill="{BERT_C}"/>')
        p.append(f'<rect x="{padl}" y="{y+rh/2:.1f}" width="{plot_w*mv/vmax:.1f}" height="{rh/2-3}" rx="2" fill="{MBERT_C}"/>')
        p.append(f'<text x="{padl+plot_w*max(bv,mv)/vmax+5:.1f}" y="{y+rh-7:.1f}" fill="#1e293b" font-size="9">{mv:.2f}</text>')
    p.append(f'<rect x="{padl}" y="6" width="11" height="11" fill="{BERT_C}"/><text x="{padl+15}" y="15" fill="#64748b">BERT</text>')
    p.append(f'<rect x="{padl+70}" y="6" width="11" height="11" fill="{MBERT_C}"/><text x="{padl+85}" y="15" fill="#64748b">MentalBERT</text>')
    p.append('</svg>')
    return f'<p class="note" style="margin-bottom:2px"><b>{title}</b></p>' + "".join(p)


def main():
    # main table
    def trow(model, mlabel, color):
        rows = ""
        for lname, lkey in LOSSES:
            m = load(model, lkey)
            if not m:
                continue
            i = m["item"]; t = m["total"]; th = t and m["total_thresholds"][">=10"]
            ib = m["item_binary_0v1_vs_2v3"]
            hl = ' style="background:#eef2ff"' if (model == "mbert" and lkey == "corn") else ""
            rows += (f"<tr{hl}><td>{mlabel}</td><td>{lname}</td>"
                     f"<td><b>{i['macro_f1']:.3f}</b></td><td>{i['mae']:.3f}</td><td class='sec'>{i['accuracy']:.3f}</td>"
                     f"<td>{i['qwk']:.3f}</td><td>{i['f1_per_class']['2']:.3f}</td><td>{i['f1_per_class']['3']:.3f}</td>"
                     f"<td>{ib['f1']:.3f}</td><td>{t['total_mae']:.2f}</td><td>{t['total_qwk']:.3f}</td>"
                     f"<td>{th['balanced_accuracy']:.3f}</td><td>{th['sensitivity']:.3f}</td></tr>")
        return rows
    main_rows = trow("bert", "BERT-base", BERT_C) + trow("mbert", "MentalBERT", MBERT_C)

    # paired deltas
    paired = ""
    for lname, lkey in LOSSES:
        b = load("bert", lkey); m = load("mbert", lkey)
        if not (b and m):
            continue
        d = m["item"]["macro_f1"] - b["item"]["macro_f1"]
        dq = m["item"]["qwk"] - b["item"]["qwk"]
        dt = m["total"]["total_qwk"] - b["total"]["total_qwk"]
        paired += (f"<tr><td>{lname}</td><td>{b['item']['macro_f1']:.3f}</td><td>{m['item']['macro_f1']:.3f}</td>"
                   f"<td><b>{d:+.3f}</b></td><td>{dq:+.3f}</td><td>{dt:+.3f}</td></tr>")

    # thresholds for best of each (BERT weighted CE vs MentalBERT CORN)
    def thr_table(model, loss, label):
        m = load(model, loss)
        rows = ""
        for thr in [">=5", ">=10", ">=15", ">=20"]:
            t = m["total_thresholds"][thr]
            rows += (f"<tr><td>{label}</td><td>total {thr}</td><td>{t['sensitivity']:.3f}</td><td>{t['specificity']:.3f}</td>"
                     f"<td>{t['precision']:.3f}</td><td><b>{t['balanced_accuracy']:.3f}</b></td><td>{t['f1']:.3f}</td>"
                     f"<td>{t['n_pos']}/{t['n_neg']}</td></tr>")
        return rows
    thr_rows = thr_table("bert", "ce", "BERT wCE") + thr_table("mbert", "corn", "MentalBERT CORN")

    # severity band metrics
    band_rows = ""
    for model, label in [("bert", "BERT wCE"), ("mbert", "MentalBERT CORN")]:
        loss = "ce" if model == "bert" else "corn"
        sb = load(model, loss)["severity_bands"]
        band_rows += (f"<tr><td>{label}</td><td>{sb['band_accuracy']:.3f}</td><td>{sb['band_qwk']:.3f}</td>"
                      f"<td>{', '.join(f'{k}:{v}' for k,v in sb['true_band_counts'].items())}</td></tr>")

    fig_f1 = grouped_bars("Fig 1 · Item macro-F1 by loss", lambda m: m["item"]["macro_f1"], 0.4,
                          note="CE variants ~tied; MentalBERT edges ahead under CORN.")
    fig_qwk = grouped_bars("Fig 2 · Item QWK by loss", lambda m: m["item"]["qwk"], 0.3,
                           note="MentalBERT higher across all three losses.")
    fig_totqwk = grouped_bars("Fig 3 · Reconstructed PHQ-8 TOTAL QWK", lambda m: m["total"]["total_qwk"], 0.4)
    fig_totmae = grouped_bars("Fig 4 · PHQ-8 TOTAL MAE (lower=better)", lambda m: m["total"]["total_mae"], 6.0, fmt="{:.2f}")
    fig_bacc = grouped_bars("Fig 5 · Screening balanced accuracy (total ≥ 10)", lambda m: m["total_thresholds"][">=10"]["balanced_accuracy"], 0.7)
    fig_cm = ('<div style="display:flex;flex-wrap:wrap;gap:18px">'
              + confusion_svg("bert", "ce", "Fig 6a · BERT (weighted CE) — best BERT", BERT_C)
              + confusion_svg("mbert", "corn", "Fig 6b · MentalBERT (CORN) — best overall", MBERT_C)
              + '</div>')
    fig_item = per_item_grouped("corn", "Fig 7 · Per-item macro-F1 (CORN): BERT vs MentalBERT")

    bce = load("bert", "ce")["item"]; mco = load("mbert", "corn")["item"]
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BERT vs MentalBERT — unified BM25 baseline</title>
<style>
 *{{box-sizing:border-box}} body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;color:#1e293b;background:#f8fafc;line-height:1.55}}
 header{{background:linear-gradient(135deg,#4f46e5,#0891b2);color:#fff;padding:32px 28px}} header h1{{margin:0 0 6px;font-size:24px}} header .sub{{opacity:.92;font-size:14px}}
 header .meta{{margin-top:12px;font-size:13px;opacity:.9;display:flex;gap:18px;flex-wrap:wrap}}
 .wrap{{max-width:1060px;margin:0 auto;padding:24px 28px 60px}}
 .card{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px 24px;margin:18px 0;box-shadow:0 1px 2px rgba(15,23,42,.04)}}
 h2{{font-size:19px;margin:4px 0 14px;padding-bottom:8px;border-bottom:2px solid #eef2ff}} h3{{font-size:14px;color:#0891b2;margin:16px 0 6px}}
 table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:12.5px}}
 th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid #e2e8f0}} th{{font-size:11px;text-transform:uppercase;letter-spacing:.03em;color:#64748b;background:#fbfcfe}}
 td.sec,th.sec{{color:#94a3b8}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:6px 0}}
 .stat{{background:#eef2ff;border-radius:10px;padding:13px 15px}} .stat .n{{font-size:21px;font-weight:700;color:#4f46e5}} .stat .l{{font-size:11.5px;color:#475569}}
 .callout{{border-left:4px solid #0891b2;background:#ecfeff;padding:12px 16px;border-radius:0 8px 8px 0;margin:12px 0}}
 .warn{{border-left-color:#d97706;background:#fef3c7}}
 .note{{color:#64748b;font-size:13px}} code{{background:#f1f5f9;padding:1px 5px;border-radius:5px;font-size:12px}}
 footer{{text-align:center;color:#64748b;font-size:12px;padding:24px}} ul{{margin:8px 0;padding-left:20px}} li{{margin:5px 0}}
</style></head><body>
<header><h1>BERT-base vs MentalBERT — unified BM25 utterance baseline</h1>
<div class="sub">PHQ-8 item severity · <code style="background:rgba(255,255,255,.15);color:#fff">phq8_item_dataset_full_bm25.csv</code> · participant-grouped 5-fold CV (shared folds, OOF)</div>
<div class="meta"><span>📅 2026-06-08</span><span>seed 42 · max_len 256 · 8 epochs</span><span>🖥️ RTX 3090</span></div></header>
<div class="wrap">

 <div class="card"><h2>TL;DR</h2>
  <p>Same dataset, evidence, folds, and hyperparameters — only the encoder changes. Three losses each.</p>
  <div class="grid">
   <div class="stat"><div class="n">{mco['macro_f1']:.3f}</div><div class="l">Best macro-F1 (MentalBERT·CORN) vs {bce['macro_f1']:.3f} best BERT</div></div>
   <div class="stat"><div class="n">+all 3</div><div class="l">losses: MentalBERT higher QWK</div></div>
   <div class="stat"><div class="n">{load('mbert','corn')['total']['total_qwk']:.2f}</div><div class="l">MentalBERT total QWK vs {load('bert','ce')['total']['total_qwk']:.2f} BERT</div></div>
   <div class="stat"><div class="n">5-fold</div><div class="l">paired CV (shared folds)</div></div>
  </div>
  <div class="callout"><b>On macro-F1 the cross-entropy variants are ~tied, but MentalBERT is consistently ahead on QWK, PHQ-8 total MAE/QWK, and screening balanced-accuracy across all three losses.</b> On this BM25 evidence MentalBERT has a small, consistent edge — unlike the earlier TF-IDF setup where the two were a confirmed tie.</div>
  <div class="callout warn"><b>⚠ Single seed (42).</b> The CORN macro-F1 gap is partly a low BERT-CORN run (0.257 here vs 0.289 on TF-IDF). The robust signal is the consistent multi-metric edge, not the exact magnitude. A 3–5 seed repeat would settle it.</div>
 </div>

 <div class="card"><h2>Full comparison (6 runs)</h2>
  <table><tr><th>Model</th><th>Loss</th><th>Macro-F1</th><th>MAE↓</th><th class="sec">Acc (2°)</th><th>QWK</th><th>F1 c2</th><th>F1 c3</th><th>sev F1<br>(0-1v2-3)</th><th>totMAE↓</th><th>totQWK</th><th>bAcc≥10</th><th>sens≥10</th></tr>
  {main_rows}</table>
  <p class="note">Accuracy greyed (secondary; majority ≈ 0.44). "sev F1" = item-binary 0-1 vs 2-3. MentalBERT·CORN row highlighted (best overall).</p>
 </div>

 <div class="card"><h2>Paired deltas (same folds, MentalBERT − BERT)</h2>
  <table><tr><th>Loss</th><th>BERT macro-F1</th><th>MentalBERT macro-F1</th><th>Δ macro-F1</th><th>Δ QWK</th><th>Δ total-QWK</th></tr>{paired}</table>
 </div>

 <div class="card"><h2>Figures</h2>
  {fig_f1}{fig_qwk}{fig_totqwk}{fig_totmae}{fig_bacc}
  <h3>Confusion matrices (best of each model)</h3>
  {fig_cm}
  {fig_item}
 </div>

 <div class="card"><h2>Clinical thresholds (PHQ-8 total) — best of each model</h2>
  <table><tr><th>Model</th><th>Threshold</th><th>Sensitivity</th><th>Specificity</th><th>Precision</th><th>Balanced Acc</th><th>F1</th><th>pos/neg</th></tr>{thr_rows}</table>
  <h3>Severity-band agreement (0-4 / 5-9 / 10-14 / 15-19 / 20-24)</h3>
  <table><tr><th>Model</th><th>Band accuracy</th><th>Band QWK</th><th>True band counts</th></tr>{band_rows}</table>
  <p class="note">All metrics from reconstructed PHQ-8 totals (sum of 8 OOF item predictions per participant; 219 participants, 65 with total ≥ 10).</p>
 </div>

 <div class="card"><h2>Takeaways</h2>
  <ul>
   <li><b>MentalBERT has a small, consistent edge on this BM25 evidence</b> — strongest on ordinal/total-score metrics (QWK, total MAE/QWK, screening balanced-acc), all three losses.</li>
   <li><b>The model verdict is retrieval-base dependent:</b> tie on TF-IDF (multi-seed confirmed), mild MentalBERT lead on BM25 (single seed here).</li>
   <li><b>Confirm with multi-seed</b> on <code>full_bm25</code> before treating the gap as definitive.</li>
  </ul>
  <p class="note">Data: <code>outputs/cv/comprehensive_metrics_bm25base_*.json</code>, <code>outputs/model_comparison_results.json</code>. Runs in <code>outputs/run_manifest.json</code> (retrieval_method = BM25 utterance retrieval).</p>
 </div>
 <footer>Generated by <code>src/evaluation/build_model_comparison_report.py</code> · inline-SVG · paired 5-fold CV</footer>
</div></body></html>"""
    out = OUT / "bert_vs_mbert_bm25_report.html"
    out.write_text(html)
    print("wrote", out, f"({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
