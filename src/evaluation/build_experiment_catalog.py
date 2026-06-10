"""
Experiment catalog — every run we did, organised like the sweep figure
(Backbone x Loss x Evidence). Reads outputs/run_manifest.json and renders a
self-contained HTML coverage matrix + run-family summary.

    python -m src.evaluation.build_experiment_catalog
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"

MODELS = ["BERT", "MentalBERT"]
LOSSES = ["plain CE", "weighted CE", "CORN"]
# evidence axis (figure order; TF-IDF utt added as the original baseline)
EVID = ["Plain (no-retrieval utt)", "TF-IDF utt", "BM25 utt", "BM25 W3", "BM25 W5", "Hybrid W3", "Hybrid W5"]


def model_of(r):
    return "MentalBERT" if "mental" in r["model_name"] else "BERT"


def loss_of(r):
    l = r["loss"]
    return "CORN" if "CORN" in l else ("weighted CE" if "weighted" in l else "plain CE")


def evid_of(r):
    rm, ds = r["retrieval_method"], r["dataset_path"]
    if "first-K" in rm:
        return "Plain (no-retrieval utt)"
    if rm.startswith("TF-IDF"):
        return "TF-IDF utt"
    if rm.startswith("BM25 (utt"):
        return "BM25 utt"
    if "context windows" in rm and rm.startswith("BM25"):
        return "BM25 W3" if "w3" in ds else "BM25 W5"
    if "Hybrid" in rm:
        return "Hybrid W3" if "w3" in ds else "Hybrid W5"
    return "other"


def family_of(rid):
    if rid.startswith("cv/seedconf_"):
        return "Multi-seed confirmation (CORN×Hybrid, 5 seeds)"
    if rid.startswith("cv/ctxm_"):
        return "MentalBERT context-window matrix"
    if rid.startswith("cv/ctx_"):
        return "BERT context-window matrix"
    if rid.startswith("cv/bm25base_"):
        return "Unified BM25 baseline (BERT vs MentalBERT)"
    if rid.startswith("cv/"):
        return "Original CV (TF-IDF baseline)"
    if rid.startswith("sweep_"):
        return "MentalBERT LR×epoch sweep"
    if rid.startswith("seeds_"):
        return "Multi-seed weighted-CE (TF-IDF)"
    return "Single split (val/test)"


def shade(f1):
    if f1 is None:
        return "#f1f5f9", "#94a3b8"
    # green scale 0.25..0.38
    t = max(0.0, min(1.0, (f1 - 0.25) / 0.13))
    r = int(220 - 200 * t); g = int(235 - 60 * t); b = int(225 - 160 * t)
    return f"rgb({r},{g},{b})", ("#fff" if t > 0.7 else "#1e293b")


def main():
    m = json.loads((OUT / "run_manifest.json").read_text())

    # coverage cell: (evidence, model, loss) -> list of macro_f1
    cell = {}
    for r in m:
        ev = evid_of(r)
        if ev == "other":
            continue
        k = (ev, model_of(r), loss_of(r))
        f1 = r["metrics"].get("macro_f1")
        cell.setdefault(k, []).append(f1)

    # build matrix table
    cols = [(mo, lo) for mo in MODELS for lo in LOSSES]
    header = "".join(f'<th class="m">{lo}</th>' for mo in MODELS for lo in LOSSES)
    rows_html = ""
    for ev in EVID:
        cells = ""
        for mo, lo in cols:
            vals = [v for v in cell.get((ev, mo, lo), []) if v is not None]
            if not vals:
                bg, fg = shade(None)
                cells += f'<td style="background:{bg};color:{fg}">·</td>'
            else:
                best = max(vals); bg, fg = shade(best)
                cells += (f'<td style="background:{bg};color:{fg}">'
                          f'<b>{best:.3f}</b><br><span style="font-size:9px;opacity:.8">{len(vals)} run{"s" if len(vals)>1 else ""}</span></td>')
        rows_html += f'<tr><td class="ev">{ev}</td>{cells}</tr>'

    # run-family summary
    fam = {}
    for r in m:
        f = family_of(r["run_id"])
        fam[f] = fam.get(f, 0) + 1
    fam_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(fam.items(), key=lambda x: -x[1]))

    # best overall cell
    best_run = max((r for r in m if r["metrics"].get("macro_f1") is not None),
                   key=lambda r: r["metrics"]["macro_f1"])
    n_total = len(m)
    n_bert = sum(1 for r in m if model_of(r) == "BERT")
    n_mbert = n_total - n_bert

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PHQ-8 Experiment Catalog</title>
<style>
 *{{box-sizing:border-box}} body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;color:#1e293b;background:#f8fafc;line-height:1.55}}
 header{{background:linear-gradient(135deg,#1e3a5f,#0891b2);color:#fff;padding:32px 28px}} header h1{{margin:0 0 6px;font-size:24px}} header .sub{{opacity:.92;font-size:14px}}
 .wrap{{max-width:1080px;margin:0 auto;padding:24px 28px 60px}}
 .card{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px 24px;margin:18px 0;box-shadow:0 1px 2px rgba(15,23,42,.04)}}
 h2{{font-size:19px;margin:4px 0 14px;padding-bottom:8px;border-bottom:2px solid #cffafe}}
 .dims{{display:flex;gap:12px;flex-wrap:wrap}}
 .dim{{flex:1;min-width:220px;border:1px solid #cbd5e1;border-radius:10px;padding:14px 16px;background:#f8fbfd}}
 .dim .l{{font-size:11px;font-weight:700;letter-spacing:.06em;color:#0891b2;text-transform:uppercase}}
 .dim .v{{font-size:16px;font-weight:600;margin-top:4px}}
 .arrow{{font-size:20px;color:#0891b2;align-self:center}}
 table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:12.5px}}
 th,td{{padding:6px 8px;border:1px solid #e2e8f0;text-align:center}} th{{font-size:10.5px;text-transform:uppercase;color:#475569;background:#f1f5f9}}
 td.ev{{text-align:left;font-weight:600;background:#fbfcfe}} th.grp{{background:#e0f2fe}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
 .stat{{background:#ecfeff;border-radius:10px;padding:13px 15px}} .stat .n{{font-size:21px;font-weight:700;color:#0891b2}} .stat .l{{font-size:11px;color:#475569}}
 .callout{{border-left:4px solid #0891b2;background:#ecfeff;padding:12px 16px;border-radius:0 8px 8px 0;margin:12px 0}}
 .note{{color:#64748b;font-size:13px}} code{{background:#f1f5f9;padding:1px 5px;border-radius:5px;font-size:12px}}
 footer{{text-align:center;color:#64748b;font-size:12px;padding:24px}}
</style></head><body>
<header><h1>PHQ-8 Item Classifier — Experiment Catalog</h1>
<div class="sub">Everything swept so far · {n_total} training runs · participant-grouped 5-fold CV (OOF) unless noted</div></header>
<div class="wrap">

 <div class="card"><h2>Dimensions swept</h2>
  <div class="dims">
   <div class="dim"><div class="l">Backbone</div><div class="v">BERT-base · MentalBERT</div></div>
   <div class="arrow">→</div>
   <div class="dim"><div class="l">Loss function</div><div class="v">plain CE · weighted CE · CORN (ordinal)</div></div>
   <div class="arrow">→</div>
   <div class="dim"><div class="l">Evidence (retrieval)</div><div class="v">Plain · TF-IDF utt · BM25 utt · BM25 W3/W5 · Hybrid W3/W5</div></div>
  </div>
  <div class="grid" style="margin-top:14px">
   <div class="stat"><div class="n">{n_total}</div><div class="l">total runs</div></div>
   <div class="stat"><div class="n">{n_bert}/{n_mbert}</div><div class="l">BERT / MentalBERT runs</div></div>
   <div class="stat"><div class="n">{best_run['metrics']['macro_f1']:.3f}</div><div class="l">best macro-F1 ({model_of(best_run)} · {best_run['run_id'].split('/')[-1]})</div></div>
   <div class="stat"><div class="n">256 · 512</div><div class="l">max_length (512 for richer context)</div></div>
  </div>
 </div>

 <div class="card"><h2>Coverage matrix — best macro-F1 per cell</h2>
  <table>
   <tr><th rowspan="2" class="ev">Evidence ↓ &nbsp; Backbone·Loss →</th><th class="grp" colspan="3">BERT-base</th><th class="grp" colspan="3">MentalBERT</th></tr>
   <tr>{header}</tr>
   {rows_html}
  </table>
  <p class="note">Cell = best macro-F1 across all variants/seeds for that (evidence × backbone × loss); greener = higher; "·" = not run. Counts include 256/512 and multi-seed variants.</p>
  <div class="callout"><b>Headline:</b> MentalBERT (right half) is greener than BERT (left half) across the context-window rows — the BM25/Hybrid evidence is where MentalBERT + CORN shines. Best overall: <b>{model_of(best_run)} · {best_run['run_id'].split('/')[-1]}</b> (macro-F1 {best_run['metrics']['macro_f1']:.3f}).</div>
 </div>

 <div class="card"><h2>Run families ({n_total} runs)</h2>
  <table style="font-size:13px"><tr><th class="ev">Experiment family</th><th>runs</th></tr>{fam_rows}</table>
  <p class="note">Beyond the main backbone×loss×evidence grid, this includes the MentalBERT hyperparameter sweep, multi-seed robustness studies, the unified-baseline fix, and the 512-token follow-ups. Full per-run config + metrics + reproduction command in <code>outputs/run_manifest.json</code>.</p>
 </div>

 <div class="card"><h2>Reports produced</h2>
  <ul style="font-size:13.5px">
   <li><code>status_report_v2.html</code> — original BERT study (TF-IDF): retrieval, class weighting, CORN, near-miss</li>
   <li><code>context_window_report.html</code> — context-window evidence comparison (Person A handoff)</li>
   <li><code>bert_vs_mbert_bm25_report.html</code> — BERT vs MentalBERT on the unified BM25 utterance baseline</li>
   <li><code>bert_vs_mbert_context_report.html</code> — BERT vs MentalBERT across the full context matrix</li>
   <li><code>person_b_handoff_response.md</code> — answers + methods · <code>run_manifest.json</code> — this catalog's source</li>
  </ul>
 </div>
 <footer>Generated by <code>src/evaluation/build_experiment_catalog.py</code> from <code>run_manifest.json</code> · {n_total} runs</footer>
</div></body></html>"""
    out = OUT / "experiment_catalog.html"
    out.write_text(html)
    print("wrote", out, f"({len(html)//1024} KB)")


if __name__ == "__main__":
    main()
