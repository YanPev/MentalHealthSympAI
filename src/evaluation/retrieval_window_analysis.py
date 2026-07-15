"""
Per-item analysis of the retrieved context windows.

Motivated by the error analysis (far-off errors concentrate on somatic/internal
items): does retrieval actually surface *item-relevant* content for every PHQ-8
item, or are some items reasoning over evidence that never mentions the symptom?

For each item we measure, over the dataset:
  * keyword-hit rate -- fraction of examples whose retrieved windows contain any
    item-relevant term (a heuristic topical-relevance proxy)
  * retrieval scores (mean top-1 / mean hybrid score)
  * evidence size (mean #windows, mean characters)
and cross-reference the per-item keyword-hit rate against the CoT far-off (>=2)
error rate to see whether thin evidence tracks the errors.

Outputs an HTML report with inline-SVG plots AND a gallery of sampled examples
per item (retrieved windows shown, item-relevant terms highlighted).

    python -m src.evaluation.retrieval_window_analysis
"""

from pathlib import Path
import argparse
import glob
import html as _html
import json
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.retrieval.bm25_retriever import tokenize
from src.retrieval.symptom_glossary import EXPAND

PR = Path(__file__).resolve().parents[2]
DEFAULT_DS = PR / "data" / "processed" / "phq8_item_dataset_context_windows_hybrid_w3.csv"
COT_W5_DIR = PR / "outputs" / "cot" / "folds_w5"
OUT_HTML = PR / "outputs" / "cot" / "retrieval_window_analysis.html"
OUT_JSON = PR / "outputs" / "cot" / "retrieval_window_analysis.json"

# Item-relevant lexicons (substring match, lowercased), derived from the SAME
# glossary used to expand BM25 queries in item-aware retrieval (symptom_glossary.
# EXPAND), tokenized the same way BM25 tokenizes queries, with generic English
# stopwords ("i", "to", "about", ...) dropped so they don't trivially match every
# transcript window. This keeps the hit-rate measurement and the retrieval lever
# it is reporting on aligned to one vocabulary instead of two hand-maintained ones.
# Still a topical proxy, not ground truth -- a window can be relevant without
# containing an exact token.
LEXICONS = {
    name: sorted({t for t in tokenize(terms)
                  if t not in ENGLISH_STOP_WORDS and len(t) > 2})
    for name, terms in EXPAND.items()
}
ITEM_COLORS = {"NoInterest": "#0ea5e9", "Depressed": "#7c3aed", "Sleep": "#16a34a",
               "Tired": "#f59e0b", "Appetite": "#dc2626", "Failure": "#db2777",
               "Concentrating": "#0891b2", "Moving": "#65a30d"}


def parse_list(v):
    try:
        return json.loads(v) if isinstance(v, str) else []
    except json.JSONDecodeError:
        return []


def windows_text(wins):
    return " ".join(str(w).replace("[TURN_SEP]", " ") for w in wins).lower()


def hit_terms(text, lex):
    return [t for t in lex if t in text]


# --------------------------------------------------------------------------- SVG
def hbar(rows, title, vmax, fmt="{:.2f}", note=""):
    n = len(rows); rh = 26; padl = 130; padr = 60; padt = 6; padb = 6
    W = 600; pw = W - padl - padr; H = padt + n * rh + padb
    p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
         f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    for i, (lab, v, col) in enumerate(rows):
        y = padt + i * rh; bw = pw * (max(v, 0) / vmax)
        p.append(f'<text x="{padl-8}" y="{y+rh/2+3:.1f}" text-anchor="end" fill="#475569">{lab}</text>')
        p.append(f'<rect x="{padl}" y="{y+3:.1f}" width="{bw:.1f}" height="{rh-7}" rx="3" fill="{col}"/>')
        p.append(f'<text x="{padl+bw+5:.1f}" y="{y+rh/2+3:.1f}" fill="#1e293b" font-size="10">{fmt.format(v)}</text>')
    p.append('</svg>')
    sub = f'<br><span style="font-size:10px;color:#94a3b8">{note}</span>' if note else ""
    return f'<p class="note" style="margin:10px 0 2px"><b>{title}</b>{sub}</p>' + "".join(p)


def scatter(points, title, xlab, ylab, note=""):
    # points: list of (x, y, label, color)
    W, H = 560, 360; padl, padb, padt, padr = 56, 46, 16, 16
    pw, ph = W - padl - padr, H - padt - padb
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    xlo, xhi = 0, max(xs) * 1.1 + 1e-6
    ylo, yhi = 0, max(ys) * 1.15 + 1e-6
    sx = lambda x: padl + pw * (x - xlo) / (xhi - xlo)
    sy = lambda y: padt + ph * (1 - (y - ylo) / (yhi - ylo))
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
         f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
    s.append(f'<line x1="{padl}" y1="{padt}" x2="{padl}" y2="{padt+ph}" stroke="#cbd5e1"/>')
    s.append(f'<line x1="{padl}" y1="{padt+ph}" x2="{padl+pw}" y2="{padt+ph}" stroke="#cbd5e1"/>')
    # trend line (least squares)
    if len(points) >= 2:
        b, a = np.polyfit(xs, ys, 1)
        s.append(f'<line x1="{sx(xlo):.1f}" y1="{sy(a+b*xlo):.1f}" x2="{sx(xhi):.1f}" '
                 f'y2="{sy(a+b*xhi):.1f}" stroke="#94a3b8" stroke-dasharray="4 3"/>')
        r = np.corrcoef(xs, ys)[0, 1]
        s.append(f'<text x="{padl+pw}" y="{padt+12}" text-anchor="end" fill="#64748b" font-size="10">r = {r:.2f}</text>')
    for x, y, lab, col in points:
        s.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="{col}"/>')
        s.append(f'<text x="{sx(x)+8:.1f}" y="{sy(y)+3:.1f}" fill="#475569" font-size="10">{lab}</text>')
    s.append(f'<text x="{padl+pw/2}" y="{H-8}" text-anchor="middle" fill="#64748b">{xlab}</text>')
    s.append(f'<text x="14" y="{padt+ph/2}" text-anchor="middle" fill="#64748b" transform="rotate(-90 14 {padt+ph/2})">{ylab}</text>')
    s.append('</svg>')
    sub = f'<br><span style="font-size:10px;color:#94a3b8">{note}</span>' if note else ""
    return f'<p class="note" style="margin:10px 0 2px"><b>{title}</b>{sub}</p>' + "".join(s)


def highlight(text, terms, color):
    """Escape text, then bold/colour any matched lexicon terms."""
    esc = _html.escape(text)
    for t in sorted(set(terms), key=len, reverse=True):
        esc = re.sub(f"({re.escape(_html.escape(t))})",
                     rf'<mark style="background:{color}22;color:{color};font-weight:600">\1</mark>',
                     esc, flags=re.IGNORECASE)
    return esc


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Per-item retrieved-window analysis")
    ap.add_argument("--dataset", default=str(DEFAULT_DS))
    ap.add_argument("--samples-per-item", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lexicon", choices=["glossary", "expanded"], default="glossary",
                    help="glossary = original EXPAND tokens (frozen baseline metric); "
                         "expanded = tiered symptom_lexicon flat_terms (incl. MWEs)")
    ap.add_argument("--evidence-column", default="retrieved_context_windows_hybrid_list")
    ap.add_argument("--scores-column", default="retrieved_context_hybrid_scores")
    ap.add_argument("--out-tag", default="",
                    help="suffix for output files; empty writes the original "
                         "retrieval_window_analysis.{html,json} paths")
    a = ap.parse_args()

    non_default = (a.lexicon != "glossary" or a.dataset != str(DEFAULT_DS)
                   or a.evidence_column != "retrieved_context_windows_hybrid_list")
    if non_default and not a.out_tag:
        raise SystemExit("--out-tag is required for non-default runs so the frozen "
                         "baseline retrieval_window_analysis.{html,json} is never overwritten")

    if a.lexicon == "expanded":
        from src.retrieval.symptom_lexicon import flat_terms
        lexicons = {name: flat_terms(name) for name in EXPAND}
    else:
        lexicons = LEXICONS
    out_html = (OUT_HTML.with_name(f"retrieval_window_analysis_{a.out_tag}.html")
                if a.out_tag else OUT_HTML)
    out_json = (OUT_JSON.with_name(f"retrieval_window_analysis_{a.out_tag}.json")
                if a.out_tag else OUT_JSON)

    df = pd.read_csv(a.dataset)
    df["participant_id"] = df["participant_id"].astype(str)
    df["wins"] = df[a.evidence_column].apply(parse_list)
    df["scores"] = df[a.scores_column].apply(parse_list)
    df["text"] = df["wins"].apply(windows_text)
    df["nchars"] = df["text"].str.len()
    df["nwins"] = df["wins"].apply(len)
    df["topscore"] = df["scores"].apply(lambda s: max(s) if s else 0.0)
    df["hits"] = df.apply(lambda r: hit_terms(r["text"], lexicons.get(r["item_name"], [])), axis=1)
    df["nhits"] = df["hits"].apply(len)
    df["any_hit"] = df["nhits"] > 0

    # ---- optional: far-off error rate per item (CoT W5) ----
    faroff_by_item = {}
    cot_files = sorted(glob.glob(str(COT_W5_DIR / "*.csv")))
    if cot_files:
        cot = pd.concat([pd.read_csv(f) for f in cot_files], ignore_index=True)
        cot["participant_id"] = cot["participant_id"].astype(str)
        cot["faroff"] = (cot["prediction"] - cot["label"]).abs() >= 2
        faroff_by_item = cot.groupby("item_name")["faroff"].mean().to_dict()

    # ---- per-item aggregation ----
    rows = []
    for iid in range(1, 9):
        g = df[df.item_id == iid]
        name = g.item_name.iloc[0]
        rows.append({
            "item_id": iid, "item": name, "n": int(len(g)),
            "keyword_hit_rate": float(g.any_hit.mean()),
            "mean_hits": float(g.nhits.mean()),
            "mean_top_score": float(g.topscore.mean()),
            "mean_nwins": float(g.nwins.mean()),
            "mean_nchars": float(g.nchars.mean()),
            "frac_label0": float((g.label == 0).mean()),
            "faroff_rate": float(faroff_by_item.get(name, float("nan"))),
        })
    agg = pd.DataFrame(rows)
    corr_kw_fo = (float(np.corrcoef(agg.keyword_hit_rate, agg.faroff_rate)[0, 1])
                  if faroff_by_item else float("nan"))
    out_json.write_text(json.dumps({"dataset": Path(a.dataset).name,
                                    "lexicon": a.lexicon,
                                    "evidence_column": a.evidence_column,
                                    "lexicons": lexicons, "per_item": rows}, indent=2))

    # ---- figures ----
    hr = agg.sort_values("keyword_hit_rate")
    fig_hit = hbar([(r["item"], r["keyword_hit_rate"], ITEM_COLORS[r["item"]]) for _, r in hr.iterrows()],
                   "Fig 1 · Keyword-hit rate per item (fraction with any item-relevant term)", 1.0,
                   note="Low = retrieval often returns windows that never mention the symptom.")
    sc = agg.sort_values("mean_top_score")
    fig_score = hbar([(r["item"], r["mean_top_score"], ITEM_COLORS[r["item"]]) for _, r in sc.iterrows()],
                     "Fig 2 · Mean top-1 hybrid retrieval score per item", 0.6)
    fig_size = hbar([(r["item"], r["mean_nchars"], ITEM_COLORS[r["item"]])
                     for _, r in agg.sort_values("mean_nchars").iterrows()],
                    "Fig 3 · Mean evidence length per item (chars)", float(agg.mean_nchars.max()) * 1.1,
                    fmt="{:.0f}")
    fig_scatter = ""
    if faroff_by_item:
        pts = [(r["keyword_hit_rate"], r["faroff_rate"], r["item"], ITEM_COLORS[r["item"]])
               for _, r in agg.iterrows()]
        fig_scatter = scatter(pts, "Fig 4 · Far-off (≥2) error rate vs keyword-hit rate, per item",
                              "keyword-hit rate", "CoT far-off rate",
                              note=f"Only weakly negative (r={corr_kw_fo:.2f}) and CONFOUNDED by label skew: "
                                   f"Moving has the worst retrieval yet low far-off because ~77% of its labels are 0 (easy).")

    # ---- per-item table ----
    def trow(r):
        fo = "" if np.isnan(r["faroff_rate"]) else f"{r['faroff_rate']:.2f}"
        return (f"<tr><td>{r['item']}</td><td class='sec'>{r['n']}</td>"
                f"<td><b>{r['keyword_hit_rate']:.2f}</b></td><td>{r['mean_hits']:.1f}</td>"
                f"<td>{r['mean_top_score']:.2f}</td><td>{r['mean_nwins']:.1f}</td>"
                f"<td>{r['mean_nchars']:.0f}</td><td class='sec'>{r['frac_label0']:.2f}</td><td>{fo}</td></tr>")
    table = ("<table><thead><tr><th>Item</th><th>n</th><th>kw-hit rate</th><th>mean #kw</th>"
             "<th>top score</th><th>#windows</th><th>chars</th><th>frac y=0</th><th>far-off</th></tr></thead><tbody>"
             + "".join(trow(r) for _, r in agg.sort_values("keyword_hit_rate").iterrows())
             + "</tbody></table>")

    # ---- sampled-examples gallery ----
    rng = np.random.RandomState(a.seed)
    gallery = ""
    for iid in range(1, 9):
        g = df[df.item_id == iid]
        name = g.item_name.iloc[0]; col = ITEM_COLORS[name]
        # sample across labels for variety: prefer one severe (>=2) and the rest random
        picks = []
        sev = g[g.label >= 2]
        if len(sev):
            picks.append(sev.iloc[rng.randint(len(sev))])
        pool = g.drop([p.name for p in picks]) if picks else g
        extra = pool.sample(min(a.samples_per_item - len(picks), len(pool)), random_state=a.seed)
        picks += [r for _, r in extra.iterrows()]
        cards = ""
        for r in picks:
            wins_html = ""
            scores = r["scores"]
            for k, w in enumerate(r["wins"]):
                txt = str(w).replace("[TURN_SEP]", "; ")
                sc_v = f"{scores[k]:.2f}" if k < len(scores) else "–"
                wins_html += (f'<div class="exc"><span class="score">{sc_v}</span> '
                              f'{highlight(txt, r["hits"], col)}</div>')
            badge = "✓ relevant term" if r["any_hit"] else "✗ no relevant term"
            bcol = "#16a34a" if r["any_hit"] else "#dc2626"
            cards += (f'<div class="ex"><div class="exhead">participant {r["participant_id"]} · '
                      f'<b>gold label {r["label"]}</b> · <span style="color:{bcol}">{badge}</span></div>'
                      f'{wins_html}</div>')
        gallery += (f'<div class="card"><h3 style="color:{col};margin-top:0">'
                    f'{name} — kw-hit {agg[agg.item==name].keyword_hit_rate.iloc[0]:.2f}, '
                    f'top score {agg[agg.item==name].mean_top_score.iloc[0]:.2f}</h3>'
                    f'<p class="note" style="margin:0 0 6px">lexicon: '
                    f'{", ".join(lexicons[name])}</p>{cards}</div>')

    worst = agg.sort_values("keyword_hit_rate").iloc[0]
    best = agg.sort_values("keyword_hit_rate").iloc[-1]
    css = """
 body{margin:0;background:#f1f5f9;color:#0f172a;font-family:-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.5}
 header{background:#0f172a;color:#fff;padding:24px 28px}header h1{margin:0;font-size:21px}
 .sub{color:#cbd5e1;font-size:13px;margin-top:4px;max-width:760px}
 .wrap{max-width:940px;margin:20px auto;padding:0 18px}
 .card{background:#fff;border-radius:12px;padding:18px 22px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
 .card h2{margin:0 0 10px;font-size:16px}.card h3{font-size:14px}.note{font-size:12px;color:#64748b}
 table{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px}
 th,td{padding:6px 9px;border-bottom:1px solid #e2e8f0;text-align:right}
 th:first-child,td:first-child{text-align:left}thead th{background:#f8fafc;color:#475569;font-weight:600}.sec{color:#94a3b8}
 .ex{border:1px solid #e2e8f0;border-radius:8px;padding:8px 10px;margin:8px 0;background:#fafafa}
 .exhead{font-size:12px;color:#334155;margin-bottom:4px}
 .exc{font-size:12px;color:#475569;padding:2px 0;border-bottom:1px dashed #eef2f7}
 .score{display:inline-block;min-width:30px;color:#94a3b8;font-variant-numeric:tabular-nums}
 .caveat{background:#fffbeb;border-left:4px solid #f59e0b;padding:8px 14px;border-radius:6px;font-size:13px}
"""
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retrieved-window analysis per PHQ-8 item</title><style>{css}</style></head><body>
<header><h1>Retrieved context windows — per-item analysis</h1>
<div class="sub">Does retrieval surface item-relevant content for every PHQ-8 item? Hybrid retrieval, {Path(a.dataset).name}, n={len(df)} item-examples. Keyword-hit rate is a heuristic topical-relevance proxy.</div></header>
<div class="wrap">
 <div class="card"><h2>Summary</h2>
 <p>Retrieval relevance varies sharply by item. <b>{best['item']}</b> surfaces a relevant term {best['keyword_hit_rate']*100:.0f}% of the time (a distinctive keyword), while <b>{worst['item']}</b> does so only {worst['keyword_hit_rate']*100:.0f}% of the time — for {worst['item']} the model is often reasoning over windows that never name the symptom. <b>Appetite</b> additionally has the weakest retrieval scores (top-1 {agg[agg.item=='Appetite'].mean_top_score.iloc[0]:.2f}).</p>
 <p>The link to errors is <b>real but not clean</b>: the two items that are <i>both</i> thin-evidence and high far-off (Failure, Appetite) are exactly the worst items in the error analysis — but across all items the keyword-hit ↔ far-off correlation is only weak (r={corr_kw_fo:.2f}), because far-off also depends on label skew. {worst['item']} is the clearest confound: worst retrieval yet low far-off, because {agg[agg.item==worst['item']].frac_label0.iloc[0]*100:.0f}% of its labels are 0, so "predict 0" is usually right regardless of evidence (see the <i>frac y=0</i> column).</p>
 {table}
 <p class="note" style="margin-top:6px">Keyword lexicons are heuristic substring proxies (a window can be relevant without the exact term), so absolute rates are approximate — the <i>ranking</i> across items is the signal.</p>
 </div>
 <div class="card"><h2>Plots</h2>{fig_hit}{fig_score}{fig_size}{fig_scatter}</div>
 <div class="card"><h2>Sampled examples per item</h2>
 <p class="note">For each item: up to {a.samples_per_item} sampled examples (one severe where available). Retrieved windows shown with their hybrid score; item-relevant terms highlighted.</p></div>
 {gallery}
 <div class="card"><div class="caveat"><b>Takeaway.</b> For several items (notably Failure and Appetite) retrieval frequently returns windows that never mention the symptom — part of the model's difficulty is <b>upstream of reasoning</b>: the evidence (and often the interview itself) lacks the signal. The relationship to far-off errors is confounded by label base rates, so this is supporting evidence, not proof, for the evidence/label ceiling. The one upstream lever still untried is <b>item-aware retrieval</b> (querying with the symptom's own vocabulary) — most promising for the low-hit-rate items.</div></div>
</div></body></html>"""
    out_html.write_text(html)
    print(f"Saved: {out_html}")
    print(agg[["item", "n", "keyword_hit_rate", "mean_top_score", "mean_nchars", "faroff_rate"]]
          .sort_values("keyword_hit_rate").to_string(index=False))


if __name__ == "__main__":
    main()
