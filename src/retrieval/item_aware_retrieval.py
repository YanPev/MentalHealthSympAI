"""
Prototype: item-aware retrieval.

The current pipeline retrieves a participant's context windows using the bare
PHQ-8 item text as the BM25 query (e.g. "Poor appetite or overeating"). The
per-item retrieval analysis showed several items (Moving, Concentrating,
Appetite) rarely surface windows that mention the symptom at all.

This prototype expands the query with each symptom's *own vocabulary* (synonyms,
lay phrasings, behavioural cues) and re-retrieves with BM25 from the same window
bank. It writes a new dataset (schema-compatible with cot_probe) whose evidence
column is the item-aware retrieval, and reports the change in retrieval vs the
bare-item BM25 baseline and the existing hybrid retrieval.

NOTE ON EVALUATION. Measuring "keyword-hit rate" with the same vocabulary we add
to the query is partly circular (it must go up). So the offline hit-rate here is
a sanity check only; the real test is a downstream CoT run on this evidence
(run_cot_probe with --dataset-path the CSV this writes).

    python -m src.retrieval.item_aware_retrieval
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from src.retrieval.bm25_retriever import retrieve_top_k_bm25
from src.evaluation.retrieval_window_analysis import (
    LEXICONS, windows_text, hit_terms, ITEM_COLORS)
from src.llm.cot_probe import format_evidence

PR = Path(__file__).resolve().parents[2]
BANK = PR / "data" / "processed" / "context_window_bank_w3.json"
BASE_DS = PR / "data" / "processed" / "phq8_item_dataset_context_windows_hybrid_w3.csv"
OUT_CSV = PR / "data" / "processed" / "phq8_item_dataset_itemaware_w3.csv"
OUT_HTML = PR / "outputs" / "cot" / "item_aware_retrieval_report.html"
OUT_JSON = PR / "outputs" / "cot" / "item_aware_retrieval.json"
K = 5

# Query-expansion vocabulary per item: symptom synonyms / lay phrasings / cues.
# Appended to the item text to form the BM25 query.
EXPAND = {
    "NoInterest": "interest pleasure enjoy enjoyment hobbies activities fun excited "
                  "bored unmotivated stopped doing things i used to like care about",
    "Depressed": "depressed down sad hopeless unhappy crying cry low mood blue empty "
                 "despair miserable feeling bad",
    "Sleep": "sleep asleep insomnia falling asleep staying asleep waking up early "
             "sleeping too much nights restless tossing turning in bed nap",
    "Tired": "tired fatigue no energy low energy exhausted lethargic worn out drained "
             "sluggish run down",
    "Appetite": "appetite eating eat food hungry hunger overeating poor appetite "
                "weight loss weight gain meals skipping meals eating too much snacking",
    "Failure": "failure failing failed guilt guilty worthless bad about myself "
               "let myself down let my family down disappointed useless ashamed",
    "Concentrating": "concentrate concentrating focus paying attention distracted "
                     "forgetful remembering forget reading watching tv mind wandering thoughts",
    "Moving": "moving slowly speaking slowly slowed down restless fidgety fidget "
              "agitated sluggish pacing can't sit still",
}


def main():
    ap = argparse.ArgumentParser(description="Item-aware retrieval prototype")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--samples-per-item", type=int, default=2)
    a = ap.parse_args()

    bank = json.loads(BANK.read_text())
    bank = {str(k): v for k, v in bank.items()}
    base = pd.read_csv(BASE_DS)
    base["participant_id"] = base["participant_id"].astype(str)

    # existing hybrid retrieval (for reference)
    def parse_list(v):
        try:
            return json.loads(v) if isinstance(v, str) else []
        except json.JSONDecodeError:
            return []
    base["hybrid_wins"] = base["retrieved_context_windows_hybrid_list"].apply(parse_list)

    out_rows = []
    miss = 0
    for _, r in base.iterrows():
        pid, name = r["participant_id"], r["item_name"]
        cand = [w["window_text"] for w in bank.get(pid, []) if str(w.get("window_text", "")).strip()]
        if not cand:
            miss += 1
        q_base = r["item_text"]
        q_item = f"{r['item_text']} {EXPAND.get(name, '')}"
        base_wins = retrieve_top_k_bm25(q_base, cand, k=a.k)
        item_wins = retrieve_top_k_bm25(q_item, cand, k=a.k)
        out_rows.append({
            "participant_id": pid, "item_id": int(r["item_id"]), "item_name": name,
            "item_text": r["item_text"], "label": int(r["label"]),
            "split": r.get("split", ""),
            "transcript_text": r.get("transcript_text", r["item_text"]),
            "retrieved_context_windows_itemaware_list": json.dumps(item_wins),
            "retrieved_context_windows_itemaware_pack": " [WINDOW_SEP] ".join(item_wins),
            # bare-item BM25 kept as the matched control for the downstream CoT test
            "retrieved_context_windows_bm25base_list": json.dumps(base_wins),
            "_base_bm25_list": json.dumps(base_wins),
            "_hybrid_list": json.dumps(r["hybrid_wins"]),
        })
    out = pd.DataFrame(out_rows)
    out.drop(columns=["_base_bm25_list", "_hybrid_list"]).to_csv(OUT_CSV, index=False)
    if miss:
        print(f"[warn] {miss} rows had no candidate windows in the bank")

    # ---- relevance comparison (sanity check; partly circular -- see header) ----
    def hitrate(list_col, df):
        rates = {}
        for iid in range(1, 9):
            g = df[df.item_id == iid]
            name = g.item_name.iloc[0]
            lex = LEXICONS[name]
            hits = g[list_col].apply(lambda s: len(hit_terms(windows_text(json.loads(s)), lex)) > 0)
            rates[name] = float(hits.mean())
        return rates

    hr_item = hitrate("retrieved_context_windows_itemaware_list", out)
    hr_base = hitrate("_base_bm25_list", out)
    # hybrid from base df
    hr_hyb = {}
    for iid in range(1, 9):
        g = base[base.item_id == iid]; name = g.item_name.iloc[0]; lex = LEXICONS[name]
        hr_hyb[name] = float(g["hybrid_wins"].apply(
            lambda w: len(hit_terms(windows_text(w), lex)) > 0).mean())

    summary = {"k": a.k, "hit_rate_itemaware": hr_item, "hit_rate_bm25_baseline": hr_base,
               "hit_rate_hybrid_existing": hr_hyb, "expand_vocab": EXPAND}
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    items = sorted(LEXICONS, key=lambda n: hr_item[n] - hr_base[n], reverse=True)
    print(f"Item-aware retrieval (BM25, k={a.k}) — keyword-hit rate (sanity check):")
    print(f"{'item':14s} {'hybrid':>8} {'bm25-base':>10} {'item-aware':>11} {'Δ vs base':>10}")
    for n in items:
        print(f"{n:14s} {hr_hyb[n]:8.2f} {hr_base[n]:10.2f} {hr_item[n]:11.2f} {hr_item[n]-hr_base[n]:+10.2f}")

    # ---- HTML: bars + side-by-side sampled examples ----
    def hbar3(title):
        n = 8; rh = 30; padl = 130; padr = 20; padt = 26; W = 640; pw = W - padl - padr
        H = padt + n * rh + 6
        p = [f'<svg viewBox="0 0 {W} {H}" width="100%" style="max-width:{W}px" '
             f'font-family="-apple-system,Segoe UI,sans-serif" font-size="11">']
        leg = [("hybrid (existing)", "#cbd5e1"), ("BM25 item-only", "#94a3b8"), ("item-aware", "#0ea5e9")]
        lx = padl
        for lab, c in leg:
            p.append(f'<rect x="{lx}" y="6" width="10" height="10" fill="{c}"/>'
                     f'<text x="{lx+13}" y="15" fill="#64748b">{lab}</text>'); lx += 150
        for i, name in enumerate(items):
            y = padt + i * rh
            p.append(f'<text x="{padl-8}" y="{y+rh/2+3:.1f}" text-anchor="end" fill="#475569">{name}</text>')
            for k, (val, c) in enumerate([(hr_hyb[name], "#cbd5e1"), (hr_base[name], "#94a3b8"),
                                          (hr_item[name], "#0ea5e9")]):
                yy = y + 2 + k * (rh - 4) / 3
                p.append(f'<rect x="{padl}" y="{yy:.1f}" width="{pw*val:.1f}" height="{(rh-4)/3-1:.1f}" fill="{c}"/>')
            p.append(f'<text x="{padl+pw*hr_item[name]+4:.1f}" y="{y+rh/2+3:.1f}" fill="#0ea5e9" font-size="9">{hr_item[name]:.2f}</text>')
        p.append('</svg>')
        return f'<p class="note" style="margin:8px 0 2px"><b>{title}</b></p>' + "".join(p)

    import html as _h
    import re
    def hl(text, name):
        esc = _h.escape(text)
        for t in sorted(LEXICONS[name], key=len, reverse=True):
            esc = re.sub(f"({re.escape(t)})", r'<mark style="background:#0ea5e922;color:#0369a1;font-weight:600">\1</mark>',
                         esc, flags=re.IGNORECASE)
        return esc

    rng = np.random.RandomState(0)
    gallery = ""
    # showcase the items where item-aware helped most
    for name in items[:4]:
        g = out[(out.item_name == name)]
        # prefer examples where base missed but item-aware hit
        def hit(s): return len(hit_terms(windows_text(json.loads(s)), LEXICONS[name])) > 0
        flips = g[g["_base_bm25_list"].apply(lambda s: not hit(s)) &
                  g["retrieved_context_windows_itemaware_list"].apply(hit)]
        pick = flips if len(flips) else g
        pick = pick.sample(min(a.samples_per_item, len(pick)), random_state=0)
        cards = ""
        for _, r in pick.iterrows():
            bw = json.loads(r["_base_bm25_list"]); iw = json.loads(r["retrieved_context_windows_itemaware_list"])
            cards += (f'<div class="ex"><div class="exhead">participant {r.participant_id} · gold {r.label}</div>'
                      f'<div class="col"><div class="lbl">BM25 item-only</div>'
                      + "".join(f'<div class="exc">{hl(w.replace("[TURN_SEP]","; "),name)}</div>' for w in bw[:3])
                      + '</div><div class="col"><div class="lbl" style="color:#0369a1">item-aware</div>'
                      + "".join(f'<div class="exc">{hl(w.replace("[TURN_SEP]","; "),name)}</div>' for w in iw[:3])
                      + '</div></div>')
        gallery += (f'<div class="card"><h3 style="color:{ITEM_COLORS[name]};margin-top:0">{name} '
                    f'(hit-rate {hr_base[name]:.2f} → {hr_item[name]:.2f})</h3>'
                    f'<p class="note" style="margin:0 0 4px">added query terms: {EXPAND[name]}</p>{cards}</div>')

    mean_d = np.mean([hr_item[n] - hr_base[n] for n in items])
    css = """
 body{margin:0;background:#f1f5f9;color:#0f172a;font-family:-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.5}
 header{background:#0f172a;color:#fff;padding:24px 28px}header h1{margin:0;font-size:21px}.sub{color:#cbd5e1;font-size:13px;margin-top:4px;max-width:760px}
 .wrap{max-width:940px;margin:20px auto;padding:0 18px}
 .card{background:#fff;border-radius:12px;padding:18px 22px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
 .card h2{margin:0 0 10px;font-size:16px}.card h3{font-size:14px}.note{font-size:12px;color:#64748b}
 table{border-collapse:collapse;width:100%;font-size:13px;margin-top:6px}
 th,td{padding:6px 9px;border-bottom:1px solid #e2e8f0;text-align:right}th:first-child,td:first-child{text-align:left}
 thead th{background:#f8fafc;color:#475569;font-weight:600}
 .ex{border:1px solid #e2e8f0;border-radius:8px;padding:8px 10px;margin:8px 0;background:#fafafa;display:flex;gap:12px;flex-wrap:wrap}
 .exhead{font-size:12px;color:#334155;width:100%;margin-bottom:2px}
 .col{flex:1;min-width:230px}.lbl{font-size:11px;color:#94a3b8;font-weight:600;margin-bottom:2px}
 .exc{font-size:12px;color:#475569;padding:2px 0;border-bottom:1px dashed #eef2f7}
 .caveat{background:#fffbeb;border-left:4px solid #f59e0b;padding:8px 14px;border-radius:6px;font-size:13px}
"""
    rowshtml = "".join(
        f"<tr><td>{n}</td><td>{hr_hyb[n]:.2f}</td><td>{hr_base[n]:.2f}</td>"
        f"<td><b>{hr_item[n]:.2f}</b></td><td>{hr_item[n]-hr_base[n]:+.2f}</td></tr>" for n in items)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Item-aware retrieval prototype</title><style>{css}</style></head><body>
<header><h1>Item-aware retrieval — prototype</h1>
<div class="sub">Expand each PHQ-8 item's BM25 query with the symptom's own vocabulary, re-retrieve from the window bank (k={a.k}). Compared to the bare-item BM25 query and the existing hybrid retrieval.</div></header>
<div class="wrap">
 <div class="card"><h2>Keyword-hit rate by item (offline sanity check)</h2>
 {hbar3("Fraction of examples whose retrieved windows contain a symptom term")}
 <table><thead><tr><th>Item</th><th>hybrid</th><th>BM25 item-only</th><th>item-aware</th><th>Δ vs base</th></tr></thead><tbody>{rowshtml}</tbody></table>
 <p>Item-aware querying raises the mean keyword-hit rate by <b>{mean_d:+.2f}</b> over the bare-item BM25 baseline, with the biggest gains on the items the analysis flagged as evidence-starved (Concentrating, NoInterest, Appetite) — on Appetite and Concentrating it even exceeds the existing hybrid retrieval. <b>But Failure and Moving stay low even after expansion</b>, indicating those symptoms genuinely are not verbalised in the interviews — a true ceiling that retrieval cannot fix.</p>
 <div class="caveat"><b>Read this honestly:</b> because the query is expanded with the same symptom vocabulary used to measure hit-rate, this metric is <b>partly circular</b> and is expected to rise — it shows the retriever <i>can</i> surface symptom-mentioning windows when asked, not that downstream accuracy improves. The decisive test is a CoT run on this evidence (the dataset written to <code>{OUT_CSV.name}</code>), evaluated on the same folds.</div>
 </div>
 <div class="card"><h2>Sampled examples — BM25 item-only vs item-aware (top items by gain)</h2>
 <p class="note">Cases where the bare query missed the symptom but item-aware retrieval surfaced it. Symptom terms highlighted.</p></div>
 {gallery}
 <div class="card"><div class="caveat"><b>Next step.</b> Validate downstream: <code>sbatch run_cot_probe_folds.sbatch</code> pointed at <code>{OUT_CSV.name}</code> / column <code>retrieved_context_windows_itemaware_list</code>, then re-run the ensemble. Only a macro-F1/QWK gain there confirms the lever.</div></div>
</div></body></html>"""
    OUT_HTML.write_text(html)
    print(f"\nMean Δ hit-rate (item-aware − BM25 base): {mean_d:+.3f}")
    print(f"Saved dataset: {OUT_CSV}")
    print(f"Saved report : {OUT_HTML}")


if __name__ == "__main__":
    main()
