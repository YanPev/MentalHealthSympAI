"""Fig 11 - "The model points at its evidence": MIL attention attribution.

The attention-MIL head (Job 3) assigns a weight to each retrieved window, so we
can show *which* utterance drove each PHQ-8 item score. This renders a handful of
correct, confidently-attended examples as cards: the PHQ-8 question, the candidate
windows as attention bars, and the predicted vs true severity. Pairs with the CoT
reasoning cards (fig5) — one model explains in words, the other points at evidence.

Joins outputs/cv/mil_attention_mil_hybw3.csv (attn_0..4) + the MIL OOF predictions
with the window texts in the hybrid-W3 dataset (_list column).

    python -m src.evaluation.presentation.fig11_mil_attribution
"""

import html
import json

import numpy as np
import pandas as pd

from src.evaluation.presentation import _style as S

ATTN = S.ROOT / "outputs" / "cv" / "mil_attention_mil_hybw3.csv"
OOF = S.ROOT / "outputs" / "cv" / "oof_predictions_mil_hybw3.csv"
DATA = S.ROOT / "data" / "processed" / "phq8_item_dataset_context_windows_hybrid_w3.csv"
LIST_COL = "retrieved_context_windows_hybrid_list"
SEVERITY = ["Not at all", "Several days", "More than half the days", "Nearly every day"]
QUESTION = {
    1: "Little interest or pleasure in doing things",
    2: "Feeling down, depressed, or hopeless",
    3: "Trouble falling or staying asleep, or sleeping too much",
    4: "Feeling tired or having little energy",
    5: "Poor appetite or overeating",
    6: "Feeling bad about yourself — or that you are a failure",
    7: "Trouble concentrating on things",
    8: "Moving or speaking slowly, or being fidgety / restless",
}


def main():
    a = pd.read_csv(ATTN); a["participant_id"] = a["participant_id"].astype(str)
    o = pd.read_csv(OOF)[["participant_id", "item_id", "label", "prediction"]]
    o["participant_id"] = o["participant_id"].astype(str)
    d = pd.read_csv(DATA, usecols=["participant_id", "item_id", LIST_COL])
    d["participant_id"] = d["participant_id"].astype(str)
    m = a.merge(o, on=["participant_id", "item_id"]).merge(d, on=["participant_id", "item_id"])
    attn_cols = [f"attn_{i}" for i in range(5)]
    m["amax"] = m[attn_cols].to_numpy().max(1)
    m["correct"] = m["label"] == m["prediction"]

    # ---- deterministic selection: correct, clearly peaked, diverse items,
    #      prefer non-trivial severities and include a moderate/severe case ----
    cand = m[(m["correct"]) & (m["amax"] >= 0.55) & (m["amax"] <= 0.97)].copy()
    cand = cand.sort_values(["label", "amax"], ascending=[False, False])
    picks, used_items = [], set()
    for require_new in (True, False):
        for _, r in cand.iterrows():
            if len(picks) >= 6:
                break
            if require_new and r["item_name"] in used_items:
                continue
            windows = json.loads(r[LIST_COL]) if isinstance(r[LIST_COL], str) else []
            if not windows:
                continue
            used_items.add(r["item_name"])
            picks.append((r, windows))

    cards = "".join(card(r, windows, attn_cols) for r, windows in picks)
    n = len(picks)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MIL attention attribution — PHQ-8</title>
<style>
 body{{margin:0;background:#f1f5f9;color:#0f172a;font-family:-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.5}}
 header{{background:#0f172a;color:#fff;padding:22px 28px}} header h1{{margin:0;font-size:21px}}
 .sub{{color:#cbd5e1;font-size:13px;margin-top:4px}}
 .wrap{{max-width:900px;margin:20px auto;padding:0 18px}}
 .card{{background:#fff;border-radius:12px;padding:16px 20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 .q{{font-size:14px;margin-bottom:10px;color:#334155}}
 .win{{display:flex;align-items:center;gap:10px;margin:5px 0;font-size:12.5px}}
 .barwrap{{flex:0 0 130px;background:#eef2f7;border-radius:5px;height:18px;overflow:hidden}}
 .bar{{height:18px;border-radius:5px}}
 .pct{{flex:0 0 38px;text-align:right;color:#475569;font-variant-numeric:tabular-nums}}
 .txt{{flex:1;color:#334155;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
 .top .txt{{font-weight:700;color:#0f172a}}
 .row{{display:flex;gap:20px;font-size:13px;color:#475569;margin-top:10px}}
</style></head><body>
<header><h1>The model points at its evidence — attention-MIL attribution</h1>
<div class="sub">Gated-attention weights over the retrieved context windows (MentalBERT, hybrid-W3).
{n} correct examples; the highlighted window is the one the model weighted most. Pairs with the CoT reasoning cards.</div></header>
<div class="wrap">{cards}
 <p style="font-size:12px;color:#64748b">Bar length = attention weight (sums to 1 across a participant's windows).
 Window text is the retrieved transcript span ([TURN_SEP] separates turns).</p>
</div></body></html>"""
    out = S.FIG_DIR / "fig11_mil_attribution.html"
    S.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    print(f"  wrote {out}")

    rows = []
    for r, windows in picks:
        w = [r[c] for c in attn_cols][:len(windows)]
        top = int(np.argmax(w))
        rows.append({"participant_id": r["participant_id"], "item_name": r["item_name"],
                     "label": int(r["label"]), "prediction": int(r["prediction"]),
                     "top_attn": round(float(max(w)), 3),
                     "top_window": windows[top][:200]})
    cp = S.companion_path("fig11_mil_attribution", "csv")
    pd.DataFrame(rows).to_csv(cp, index=False)
    print(f"  wrote {cp}")
    for x in rows:
        print(f"  {x['item_name']:13s} true={x['label']} pred={x['prediction']} "
              f"attn={x['top_attn']:.2f}")


def card(r, windows, attn_cols):
    weights = [float(r[c]) for c in attn_cols][:len(windows)]
    top = int(np.argmax(weights))
    iid = int(r["item_id"])
    correct_col = S.GOOD if r["label"] == r["prediction"] else S.BAD
    rows = []
    for i, (w, txt) in enumerate(zip(weights, windows)):
        is_top = i == top
        col = S.COT_COLOR if is_top else "#cbd5e1"
        cls = " top" if is_top else ""
        rows.append(
            f'<div class="win{cls}"><div class="barwrap">'
            f'<div class="bar" style="width:{max(w*100,2):.0f}%;background:{col}"></div></div>'
            f'<div class="pct">{w*100:.0f}%</div>'
            f'<div class="txt">{html.escape(txt)}</div></div>')
    return f"""
 <div class="card">
   <div class="q"><b>PHQ-8 · {r['item_name']}</b> — “{QUESTION[iid]}”</div>
   {''.join(rows)}
   <div class="row"><div>True: <b>{SEVERITY[int(r['label'])]}</b></div>
     <div>MIL: <b style="color:{correct_col}">{SEVERITY[int(r['prediction'])]}</b></div>
     <div>top window weight: <b>{max(weights)*100:.0f}%</b></div></div>
 </div>"""


if __name__ == "__main__":
    main()
