"""Fig 5 - Inside the reasoning: a curated CoT showcase.

Selects illustrative Chain-of-Thought rationales from the Qwen-7B probe and
renders them as slide-ready cards (PHQ-8 question -> CoT reasoning -> predicted
vs true, with the encoder's call alongside). Prioritises (a) severe-class cases
the CoT rescues that the encoder misses, (b) confident correct calls, and
(c) one or two instructive failures. Deterministic selection.

Outputs an HTML deck (outputs/figures/fig5_reasoning_showcase.html) plus a CSV
of the chosen exemplars.

    python -m src.evaluation.presentation.fig5_reasoning_showcase
"""

import glob
import html

import pandas as pd

from src.evaluation.presentation import _style as S

COT_DIR = S.ROOT / "outputs" / "cot" / "folds"
COT_GLOB = "cot_probe_qwen_hybw3_fold*.csv"
ENC_OOF = S.ROOT / "outputs" / "cv" / "oof_predictions_ctxm_corn_hybw3.csv"
KEY = ["participant_id", "item_id"]

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


def load():
    files = sorted(glob.glob(str(COT_DIR / COT_GLOB)))
    cot = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    cot["participant_id"] = cot["participant_id"].astype(str)
    enc = pd.read_csv(ENC_OOF)
    enc["participant_id"] = enc["participant_id"].astype(str)
    enc = enc[KEY + ["prediction"]].rename(columns={"prediction": "enc_pred"})
    df = cot.merge(enc, on=KEY, how="left")
    df["cot_correct"] = df["prediction"] == df["label"]
    df["enc_correct"] = df["enc_pred"] == df["label"]
    df["dist"] = (df["prediction"] - df["label"]).abs()
    df["conf"] = df[[f"prob_{c}" for c in S.LABELS]].max(axis=1)
    df["reasoning"] = df["reasoning"].fillna("").astype(str)
    df = df[df["reasoning"].str.len() > 20]
    return df


def select(df):
    """Deterministic exemplar selection -> list of (kind, row)."""
    picks, used, used_items = [], set(), set()

    def take(sub, kind, n, diverse=True):
        # two passes: first prefer unseen PHQ-8 items, then fill if short
        for require_new in ([True, False] if diverse else [False]):
            for _, r in sub.iterrows():
                if n == 0:
                    return
                k = (r["participant_id"], r["item_id"])
                if k in used:
                    continue
                if require_new and r["item_name"] in used_items:
                    continue
                used.add(k)
                used_items.add(r["item_name"])
                picks.append((kind, r))
                n -= 1

    # (a) severe rescue: CoT right, encoder wrong, true class severe/moderate
    rescue = df[(df.cot_correct) & (~df.enc_correct) & (df.label >= 2)] \
        .sort_values(["label", "conf"], ascending=[False, False])
    take(rescue, "CoT rescues a severe case the encoder missed", 2)

    # (b) confident correct (mild/moderate), prefer CoT-beats-encoder
    conf = df[(df.cot_correct) & (df.label.isin([1, 2]))].copy()
    conf["beats"] = (~conf.enc_correct).astype(int)
    conf = conf.sort_values(["beats", "conf"], ascending=[False, False])
    take(conf, "Confident, correct reading of the transcript", 2)

    # (c) instructive failures: far-off miss (one over-call, one under-call)
    over = df[(~df.cot_correct) & (df["prediction"] > df["label"]) & (df.dist >= 2)] \
        .sort_values("dist", ascending=False)
    take(over, "Instructive miss — CoT over-calls on thin evidence", 1)
    under = df[(~df.cot_correct) & (df["prediction"] < df["label"]) & (df.dist >= 2)] \
        .sort_values("dist", ascending=False)
    take(under, "Instructive miss — CoT under-calls a severe case", 1)
    return picks


def prob_bar(r):
    segs = []
    colors = ["#cbd5e1", "#93c5fd", "#fbbf24", "#ef4444"]
    for c in S.LABELS:
        p = float(r[f"prob_{c}"])
        if p < 0.005:
            continue
        segs.append(f'<div style="width:{p*100:.1f}%;background:{colors[c]};" '
                    f'title="{SEVERITY[c]}: {p:.2f}">{S.CLASS_SHORT[c][:3]} {p:.0%}</div>')
    return ('<div style="display:flex;height:20px;border-radius:5px;overflow:hidden;'
            'font-size:10px;color:#1e293b;line-height:20px;text-align:center">'
            + "".join(segs) + "</div>")


def card(kind, r):
    correct = bool(r["cot_correct"])
    badge_col = S.GOOD if correct else S.BAD
    verdict = "correct" if correct else "miss"
    iid = int(r["item_id"])
    enc = int(r["enc_pred"]) if pd.notna(r["enc_pred"]) else None
    enc_note = ""
    if enc is not None:
        enc_ok = enc == int(r["label"])
        enc_note = (f'<span style="color:{S.ENC_COLOR if enc_ok else S.MUTED}">'
                    f'Encoder said <b>{SEVERITY[enc]}</b> '
                    f'({"correct" if enc_ok else "wrong"})</span>')
    reasoning = html.escape(r["reasoning"]).strip().strip('"')
    return f"""
 <div class="card">
   <div class="kind" style="border-color:{badge_col};color:{badge_col}">{html.escape(kind)}</div>
   <div class="q"><b>PHQ-8 · {r['item_name']}</b> — “{QUESTION[iid]}”</div>
   <div class="reason">🧠 {reasoning}</div>
   <div class="row">
     <div>True: <b>{SEVERITY[int(r['label'])]}</b></div>
     <div>CoT: <b style="color:{badge_col}">{SEVERITY[int(r['prediction'])]}</b> ({verdict})</div>
     <div>{enc_note}</div>
   </div>
   <div class="pb">{prob_bar(r)}</div>
 </div>"""


def main():
    df = load()
    picks = select(df)

    cards = "".join(card(k, r) for k, r in picks)
    n = len(picks)
    n_rescue = sum(1 for k, _ in picks if k.startswith("CoT rescues"))
    html_doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Inside the model's reasoning — PHQ-8 CoT</title>
<style>
 body{{margin:0;background:#f1f5f9;color:#0f172a;font-family:-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.5}}
 header{{background:#0f172a;color:#fff;padding:22px 28px}} header h1{{margin:0;font-size:21px}}
 .sub{{color:#cbd5e1;font-size:13px;margin-top:4px}}
 .wrap{{max-width:880px;margin:20px auto;padding:0 18px}}
 .card{{background:#fff;border-radius:12px;padding:16px 20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 .kind{{display:inline-block;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.03em;
        border:1.5px solid;border-radius:20px;padding:2px 10px;margin-bottom:10px}}
 .q{{font-size:14px;margin-bottom:8px;color:#334155}}
 .reason{{background:#f8fafc;border-left:3px solid #7c3aed;padding:10px 14px;border-radius:6px;
          font-size:13.5px;color:#1e293b;margin-bottom:10px}}
 .row{{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;color:#475569;margin-bottom:8px}}
 .pb{{margin-top:4px}}
</style></head><body>
<header><h1>Inside the model's reasoning — PHQ-8 item severity</h1>
<div class="sub">Curated Chain-of-Thought rationales (Qwen-7B, no task training, hybrid-W3 evidence).
{n} exemplars · {n_rescue} severe-class rescues the encoder missed. 🧠 = the model's own words.</div></header>
<div class="wrap">{cards}
 <p style="font-size:12px;color:#64748b">Reasoning text is verbatim model output; quotes inside it are
 snippets the retriever surfaced from the interview transcript. Encoder = MentalBERT+CORN on the same OOF fold.</p>
</div></body></html>"""

    out = S.FIG_DIR / "fig5_reasoning_showcase.html"
    S.FIG_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc)
    print(f"  wrote {out}")

    cols = KEY + ["item_name", "label", "prediction", "enc_pred", "conf", "reasoning"]
    sel = pd.DataFrame([{"kind": k, **{c: r[c] for c in cols}} for k, r in picks])
    cp = S.companion_path("fig5_reasoning_showcase", "csv")
    sel.to_csv(cp, index=False)
    print(f"  wrote {cp}")
    for k, r in picks:
        print(f"  [{k[:38]:38s}] {r['item_name']:13s} true={int(r['label'])} "
              f"cot={int(r['prediction'])} enc={int(r['enc_pred']) if pd.notna(r['enc_pred']) else '-'}")


if __name__ == "__main__":
    main()
