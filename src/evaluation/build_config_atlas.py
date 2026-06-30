"""Build the Configuration Atlas HTML from outputs/config_atlas_data.json.

One box per configuration, grouped by family, each showing the ordered STEPS,
exact methodology, settings, a results table and the verdict. Pure stdlib.

    python -m src.evaluation.build_config_atlas
    -> outputs/config_atlas.html
"""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs" / "config_atlas_data.json"
OUT = ROOT / "outputs" / "config_atlas.html"

# family display metadata -------------------------------------------------- #
FAMILY_META = {
    "encoder": ("1", "Encoder backbone, loss &amp; evidence",
                "Fine-tuned transformer classifiers (BERT-base / MentalBERT) over retrieved "
                "evidence — backbone, retrieval, class-weighting, CORN ordinal loss, multi-seed "
                "&amp; CV corrections, and the production context-window encoder."),
    "cot_peritem": ("2", "Per-item chain-of-thought (Qwen-7B, no training)",
                    "Each PHQ-8 item scored independently by a frozen 7B reasoner, plus the full "
                    "11-step investigation: self-consistency, distillation, ensembling, error "
                    "analysis, evidence &amp; retrieval levers."),
    "cot_whole": ("3", "Whole-picture CoT prompting variants",
                  "Score all 8 items in one pass to use symptom correlations — joint, staged, "
                  "severe-nudge, tolerant, and self-consistency variants."),
    "bucketb": ("4", "Bucket-B advanced experiments",
                "Long-tail losses, attention-MIL and cross-encoder reranking on the MentalBERT / "
                "Hybrid-W3 base — no single method dominates."),
    "fusion": ("5", "Confidence-gated cascade &amp; statistical validation",
               "Fusing the SC-tolerant LLM with the encoder via top-2 tiebreak gates, plus this "
               "session's cluster-bootstrap CIs, near-miss breakdown and PHQ-8 screening curves."),
    "figures": ("6", "Presentation insight figures",
                "The deck's insight figures — each isolates one finding (retrieval lever, ordinal "
                "near-miss, calibrated fusion, clinical screener, selective prediction, CIs)."),
}

# status chip metadata: (label, css-class) --------------------------------- #
STATUS = {
    "best":       ("BEST",       "s-best"),
    "robust":     ("ROBUST",     "s-robust"),
    "lever":      ("LEVER",      "s-lever"),
    "baseline":   ("BASELINE",   "s-base"),
    "seed-luck":  ("SEED-LUCK",  "s-seed"),
    "failed":     ("DID NOT WORK","s-failed"),
    "superseded": ("SUPERSEDED", "s-super"),
    "tradeoff":   ("TRADE-OFF",  "s-trade"),
}

CSS = """
:root{
 --bg:#0e1015;--panel:#161922;--panel2:#1c2029;--ink:#e7ebf2;--muted:#9aa4b2;
 --line:#2a2f3a;--accent:#5eb1ff;--good:#54d18c;--warn:#ffce6b;--bad:#ff7a85;
 --lever:#46d6d6;--purple:#b79bff;--chip:#222836;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
 font:14.5px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:36px 22px 90px}
header{border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:10px}
h1{font-size:30px;margin:0 0 6px;letter-spacing:-.3px}
.sub{color:var(--muted);max-width:90ch}
.meta{color:var(--muted);font-size:12.5px;margin-top:10px}
.stats{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0 4px}
.stat{background:var(--panel2);border:1px solid var(--line);border-radius:10px;padding:8px 14px;text-align:center}
.stat b{font-size:21px;display:block;letter-spacing:-.5px}
.stat span{color:var(--muted);font-size:11.5px}
/* legend + filters */
.bar{position:sticky;top:0;z-index:5;background:rgba(14,16,21,.93);backdrop-filter:blur(6px);
 border-bottom:1px solid var(--line);padding:10px 0;margin:14px 0 8px}
.barwrap{max-width:1200px;margin:0 auto;padding:0 22px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.flab{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.5px;margin-right:4px}
.fbtn{background:var(--chip);border:1px solid var(--line);color:var(--ink);border-radius:999px;
 padding:4px 11px;font-size:12.5px;cursor:pointer;user-select:none}
.fbtn:hover{border-color:var(--accent)}
.fbtn.off{opacity:.32}
.sep{width:1px;height:18px;background:var(--line);margin:0 6px}
/* family section */
.fam{margin:36px 0 8px}
.fam h2{font-size:22px;margin:0 0 4px;letter-spacing:-.2px;display:flex;align-items:baseline;gap:10px}
.fam .num{display:inline-flex;width:30px;height:30px;align-items:center;justify-content:center;
 background:var(--accent);color:#06121f;border-radius:8px;font-weight:800;font-size:15px}
.fam .fdesc{color:var(--muted);max-width:92ch;margin:2px 0 14px}
.fcount{color:var(--muted);font-size:12.5px;font-weight:400}
/* config grid */
.cards{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
@media(max-width:900px){.cards{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:0;overflow:hidden;
 border-left:4px solid var(--line)}
.card.s-best{border-left-color:var(--good)} .card.s-robust{border-left-color:var(--accent)}
.card.s-lever{border-left-color:var(--lever)} .card.s-base{border-left-color:#5b6472}
.card.s-seed{border-left-color:var(--warn)} .card.s-failed{border-left-color:var(--bad)}
.card.s-super{border-left-color:var(--purple)} .card.s-trade{border-left-color:#e0a96b}
.chead{padding:14px 16px 10px;border-bottom:1px solid var(--line);background:linear-gradient(180deg,var(--panel2),var(--panel))}
.nick{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:15.5px;font-weight:700;color:#eaf1fb;
 word-break:break-word}
.fname{color:#c7d0de;font-size:12.5px;margin-top:3px}
.summ{color:var(--muted);font-size:12.5px;margin-top:7px;font-style:italic}
.chip{float:right;margin-left:8px;font-size:10px;font-weight:800;letter-spacing:.5px;border-radius:6px;
 padding:3px 8px;white-space:nowrap}
.s-best .chip,.chip.s-best{background:rgba(84,209,140,.16);color:var(--good);border:1px solid rgba(84,209,140,.4)}
.chip.s-robust{background:rgba(94,177,255,.16);color:var(--accent);border:1px solid rgba(94,177,255,.4)}
.chip.s-lever{background:rgba(70,214,214,.16);color:var(--lever);border:1px solid rgba(70,214,214,.4)}
.chip.s-base{background:rgba(154,164,178,.14);color:#aeb7c4;border:1px solid rgba(154,164,178,.35)}
.chip.s-seed{background:rgba(255,206,107,.16);color:var(--warn);border:1px solid rgba(255,206,107,.4)}
.chip.s-failed{background:rgba(255,122,133,.16);color:var(--bad);border:1px solid rgba(255,122,133,.4)}
.chip.s-super{background:rgba(183,155,255,.16);color:var(--purple);border:1px solid rgba(183,155,255,.4)}
.chip.s-trade{background:rgba(224,169,107,.16);color:#e0a96b;border:1px solid rgba(224,169,107,.4)}
.cbody{padding:13px 16px 15px}
.lbl{font-size:10.5px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);font-weight:700;margin:0 0 7px}
/* steps */
.steps{margin:0 0 14px;position:relative}
.step{display:flex;gap:10px;padding:0 0 11px;position:relative}
.step:not(:last-child)::before{content:"";position:absolute;left:11px;top:24px;bottom:0;width:2px;background:var(--line)}
.step .n{flex:none;width:24px;height:24px;border-radius:50%;background:var(--panel2);border:1.5px solid var(--accent);
 color:var(--accent);font-size:12px;font-weight:800;display:flex;align-items:center;justify-content:center;z-index:1}
.step .stx b{font-size:13px;color:#dfe6f1;display:block}
.step .stx span{font-size:12.5px;color:var(--muted)}
/* two-col lower */
.lower{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:560px){.lower{grid-template-columns:1fr}}
.method{font-size:12.5px;color:#cdd6e4;margin:0 0 10px}
.settings{display:flex;flex-direction:column;gap:3px;margin-top:4px}
.kv{display:flex;gap:8px;font-size:11.5px;border-bottom:1px dashed var(--line);padding:2px 0}
.kv .k{color:var(--muted);flex:none;min-width:74px}
.kv .v{color:#d6deeb;word-break:break-word}
table{width:100%;border-collapse:collapse;font-size:11.8px}
th,td{text-align:left;padding:4px 6px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.4px}
td.m{color:#c7d0de;white-space:nowrap}
td.v{color:#eaf1fb;font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap}
td.nt{color:var(--muted);font-size:11px}
.verdict{margin-top:13px;padding:10px 12px;background:var(--panel2);border-radius:8px;font-size:12.5px;color:#dbe2ee;
 border-left:3px solid var(--line)}
.card.s-best .verdict{border-left-color:var(--good)} .card.s-failed .verdict{border-left-color:var(--bad)}
.card.s-seed .verdict{border-left-color:var(--warn)} .card.s-lever .verdict{border-left-color:var(--lever)}
.card.s-robust .verdict{border-left-color:var(--accent)} .card.s-trade .verdict{border-left-color:#e0a96b}
.card.s-super .verdict{border-left-color:var(--purple)}
.verdict b{color:#fff}
.arts{margin-top:9px}
.art{display:inline-block;background:#0c0e12;border:1px solid var(--line);border-radius:5px;
 padding:1px 7px;font-size:10.5px;color:#9fb0c6;font-family:ui-monospace,Menlo,monospace;margin:2px 4px 0 0;word-break:break-all}
.foot{margin-top:54px;border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:12px}
code{background:#0c0e12;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:11.5px;color:#bcd}
"""

JS = """
const famBtns=[...document.querySelectorAll('[data-fam]')];
const stBtns=[...document.querySelectorAll('[data-status]')];
const cards=[...document.querySelectorAll('.card')];
const fams=[...document.querySelectorAll('.fam')];
let offFam=new Set(), offSt=new Set();
function apply(){
  cards.forEach(c=>{
    const hide=offFam.has(c.dataset.fam)||offSt.has(c.dataset.status);
    c.style.display=hide?'none':'';
  });
  fams.forEach(f=>{
    const vis=[...f.querySelectorAll('.card')].some(c=>c.style.display!=='none');
    f.style.display=vis?'':'none';
  });
}
famBtns.forEach(b=>b.onclick=()=>{const k=b.dataset.fam;
  if(offFam.has(k)){offFam.delete(k);b.classList.remove('off')}else{offFam.add(k);b.classList.add('off')}apply()});
stBtns.forEach(b=>b.onclick=()=>{const k=b.dataset.status;
  if(offSt.has(k)){offSt.delete(k);b.classList.remove('off')}else{offSt.add(k);b.classList.add('off')}apply()});
"""


def e(s):
    return html.escape(str(s), quote=True)


def card_html(c, fam_key):
    st = c.get("status", "baseline")
    scls = STATUS.get(st, ("?", "s-base"))[1]
    slab = STATUS.get(st, (st.upper(), "s-base"))[0]
    steps = "".join(
        f'<div class="step"><div class="n">{i+1}</div>'
        f'<div class="stx"><b>{e(s.get("title",""))}</b><span>{e(s.get("detail",""))}</span></div></div>'
        for i, s in enumerate(c.get("steps", [])))
    settings = "".join(
        f'<div class="kv"><span class="k">{e(s.get("field",""))}</span>'
        f'<span class="v">{e(s.get("value",""))}</span></div>'
        for s in c.get("settings", []))
    rows = "".join(
        f'<tr><td class="m">{e(r.get("metric",""))}</td><td class="v">{e(r.get("value",""))}</td>'
        f'<td class="nt">{e(r.get("note",""))}</td></tr>'
        for r in c.get("results", []))
    arts = "".join(f'<span class="art">{e(a)}</span>' for a in c.get("artifacts", []) or [])
    return f"""
    <div class="card {scls}" data-fam="{fam_key}" data-status="{st}">
      <div class="chead">
        <span class="chip {scls}">{e(slab)}</span>
        <div class="nick">{e(c.get("nickname",""))}</div>
        <div class="fname">{e(c.get("full_name",""))}</div>
        <div class="summ">{e(c.get("summary",""))}</div>
      </div>
      <div class="cbody">
        <p class="lbl">Steps</p>
        <div class="steps">{steps}</div>
        <div class="lower">
          <div>
            <p class="lbl">Methodology</p>
            <p class="method">{e(c.get("methodology",""))}</p>
            <p class="lbl">Settings</p>
            <div class="settings">{settings}</div>
          </div>
          <div>
            <p class="lbl">Results</p>
            <table><thead><tr><th>Metric</th><th>Value</th><th>Note</th></tr></thead>
            <tbody>{rows}</tbody></table>
          </div>
        </div>
        <div class="verdict"><b>Verdict.</b> {e(c.get("verdict",""))}</div>
        {f'<div class="arts">{arts}</div>' if arts else ''}
      </div>
    </div>"""


def main():
    d = json.loads(DATA.read_text())
    fams = {f["family_key"]: f for f in d["families"]}
    meta = d.get("meta", {})
    total = meta.get("total_configs", sum(len(f["configs"]) for f in d["families"]))

    # status counts
    counts = {}
    for f in d["families"]:
        for c in f["configs"]:
            counts[c.get("status", "baseline")] = counts.get(c.get("status", "baseline"), 0) + 1

    fam_filters = "".join(
        f'<span class="fbtn" data-fam="{k}">{FAMILY_META[k][0]} · {FAMILY_META[k][1].split(",")[0]}'
        f' <b>{len(fams[k]["configs"])}</b></span>'
        for k in FAMILY_META if k in fams)
    st_filters = "".join(
        f'<span class="fbtn" data-status="{k}">{STATUS[k][0]} <b>{counts.get(k,0)}</b></span>'
        for k in STATUS if counts.get(k, 0))

    sections = ""
    for k in FAMILY_META:
        if k not in fams:
            continue
        num, title, desc = FAMILY_META[k]
        cards = "".join(card_html(c, k) for c in fams[k]["configs"])
        sections += f"""
        <section class="fam" id="fam-{k}">
          <h2><span class="num">{num}</span>{title}
            <span class="fcount">· {len(fams[k]["configs"])} configurations</span></h2>
          <p class="fdesc">{desc}</p>
          <div class="cards">{cards}</div>
        </section>"""

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MentalHealthSympAI — Configuration Atlas</title>
<style>{CSS}</style></head><body><div class="wrap">
<header>
  <h1>MentalHealthSympAI — Configuration Atlas</h1>
  <div class="sub">Every configuration we built for PHQ-8 item-level depression-severity scoring, as a box
   with its <b>ordered steps</b>, exact methodology, settings, results and verdict. Evaluated on 5-fold
   participant-grouped CV (pooled out-of-fold, n=1752 item-rows / 219 participants), paired across methods.</div>
  <div class="stats">
    <div class="stat"><b>{total}</b><span>configurations</span></div>
    <div class="stat"><b>{len(fams)}</b><span>families</span></div>
    <div class="stat"><b>{counts.get('best',0)}</b><span>best-in-class</span></div>
    <div class="stat"><b>{counts.get('failed',0)}</b><span>did not work</span></div>
    <div class="stat"><b>0.418</b><span>best QWK (merged cascade)</span></div>
    <div class="stat"><b>~0.40</b><span>best macro-F1</span></div>
  </div>
  <div class="meta">Verified by a 13-agent extract→verify→completeness workflow; headline numbers recomputed from
   the OOF / JSON files on disk where they exist. Generated {e(meta.get('generated','2026-06-30'))} ·
   data: <code>outputs/config_atlas_data.json</code> · builder: <code>src/evaluation/build_config_atlas.py</code></div>
</header>
<div class="bar"><div class="barwrap">
  <span class="flab">Family</span>{fam_filters}
  <span class="sep"></span>
  <span class="flab">Status</span>{st_filters}
</div></div>
<p class="meta" style="margin:6px 0 0">Click any chip to toggle it off and filter the boxes below.</p>
{sections}
<div class="foot">
  Status legend — <b style="color:var(--good)">BEST</b> best-in-family ·
  <b style="color:var(--accent)">ROBUST</b> held up under seeds/CV ·
  <b style="color:var(--lever)">LEVER</b> a validated knob ·
  <b style="color:#aeb7c4">BASELINE</b> reference point ·
  <b style="color:var(--warn)">SEED-LUCK</b> single-seed artifact ·
  <b style="color:#e0a96b">TRADE-OFF</b> wins one metric, loses another ·
  <b style="color:var(--purple)">SUPERSEDED</b> beaten by a later config ·
  <b style="color:var(--bad)">DID NOT WORK</b> failed to deliver.<br><br>
  {e(meta.get('note',''))} Companion: <code>outputs/results_summary.html</code> (narrative summary + CIs).
</div>
</div><script>{JS}</script></body></html>"""

    OUT.write_text(doc)
    print(f"wrote {OUT}  ({len(doc)//1024} KB, {total} configs across {len(fams)} families)")


if __name__ == "__main__":
    main()
