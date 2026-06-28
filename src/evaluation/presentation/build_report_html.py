"""Build a single self-contained HTML report.

Explains, in plain language, (a) every illustration we produced and (b) all the
architectures and methodologies — which language models, how retrieval works, how
Chain-of-Thought was run, the ordinal loss, MIL, reranking, fusion, and the
evaluation protocol. Figures are base64-embedded so the file is fully portable.

    python -m src.evaluation.presentation.build_report_html
"""

import base64
from pathlib import Path

from src.evaluation.presentation import _style as S

FIGDIR = S.FIG_DIR
OUT = S.ROOT / "outputs" / "project_report.html"


def img(fname):
    p = FIGDIR / fname
    if not p.exists():
        return f'<div class="missing">[{fname} not found]</div>'
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<img loading="lazy" src="data:image/png;base64,{b64}" alt="{fname}">'


# ---- illustration gallery: (file, number, title, what-it-shows, takeaway) ----
ARCH = [
    ("fig_arch_core.png", "A", "The two core strategies",
     "Every method begins the same way: the interview is split into short windows and a retriever pulls the most relevant ones for each symptom. From there it splits in two. <b>Strategy A</b> (green) feeds the symptom + its evidence into a trained MentalBERT network that outputs a 0–3 score. <b>Strategy B</b> (violet) hands the same evidence to a frozen chatbot LLM that reasons in words and states a score.",
     "One trained network, one frozen LLM — both reading the same retrieved evidence."),
    ("fig_arch_variants.png", "B", "Three refinements built from the same parts",
     "The advanced experiments reuse the same building blocks. <b>Attention-MIL</b> reads each window separately and learns which one matters (giving an explanation). <b>Cross-encoder rerank</b> re-sorts all transcript windows to keep the most telling ones. <b>Calibrated fusion</b> fixes each model's confidence, then combines the encoder and the LLM.",
     "MIL = interpretability; rerank = sharper evidence; fusion = combine the two brains."),
]

FIGS = [
    ("fig1_retrieval_waterfall.png", 1, "The evidence is the hero",
     "We keep the model and training fixed and change <i>only the evidence it reads</i>. Plain sentences are weak; keyword (BM25) search barely helps; <b>semantic ‘hybrid’ search makes the big jump.</b>",
     "Agreement with the true total score rises 6.5× (QWK 0.074 → 0.485) — from better evidence, not a bigger model."),
    ("fig2_ordinal_nearmiss.png", 2, "Almost always almost right",
     "Severity is ordered (0–3), so we ask how <i>far</i> the mistakes are. The heatmap hugs the diagonal: the model almost never confuses ‘minimal’ with ‘severe’.",
     "Exact accuracy is ~45%, but 88% of predictions are within one level. The honest weak spot: it under-rates some truly severe people."),
    ("fig3_evidence_quality_map.png", 3, "Good evidence isn’t enough",
     "Each bubble is one symptom: how on-topic its evidence is (left–right) vs how well we predict it (up–down). Bubble size = how lopsided the classes are.",
     "Having rich evidence doesn’t make a symptom easy — ‘Sleep’ has plenty yet stays hard. Some difficulty is intrinsic to the symptom."),
    ("fig4_cot_complementarity.png", 4, "Two minds, different routes",
     "The trained network and the frozen LLM reach similar accuracy but <b>agree less than half the time</b>. Left: who is right when they disagree, by true severity.",
     "The network plays it safe in the middle; the LLM commits to the extremes and recovers more severe cases (F1 0.16 → 0.30)."),
    (None, 5, "Inside the LLM’s reasoning",
     "Real chain-of-thought rationales, in the model’s own words (shown as cards in <code>fig5_reasoning_showcase.html</code>). Includes severe cases the trained model missed, plus one honest failure.",
     "When the retriever surfaces no evidence (e.g. Appetite), even sound reasoning can’t score the item — tying back to illustration 3."),
    ("fig6_calibrated_fusion.png", 6, "Calibrate, then combine",
     "Left: the model’s confidence was untrustworthy (over-confident); one simple ‘temperature’ fix corrects it. Right: ways of combining the two models.",
     "Calibration error drops 79% (0.40 → 0.08). A plain average doesn’t help — you must combine by symptom/severity, not with one global weight."),
    ("fig7_clinical_screener.png", 7, "As a depression screener",
     "We reconstruct each person’s total PHQ-8 score and flag clinically significant depression (≥10). Left: the trade-off curve; right: where to set the cut-off.",
     "The default rule catches only 37% of true cases, but the model ranks people well (AUC 0.78) — a tuned cut-off reaches 86% sensitivity."),
    ("fig8_selective_prediction.png", 8, "Knowing when to ask a human",
     "If the model abstains on its least-confident items and routes them to a clinician, how good is it on the rest? And <i>which</i> cases does it hand off?",
     "Deferring the hardest 40% raises accuracy on the rest (0.46 → 0.52), and the deferred cases are disproportionately the ambiguous severe ones."),
    ("fig9_bootstrap_ci.png", 9, "Are the gains real?",
     "With only 219 participants, we resample them 2,000 times to put error bars on each improvement. Green = clears zero (significant); grey = not.",
     "Retrieval gains are unambiguous; the LLM’s small macro-F1 edge on its own is <i>not</i> statistically certain — a claim to state carefully."),
    ("fig10_bucketb_summary.png", 10, "Four upgrade experiments",
     "Each cell is an improvement (green) or regression (red) versus the baseline, across five metrics. Boxes mark the best in each column.",
     "No single winner: attention-MIL is best overall, focal loss best for ordering (QWK), and reranking / class-balancing best for catching severe cases."),
    (None, 11, "The model points at its evidence",
     "The attention-MIL model highlights which retrieved window drove each score (cards in <code>fig11_mil_attribution.html</code>).",
     "Spot-on for ‘Tired’ (88% of weight on the right snippet); visibly weaker for thin-evidence items like Appetite — honest about its limits."),
]

# inline example cards (from fig5 / fig11 data) so the report is self-contained
COT_CARDS = [
    ("Depressed", "Severe — rescued", True,
     "“Several snippets indicate persistent feelings of depression and hopelessness… ‘happy but then I went back to being’ — maps to nearly every day.”",
     "True 3 · LLM 3 · Encoder said 1"),
    ("Appetite", "Instructive miss", False,
     "“No direct mention of poor appetite or overeating in the snippets…” — correct reasoning, but the retriever surfaced no appetite evidence.",
     "True 3 · LLM 0"),
]
MIL_CARDS = [
    ("Tired", "88% attention", True,
     "“tired very tired and hard to focus… it just takes me longer to get things done cuz I don’t have my energy”"),
    ("Appetite", "59% attention", False,
     "“depressed [TURN_SEP] mood swings [TURN_SEP] high anxiety” — only loosely on-symptom: appetite evidence is genuinely thin."),
]


def gallery_item(fig):
    fname, num, title, what, take = fig
    media = img(fname) if fname else cards_for(num)
    return f"""
    <article class="card">
      <div class="cardhead"><span class="num">{num}</span><h3>{title}</h3></div>
      <div class="media">{media}</div>
      <p class="what">{what}</p>
      <p class="take"><span>Takeaway</span> {take}</p>
    </article>"""


def cards_for(num):
    if num == 5:
        rows = "".join(
            f'<div class="mini {"ok" if ok else "bad"}"><div class="mh"><b>PHQ-8 · {it}</b>'
            f'<span class="tag">{tag}</span></div><p>{txt}</p><div class="meta">{meta}</div></div>'
            for it, tag, ok, txt, meta in COT_CARDS)
    else:
        rows = "".join(
            f'<div class="mini {"ok" if ok else "bad"}"><div class="mh"><b>PHQ-8 · {it}</b>'
            f'<span class="tag">{tag}</span></div><p>{txt}</p></div>'
            for it, tag, ok, txt in MIL_CARDS)
    return f'<div class="minis">{rows}</div>'


def main():
    arch_html = "".join(arch_item(a) for a in ARCH)
    gallery = "".join(gallery_item(f) for f in FIGS)

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PHQ-8 Severity from Interviews — Methods & Illustrations</title>
<style>
  :root{{--teal:#0D7D87;--dark:#0B3B45;--mint:#16B5A5;--ink:#0F172A;--mute:#5B7079;
         --tint:#EEF6F6;--line:#D9E6E6;--good:#16A34A;--coral:#E2683C;}}
  *{{box-sizing:border-box}}
  body{{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
        color:var(--ink);line-height:1.6;background:#F7FAFA}}
  .wrap{{max-width:1040px;margin:0 auto;padding:0 22px}}
  header.hero{{background:var(--dark);color:#fff;padding:54px 0 46px}}
  .hero .kick{{color:var(--mint);font-weight:700;letter-spacing:2px;font-size:13px}}
  .hero h1{{font-family:Cambria,Georgia,serif;font-size:38px;margin:.25em 0 .15em;line-height:1.12}}
  .hero p{{color:#CDE6E6;font-size:18px;max-width:760px;margin:.2em 0}}
  .hero .meta{{color:#9FC3C3;font-size:13.5px;margin-top:14px}}
  nav{{position:sticky;top:0;background:#fff;border-bottom:1px solid var(--line);z-index:10}}
  nav .wrap{{display:flex;flex-wrap:wrap;gap:8px 18px;padding:11px 22px;font-size:13.5px}}
  nav a{{color:var(--teal);text-decoration:none;font-weight:600}} nav a:hover{{text-decoration:underline}}
  section{{padding:40px 0 8px}}
  h2{{font-family:Cambria,Georgia,serif;font-size:26px;color:var(--dark);margin:0 0 4px}}
  .lead{{color:var(--mute);font-size:15.5px;margin:0 0 22px;max-width:820px}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
  @media(max-width:820px){{.grid2{{grid-template-columns:1fr}} .hero h1{{font-size:30px}}}}
  .card,.method,.archcard{{background:#fff;border:1px solid var(--line);border-radius:14px;
        box-shadow:0 1px 3px rgba(11,59,69,.06)}}
  .card{{padding:16px 18px;margin-bottom:20px}}
  .cardhead{{display:flex;align-items:center;gap:12px;margin-bottom:10px}}
  .num{{flex:0 0 34px;height:34px;border-radius:50%;background:var(--teal);color:#fff;
        font-weight:700;display:flex;align-items:center;justify-content:center;font-size:15px}}
  .card h3{{font-size:19px;margin:0;color:var(--ink)}}
  .media{{margin:6px 0 12px}} .media img{{width:100%;border-radius:10px;border:1px solid var(--line)}}
  .what{{margin:.2em 0 .7em;font-size:15px}}
  .take{{background:var(--tint);border-radius:9px;padding:9px 13px;font-size:14.5px;margin:0}}
  .take span{{display:inline-block;font-weight:700;color:var(--teal);text-transform:uppercase;
        font-size:11px;letter-spacing:1px;margin-right:8px}}
  .archcard{{padding:16px 18px;margin-bottom:18px}}
  .archcard img{{width:100%;border-radius:10px;border:1px solid var(--line);margin:8px 0}}
  .archcard .badge{{display:inline-block;background:var(--dark);color:#fff;border-radius:6px;
        padding:2px 10px;font-weight:700;font-size:12px;letter-spacing:1px}}
  .archcard h3{{font-size:19px;margin:8px 0 2px}}
  table{{border-collapse:collapse;width:100%;font-size:14px;background:#fff;border-radius:12px;overflow:hidden}}
  th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
  thead th{{background:var(--dark);color:#fff;font-weight:600}}
  tbody tr:nth-child(even){{background:#F4FAFA}}
  .pill{{display:inline-block;border-radius:20px;padding:1px 9px;font-size:11.5px;font-weight:700}}
  .pill.no{{background:#E0F2F1;color:#0d7d87}} .pill.yes{{background:#FDEAE0;color:#c2410c}}
  .method{{padding:15px 17px}} .method h4{{margin:0 0 5px;font-size:16px;color:var(--dark)}}
  .method p{{margin:.2em 0;font-size:14px}}
  .steps{{counter-reset:s;list-style:none;padding:0;margin:8px 0}}
  .steps li{{counter-increment:s;position:relative;padding:6px 0 6px 38px;font-size:14.5px;border-bottom:1px dashed #E2EAEA}}
  .steps li:before{{content:counter(s);position:absolute;left:0;top:6px;width:26px;height:26px;
        background:var(--mint);color:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px}}
  .anchor{{background:var(--tint);border-radius:10px;padding:12px 16px;font-size:14px}}
  .anchor b{{color:var(--teal)}}
  pre.prompt{{background:#0B3B45;color:#CDE6E6;border-radius:10px;padding:14px 16px;font-size:12.5px;
        overflow:auto;white-space:pre-wrap;line-height:1.5}}
  pre.prompt .k{{color:var(--mint);font-weight:700}}
  .minis{{display:grid;gap:10px}}
  .mini{{border-radius:10px;padding:11px 14px;background:var(--tint);font-size:13.5px}}
  .mini.bad{{background:#FBEEE8}}
  .mini .mh{{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}}
  .mini .tag{{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--teal)}}
  .mini.bad .tag{{color:var(--coral)}}
  .mini p{{margin:.2em 0;font-style:italic;color:var(--ink)}}
  .mini .meta{{font-size:12px;color:var(--mute);margin-top:4px}}
  footer{{background:var(--dark);color:#9FC3C3;padding:30px 0;margin-top:40px;font-size:13px}}
  .note{{font-size:13px;color:var(--mute)}}
</style></head><body>

<header class="hero"><div class="wrap">
  <div class="kick">DEPRESSION SCREENING · NLP</div>
  <h1>Reading Depression Severity from Clinical Interviews</h1>
  <p>A plain-language guide to every illustration we produced — and to the models and methods behind them: which language models, how retrieval works, and how we ran Chain-of-Thought.</p>
  <div class="meta">PHQ-8 item-level severity (0–3) · 219 participants · 1,752 item-examples · 5-fold cross-validation</div>
</div></header>

<nav><div class="wrap">
  <a href="#problem">The task</a><a href="#arch">Architectures</a><a href="#models">Language models</a>
  <a href="#retrieval">Retrieval</a><a href="#cot">Chain-of-Thought</a><a href="#methods">Other methods</a>
  <a href="#eval">Evaluation</a><a href="#gallery">All illustrations</a>
</div></nav>

<div class="wrap">

<section id="problem">
  <h2>The task &amp; the data</h2>
  <p class="lead">From one clinical interview transcript, predict the severity (0–3) of each of the eight PHQ-8 depression symptoms, then sum them to screen for clinically significant depression (total ≥ 10).</p>
  <div class="grid2">
    <div class="method"><h4>The 0–3 scale is ordinal</h4>
      <p>0 = Not at all · 1 = Several days · 2 = More than half the days · 3 = Nearly every day. The order matters, so being “one off” is a small error and being three off is a big one.</p></div>
    <div class="method"><h4>Small &amp; imbalanced</h4>
      <p>219 participants (DAIC-WOZ), 1,752 item-examples. Only <b>9.5%</b> of items are severe — that rare tail is the project’s recurring bottleneck.</p></div>
  </div>
</section>

<section id="arch">
  <h2>Architectures</h2>
  <p class="lead">Two core strategies share one evidence front-end; three refinements recombine the same parts.</p>
  {arch_html}
</section>

<section id="models">
  <h2>Which language models?</h2>
  <p class="lead">Four off-the-shelf models, used in different roles. Only the encoder is fine-tuned on our task; everything else is used as-is.</p>
  <table>
    <thead><tr><th>Model</th><th>Role</th><th>Trained on our data?</th><th>Why this one</th></tr></thead>
    <tbody>
      <tr><td><b>MentalBERT</b><br><span class="note">mental/mental-bert-base-uncased</span></td><td>Main encoder (Strategy A)</td><td><span class="pill yes">Fine-tuned</span></td><td>A BERT pre-trained on mental-health text; reads symptom + evidence and outputs a score.</td></tr>
      <tr><td><b>BERT-base</b><br><span class="note">bert-base-uncased</span></td><td>Baseline encoder</td><td><span class="pill yes">Fine-tuned</span></td><td>General-domain control, to test whether domain pre-training helps.</td></tr>
      <tr><td><b>Qwen2.5-7B-Instruct</b><br><span class="note">open-weight, 7B params</span></td><td>Chain-of-Thought probe (Strategy B)</td><td><span class="pill no">Frozen</span></td><td>An instruction-tuned chat LLM that reasons about the evidence in natural language — no training.</td></tr>
      <tr><td><b>MiniLM cross-encoder</b><br><span class="note">ms-marco-MiniLM-L-6-v2</span></td><td>Evidence reranker</td><td><span class="pill no">Off-the-shelf</span></td><td>Scores how relevant each window is to the symptom question, to keep the most telling evidence.</td></tr>
      <tr><td><b>BM25 + dense embeddings</b></td><td>Hybrid retriever</td><td><span class="pill no">No training</span></td><td>Finds the most relevant interview windows for each symptom (keyword + meaning).</td></tr>
    </tbody>
  </table>
</section>

<section id="retrieval">
  <h2>How retrieval works</h2>
  <p class="lead">A whole interview is too long to feed a model, and most of it is irrelevant to any one symptom. So for each symptom we retrieve only the relevant pieces.</p>
  <ol class="steps">
    <li><b>Window the interview.</b> Split the transcript into overlapping <i>context windows</i> of a few consecutive turns (we tried 3- and 5-turn windows).</li>
    <li><b>Query per symptom.</b> Use the PHQ-8 item (e.g. “Trouble falling or staying asleep”) as the search query — this is the <i>item-aware</i> part.</li>
    <li><b>Hybrid scoring.</b> Rank windows by a mix of <b>BM25</b> (keyword overlap) and <b>dense embeddings</b> (semantic similarity). Illustration 1 shows the semantic half is what really helps.</li>
    <li><b>Pack the top-k.</b> Join the best windows into one evidence block, given to both the encoder and the LLM — so the two strategies are compared on identical evidence.</li>
  </ol>
</section>

<section id="cot">
  <h2>How we performed Chain-of-Thought</h2>
  <p class="lead">We ask a frozen chat LLM (Qwen2.5-7B-Instruct) to score one symptom at a time by reasoning step-by-step over the same retrieved evidence — with no fine-tuning.</p>

  <div class="grid2">
    <div>
      <div class="method"><h4>The instructions (system prompt, paraphrased)</h4>
        <p>“You are a careful clinical-research assistant scoring a single PHQ-8 item from short, keyword-like retrieved snippets. Treat them as weak cues; do <b>not</b> invent symptoms. Reason step-by-step, then map to the closest frequency anchor; when evidence is thin, prefer the lower adjacent score. Reply as JSON: <code>{{reasoning, label}}</code>.”</p></div>
      <div class="anchor" style="margin-top:12px">
        <b>The frequency anchors the model maps onto</b><br>
        <b>0</b> Not at all — absent, or the opposite is reported<br>
        <b>1</b> Several days — occasional / mild<br>
        <b>2</b> More than half the days — frequent<br>
        <b>3</b> Nearly every day — persistent / dominant
      </div>
    </div>
    <div>
      <div class="method"><h4>Reasoning steps it is told to follow</h4>
        <ol class="steps">
          <li>Note which snippets are relevant to <i>this</i> symptom.</li>
          <li>Judge whether they show it absent, occasional, frequent, or persistent.</li>
          <li>Map that to the closest 0–3 anchor (prefer lower when ambiguous).</li>
        </ol></div>
      <div class="method" style="margin-top:12px"><h4>Decoding</h4>
        <p><b>Greedy</b> = one deterministic answer. <b>Self-consistency</b> = sample 5 reasoning chains at temperature 0.7 and take the majority/soft vote — more robust, and gives a probability per class.</p></div>
    </div>
  </div>

  <h4 style="margin:22px 0 6px;color:var(--dark)">Few-shot teaching examples</h4>
  <p class="note" style="margin-top:0">Four hand-crafted demos (one per severity level, with synthetic evidence) are prepended so the model learns the format and the anchor mapping. The “severe” demo, verbatim:</p>
  <pre class="prompt"><span class="k">item:</span> Feeling down, depressed, or hopeless
<span class="k">evidence:</span> I feel down every single day /// nothing is going to get better /// cry most mornings /// can't see a way out /// hopeless
<span class="k">reasoning:</span> The snippets are pervasive and severe: 'every single day', 'cry most mornings', explicit 'hopeless' and 'nothing is going to get better'. Persistent and dominant -&gt; 'Nearly every day'.
<span class="k">label:</span> 3</pre>
  <p class="note">We also tested variants: a <i>strict</i> prompt (an explicit “not discussed → 0” demo to curb over-attribution) and a <i>full-transcript</i> mode (the LLM reads the whole interview instead of retrieved snippets).</p>
</section>

<section id="methods">
  <h2>How each method works</h2>
  <p class="lead">The mechanism behind every model in the leaderboard — what problem it solves and how.</p>

  <div class="method"><h4>Encoder + CORN ordinal head &nbsp;<span class="note">— the baseline model</span></h4>
    <p>Input is <code>[CLS] symptom-question [SEP] evidence [SEP]</code>, read by MentalBERT. Instead of a normal 4-way softmax (which would treat 0,1,2,3 as unrelated), the <b>CORN</b> head predicts three <i>ordered</i> yes/no thresholds — “severity &gt; 0?”, “&gt; 1?”, “&gt; 2?” — each trained only on the examples that reached it. The probabilities are cumulative, so they’re guaranteed rank-consistent, and the score is just how many thresholds clear 50%. This ordering is <i>why</i> almost all mistakes are only one level off.</p></div>

  <div class="grid2">
    <div class="method"><h4>Balanced CORN <span class="note">— severe recall</span></h4>
      <p>Plain CORN weights every example equally, so the 9.5% severe items get drowned out. Here each example’s loss is multiplied by the <b>inverse frequency of its class</b>, pushing the decision boundaries to catch rare severe cases. Lifts severe recall, costs some MAE.</p></div>
    <div class="method"><h4>Focal CORN <span class="note">— hard cases</span></h4>
      <p>Adds a focal factor <code>(1−p)<sup>γ</sup></code> to each threshold decision: easy, already-confident cases are down-weighted so training <b>focuses on the ambiguous boundary cases</b>. Best ordinal agreement (QWK) at no extra cost.</p></div>
    <div class="method"><h4>Attention-MIL <span class="note">— best single model</span></h4>
      <p>Instead of gluing windows into one block (which truncates), <b>each window is encoded separately</b> by the shared MentalBERT. A gated-attention unit scores every window and a softmax turns those into weights that sum to 1; the windows are combined by that weighted average into one “bag” embedding, fed to the CORN head. The weights double as an <b>explanation</b> — the highest-weighted window is the evidence it used.</p></div>
    <div class="method"><h4>Cross-encoder reranking <span class="note">— sharper evidence</span></h4>
      <p>Similarity isn’t the same as being <i>diagnostic</i>. A cross-encoder (MiniLM) re-reads every (symptom-question, window) pair in the whole transcript (~90 windows) and scores its relevance; the top-5 most diagnostic windows become a new evidence pack for the encoder.</p></div>
  </div>

  <div class="method" style="margin-top:16px"><h4>Calibrated fusion <span class="note">— combining the encoder &amp; the LLM</span></h4>
    <p><b>Step 1 — calibration.</b> The raw CORN probabilities are over-confident, so we divide the logits by one learned “temperature” (fit to maximise likelihood on the training folds) before the softmax. This pulls confidence in line with real accuracy (error 0.40 → 0.08) — essential because the next step combines probabilities. <b>Step 2 — combine</b>, three ways:</p>
    <ul class="steps" style="counter-reset:none">
      <li style="padding-left:14px"><b>Probability blend</b> — a weighted average of the two calibrated probability vectors (weight chosen leakage-free by nested CV), decoded by expected-value rounding. A single global weight ≈ just the encoder — no real gain.</li>
      <li style="padding-left:14px"><b>Severity-routing</b> <i>(the best system)</i> — a simple rule: if the LLM predicts severe (≥ 2) take its label, else take the encoder’s. Exploits the LLM’s known severe-class strength; beats both parents on F1 and QWK at once.</li>
      <li style="padding-left:14px"><b>Learned stacker</b> — a small logistic-regression meta-model over [encoder probs + LLM probs + which-symptom], trained nested-CV, that learns <i>when</i> to trust which model. Best severe-class F1.</li>
    </ul>
    <p class="note">The unifying lesson: the two models are right on <i>different</i> items, so a global blend can’t help — only combiners that <b>condition</b> (on severity or symptom) capture the complementarity.</p></div>
</section>

<section id="eval">
  <h2>How we evaluated</h2>
  <ol class="steps">
    <li><b>Participant-grouped 5-fold cross-validation.</b> No person appears in both training and test, so the model can’t memorise individuals. Every example is predicted exactly once (out-of-fold).</li>
    <li><b>Ordinal-aware metrics.</b> Accuracy and macro-F1, but also MAE and Quadratic-Weighted Kappa (QWK), which reward being <i>close</i> on this ordered scale, plus the “within-one” / off-by-one rate.</li>
    <li><b>Reconstructed total &amp; screening.</b> Sum the 8 item predictions per person and evaluate the ≥10 depression-screening decision (sensitivity / specificity, AUC).</li>
    <li><b>Honest uncertainty.</b> With only 219 people, we cluster-bootstrap (resample participants 2,000×) to put confidence intervals on every gain (illustration 9).</li>
  </ol>
</section>

<section id="gallery">
  <h2>All the illustrations</h2>
  <p class="lead">Every figure we produced, in plain language: what it shows and the one thing to remember. Two of them (5 and 11) are interactive card pages — representative examples are shown inline.</p>
  {gallery}
</section>

</div>
<footer><div class="wrap">
  Generated from <code>outputs/figures/</code> · figures embedded for portability ·
  MentalHealthSympAI · PHQ-8 item-level severity prediction.
</div></footer>
</body></html>"""

    OUT.write_text(html)
    size_mb = OUT.stat().st_size / 1e6
    print(f"  wrote {OUT}  ({size_mb:.1f} MB, self-contained)")


def arch_item(a):
    fname, badge, title, desc, take = a
    return f"""
    <div class="archcard">
      <span class="badge">STRATEGY {badge}</span>
      <h3>{title}</h3>
      {img(fname)}
      <p class="what">{desc}</p>
      <p class="take"><span>In short</span> {take}</p>
    </div>"""


if __name__ == "__main__":
    main()
