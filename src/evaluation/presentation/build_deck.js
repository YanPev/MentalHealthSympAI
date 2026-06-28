const pptxgen = require("pptxgenjs");

const RATIOS = {
  "fig1_retrieval_waterfall.png": 2.062, "fig2_ordinal_nearmiss.png": 2.348,
  "fig3_evidence_quality_map.png": 1.680, "fig4_cot_complementarity.png": 2.326,
  "fig6_calibrated_fusion.png": 2.282, "fig7_clinical_screener.png": 2.369,
  "fig8_selective_prediction.png": 2.419, "fig9_bootstrap_ci.png": 2.545,
  "fig10_bucketb_summary.png": 2.194,
  "fig_arch_core.png": 2.249, "fig_arch_variants.png": 2.039,
};

const p = new pptxgen();
p.layout = "LAYOUT_WIDE";        // 13.3 x 7.5
const PW = 13.3, PH = 7.5;
p.author = "MentalHealthSympAI";
p.title = "PHQ-8 Item Severity — Insights & Experiments";

// ---- palette (clinical teal) ----
const DARK = "0B3B45", TEAL = "0D7D87", MINT = "16B5A5";
const INK = "0F172A", MUTE = "5B7079", FAINT = "94A3B8";
const WHITE = "FFFFFF", TINT = "EEF6F6", CORAL = "E2683C", GOOD = "16A34A";
const HEAD = "Cambria", BODY = "Calibri";
const shadow = () => ({ type: "outer", color: "0B3B45", blur: 9, offset: 3, angle: 90, opacity: 0.14 });

const MX = 0.65;                 // side margin
function kicker(slide, text) {
  slide.addText(text.toUpperCase(), { x: MX, y: 0.42, w: PW - 2 * MX, h: 0.32,
    fontFace: BODY, fontSize: 12.5, bold: true, color: TEAL, charSpacing: 2, margin: 0 });
}
function title(slide, text) {
  slide.addText(text, { x: MX, y: 0.74, w: PW - 2 * MX, h: 0.78,
    fontFace: HEAD, fontSize: 26, bold: true, color: INK, margin: 0 });
}
function takeaway(slide, rich) {
  const y = 6.18, h = 0.96;
  slide.addShape(p.shapes.ROUNDED_RECTANGLE, { x: MX, y, w: PW - 2 * MX, h,
    fill: { color: TINT }, line: { type: "none" }, rectRadius: 0.09, shadow: shadow() });
  slide.addText(rich, { x: MX + 0.25, y, w: PW - 2 * MX - 0.5, h,
    fontFace: BODY, fontSize: 13.5, color: INK, valign: "middle", margin: 0, lineSpacingMultiple: 1.02 });
}
function figure(slide, file, top, maxH) {
  const ratio = RATIOS[file];
  const mh = maxH || 4.25, mw = PW - 2 * MX;
  let w = mh * ratio, h = mh;
  if (w > mw) { w = mw; h = w / ratio; }
  slide.addImage({ path: file, x: (PW - w) / 2, y: top + (mh - h) / 2, w, h });
}

function contentSlide(opts) {
  const s = p.addSlide();
  s.background = { color: WHITE };
  kicker(s, opts.kicker);
  title(s, opts.title);
  if (opts.image) figure(s, opts.image, 1.7, opts.maxH || 4.2);
  takeaway(s, opts.takeaway);
  if (opts.notes) s.addNotes(opts.notes);
  return s;
}

// ============ 1 · TITLE ============
{
  const s = p.addSlide();
  s.background = { color: DARK };
  s.addShape(p.shapes.OVAL, { x: PW - 3.6, y: -2.0, w: 5.2, h: 5.2, fill: { color: TEAL, transparency: 78 }, line: { type: "none" } });
  s.addShape(p.shapes.OVAL, { x: -1.7, y: PH - 2.6, w: 4.2, h: 4.2, fill: { color: MINT, transparency: 84 }, line: { type: "none" } });
  s.addText("DEPRESSION SCREENING · NLP", { x: MX, y: 1.55, w: 10, h: 0.4,
    fontFace: BODY, fontSize: 14, bold: true, color: MINT, charSpacing: 3, margin: 0 });
  s.addText("Reading Depression Severity from Clinical Interviews",
    { x: MX, y: 2.05, w: 11.4, h: 1.9, fontFace: HEAD, fontSize: 41, bold: true, color: WHITE, margin: 0, lineSpacingMultiple: 1.02 });
  s.addText("PHQ-8 item-level severity prediction — what drives performance, how robust it is, and how far it can be pushed",
    { x: MX, y: 3.95, w: 11.0, h: 0.9, fontFace: BODY, fontSize: 17, color: "CDE6E6", margin: 0, lineSpacingMultiple: 1.05 });
  s.addText([
    { text: "MentalBERT + CORN", options: { color: WHITE, bold: true } },
    { text: "   ·   hybrid context-window retrieval   ·   ", options: { color: "9FC3C3" } },
    { text: "frozen-LLM Chain-of-Thought", options: { color: WHITE, bold: true } },
    { text: "      |      219 participants · 1,752 item-examples · 5-fold CV", options: { color: "9FC3C3" } },
  ], { x: MX, y: 6.35, w: 12, h: 0.5, fontFace: BODY, fontSize: 12.5, margin: 0 });
  s.addNotes("Welcome. This project reads depression-symptom severity from clinical interview transcripts — PHQ-8, one symptom at a time. I'll cover three things: what actually drives performance, how robust and clinically usable it is, and four experiments that push the frontier.");
}

// ============ 2 · SETUP / DATA ============
{
  const s = p.addSlide();
  s.background = { color: WHITE };
  kicker(s, "The task");
  title(s, "Eight depression symptoms, rated from one conversation");
  s.addText("For each clinical interview transcript, predict the PHQ-8 score (0–3, ordinal) of all eight depression symptoms — then reconstruct the total to screen for clinically significant depression (≥ 10).",
    { x: MX, y: 1.55, w: PW - 2 * MX, h: 0.85, fontFace: BODY, fontSize: 15.5, color: INK, margin: 0, lineSpacingMultiple: 1.06 });

  const stats = [
    ["219", "participants\n(DAIC-WOZ)"],
    ["1,752", "item-level\nexamples"],
    ["4", "ordinal severity\nclasses (0–3)"],
    ["9.5%", "are severe —\nthe hard tail"],
  ];
  const cw = 2.78, gap = 0.33, sx = MX, sy = 2.6, ch = 1.55;
  stats.forEach((st, i) => {
    const x = sx + i * (cw + gap);
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y: sy, w: cw, h: ch, fill: { color: i === 3 ? DARK : TINT }, line: { type: "none" }, rectRadius: 0.1, shadow: shadow() });
    s.addText(st[0], { x, y: sy + 0.18, w: cw, h: 0.7, fontFace: HEAD, fontSize: 34, bold: true, color: i === 3 ? MINT : TEAL, align: "center", margin: 0 });
    s.addText(st[1], { x, y: sy + 0.88, w: cw, h: 0.6, fontFace: BODY, fontSize: 12.5, color: i === 3 ? "CDE6E6" : MUTE, align: "center", margin: 0, lineSpacingMultiple: 0.95 });
  });

  s.addText("THREE MODELLING INGREDIENTS", { x: MX, y: 4.5, w: 10, h: 0.3, fontFace: BODY, fontSize: 12, bold: true, color: TEAL, charSpacing: 2, margin: 0 });
  const ing = [
    ["Encoder", "MentalBERT with a rank-consistent CORN ordinal head"],
    ["Retrieval", "Hybrid (lexical + semantic) context-window evidence per item"],
    ["CoT probe", "Frozen Qwen-7B, few-shot chain-of-thought — no task training"],
  ];
  const iw = 3.9, ig = 0.3, iy = 4.85, ih = 1.15;
  ing.forEach((it, i) => {
    const x = MX + i * (iw + ig);
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y: iy, w: iw, h: ih, fill: { color: WHITE }, line: { color: "D9E6E6", width: 1 }, rectRadius: 0.08 });
    s.addText(it[0], { x: x + 0.22, y: iy + 0.14, w: iw - 0.4, h: 0.34, fontFace: HEAD, fontSize: 16, bold: true, color: INK, margin: 0 });
    s.addText(it[1], { x: x + 0.22, y: iy + 0.5, w: iw - 0.4, h: 0.6, fontFace: BODY, fontSize: 12, color: MUTE, margin: 0, lineSpacingMultiple: 1.0 });
  });
  s.addNotes("The task is fine-grained: rate all eight PHQ-8 symptoms 0 to 3 from one interview, then sum them to screen at the clinical threshold of 10. Two things to flag: the data is small — 219 participants — and badly imbalanced, only 9.5% of items are severe. That severe tail becomes the recurring theme of the talk. Three ingredients power everything: a MentalBERT encoder with a rank-consistent ordinal head, hybrid retrieval for evidence, and a frozen large language model doing chain-of-thought with no training.");
}

// ============ ARCHITECTURE ============
contentSlide({ kicker: "How the models work", title: "Two strategies over one shared evidence front-end",
  image: "fig_arch_core.png", maxH: 4.0,
  takeaway: [ { text: "Two paradigms, same evidence.  ", options: { bold: true, color: TEAL } },
    { text: "Both read item-aware retrieved windows. ", options: {} },
    { text: "Strategy A trains MentalBERT end-to-end with a rank-consistent ordinal head; Strategy B is a frozen LLM that reasons over the same evidence — no task training.", options: {} } ],
  notes: "Before the results, here's how the two core models work — and they share a front-end. Every strategy starts the same way: the interview is chopped into context windows and an item-aware hybrid retriever pulls the most relevant ones for each PHQ-8 symptom. From there it splits. Strategy A, in green, feeds the item and its evidence into MentalBERT and a CORN ordinal head that's trained end-to-end. Strategy B, in violet, hands the same evidence to a frozen Qwen-7B with a few-shot prompt; it reasons in natural language and we take a self-consistency vote. One is trained, one does no training at all." });

contentSlide({ kicker: "How the models work", title: "Three refinements reuse the same building blocks",
  image: "fig_arch_variants.png", maxH: 4.3,
  takeaway: [ { text: "Same parts, recombined.  ", options: { bold: true, color: TEAL } },
    { text: "Attention-MIL pools the windows separately and exposes which one mattered; reranking sharpens the evidence; calibrated fusion combines the encoder and CoT — conditionally, not by a scalar blend.", options: {} } ],
  notes: "The advanced experiments don't introduce new machinery — they recombine these parts. Refinement 1, attention-MIL, encodes each window separately and learns an attention weight over them, which doubles as an explanation of which window drove the score. Refinement 2 reranks the full pool of transcript windows with a cross-encoder before they ever reach the model. Refinement 3 takes the encoder's and CoT's probabilities, calibrates them, and combines them conditionally. Keep this picture in mind for Part 3 and Part 4." });

// ============ PART 1 — figures 1,2,3 ============
contentSlide({ kicker: "Part 1 · What drives performance", title: "The evidence is the hero — not the model",
  image: "fig1_retrieval_waterfall.png",
  takeaway: [ { text: "Takeaway.  ", options: { bold: true, color: TEAL } },
    { text: "Same encoder, same loss — only the retrieved evidence changes. Lexical windows barely help; ", options: {} },
    { text: "semantic hybrid retrieval lifts total-score agreement 6.5× (QWK 0.074 → 0.485).", options: { bold: true } } ],
  notes: "The biggest lever isn't the model — it's the evidence. Holding the model and loss fixed, lexical BM25 windows barely move the needle, but adding semantic hybrid retrieval lifts total-score agreement more than six-fold. The headline of this whole project is: better evidence, not a bigger network. Error bars are across the five folds." });

contentSlide({ kicker: "Part 1 · What drives performance", title: "The model is almost always almost right",
  image: "fig2_ordinal_nearmiss.png",
  takeaway: [ { text: "Takeaway.  ", options: { bold: true, color: TEAL } },
    { text: "Exact accuracy (~45%) understates an ordinally coherent model: ", options: {} },
    { text: "88% of predictions are within one severity level", options: { bold: true } },
    { text: " and minimal↔severe confusions are ≈ 0. The honest weak spot is under-calling severe.", options: {} } ],
  notes: "Don't be fooled by 45% accuracy. Because severity is ordinal, what matters is how far off the errors are — and 88% land within one level. The model essentially never confuses minimal with severe. The honest weakness, boxed in amber on the left, is that it under-calls genuinely severe cases — that's the class imbalance showing up again." });

contentSlide({ kicker: "Part 1 · What drives performance", title: "Good evidence is necessary, not sufficient",
  image: "fig3_evidence_quality_map.png", maxH: 4.2,
  takeaway: [ { text: "Takeaway.  ", options: { bold: true, color: TEAL } },
    { text: "Evidence presence ≠ predictability (r = −0.30). ‘Moving’ looks easy only through class imbalance; ‘Sleep’ stays hard despite rich on-topic evidence — symptom difficulty is intrinsic.", options: {} } ],
  notes: "We asked why some symptoms are harder than others. Plotting evidence quality against predictability, the correlation is actually slightly negative. 'Moving' looks easy only because one class dominates; 'Sleep' has plenty of on-topic evidence yet stays hard. So part of the difficulty is intrinsic to the symptom, not a retrieval problem — keep that in mind for Part 4, where reranking tries and fails to fix exactly these items." });

// ============ PART 2 — figures 4, 5 ============
contentSlide({ kicker: "Part 2 · Two paradigms", title: "Two minds, different routes",
  image: "fig4_cot_complementarity.png",
  takeaway: [ { text: "Takeaway.  ", options: { bold: true, color: TEAL } },
    { text: "A frozen LLM and the trained encoder agree only 45% of the time: ", options: {} },
    { text: "the encoder hedges to the middle, CoT commits to the extremes", options: { bold: true } },
    { text: " and recovers the severe class (F1 0.16 → 0.30).", options: {} } ],
  notes: "This is the most interesting scientific result. A frozen language model and the trained encoder reach similar accuracy, but they agree only 45% of the time. Look at the left panel: the encoder wins the middle 'mild' class — it hedges — while the LLM wins both extremes, minimal and severe. Two very different routes to the same overall score, which is exactly the signature of complementary models." });

// ---- Slide: CoT reasoning cards (native) ----
{
  const s = p.addSlide();
  s.background = { color: WHITE };
  kicker(s, "Part 2 · Two paradigms");
  title(s, "Inside the reasoning: the LLM explains its rating");
  const cards = [
    { tag: "SEVERE — RESCUED", item: "Depressed", t: 3, m: 3, enc: 1, ok: true,
      r: "“Several snippets indicate persistent feelings of depression and hopelessness… ‘happy but then I went back to being’ — maps to nearly every day.”" },
    { tag: "SEVERE — RESCUED", item: "Sleep", t: 3, m: 3, enc: 1, ok: true,
      r: "“Significant trouble sleeping — ‘I don’t sleep very well’, ‘waking up with anxiety attacks’, woken by stress and dreams.”" },
    { tag: "INSTRUCTIVE MISS", item: "Appetite", t: 3, m: 0, enc: 0, ok: false,
      r: "“No direct mention of poor appetite or overeating in the snippets…” — correct reasoning, but the retriever surfaced no appetite evidence (see Sleep/Appetite, Part 1)." },
  ];
  const y0 = 1.62, ch = 1.4, gap = 0.27, w = PW - 2 * MX;
  cards.forEach((c, i) => {
    const y = y0 + i * (ch + gap);
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: MX, y, w, h: ch, fill: { color: c.ok ? TINT : "FBEEE8" }, line: { type: "none" }, rectRadius: 0.07, shadow: shadow() });
    s.addText(c.tag, { x: MX + 0.25, y: y + 0.16, w: 2.6, h: 0.3, fontFace: BODY, fontSize: 10.5, bold: true, color: c.ok ? GOOD : CORAL, charSpacing: 1, margin: 0 });
    s.addText([{ text: "PHQ-8 · " + c.item, options: { bold: true, color: INK } }],
      { x: MX + 0.25, y: y + 0.44, w: 3.2, h: 0.4, fontFace: HEAD, fontSize: 16, margin: 0 });
    s.addText([
      { text: "True ", options: { color: MUTE } }, { text: String(c.t), options: { bold: true, color: INK } },
      { text: "    CoT ", options: { color: MUTE } }, { text: String(c.m), options: { bold: true, color: c.ok ? GOOD : CORAL } },
      { text: "    Enc ", options: { color: MUTE } }, { text: String(c.enc), options: { bold: true, color: INK } },
    ], { x: MX + 0.25, y: y + 0.95, w: 3.0, h: 0.32, fontFace: BODY, fontSize: 12.5, margin: 0 });
    s.addText(c.r, { x: MX + 3.6, y: y + 0.16, w: w - 3.9, h: ch - 0.32, fontFace: BODY, fontSize: 12.5, italic: true, color: INK, valign: "middle", margin: 0, lineSpacingMultiple: 1.03 });
  });
  s.addNotes("To make complementarity concrete — here's the language model explaining its own ratings, in its own words. The top two are severe cases it caught that the encoder rated only mild. The bottom one is an instructive failure: its reasoning is actually sound — it says there's no appetite evidence in the snippets — but that's because the retriever surfaced none. It ties straight back to Part 1: good reasoning can't compensate for missing evidence.");
}

// ============ PART 3 — figures 6,7,8,9 ============
contentSlide({ kicker: "Part 3 · Calibrate & deploy", title: "Calibrate first — then condition, don’t just blend",
  image: "fig6_calibrated_fusion.png",
  takeaway: [ { text: "Takeaway.  ", options: { bold: true, color: TEAL } },
    { text: "Temperature scaling cuts calibration error 79% (ECE 0.40 → 0.08). A scalar blend still ≈ the encoder; ", options: {} },
    { text: "only combiners that condition on item / severity beat both parents.", options: { bold: true } } ],
  notes: "Before combining models, you have to calibrate. The raw probabilities are badly overconfident — calibration error 0.40, cut to 0.08 by one temperature parameter. But here's the punchline on the right: even calibrated, a naive scalar blend still doesn't beat either parent. Only combiners that condition — on the item, or on predicted severity — do. The complementarity is structural, not something a single global weight can capture." });

contentSlide({ kicker: "Part 3 · Calibrate & deploy", title: "As a screener, the default threshold hides a usable tool",
  image: "fig7_clinical_screener.png",
  takeaway: [ { text: "Takeaway.  ", options: { bold: true, color: TEAL } },
    { text: "Default argmax ≥ 10 catches only 37% of cases, but the ranking is strong (AUC 0.78): ", options: {} },
    { text: "a tuned threshold reaches 86% sensitivity", options: { bold: true } },
    { text: " — report the metric that matches the deployment question.", options: {} } ],
  notes: "Now reframe it as a screener. The naive 'sum the argmax predictions, flag if ≥ 10' rule catches only 37% of true cases — useless clinically. But the model's ranking of who is depressed is good: AUC 0.78. So simply moving the decision threshold gets sensitivity up to 86%. The lesson for the deck: report the metric that matches the deployment question, not the default one." });

contentSlide({ kicker: "Part 3 · Calibrate & deploy", title: "It knows what it doesn’t know",
  image: "fig8_selective_prediction.png", maxH: 4.2,
  takeaway: [ { text: "Takeaway.  ", options: { bold: true, color: TEAL } },
    { text: "Abstaining on the least-confident 40% raises accuracy on the rest (0.46 → 0.52) and ", options: {} },
    { text: "defers the ambiguous severe cases to a clinician", options: { bold: true } },
    { text: " — a credible human-in-the-loop design.", options: {} } ],
  notes: "A deployable system doesn't have to answer everything. If it abstains on its least-confident 40% and routes them to a clinician, accuracy on what it does decide climbs from 0.46 to 0.52. And critically — right panel — the cases it defers are disproportionately the ambiguous moderate and severe ones, exactly who a human should review. It knows what it doesn't know." });

contentSlide({ kicker: "Part 3 · Calibrate & deploy", title: "Which gains survive uncertainty?",
  image: "fig9_bootstrap_ci.png", maxH: 4.1,
  takeaway: [ { text: "Takeaway.  ", options: { bold: true, color: TEAL } },
    { text: "Cluster bootstrap over 219 participants: ", options: {} },
    { text: "retrieval gains are unambiguous", options: { bold: true } },
    { text: "; the CoT macro-F1 edge alone is not significant — a claim to state carefully.", options: {} } ],
  notes: "With only 219 participants, point estimates need error bars. We resample participants two thousand times — a cluster bootstrap — and put a 95% interval on every headline gain. The retrieval gains are unambiguous, well clear of zero. But the CoT macro-F1 edge on its own is not statistically significant. So we say that one carefully. This slide is really about intellectual honesty for a reviewer audience." });

// ============ PART 4 — figures 10, 11 ============
contentSlide({ kicker: "Part 4 · Pushing the frontier", title: "Four targeted experiments — each wins a different metric",
  image: "fig10_bucketb_summary.png", maxH: 4.2,
  takeaway: [ { text: "Takeaway.  ", options: { bold: true, color: TEAL } },
    { text: "No single winner: ", options: {} },
    { text: "attention-MIL is best overall", options: { bold: true } },
    { text: ", focal CORN best ordinal (QWK), and reranking / balanced loss recover severe-class recall. Reranking did not fix the intrinsically hard items.", options: {} } ],
  notes: "Four targeted experiments, all run on the cluster with identical setup — only one factor changes each time. The near-diagonal of boxed cells is the whole point: there's no single winner. Attention-MIL is best overall, focal loss best on the ordinal metric, and reranking or class-balancing buy back severe-class recall. And note — reranking did not rescue the Sleep and Appetite items from Part 1, confirming that difficulty really is intrinsic." });

// ---- Slide: MIL attribution cards (native) ----
{
  const s = p.addSlide();
  s.background = { color: WHITE };
  kicker(s, "Part 4 · Pushing the frontier");
  title(s, "The model points at its evidence");
  s.addText("Attention-MIL weights each retrieved window, so we can see which transcript span drove every score.",
    { x: MX, y: 1.5, w: PW - 2 * MX, h: 0.45, fontFace: BODY, fontSize: 14, color: MUTE, margin: 0 });
  const cards = [
    { item: "Tired", t: 2, a: 88, txt: "“tired very tired and hard to focus… it just takes me longer to get things done cuz I don’t have my energy”", good: true },
    { item: "Depressed", t: 3, a: 60, txt: "“…nervous and depressed and not happy [TURN_SEP] have you noticed any changes [TURN_SEP] yes”", good: true },
    { item: "Appetite", t: 3, a: 59, txt: "“depressed [TURN_SEP] mood swings [TURN_SEP] high anxiety” — only loosely on-symptom: appetite evidence is genuinely thin.", good: false },
  ];
  const y0 = 2.1, ch = 1.28, gap = 0.26, w = PW - 2 * MX;
  cards.forEach((c, i) => {
    const y = y0 + i * (ch + gap);
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: MX, y, w, h: ch, fill: { color: c.good ? TINT : "FBEEE8" }, line: { type: "none" }, rectRadius: 0.07, shadow: shadow() });
    s.addText([{ text: "PHQ-8 · " + c.item, options: { bold: true, color: INK } }],
      { x: MX + 0.25, y: y + 0.16, w: 2.7, h: 0.4, fontFace: HEAD, fontSize: 16, margin: 0 });
    s.addText([{ text: "True severity ", options: { color: MUTE } }, { text: String(c.t), options: { bold: true, color: INK } }],
      { x: MX + 0.25, y: y + 0.62, w: 2.7, h: 0.3, fontFace: BODY, fontSize: 12, margin: 0 });
    // attention chip
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x: MX + 0.25, y: y + 0.92, w: 2.3, h: 0.28, fill: { color: c.good ? GOOD : CORAL }, line: { type: "none" }, rectRadius: 0.06 });
    s.addText(c.a + "% attention on top window", { x: MX + 0.25, y: y + 0.92, w: 2.3, h: 0.28, fontFace: BODY, fontSize: 9.5, bold: true, color: WHITE, align: "center", valign: "middle", margin: 0 });
    s.addText(c.txt, { x: MX + 3.0, y: y + 0.16, w: w - 3.3, h: ch - 0.32, fontFace: BODY, fontSize: 12.5, italic: true, color: INK, valign: "middle", margin: 0, lineSpacingMultiple: 1.03 });
  });
  s.addNotes("And finally, interpretability — the counterpart to the LLM's words. The MIL attention head tells us which transcript window drove each score. For 'Tired' it put 88% of its weight on exactly the right utterance about low energy. Honestly, for Appetite — a thin-evidence item — it can only grab a loosely related window. So the model points at its evidence when the evidence exists, and visibly struggles on the items we already know are hard.");
}

// ============ FINAL · CONCLUSIONS ============
{
  const s = p.addSlide();
  s.background = { color: DARK };
  s.addShape(p.shapes.OVAL, { x: PW - 3.2, y: PH - 3.0, w: 4.8, h: 4.8, fill: { color: TEAL, transparency: 80 }, line: { type: "none" } });
  s.addText("WHAT WE LEARNED", { x: MX, y: 0.55, w: 10, h: 0.4, fontFace: BODY, fontSize: 13, bold: true, color: MINT, charSpacing: 3, margin: 0 });
  s.addText("Strength: the retrieval pipeline and an honest clinical framing.  Frontier: the severe class.",
    { x: MX, y: 0.98, w: 12, h: 0.85, fontFace: HEAD, fontSize: 24, bold: true, color: WHITE, margin: 0, lineSpacingMultiple: 1.03 });
  const pts = [
    ["1", "Retrieval is the one rock-solid lever", "Semantic evidence — not a bigger model — drives the gain and survives the bootstrap with room to spare."],
    ["2", "Ordinally coherent, but severe-blind", "Errors stay adjacent (88% within-one); the 9.5% severe tail is the recurring bottleneck."],
    ["3", "Complementary models must be conditioned, not blended", "Encoder + CoT are right on different items; routing / MIL exploit this, a scalar blend does not."],
    ["4", "Clinically usable as a calibrated, deferring screener", "ECE 0.40→0.08, AUC 0.78, tunable to 86% sensitivity, defers its hardest cases to a human."],
    ["5", "The frontier is interpretable + severe-targeted", "Attention-MIL points at its evidence and wins overall; Sleep/Appetite difficulty is intrinsic, not retrieval."],
  ];
  const y0 = 2.1, rh = 1.0;
  pts.forEach((pt, i) => {
    const y = y0 + i * rh;
    s.addShape(p.shapes.OVAL, { x: MX, y: y + 0.04, w: 0.62, h: 0.62, fill: { color: MINT }, line: { type: "none" } });
    s.addText(pt[0], { x: MX, y: y + 0.04, w: 0.62, h: 0.62, fontFace: HEAD, fontSize: 22, bold: true, color: DARK, align: "center", valign: "middle", margin: 0 });
    s.addText([{ text: pt[1] + "   ", options: { bold: true, color: WHITE } }, { text: pt[2], options: { color: "AFCFCF" } }],
      { x: MX + 0.85, y: y, w: 11.0, h: rh - 0.12, fontFace: BODY, fontSize: 13.5, valign: "middle", margin: 0, lineSpacingMultiple: 1.0 });
  });
  s.addNotes("To wrap up — five takeaways. One: retrieval is the rock-solid lever, semantic evidence over a bigger model. Two: the model is ordinally sane but severe-blind. Three: the encoder and the LLM are complementary, but you must combine them by conditioning, not blending. Four: framed as a calibrated, deferring screener, it's genuinely clinically usable. Five: the frontier is interpretable, severe-targeted modelling. The one-line version: the strength is the pipeline and an honest framing — the frontier is the severe class. Thank you.");
}

// ============ APPENDIX · HOW EACH METHOD WORKS ============
function methodSlide(titleText, cards, notes) {
  const s = p.addSlide();
  s.background = { color: WHITE };
  kicker(s, "Appendix · How each method works");
  title(s, titleText);
  const cw = (PW - 2 * MX - 0.4) / 2, ch = 2.4, gx = 0.4, gy = 0.32;
  const x0 = MX, y0 = 1.72;
  cards.forEach((c, i) => {
    const x = x0 + (i % 2) * (cw + gx);
    const y = y0 + Math.floor(i / 2) * (ch + gy);
    s.addShape(p.shapes.ROUNDED_RECTANGLE, { x, y, w: cw, h: ch, fill: { color: TINT }, line: { type: "none" }, rectRadius: 0.07, shadow: shadow() });
    s.addText([{ text: c[0], options: { bold: true, color: INK } },
               c[1] ? { text: "   " + c[1], options: { italic: true, color: TEAL } } : { text: "" }],
      { x: x + 0.25, y: y + 0.18, w: cw - 0.5, h: 0.4, fontFace: HEAD, fontSize: 15, margin: 0 });
    s.addText(c[2], { x: x + 0.25, y: y + 0.66, w: cw - 0.5, h: ch - 0.85, fontFace: BODY, fontSize: 12, color: INK, margin: 0, valign: "top", lineSpacingMultiple: 1.04 });
  });
  if (notes) s.addNotes(notes);
}

methodSlide("Models & losses", [
  ["Encoder + CORN", "baseline", "MentalBERT reads [CLS] symptom + evidence. The CORN head predicts three ordered thresholds — “> 0?”, “> 1?”, “> 2?” — instead of a flat 4-way softmax. The ordering is why almost all errors are only one level off."],
  ["Balanced CORN", "severe recall", "Weights each example by the inverse frequency of its class, so the rare 9.5% severe items aren’t drowned out. Pushes the boundaries to catch severe cases; costs some MAE."],
  ["Focal CORN", "hard cases", "Adds a focal factor (1−p)^γ that down-weights easy, already-confident threshold decisions, so training focuses on the ambiguous boundary cases. Best ordinal agreement (QWK)."],
  ["CoT probe — Qwen2.5-7B", "frozen LLM", "A chat LLM reasons step-by-step over the same retrieved evidence, mapping it to the 0–3 frequency anchors via four few-shot demos. Decoded greedily or by a 5-sample self-consistency vote. No task training."],
], "These are the four ways of turning evidence into a score. The first three share the trained encoder and the rank-consistent ordinal head, differing only in the loss; the fourth is the frozen LLM. Note the through-line: balanced and focal CORN both exist to attack the severe class, the project's bottleneck.");

methodSlide("Architecture, evidence & fusion", [
  ["Attention-MIL", "best single model", "Encodes each retrieved window separately, then gated attention learns weights (summing to 1) over them → one ‘bag’ embedding → CORN head. The weights double as an explanation: which window drove the score."],
  ["Cross-encoder rerank", "sharper evidence", "A MiniLM cross-encoder re-scores all ~90 transcript windows against the symptom question and keeps the top-5 most diagnostic — in case the first retriever missed the telling snippet."],
  ["Calibration", "trust the confidence", "Divides the logits by one learned ‘temperature’ so predicted confidence matches real accuracy (error 0.40 → 0.08). A prerequisite for combining probabilities."],
  ["Fusion: condition, don’t blend", "best system", "Severity-routing (use the LLM’s call when it predicts severe) beats both parents; a learned stacker conditions on the symptom; a flat probability average gains nothing."],
], "And these recombine the parts. MIL changes how evidence is pooled and gives interpretability; reranking changes which evidence arrives; and fusion combines the encoder with the LLM — but only after calibration, and only by conditioning on severity or symptom, never a flat blend. That conditioning point is the single most important takeaway of the combination work.");

p.writeFile({ fileName: "phq8_final_deck.pptx" }).then(f => console.log("wrote", f));
