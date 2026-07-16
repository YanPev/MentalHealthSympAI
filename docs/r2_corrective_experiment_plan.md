# R2 corrective retrieval experiment — plan

**Status:** planning complete; retrieval construction not yet started. **No GPU
job may be launched until this document is committed and all retrieval CPU /
small-GPU smoke tests pass** (per the experiment brief, §0.4).

**Date:** 2026-07-15. **Branch:** `models/bert-classifier`. **Working dir:**
`/home/yanivpev/MentalHealthSympAI`.

This is a *focused corrective follow-up* to the previous 8-phase expanded-retrieval
study. It does **not** rebuild that study. It repairs the methodological gaps in
how the retrieval configuration (previous **R1 = `exphyb_bm25q`**) was constructed
and selected, then trains only the minimum downstream models needed to close the
research story. All new artifacts live under `outputs/r2_systematic/`.

---

## 1. Research question

Can a **systematically constructed and selected** retrieval configuration (**R2**)
improve PHQ-8 item prediction beyond the frozen **R0** baseline, when (1) lay vs
clinical lexicon contributions are separated; (2) clinical terms have documented
provenance; (3) semantic queries are natural-language multi-queries, not
concatenated keyword lists; (4) BM25 and semantic retrieval are evaluated
separately before any hybrid fusion; (5) evidence-set size is chosen from
evidence quality + the encoder token budget; and (6) exactly one final R2 is used
for training?

Answered in order: **A** which lexicon tier helps → **B** BM25 vs semantic vs
hybrid → **C** evidence budget → **D** encoder effect → **E** LLM effect → **F**
encoder missing-evidence policy → **G** LLM evidence-filtering policy → **H**
encoder-first vs LLM-first routing.

---

## 2. What already exists (verified by inspection, 2026-07-15)

### 2.1 Data & folds (frozen, reused unchanged)
- **Dataset:** `data/processed/phq8_item_dataset_context_windows_hybrid_w3.csv` —
  **1752 rows = 219 participants × 8 items**. Item texts fixed in
  `src/data/phq8_items.py`. Gold `label ∈ {0,1,2,3}`; distribution
  {0:852, 1:513, 2:220, 3:167}. **Severe = gold-3 (n=167)**; per item:
  Sleep 35, Tired 28, Appetite 24, Failure 24, Concentrating 22, Depressed 18,
  NoInterest 9, Moving 7.
- **Context bank (w3):** `data/processed/context_window_bank_w3.json` — the
  window bank used everywhere. **w3 stays fixed**; window construction is *not*
  changed in this experiment.
- **Folds:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`,
  grouped by `participant_id`, stratified by `label`, computed on the FULL
  dataset (`src/models/cross_validate.py:164,178`). The canonical fold column
  lives in every OOF file (`fold ∈ {1..5}`). Every downstream comparison reuses
  these exact folds.

### 2.2 Retrieval stack (`src/retrieval/prod_context/`)
- **BM25** (`bm25_context_retriever.py`): Okapi BM25, **k1=1.5, b=0.75**, custom
  tokenizer (lowercase, strip non-alnum, 20-word stoplist), participant-local,
  turn-overlap diversity (`max_turn_overlap_ratio=0.5`), `candidate_multiplier=5`.
  Query = the row's `item_text`. Zero-score fallback returns first-k windows.
- **Semantic** (`semantic_context_retriever.py`): `sentence-transformers/all-MiniLM-L6-v2`,
  **normalized embeddings + dot product = cosine**. **The query is a single
  encoded string** (`item_text`); window embeddings encoded once per participant;
  score = `window_emb @ query_emb`. Same diversity/candidate settings.
- **Hybrid** (`hybrid_context_retriever.py`): takes *separate* pre-computed BM25
  and semantic JSONs, unions candidate window-ids, fills the missing modality
  with 0, **per-row min-max normalizes each modality over the candidate union**
  (`min_max_normalize`: if min==max → all 0), then
  `hybrid_score = α·bm25_norm + (1−α)·semantic_norm`, **α=0.5**, k=5, diversity
  0.5. This is the normalization/fusion implementation R2 will reuse unchanged.
- **Query-expansion trick** (`build_expanded_prod_hybrid.py`): the expansion is
  injected by writing the expanded string into the query CSV `item_text`; BM25
  and semantic can receive different query CSVs. This is how the previous R1 fed
  an expanded BM25 query while keeping a bare semantic query.

### 2.3 Lexicon (`src/retrieval/`)
- **`symptom_glossary.py` — `EXPAND`**: the frozen original glossary, 8 items,
  ~15 terms/item. **This is Core (L0). It must stay byte-identical.**
- **`symptom_lexicon.py`**: 3 tiers — `core` (imports `EXPAND`, never copies),
  `synonyms` (lay paraphrases/MWEs), `clinical` (DSM-register). ~506 unique
  terms total. **`clinical` was augmented by an offline Qwen brainstorm pass and
  merged with only light review** — this is the provenance defect R2 fixes.
  Flagged terms present in `clinical`: *imposter syndrome, inferiority complex*
  (Failure); *cynicism, passivity* (Depressed); *hyperactivity, bradykinesia*
  (Moving); *underarousal* (NoInterest) — all tagged "reviewed brainstorm
  additions", i.e. LLM-sourced.

### 2.4 Evidence judge (`src/evaluation/llm_evidence_quality.py`)
- 4-way {supports, against, ambiguous, none} + `severity_hint` + reason, over the
  *combined* evidence set (`format_evidence` of the window list). SC majority
  vote. **Previous config: `Qwen/Qwen2.5-7B-Instruct`, n-samples 3, T=0.7.**
- **Defect:** the judge shares the **same model family (Qwen2.5)** as the PHQ
  predictor, violating judge/predictor independence. It is also **set-level only**
  (no per-window judgments) and used T=0.7.

### 2.5 Encoder (`src/models/cross_validate.py`, `ordinal.py`)
- **MentalBERT + CORN**: `mental/mental-bert-base-uncased`, `--loss corn` (K−1=3
  ordinal head, `corn_predict` = #thresholds with P(y>k)>0.5), **5 folds, 8
  epochs, batch 16, max_length 256, lr 2e-5 (AdamW), seed 42**. Input =
  sentence-pair `[CLS] item_text [SEP] evidence [SEP]`, `truncation="only_second"`
  (trims evidence, keeps item), evidence column
  `retrieved_context_windows_hybrid_pack` (windows joined ` [WINDOW_SEP] `).
  OOF schema `participant_id,item_id,item_name,label,prediction,fold,prob_0..3`.
- **Attention-MIL** (`mil_classifier.py`): batch 8, max_length 128, 8 epochs,
  seed 42. **NOT retrained in R2** (brief §2.4).
- **Missing policies** (`missing_policy.py`): `zero` = **status-quo, keeps the
  gold label** (no-op — it does *not* relabel to 0); `mask` = label→−1 on the
  training copy, filtered from the loss (≡ zero-weighting); `drop` = rows removed
  from the training subset + class-weight calc. Applied **after** the split, to
  train only; validation untouched.

### 2.6 LLM predictor (`src/llm/cot_joint.py`)
- `Qwen/Qwen2.5-7B-Instruct`, **bf16, no quantization**, single-GPU, native 32K
  context (no truncation in staged mode), **`--mode staged --tolerant`,
  max_new_tokens=1100, self-consistency 5 chains @ T=0.7 / top_p=0.9**.
- Staged flow: confident items → clinical context (from confident only) →
  difficult items → final 8 scores. Tolerant = optimized for off-by-≤1 (hedge to
  1–2 when uncertain, extremes only on clear evidence).
- Per item: plurality vote over chains, `prob_0..3` = vote fractions (graded
  confidence), lower-label tie-break; `difficult_frac` = fraction of chains that
  self-flagged the item difficult. Evidence column
  `retrieved_context_windows_hybrid_list`; eval slice = the `fold` column of
  `oof_predictions_ctxm_corn_hybw3.csv`. **Generations are recomputed per
  retrieval variant (not cached).**
- **Long-context infra exists:** `cot_probe.py --full-transcript` uses
  `fit_transcript` (deterministic head+tail, drop-middle) at max-context 4096;
  `cot_joint.py --full-transcript-items` appends a head+tail-truncated transcript
  (6000 tok) for named thin items. Median transcript ≈ 1.3k tokens.

### 2.7 Cascade (`src/evaluation/cot_cascade.py`)
- **LLM-first, encoder breaks the top-2 tie.** `merged` gate routes an item iff
  `(max LLM vote-fraction < τ) OR (difficult_frac ≥ diff_thresh)`, **τ=0.8,
  diff=0.5**, ~**41.5% routed**; routed items: encoder picks the higher-prob of
  the LLM's top-2 classes (never a third class). Join on `(participant_id,
  item_id)`; encoder OOF supplies `label` + `fold`.
- **Leakage note:** (0.8, 0.5) were **chosen because they were best on this OOF**
  (`--sweep` grid on the same rows), then frozen. Low capacity (2 global scalars)
  but *not* nested-CV clean. **R2 cascades must fit routing within training folds.**

### 2.8 Metrics & bootstrap (`src/evaluation/model_comparison_eval.py`, `retrieval_effect_analysis.py`) — reused verbatim
- `comprehensive(df)` (asserts 8 items/participant): Accuracy, Macro-F1
  (labels [0-3], zero_division=0), **QWK = quadratic-weighted Cohen κ**, MAE,
  per-class F1 (incl. **class-3 F1**), top-2 acc, off-by-one, total-score metrics,
  threshold/band analyses.
- **Severe metrics** (`severe_calling`, severe = **class 3**): `severe_recall =
  P(pred=3|true=3)`, `false_severe_rate = P(pred=3|true<3)`, undercall, signed err.
- `comprehensive_with_abstention(df)`: coverage rate, covered-subset item metrics,
  **headline 0-imputed** total + **prorated** total (for the LLM filter/abstain
  conditions where `prediction` may be NaN).
- **`_boot_delta`**: participant-cluster bootstrap, **B=2000, seed=42**, paired on
  participant×item, percentile 95% CI `[2.5, 97.5]`. "Significant" ⇔ CI excludes 0.
- **CCC is NOT implemented** anywhere in the repo → per brief §11.2 ("CCC if
  already supported reliably"), **CCC is omitted**; total-score agreement uses
  `total_qwk`, `total_mae`, `pearson_r`, `spearman_rho` (already computed).

---

## 3. What the previous experiment learned (baseline to beat)

From `outputs/cot/retrieval_effect_analysis.json` + `docs/results_summary.md`:

- **R0 → R1 (`exphyb_bm25q`) encoder (MentalBERT+CORN):** macro-F1 .347→.372,
  QWK .354→.384, MAE .680→.671, class-3 F1 .159→.223, severe-recall .102→.150.
  **Every bootstrap CI straddles 0** (macro-F1 [-0.010,+0.057], QWK
  [-0.022,+0.078]) — the retrieval move is *not* statistically significant.
- **LLM R0→R1:** macro-F1 .400→.410, QWK .406→**.397** (down), MAE up. CIs straddle 0.
- **Cascade (MIL merged) R0→R1:** **regresses** — QWK .447→.410, MAE .606→.640
  (MAE CI [+0.002,+0.066] excludes 0 → significantly worse). So R1 *hurt* the
  project-best cascade.
- **Missing policy (R1 encoder):** `drop` is clearly best (macro-F1 .393, QWK
  .421, MAE .645, severe-recall .192) vs `zero` (.372/.384/.671) and `mask`
  (.364/.371/.684).
- **Evidence quality:** mean informative rate 0.287→0.294 (+0.6pt); gains
  concentrated on Concentrating (+.036), NoInterest (+.027), Moving (+.018).
- **Project-best (frozen):** Attention-MIL + merged cascade **macro-F1 .409 / QWK
  .447 / MAE .606** on **R0** retrieval.

**Interpretation:** the previous expansion helped keyword coverage and weak-item
evidence a little, helped the standalone encoder's severe classes a little, but
never significantly, and *degraded* the cascade. And its selection was
methodologically weak — which is what R2 corrects.

---

## 4. Methodological gaps R2 must resolve

| # | Gap in previous R1 | R2 correction | Stage |
|---|---|---|---|
| G1 | Lay + clinical tiers merged; contribution of each unknown | Separate **L0/L1/L2/L3**, compare each | A |
| G2 | Clinical tier augmented by **LLM brainstorm**; ~7 non-provenanced terms | **Provenance audit**; drop LLM-only terms; `related_nonspecific` excluded from primary clinical query | A |
| G3 | Semantic query = **concatenated keyword list** (V3) | **Natural-language multi-queries**; SemanticScore = **max** cosine over queries; separate poles for bipolar items | B |
| G4 | Semantic-only **never evaluated**; selection only over {BM25-only, 2 hybrids} | Evaluate **BM25-only AND semantic-only** for every lexicon first | C |
| G5 | Hybrid assumed; α never justified | **Complementarity analysis** decides if hybrid is needed at all; α∈{.25,.5,.75} only if justified | G |
| G6 | Evidence budget fixed at **K=5**; never selected | Select budget from evidence quality + **encoder TOKEN_BUDGET** (Top-3/Top-5/budget) | C/E/H |
| G7 | **Judge = Qwen2.5**, same family as predictor | **Independent judge** — MentaLLaMA (LLaMA-2 family), T∈[.3,.5] | D |
| G8 | Judge set-level only | **Per-window + set-level** judgments | D |
| G9 | R0 excluded from the argmax; R1 = argmax over expanded only | Explicit **label-free selection** with defined tie-breakers, R0 in contention | H |
| G10 | Cascade thresholds picked on OOF | **Fit routing within training folds** (leakage-safe) | L |
| G11 | `zero` policy mislabeled "absent→0" in the report | Correct terminology: **status_quo / mask_none / drop_none** | I |

---

## 5. Evidence-judge model decision (independence constraint)

**Requirement (brief §6, constraint 11):** the judge must be a *different model
family* from the PHQ predictor (Qwen2.5-7B-Instruct). The brief's preferred judges
(Llama-3.1-8B, Mistral-7B-v0.3, Gemma-2-9B) are **not** in the offline HF cache,
and the SLURM nodes have **no network**.

**Locally cached instruct models** (`.hf_cache/hub/`): `Qwen2.5-7B-Instruct`
(=predictor, disqualified), `Qwen2.5-1.5B-Instruct` (same family, disqualified),
and two non-Qwen options — **`klyang/MentaLLaMA-chat-13B`** and
**`klyang/MentaLLaMA-chat-7B`** (LLaMA-2 family, mental-health instruction-tuned,
4096 ctx, both fully downloaded).

**Attempted independent judge → FAILED validation.** `MentaLLaMA-chat-7B` and
`-13B` are the only non-Qwen instruct models cached. Empirical validation (see
`outputs/r2_systematic/judge/window_judgments_qcheck*.csv`) showed **both are
unsuitable for the 4-way evidence task**:
- **Valence inversion:** both label *"still enjoys sports / enjoys cooking / enjoys
  going out with friends"* as **supports** for loss-of-interest (severity 3),
  when enjoyment is evidence **against**. A worked few-shot `enjoy→against` example
  did not correct it.
- **Degenerate distributions:** 7B → 79% supports / ~6% none (no dynamic range to
  distinguish configs); 13B → 73% ambiguous. The prior Qwen judge had ~36% none.
- **13B also computationally infeasible:** ~10 min float32 load + ~0.45 judgments/s;
  the 88k-judgment run is impractical even sharded.

**Decision (user-approved, 2026-07-15): use `Qwen/Qwen2.5-7B-Instruct` as the
judge, relaxing constraint 11**, with the limitation reported prominently. This is
a *documented, explicit* substitution — not a silent one:
- **Limitation:** the judge shares the PHQ predictor's model family, so gap **G7
  (judge/predictor independence) is NOT fixed** by R2 — it is inherited from the
  previous experiment. All judge-derived results (R2 selection, missing-evidence
  status, evidence-status stratification) carry this caveat. The other six
  methodological corrections (G1–G6, G8–G11) are unaffected.
- The judge still sees **no gold labels**, uses SC3 @ T=0.4, the improved
  few-shot 4-way prompt, and a chat-template-aware formatter (Qwen's native
  template). Qwen 7B bf16 fits a 24 GB rtx_3090.
- Alternatives considered and recorded: stop at the boundary (rejected — user
  chose to proceed); keyword/cross-encoder relevance proxy (rejected — not a 4-way
  judge). A genuine independent judge (Llama-3.1-8B/Mistral-7B/Gemma-2) would
  require network access to download.

**Judge config (new, brief §6):** SC **3 samples @ T=0.4**, majority vote,
ties→ambiguous, JSON-only, robust regex fallback; distinguishes participant vs
interviewer/third-party evidence; keyword presence ≠ evidence; `severity_hint`
only when frequency explicit. Raw generations + parsed + vote counts saved.

---

## 6. TOKEN_BUDGET (encoder evidence budget, brief §6.3)

Computed from the **actual MentalBERT tokenizer + configured `max_length=256`**,
per item, reserving: `[CLS]` + tokenized `item_text` + `[SEP]` + `[SEP]` + a fixed
**safety margin (8 tokens)**. Evidence budget = `256 − reserved`. Windows are added
in rank order until the *next whole window* would exceed the budget; **no window
is silently truncated** (matching the encoder's `truncation="only_second"` reality
that anything past 256 is dropped). Stored per (participant, item, retriever):
selected window count, total evidence tokens, unused capacity, excluded-next-window
length. This "budget" prefix is the third judged/評価 set alongside Top-3 and Top-5.

---

## 7. Files to be added / changed

**Preserved byte-identical (never edited):** `symptom_glossary.py`; all frozen OOF
CSVs (`oof_predictions_ctxm_corn_hybw3.csv`, `mil_hybw3.csv`, R1 variants);
`folds_tolerant_sc5*/`; `folds_exphybw3/`; `evidence_quality_*.json`;
`retrieval_effect_analysis.json`; `retrieval_window_analysis.json`;
`docs/results_summary.md`, `docs/retrieval_method.md`, `docs/results_report.html`.
`symptom_lexicon.py` is **left as-is** (a frozen record of the old R1 lexicon); R2
uses new modules instead.

**New source:**
- `src/retrieval/clinical_lexicon_curated.py` — audited clinical tier (L2/L3).
- `src/retrieval/lay_lexicon.py` — manifested lay tier (L1/L3) (reuses reviewed
  terms from `symptom_lexicon.SYNONYMS`, with an explicit manifest).
- `src/retrieval/r2_lexicons.py` — assembles L0/L1/L2/L3 configs + `flat_terms`.
- `src/retrieval/semantic_queries.py` — natural-language multi-queries per item
  (Q0/Q1/Q2, bipolar poles) + builder for `semantic_queries.json`.
- `src/retrieval/prod_context/semantic_multiquery_retriever.py` — semantic
  retriever variant that scores each window by **max** cosine over a query set and
  records per-query scores + winning query.
- `src/retrieval/build_r2_rankings.py` — Stage C: Top-20 BM25-only + semantic-only
  rankings per lexicon, full per-window score records.
- `src/evaluation/judge_windows.py` — Stage D: MentaLLaMA per-window + set-level
  judge (independent family), TOKEN_BUDGET set assembly.
- `src/evaluation/r2_retrieval_metrics.py` — Stages E/F/G/H: per-window + set
  metrics, redundancy, complementarity, lexicon shortlist, hybrid decision, R2
  selection → `r2_selection.json`.
- `src/evaluation/r2_encoder_eval.py`, `r2_llm_eval.py`, `r2_complementarity.py`,
  `r2_cascade.py` — Stages I/J/K/L (reuse `comprehensive*` + `_boot_delta`).
- `src/evaluation/build_r2_report.py` — final HTML report + `run_manifest.json` +
  `status_report.md`.
- Tests: `tests/test_semantic_queries.py` (brief §4 assertions),
  `tests/test_r2_lexicons.py`, `tests/test_token_budget.py`.

**New docs:** `docs/r2_clinical_lexicon_method.md`, `docs/r2_semantic_query_method.md`,
`docs/r2_retrieval_selection.md`.

**New sbatch:** `run_r2_judge.sbatch` (array over lexicon×retriever), the encoder
runs (`run_r2_encoder.sbatch` — E0/E1/E2 on R2), the LLM runs
(`run_r2_llm.sbatch` — L1..L5 folds), all mirroring the frozen hyperparameters,
only the dataset/evidence changing.

**Output tree:** `outputs/r2_systematic/{lexicons,semantic_queries,retrieval,judge,
encoder,llm,cascade,analysis,logs}/` (created).

---

## 8. Jobs & dependencies (compute plan)

```
A lexicon build + clinical audit         CPU (no job)            ── gates everything
B semantic multi-queries + tests         CPU (MiniLM local)      ── smoke on login/CPU
C BM25 + semantic Top-20 rankings (L0-L3) CPU/MiniLM              ── smoke: 1 participant
   └─ smoke tests must pass ──►  (only now may GPU jobs start)
D judge (MentaLLaMA 7B)  run_r2_judge     GPU array               ── the big job
   per-window: Top-10 BM25 + Top-10 sem per (part,item) for L0-L3
   set-level : Top-3 / Top-5 / TOKEN_BUDGET
E-H metrics + complementarity + SELECT R2 CPU                     ── label-free, one R2
   └─ exactly ONE R2 frozen ──►
I encoder MentalBERT+CORN on R2  run_r2_encoder  GPU (3 runs: E0/E1/E2)
J LLM staged-tolerant SC5 on R2  run_r2_llm      GPU (folds×conditions)
   L5 long-context: one strategy, fit-fraction reported first
K complementarity  CPU     L cascades (fit-in-fold)  CPU
Final report + manifest    CPU
```

**Judge compute estimate:** ~1752 (part,item) × [≤10 BM25 + ≤10 semantic unique
windows ≈ 12–16 after dedup] per-window judgments + 1752 × {Top-3,Top-5,budget}
set judgments, × 4 lexicons for per-window (but set-level only for shortlisted
lexicons), × SC3. Batched short-JSON generations on MentaLLaMA-7B. Run as a fold-
or lexicon-sharded array; **log any subsampling/cap explicitly** (brief §15).
To bound cost: per-window judging is the diagnostic that drives the BM25-vs-
semantic complementarity decision, so it is run for **L0 + the ≤2 shortlisted
lexicons**, not all four (documented in the selection doc). Set-level judging (the
selection driver) is run for the shortlisted lexicons × {BM25, semantic, and hybrid
only if Stage G justifies it}.

---

## 9. Selection protocol (label-free, defined BEFORE any downstream metric)

**Primary criterion:** **set informative rate = P(final status ∈ {supports,
against})** from the independent judge. Keyword hit-rate is diagnostic only.

- **Lexicon shortlist (Stage F):** rank L0–L3 by **Top-5 set informative rate**
  (BM25-only and semantic-only). Tie tolerance **0.005**. Among ties → lower none,
  lower ambiguous, lower redundancy, simpler lexicon. Keep ≤2 lexicons. An
  expansion is retained only if it lifts informative ≥0.005, or none-rate ≥0.01,
  or a predefined weak item (NoInterest / Moving / Concentrating) materially.
- **Hybrid decision (Stage G):** pick a **single retriever** if it beats the other
  by ≥0.01 informative without worsening none by >0.01 and the other has few unique
  informative wins. Test hybrid (α∈{.25,.5,.75}) only when both retrievers show
  meaningful unique wins. If hybrid improves <0.005 over the best single, keep the
  single retriever.
- **Budget (Stage H):** if Top-5 vs TOKEN_BUDGET differ <0.005 informative and
  <0.01 none → choose **Top-5**.
- **R2 (Stage H):** highest set informative rate; ties (≤0.005) → lower none →
  lower ambiguous → lower redundancy → fewer windows → simpler retriever → α=0.5.
  **Downstream PHQ metrics never revise R2.**

R2 may legitimately turn out to be **L0/Core** or a single retriever — that is an
acceptable, documented outcome (brief §15.2).

---

## 10. Downstream selection rules (defined now, before observing PHQ metrics)

- **E_FINAL** (encoder policy, Stage I): primary QWK then Macro-F1. Constraints:
  MAE not worse by >0.02 without a primary gain; severe-recall not materially down;
  false-severe up ≤0.015; policy must not depend on pathological removal of severe
  rows (audited in `missing_policy_audit.csv`).
- **L_FINAL** (LLM policy, Stage J): primary QWK then Macro-F1; same MAE/severe/
  false-severe constraints; compute reported but secondary.
- **Cascade acceptance** (Stage L): accept only if QWK > better component,
  Macro-F1 down ≤0.005, MAE up ≤0.02, severe-recall not down, false-severe up
  ≤0.015, and paired participant bootstrap shows no clear regression. Routing uses
  **one** encoder-confidence threshold, **one** LLM vote-margin threshold, **one**
  item-grouping rule, **one** evidence-status rule, all **fit inside training
  folds** (no OOF tuning). If neither cascade passes → best single model is the
  recommendation.

---

## 11. Non-negotiable invariants (brief §2)

1. R0 and all baseline artifacts preserved unchanged (§7 list).
2. Exact existing participant folds for every downstream comparison.
3. **No gold labels** select lexicon / BM25-vs-semantic / α / budget / R2.
4. **Attention-MIL not retrained.** 5. Clinician reports not regenerated.
6. Presentation figures not recreated (≤1 summary figure if needed).
7. Not every model on every candidate. 8. Retrieval selection completes first.
9. Exactly one R2 before training. 10. MentalBERT+CORN and the existing Qwen
predictor only. 11. Judge family ≠ predictor family. 12. No silent substitution of
models / folds / hyperparameters / metrics. 13. Every selection rule defined before
observing downstream PHQ metrics (this document does so).

---

## 12. Explicit stopping rules (brief §15, §18)

- No prediction model trained until **R2 is selected**.
- If expanded lexicons fail the retrieval thresholds → **keep Core (L0)**.
- If one retriever dominates and hybrid < 0.005 gain → **skip hybrid downstream**.
- Top-5 ≈ TOKEN_BUDGET → **use Top-5**. Don't train per budget. Don't rerun R0
  models (valid frozen OOF exists). Don't rerun MIL. No clinician reports.
- **Hard stops (preserve artifacts, report the boundary):** no independent judge
  model *(not triggered — MentaLLaMA available)*; clinical sources unavailable so
  the clinical tier can't be validated; folds not reproducible; essential data
  missing; a baseline metric not reproducible; a decision outside this spec.
- **Early candidate stop (after 2 folds)** only if clearly inferior on *both*
  primary metrics *and* worse MAE, non-marginal — documented each time.
- Continue-guards: stop a candidate on failed fold identity, duplicate OOF rows,
  label mismatch, missing participant coverage, unrecoverable parse failure, or an
  invalid judge/predictor family overlap.

---

## 13. Required outputs (brief §16) — target paths

`outputs/r2_systematic/lexicons/{lexicon_manifest.csv, clinical_audit.csv}` ·
`semantic_queries/semantic_queries.json` ·
`retrieval/{retrieval_window_scores.parquet|csv, all_candidates.csv,
retrieval_metrics_overall.csv, retrieval_metrics_by_item.csv,
retrieval_complementarity.csv, retrieval_redundancy.csv, r2_selection.json}` ·
`encoder/{missing_policy_audit.csv, encoder_metrics_overall.csv,
encoder_metrics_by_item.csv}` · `llm/{llm_metrics_overall.csv,
llm_metrics_by_item.csv}` · `analysis/{model_complementarity.csv,
complementarity_by_item.csv, complementarity_by_evidence.csv,
complementarity_by_severity.csv, paired_bootstrap_ci.csv}` ·
`cascade/cascade_metrics.csv` · `{run_manifest.json, status_report.md,
results_report.html}`. Report = 17 decision-oriented sections (brief §17); every
claim tagged with config + source artifact + label-free? + within-fold? + CI;
no "significant" when the CI includes 0.

---

## 14. First checkpoint

Stage A (lexicons) → Stage B (semantic queries + tests) → Stage C (rankings +
smoke). **Only after all retrieval CPU/small-GPU smoke tests pass** does
`run_r2_judge.sbatch` (the first GPU job) launch. This document is the gate.
