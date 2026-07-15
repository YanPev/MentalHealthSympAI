# Expanded retrieval + evidence-quality method

This documents the retrieval-improvement work (research-plan phases 1–5) and how
the pieces avoid cross-validation leakage.

## 1. Tiered symptom lexicon

`src/retrieval/symptom_lexicon.py` replaces the single hand-curated glossary with
a three-tier, provenance-tagged dictionary per PHQ-8 item:

| tier | source | example (Sleep) |
|---|---|---|
| `core` | imported verbatim from the frozen `EXPAND` glossary (`symptom_glossary.py`) — never copied, so the old keyword-hit-rate metric stays comparable | insomnia, waking up early, sleeping too much |
| `synonyms` | general synonyms / paraphrases / multi-word expressions | can't stay asleep, tossing and turning, lying awake |
| `clinical` | DSM-5 / clinical register | initial insomnia, terminal insomnia, hypersomnia, non-restorative sleep |

The clinical tier was **hand-curated from DSM-5 construct definitions and then
augmented by one offline Qwen-2.5-7B brainstorming pass** (`src/llm/brainstorm_lexicon.py`,
`run_lexicon_brainstorm.sbatch`); the model's ~230 suggestions were manually
reviewed and only symptom-specific terms merged in. The result is **506 unique
terms (~63/item)** across the three tiers, up from the ~15/item glossary.

API: `expanded_query(item_name)` → the space-joined query string;
`flat_terms(item_name)` → the term list (multi-word expressions kept intact) used
for keyword-hit-rate matching.

### Why this is leakage-free
The lexicon is derived only from the PHQ-8 / DSM-5 symptom *constructs* — never
from DAIC-WOZ labels or any validation-fold transcript. It is therefore a single
global artifact that is fold-independent by construction. The rule enforced: term
lists are never tuned against out-of-fold metrics.

## 2. BM25 → hybrid integration

The expansion is injected as **query expansion**: the expanded query is written
into the dataset's `item_text` column and fed through the untouched production
retrieval code (`src/retrieval/prod_context/`, min-max hybrid fusion, α=0.5,
diversity control). Because BM25 and the semantic retriever take *separate* query
files, we could isolate the BM25 lever. `src/retrieval/build_expanded_prod_hybrid.py`
builds three variants on the W3 window bank:

| variant | BM25 query | semantic query | fusion |
|---|---|---|---|
| **V1** `expbm25` | expanded | — | none (BM25 only) |
| **V2** `exphyb_bm25q` | expanded | bare item text | hybrid |
| **V3** `exphyb_bothq` | expanded | expanded | hybrid |

**Decision: expanded BM25 is integrated into the hybrid fusion (V2/V3), not run
standalone.** V1 (BM25-only) was the *worst* on judged evidence quality,
confirming the semantic component is still needed; expanding the semantic query
too (V3) did not beat expanding only BM25 (V2).

## 3. Label-free variant selection

`src/evaluation/select_retrieval_variant.py` picks the winner by **mean
LLM-judged informative rate** — `P(supports ∪ against)` from the evidence judge —
which uses no gold labels, so no nested CV is required:

| variant | informative rate | no-evidence rate |
|---|---|---|
| R0 `hybw3` (baseline) | 0.2873 | 0.3956 |
| V1 `expbm25` | 0.2686 | 0.4533 |
| **V2 `exphyb_bm25q` → R1** | **0.2937** | 0.3597 |
| V3 `exphyb_bothq` | 0.2925 | 0.3483 |

**R1 = `exphyb_bm25q`.** The aggregate lift is modest (+0.6pt informative), with
per-item gains concentrated on Concentrating (+0.036), NoInterest (+0.027), and
Moving (+0.018) — the items the keyword-vs-LLM audit had flagged as poorly
covered.

## 4. Evidence-quality judge

`src/evaluation/llm_evidence_quality.py` (Qwen-2.5-7B, SC×3 majority vote)
classifies every (participant, item) pair's retrieved evidence into one of
`{supports, against, ambiguous, none}` plus a `severity_hint`. It sees only the
item text and the retrieved windows — never the gold label — so, like the
lexicon, one global pass is fold-safe. Empty-evidence rows are assigned `none`
deterministically without an LLM call.

This replaces the keyword hit-rate proxy, which the earlier audit showed
massively over-counts relevance (NoInterest 0.995 keyword vs 0.02 LLM-judged).
The judge output feeds three consumers: the NaN training policies, the fig3b
evidence-vs-performance figure, and the clinician report's per-item evidence
status.

## 5. Missing-evidence training policies

`src/models/missing_policy.py` + the `--missing-policy {zero,mask,drop}` flag on
`cross_validate.py` / `mil_classifier.py`. Folds are computed once on the full
dataset (StratifiedGroupKFold, seed 42); the policy is applied **after the split,
to the training portion only**, so fold assignments and validation rows are
identical across policies and no evidence assessment ever touches a held-out row.
Gold-label validators are untouched — "NaN" exists only as a training loss-mask
(`mask`), a dropped training row (`drop`), or an abstained prediction in the
LLM-only report path.

## 6. Artifacts

- Datasets: `data/processed/phq8_item_dataset_{expbm25,exphyb_bm25q,exphyb_bothq}_w3.csv`
- Judge: `outputs/cot/evidence_quality_{r0hybw3,expbm25,exphyb_bm25q,exphyb_bothq}.{csv,json}`
- Selection: `outputs/cot/retrieval_variant_selection.{csv,json}`
- Keyword hit-rate (old + new lexicon): `outputs/cot/retrieval_window_analysis_*_{glossary,expanded}.json`
- SLURM: `run_lexicon_brainstorm.sbatch`, `run_expanded_retrieval.sbatch`,
  `run_evidence_quality.sbatch`, `run_cv_expanded.sbatch`, `run_mil_expanded.sbatch`,
  `run_cot_probe_expfolds*.sbatch`, `run_cv_policies.sbatch`
