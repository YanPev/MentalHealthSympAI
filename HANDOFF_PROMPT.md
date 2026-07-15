# Continuation prompt for Claude Code (PHQ-8 retrieval/evidence research plan)

Paste everything below the line into a fresh Claude Code session in
`/home/yanivpev/MentalHealthSympAI` (branch `models/bert-classifier`).

---

We are mid-way through an 8-phase research plan on this repo. The approved plan
lives at `/home/yanivpev/.claude/plans/let-s-try-to-improve-abstract-bee.md` —
**read it first**, it has the full context, design decisions and phase list.
Continue executing it from where the previous session stopped. Work
autonomously: submit the SLURM jobs yourself, phase by phase, without pausing
for approval between phases.

## Project background (short)

PHQ-8 item severity prediction (8 items × 0–3) from DAIC-WOZ interviews.
Pipeline = retrieval (BM25 + semantic hybrid over context windows) → encoder
(MentalBERT + CORN / Attention-MIL) and/or LLM CoT (Qwen2.5-7B) → merged
cascade. Project best = Attention-MIL + merged-gate cascade:
**macro-F1 .409 / QWK .447 / MAE .606** (`outputs/cot/cascade_retrieval_variants/summary.csv`).
Frozen baselines to compare against: `outputs/cv/oof_predictions_ctxm_corn_hybw3.csv`
(encoder), `outputs/cot/folds_tolerant_sc5*/` (LLM pools).

The goal: expand the retrieval dictionary, add a 4-way LLM evidence-quality
judge, test missing-evidence training policies, analyse the effect, and produce
clinician-facing reports.

## Non-negotiable invariants (from the plan)

- `src/retrieval/symptom_glossary.py` (`EXPAND`), `src/evaluation/presentation/fig3_evidence_quality_map.py`,
  `outputs/cot/retrieval_window_analysis.json` and all existing OOF/metrics
  files stay **byte-identical** — they are the comparison baseline.
- Window size **w3** everywhere; use only the `src/retrieval/prod_context/`
  hybrid path (the other `hybrid_retrieval.py` has opposite alpha semantics).
- Folds: `StratifiedGroupKFold(5, shuffle=True, random_state=42)` grouped by
  `participant_id`, computed on the FULL dataset; missing-evidence policies
  apply **after** the split, to the training portion only.
- The lexicon and the evidence judge never see gold labels → globally
  applicable, no leakage. Never tune lexicon terms against OOF metrics.
  Cascade gate params (τ=0.8, diff=0.5) are reused from the frozen sweep, not
  re-tuned.

## SLURM conventions

`sbatch run_*.sbatch` from the repo root. Headers:
`--partition=gpu --gres=gpu:rtx_3090:1 --cpus-per-task=4 --mem=32G`,
`source .venv/bin/activate`, `export HF_HOME=/home/yanivpev/MentalHealthSympAI/.hf_cache`,
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. Logs → `outputs/logs/slurm_%j.out`.
Nodes have no network. Use `.venv/bin/python` for local CPU work.

## DONE so far

**Phase 1 — tiered lexicon (complete).**
- `src/retrieval/symptom_lexicon.py`: tiers `core` (imports EXPAND, never
  copies), `synonyms` (lay paraphrases + MWEs), `clinical` (DSM-5 register).
  API `expanded_query(item_name)` / `flat_terms(item_name)`. Smoke check passes
  (`python -m src.retrieval.symptom_lexicon`).
- `src/llm/brainstorm_lexicon.py` + `run_lexicon_brainstorm.sbatch`: ran as job
  19369377, **COMPLETED**. Output `outputs/cot/lexicon_brainstorm.json`
  (~30 lay + ~25 clinical suggestions per item). **These are SUGGESTIONS ONLY
  and have NOT been reviewed or merged yet** — nothing reads that file.

**Phase 2 — expanded retrieval (partially complete).**
- `src/retrieval/build_expanded_prod_hybrid.py`: builds 3 variants on the w3
  bank by feeding different query CSVs to BM25 vs semantic —
  `expbm25` (V1, BM25-only, diagnostic), `exphyb_bm25q` (V2, expanded query →
  BM25 only), `exphyb_bothq` (V3, expanded → both). Writes
  `data/processed/phq8_item_dataset_{variant}_w3.csv` using the STANDARD
  evidence column names (`retrieved_context_windows_hybrid_pack` / `_list`) so
  all downstream consumers work unmodified.
  - **V1 already built** on CPU: `phq8_item_dataset_expbm25_w3.csv` (1752 rows,
    0 empty-evidence). V2/V3 not built yet (need MiniLM → run the sbatch).
- `src/evaluation/retrieval_window_analysis.py`: extended with
  `--lexicon {glossary,expanded}`, `--dataset`, `--evidence-column`,
  `--scores-column`, `--out-tag`. It now **refuses to run non-default configs
  without `--out-tag`**, so the frozen baseline JSON can't be overwritten.
  Early signal: the expanded query already lifts keyword hit-rate a lot
  (e.g. Moving 0.64 → 0.83, Tired 0.55 → ...) on V1.
- `run_expanded_retrieval.sbatch`: builds all 3 variants + computes keyword
  hit-rates under both lexicons per variant. **Created, NOT yet submitted.**

**Phase 3 — evidence judge (code complete, not run).**
- `src/evaluation/llm_evidence_quality.py`: 4-way judge
  {supports, against, ambiguous, none} + `severity_hint` (0–3 or null) +
  reason. SC voting (3 chains @ T=0.7, ties → ambiguous). Empty evidence →
  `none` with no LLM call. Writes `outputs/cot/evidence_quality_<tag>.{csv,json}`.
  Parsing/voting unit-tested.
- `run_evidence_quality.sbatch`: array 1–4 over `r0hybw3`, `expbm25`,
  `exphyb_bm25q`, `exphyb_bothq`. **NOT yet submitted** (needs V2/V3 datasets).
- `src/evaluation/select_retrieval_variant.py`: picks **R1** = argmax mean
  informative rate (= P(supports ∪ against)) over the expanded candidates —
  label-free selection, no nested CV needed. Also anchors against the earlier
  binary audit `outputs/cot/llm_evidence_relevance.json` (which showed keyword
  hit-rate massively overcounts: NoInterest 0.995 keyword vs 0.02 LLM).

**Phase 4 — missing-evidence policies (code complete, verification pending).**
- `src/models/missing_policy.py` (new): `attach_evidence_status`,
  `apply_policy` (zero/mask/drop), `training_labels`, `wrap_loss_fn`.
  `mask` sets `label = -1` on the training copy and `wrap_loss_fn` filters those
  rows per batch (numerically identical to zero-weighting; all-masked batch → 0
  loss still attached to the graph). Unit checks pass: masked-CE == filtered-CE,
  masked-CORN == filtered-CORN.
- Wired into `src/models/cross_validate.py` and `src/models/mil_classifier.py`:
  new flags `--missing-policy {zero,mask,drop}` and `--evidence-status-csv`.
  Policy applied after the split to the train subset only; class weights use
  `training_labels()`; `evidence_status` is carried into the OOF CSV;
  `mil_classifier.run_fold` now takes an optional `loss_fn`.

## TODO — pick up here

1. **Review + merge the brainstorm suggestions.** Read
   `outputs/cot/lexicon_brainstorm.json`, keep terms that are specific to the
   symptom (drop generic/depression-wide ones and near-duplicates), and paste
   the accepted ones into the `SYNONYMS` / `CLINICAL` tiers of
   `src/retrieval/symptom_lexicon.py` with a provenance comment. Re-run
   `python -m src.retrieval.symptom_lexicon` (it asserts no cross-tier
   duplicates).
2. **Phase 2 finish:** `sbatch run_expanded_retrieval.sbatch` → builds V1/V2/V3
   (V1 will be rebuilt with the merged lexicon) + hit-rate JSONs. Verify row
   counts (1752) and empty-evidence counts.
3. **Phase 3:** `sbatch run_evidence_quality.sbatch` (array ×4, ~1.5–2 h each),
   then `python -m src.evaluation.select_retrieval_variant` → **R1**. Check
   parse rate ≥ 0.95 and that empty-evidence rows are all `none`.
4. **Phase 4 finish:** add `comprehensive_with_abstention()` to
   `src/evaluation/model_comparison_eval.py` (wrapper — leave `comprehensive()`
   untouched; note it asserts 8 items/participant at ~line 107): coverage rate,
   covered-subset item metrics, PHQ-total 0-imputed (headline) + prorated
   (mean of observed × 8, sensitivity). Then run a 1-epoch CPU/GPU sanity run of
   `cross_validate.py --missing-policy zero` and confirm the `fold` column is
   identical to the frozen baseline OOF (fold-identity is the comparability
   guarantee) **before** spending GPU on Phase 5.
5. **Phase 5 (≈9 GPU jobs):** new sbatch copies — `ctxm_corn` CV on R1 (tag
   `ctxm_corn_exphybw3`) and on V1 (`ctxm_corn_expbm25w3`); MIL on R1
   (`mil_exphybw3`); CoT probe 5-fold array SC5 on R1 →
   `outputs/cot/folds_exphybw3/` (reuse the existing R0 LLM pools, don't
   recompute); `mask` + `drop` policy runs on R1. Then CPU: cascade sweep
   (`src/evaluation/cot_cascade.py`, `cascade_variant_table_ci.py`) and the
   LLM-only-with-abstention scoring. **First re-run `comprehensive()` on the
   frozen R0 OOFs and confirm it reproduces QWK .447 / MAE .606 / mF1 .409** —
   guards against silent eval drift before quoting any deltas.
6. **Phases 6–8:** `src/evaluation/retrieval_effect_analysis.py` (R0-vs-R1
   deltas across the full metric set + bootstrap CIs),
   `src/evaluation/presentation/fig3b_retrieval_trajectory.py` (per-item arrows
   old→new retrieval, x = LLM-judged informative rate, y = per-item F1; fig3 and
   its CSV stay untouched), `src/evaluation/failure_patterns.py`,
   `src/reporting/{report_schema.py,generate_clinician_reports.py}` +
   `run_clinician_reports.sbatch`, then `docs/retrieval_method.md` and the
   deliverables assembly (see plan Phase 8).

Task list state (TaskCreate/TaskUpdate): #1 completed, #2/#3/#4 in_progress,
#5–#8 pending.
