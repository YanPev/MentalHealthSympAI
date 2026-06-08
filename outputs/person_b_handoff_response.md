# Person B → Person A: Handoff Response (Context-Window Evidence)

Branch: `models/bert-classifier` (training + evaluation pipeline) · datasets from `data/context-windows`
Date: 2026-06-08

This answers your Section 12 open questions and documents the exact setup used to
evaluate your context-window evidence variants. Results tables follow in
`context_window_results.json` / the HTML report once the GPU runs complete.

---

## A. Answers to your open questions (Section 12)

**Previous experiment dataset_path / evidence_column**
- The prior reported results used `data/processed/phq8_item_dataset_full.csv` with
  `evidence_column = retrieved_utterances`.
- For THIS comparison I switched the baseline to your specified
  `data/processed/phq8_item_dataset_full_bm25.csv` (`retrieved_utterances`) so the
  baseline and your new BM25/hybrid context windows share the same retrieval base.

**Was `retrieved_utterances` TF-IDF or BM25?**
- `phq8_item_dataset_full.csv` → TF-IDF retrieval.
- `phq8_item_dataset_full_bm25.csv` → BM25 retrieval. I use the **BM25** one as the
  baseline here for a fair comparison to your BM25/hybrid context windows.

**Full training command / config (the "previous best", reused verbatim except dataset/column):**
```
python -m src.models.cross_validate \
  --model-name bert-base-uncased \
  --dataset-path <DATASET> --evidence-column <COLUMN> \
  --k-folds 5 --num-epochs 8 --batch-size 16 --max-length 256 --seed 42 \
  --loss {cross_entropy --class-weights balanced | corn} --tag <TAG>
```

**Does the training script accept dataset_path and evidence_column as arguments?**
- Yes. Both `src/models/train_transformer_classifier.py` and
  `src/models/cross_validate.py` take `--dataset-path` and `--evidence-column`.
  Running your variants needed **zero code changes** — only those two flags.

**max_length / truncation strategy / is item_text preserved / is only evidence truncated?**
- `max_length = 256` (matching the previous best; same for all conditions so the
  comparison is fair).
- Truncation is `truncation="only_second"` in `src/models/input_formatting.py`:
  the input is the pair `[CLS] item_text [SEP] evidence [SEP]`, and **only the
  evidence (second segment) is truncated. `item_text` is always preserved in full.**
- Note: your context packs are richer, so at 256 tokens they are **heavily
  truncated** (measured below). The 256 run is the fair same-budget comparison
  you asked for; a `max_length=512` follow-up is run to give the richer windows a
  fair chance.

**Truncation rate at max_length=256 (evidence-only):**

| Condition | % examples truncated | median evidence tokens | p95 tokens |
|---|---|---|---|
| Old BM25 utterances | 13.5% | 102 | 363 |
| BM25 W3 | 55.9% | 258 | 414 |
| Hybrid W3 | 69.6% | 295 | 425 |
| BM25 W5 | 82.4% | 351 | 458 |
| Hybrid W5 | 94.5% | 377 | 468 |

Implication: at 256 the W5 variants lose most of their content. Interpret the
256 results as a same-budget test; the 512 follow-up tests the richer context.

**loss_type / how class weights are computed / train-only / overall or per-item?**
- Three loss settings are run: plain CE, **weighted CE**, and **CORN** (ordinal).
- Class weights = sklearn `compute_class_weight("balanced", ...)` =
  inverse-frequency over the 4 classes.
- Computed **from the training split only** (per fold, from that fold's training
  rows) — never from validation/test, no leakage.
- Computed **overall (pooled across items), not per item.**

**Which seeds were used?**
- Headline CV uses `seed=42`. The earlier robustness study used 5 seeds
  `[7, 31, 42, 123, 2024]` (model choice was a tie across all).

**Are predictions saved per run? Per-item metrics available?**
- Yes. Each CV run writes out-of-fold predictions for **all 1,752 items**
  (`outputs/cv/oof_predictions_<tag>.csv`) with `participant_id, item_id,
  item_name, label, prediction, fold, prob_0..prob_3`. Per-item metrics are
  computed from these.

**Is QWK implemented?**
- Yes (added for this handoff) — quadratic-weighted Cohen's kappa at both the
  item level and the reconstructed-total level.

**Is reconstructed PHQ-8 total score implemented? Clinical threshold (≥10)?**
- Yes (added for this handoff). Per participant I sum the 8 predicted item labels
  → predicted total, compare to the true total, and report total MAE, total QWK,
  Pearson r, and **threshold ≥ 10** sensitivity / specificity / balanced accuracy
  / F1. (`src/evaluation/context_window_eval.py`.)

**Can the same setup be rerun on the new context-window datasets?**
- Yes — that is exactly what this deliverable does (5 evidence conditions ×
  {weighted-CE, CORN}).

**Can the script later support `retrieved_context_windows_hybrid_list` for MIL?**
- Not yet — the current dataset class flattens one evidence string per item. MIL
  over individual windows would need a new dataset/collator (one instance per
  window) and a pooling head. Feasible as a follow-up; flagged as future work.

---

## B. Setup summary (for your records)

| Field | Value |
|---|---|
| Model | `bert-base-uncased` (BERT≈MentalBERT confirmed tied; cheaper default) |
| Eval protocol | Participant-grouped, stratified **5-fold CV** (no leakage) |
| Predictions | Out-of-fold over all 1,752 items |
| Epochs / batch / LR | 8 / 16 / 2e-5 |
| max_length | 256 (evidence-only truncation) |
| Losses | weighted-CE and CORN (ordinal) |
| Seed | 42 |

## C. Conditions evaluated

| Tag | dataset_path | evidence_column |
|---|---|---|
| old | phq8_item_dataset_full_bm25.csv | retrieved_utterances |
| hybw3 | phq8_item_dataset_context_windows_hybrid_w3.csv | retrieved_context_windows_hybrid_pack |
| hybw5 | phq8_item_dataset_context_windows_hybrid_w5.csv | retrieved_context_windows_hybrid_pack |
| bm25w3 | phq8_item_dataset_context_windows_bm25_w3.csv | retrieved_context_windows_bm25_pack |
| bm25w5 | phq8_item_dataset_context_windows_bm25.csv | retrieved_context_windows_bm25_pack |

---

## D. RESULTS

### D.1 Headline: context windows beat isolated utterances (256-token CV, OOF)

| Condition | Loss | Macro-F1 | Acc | MAE↓ | QWK | F1 c2 | F1 c3 | totMAE↓ | totQWK | Bal-Acc(≥10) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Old BM25 utterances** | CE | 0.276 | 0.389 | 0.861 | 0.173 | 0.183 | 0.066 | 5.23 | 0.174 | 0.523 |
| **Old BM25 utterances** | CORN | 0.274 | 0.382 | 0.847 | 0.119 | 0.136 | 0.093 | 5.58 | 0.074 | 0.543 |
| Hybrid W3 | CE | 0.336 | 0.418 | 0.765 | 0.319 | 0.205 | 0.183 | 4.32 | 0.459 | 0.633 |
| **Hybrid W3** | **CORN** | **0.348** | 0.436 | 0.740 | **0.336** | 0.222 | 0.185 | **4.14** | **0.485** | 0.631 |
| Hybrid W5 | CE | 0.345 | 0.455 | 0.728 | 0.319 | 0.223 | 0.187 | 4.17 | 0.425 | 0.600 |
| Hybrid W5 | CORN | 0.325 | 0.398 | 0.785 | 0.293 | 0.181 | 0.212 | 4.63 | 0.383 | **0.680** |
| BM25 W3 | CE | 0.324 | 0.439 | 0.813 | 0.227 | 0.143 | 0.203 | 4.43 | 0.332 | 0.570 |
| BM25 W3 | CORN | 0.295 | 0.453 | 0.751 | 0.210 | 0.137 | 0.050 | 4.38 | 0.274 | 0.555 |
| BM25 W5 | CE | 0.314 | 0.411 | 0.849 | 0.179 | 0.194 | 0.143 | 4.97 | 0.207 | 0.546 |
| BM25 W5 | CORN | 0.305 | 0.447 | 0.739 | 0.230 | 0.212 | 0.023 | 4.35 | 0.288 | 0.529 |

**Best overall: Hybrid W3 + CORN** (macro-F1 0.348, QWK 0.336, total MAE 4.14).

### D.2 Does the richer 512-token context help? (CORN)

| Condition | macro-F1 256→512 | MAE 256→512 | totMAE 256→512 |
|---|---|---|---|
| Old | 0.274 → 0.292 | 0.847 → 0.809 | 5.58 → 5.08 |
| Hybrid W3 | 0.348 → 0.347 | 0.740 → 0.703 | 4.14 → 4.05 |
| Hybrid W5 | 0.325 → 0.347 | 0.785 → 0.717 | 4.63 → 4.31 |
| BM25 W3 | 0.295 → 0.296 | 0.751 → 0.765 | 4.38 → 4.57 |
| BM25 W5 | 0.305 → 0.295 | 0.739 → 0.744 | 4.35 → 4.64 |

512 mainly lowers MAE and **recovers W5** (it was 82–95% truncated at 256). Hybrid W3
is already near-optimal at 256, so it is the efficient choice. **W3 ≈ W5 at 512.**

### D.3 Hybrid is what wins — not BM25 context windows alone
BM25-only context windows are only marginally above the old utterance baseline
(macro-F1 0.295–0.324 vs 0.274–0.276). The **semantic component of the hybrid** drives
the gain (Hybrid 0.347–0.348). Takeaway: the win is *better retrieval*, not just
*more context*.

### D.4 Per-item: Old vs Hybrid W3 (CORN) — improves 7/8 items
| Item | old F1 | hyb F1 | Δ | old MAE | hyb MAE |
|---|---|---|---|---|---|
| Depressed | 0.235 | 0.374 | **+0.139** | 0.763 | 0.676 |
| Sleep | 0.239 | 0.357 | **+0.118** | 1.091 | 0.840 |
| Failure | 0.233 | 0.336 | +0.103 | 1.014 | 0.799 |
| Tired | 0.213 | 0.296 | +0.084 | 0.909 | 0.849 |
| Appetite | 0.245 | 0.299 | +0.054 | 1.005 | 0.877 |
| Concentrating | 0.264 | 0.311 | +0.047 | 0.831 | 0.790 |
| NoInterest | 0.241 | 0.257 | +0.016 | 0.712 | 0.685 |
| Moving | 0.310 | 0.265 | −0.045 | 0.452 | 0.406 |

**Appetite (your flagged weak item) improved** (F1 +0.054, MAE −0.128). Only Moving
dipped slightly — it is the heavily skewed "rarely endorsed" item.

### D.5 Confusion matrices (CORN, OOF, rows=true / cols=pred)
```
OLD utterances                 HYBRID W3 (best)
      p0   p1   p2   p3              p0   p1   p2   p3
 t0  399  337   96   20         t0  429  322   82   19
 t1  208  230   64   11         t1  139  254   91   29
 t2   79  103   30    8         t2   42   93   55   30
 t3   50   75   32   10         t3   23   71   48   25
```
Hybrid pulls real mass onto the severe diagonal: true-2→pred-2 30→55, true-3→pred-3 10→25.

### D.6 Reconstructed PHQ-8 total & clinical screening (threshold ≥ 10)
Per participant = sum of 8 predicted item severities (OOF, all 8 items present).

| Condition (CORN) | total MAE↓ | total QWK | Sensitivity | Specificity | Balanced-Acc |
|---|---|---|---|---|---|
| Old utterances | 5.58 | 0.074 | 0.215 | 0.825 | 0.543 |
| **Hybrid W3** | **4.14** | **0.485** | 0.431 | 0.831 | 0.631 |
| Hybrid W5 | 4.63 | 0.383 | 0.323 | 0.805 | **0.680** |

Screening improves markedly: total QWK 0.07 → 0.49, balanced accuracy 0.54 → 0.63–0.68.
Sensitivity is still modest (~0.32–0.43) — the model under-calls depression — so for a
screening deployment a **lower decision threshold** should be tuned. (Caveat: 219
participants, 65 positive; treat threshold metrics as indicative.)

---

## E. Recommendation to Person A
1. **Adopt Hybrid context windows** — a clear, consistent win over isolated utterances
   on every metric (macro-F1 +27%, QWK ~2×, severe-class F1 ~2×, total-score QWK 0.07→0.49).
2. **Use W3 (not W5) by default** — equal quality, far cheaper, fits the BERT window
   (W5 needs max_length=512 just to match W3).
3. **The lever is hybrid (semantic) retrieval, not BM25 windows alone** — invest there.
4. **Pair with CORN** for best ordinal/MAE behaviour; weighted-CE is comparable on F1.
5. Next: tune the screening threshold for sensitivity; consider the MIL idea over
   `retrieved_context_windows_hybrid_list` as a future extension.

Artifacts: `outputs/context_window_results.json`, OOF predictions in `outputs/cv/oof_predictions_ctx_*`,
evaluation in `src/evaluation/context_window_eval.py`, HTML report `outputs/context_window_report.html`.
