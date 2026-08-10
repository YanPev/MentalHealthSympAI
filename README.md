# MentalHealthSympAI

**AI-driven symptom retrieval and PHQ-8 severity prediction from clinical interviews**

Predicting the eight PHQ-8 depression items (each scored 0 to 3) from DAIC-WOZ
interview transcripts, by retrieving the passages that actually carry symptom
evidence and reasoning over them with an encoder, an LLM, or a cascade of both.

---

## The problem

A PHQ-8 score is eight ordinal judgements, one per depressive symptom. An interview
transcript is long, mostly irrelevant to any given item, and the evidence for
"trouble sleeping" sits in a few sentences buried among thousands. Feeding the whole
transcript to a classifier drowns the signal.

So the pipeline is retrieval first, then prediction:

```
DAIC-WOZ transcript
        │
        ▼
  context windows            participant-side turns, windowed
        │
        ▼
  RETRIEVAL                  per PHQ-8 item, find the evidence
    BM25 (lexical)  +  semantic (embedding)  ->  hybrid
    symptom lexicons: curated clinical, lay, expanded
        │
        ▼
  ┌─────┴─────────────────┐
  │                       │
ENCODER                  LLM
MentalBERT               Qwen2.5-7B chain-of-thought
+ CORN ordinal loss      item-aware / joint prompting
+ Attention-MIL          self-consistency pooling
  │                       │
  └─────┬─────────────────┘
        ▼
   CASCADE                  merged gate between encoder and LLM
        ▼
   8 item scores (0-3)
```

The ordinal structure matters: predicting 3 when the truth is 0 is a worse error
than predicting 1. That is why CORN ordinal loss is used rather than plain
cross-entropy, and why QWK and far-off rate are reported alongside macro-F1.

---

## Results

Frozen Hybrid-W3 retrieval (R0), participant-level cross-validation.
Full tables and bootstrap confidence intervals: `docs/results_summary.md`.

| Configuration | macro-F1 | QWK | MAE | accuracy | severe recall |
|---|---|---|---|---|---|
| Encoder, MentalBERT + CORN | 0.347 | 0.354 | 0.680 | 0.455 | 0.102 |
| Encoder, Attention-MIL | 0.360 | 0.362 | 0.671 | 0.475 | 0.114 |
| LLM only, Qwen2.5-7B CoT | 0.400 | 0.406 | 0.641 | 0.513 | 0.204 |
| **Cascade, MIL + merged gate** | **0.409** | **0.447** | **0.606** | **0.530** | 0.198 |

The cascade is the project best on every headline metric. The LLM alone beats both
encoders, and the cascade beats the LLM, so the two error profiles are complementary
rather than redundant.

**Expanded retrieval (R1) is not an improvement.** Widening the retrieval dictionary
helps the encoders (macro-F1 +0.025 for CORN) and helps severe-class recall almost
everywhere, but it *degrades* the cascade (QWK -0.037, MAE +0.034). Only the cascade
MAE change has a bootstrap CI excluding zero, so most of these deltas are not
individually significant. Treated as a negative result and reported as such.

---

## Repository structure

```
MentalHealthSympAI/
├── src/
│   ├── data/               # dataset construction and integrity
│   │   ├── preprocess_transcripts.py   # DAIC-WOZ -> clean turns
│   │   ├── build_item_dataset.py       # per-PHQ8-item examples
│   │   ├── create_splits.py            # participant-level splits
│   │   ├── check_leakage.py            # split leakage guard
│   │   └── phq8_items.py               # the 8 item definitions
│   │
│   ├── retrieval/          # find the evidence for each item
│   │   ├── bm25_retriever.py           # lexical baseline
│   │   ├── hybrid_retrieval.py         # BM25 + semantic fusion
│   │   ├── item_aware_retrieval.py     # per-item query construction
│   │   ├── rerank_windows.py           # cross-encoder reranking
│   │   ├── prod_context/               # production context retrievers
│   │   │   ├── bm25_context_retriever.py
│   │   │   ├── semantic_context_retriever.py
│   │   │   ├── semantic_multiquery_retriever.py
│   │   │   └── hybrid_context_retriever.py
│   │   ├── symptom_lexicon.py          # curated clinical terms
│   │   ├── clinical_lexicon_curated.py
│   │   ├── lay_lexicon.py              # lay/colloquial phrasing
│   │   └── r2_lexicons.py              # expanded (R1) dictionaries
│   │
│   ├── models/             # encoder side
│   │   ├── train_transformer_classifier.py  # MentalBERT training
│   │   ├── ordinal.py                       # CORN ordinal loss
│   │   ├── mil_classifier.py                # Attention-MIL over windows
│   │   ├── cross_validate.py                # participant-level CV
│   │   ├── missing_policy.py                # no-evidence handling
│   │   └── data_module.py / phq8_torch_dataset.py
│   │
│   ├── llm/                # LLM side
│   │   ├── cot_joint.py                # joint 8-item chain-of-thought
│   │   ├── cot_probe.py                # per-item probing
│   │   ├── generate_rationales.py      # rationale generation
│   │   ├── distill_student.py          # distillation to a smaller model
│   │   └── brainstorm_lexicon.py       # LLM-assisted lexicon expansion
│   │
│   ├── evaluation/  (74)   # scoring, cascades, reports, error analysis
│   │   ├── cot_cascade.py              # encoder/LLM merged gate
│   │   ├── cot_ensemble.py / cot_stacker.py
│   │   ├── cascade_bootstrap_ci.py     # participant-cluster bootstrap
│   │   ├── per_item_analysis.py        # per-PHQ8-item breakdown
│   │   ├── failure_patterns.py         # where and how it fails
│   │   ├── confusion_matrices.py
│   │   ├── llm_evidence_quality.py     # is the retrieved evidence any good
│   │   ├── judge_windows.py            # LLM-as-judge on windows
│   │   └── build_*_report.py           # HTML/summary report builders
│   │
│   └── reporting/          # clinician-facing report generation
│
├── notebooks/              # exploration: preprocessing, TF-IDF/BM25, MentalBERT PoC
├── docs/                   # method notes and generated reports
│   ├── retrieval_method.md
│   ├── r2_semantic_query_method.md
│   ├── r2_clinical_lexicon_method.md
│   ├── r2_corrective_experiment_plan.md
│   └── results_summary.md              # the numbers above, auto-generated
├── outputs/         (149)  # results, figures, run manifests, reports
│   ├── cv/                 # cross-validation out-of-fold predictions
│   ├── cot/                # LLM chain-of-thought runs and cascades
│   ├── r2_systematic/      # expanded-retrieval (R1) sweep
│   └── figures/            # generated figures
├── tests/                  # smoke tests
└── requirements.txt
```

---

## Data

**DAIC-WOZ** (Distress Analysis Interview Corpus, Wizard-of-Oz). Not redistributed
here and not in this repository. Obtain it from its maintainers, then run the
preprocessing scripts to build the item dataset.

Splits are **participant-level**, never utterance-level, so no participant appears
in both train and test. `src/data/check_leakage.py` enforces this, and it is worth
re-running after any change to split construction.

---

## Getting started

```bash
pip install -r requirements.txt
```

Build the dataset from raw transcripts:

```bash
python -m src.data.preprocess_transcripts && python -m src.data.build_item_dataset
```

Verify no participant leaks across splits:

```bash
python -m src.data.check_leakage
```

Train the encoder with participant-level cross-validation, on the frozen
Hybrid-W3 retrieval used for the headline results:

```bash
python -m src.models.cross_validate --model-name bert-base-uncased --k-folds 5 --num-epochs 8 --batch-size 16 --max-length 256 --seed 42 --dataset-path data/processed/phq8_item_dataset_context_windows_hybrid_w3.csv --evidence-column retrieved_context_windows_hybrid_pack --tag ctx_corn_hybw3 --loss corn
```

Swap `--loss corn` for `--loss cross_entropy --class-weights balanced` to get the
weighted-CE variant, and change `--dataset-path` / `--evidence-column` to compare
retrieval conditions.

Run the LLM chain-of-thought pipeline, one fold at a time (folds 1 to 5 make up the
out-of-fold predictions):

```bash
python -m src.llm.cot_joint --fold 1 --dataset-path data/processed/phq8_item_dataset_context_windows_hybrid_w3.csv --evidence-column retrieved_context_windows_hybrid_list --output outputs/cot/folds_joint/cot_joint_fold1.csv
```

Regenerate the results summary from the analysis JSONs:

```bash
python -m src.evaluation.build_results_summary
```

The encoder and LLM steps need a GPU. On a cluster node with a pre-populated
Hugging Face cache, export these before running so nothing reaches for the network:

```bash
export HF_HOME="$PWD/.hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
```

Each command above is one experiment configuration; vary the dataset, evidence
column and loss flags to reproduce the rest. `outputs/run_manifest.json` records
which configurations were actually run and where their outputs landed.

---

## Authors and contributors

This project was jointly developed by **Daniel Schmidt** and **Yaniv Pevzner** as part of the graduate course *Advanced Artificial Intelligence for Medicine* at Ben-Gurion University of the Negev, under the guidance of Prof. Lior Rokach.

| Contributor        | Main contributions                                                                                                                                                                                                                                                                                                |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Daniel Schmidt** | Project co-author. ML/AI methodology and experimental design; data and PHQ-8 item pipeline; lexical, semantic, and hybrid evidence retrieval; model and LLM experiment design; model evaluation, statistical analysis, and error analysis; methodology development, interpretation, and scientific communication. |
| **Yaniv Pevzner**  | Project co-author. Primary ML implementation and experiment execution; encoder and LLM pipelines; ordinal and Attention-MIL modelling; cascade and ensemble implementation; evaluation infrastructure and engineering.                                                                                            |

Both authors jointly contributed to the research formulation, algorithmic and experimental design, model-selection decisions, interpretation of results, and development of the final methodology.


---

## Notes on metrics

- **macro-F1** treats all four severity levels equally, so it is sensitive to the
  rare severe classes.
- **QWK** (quadratic weighted kappa) respects the ordinal scale and penalises
  distant errors more heavily.
- **MAE** is on the 0 to 3 item scale.
- **far-off rate** is the fraction of predictions off by 2 or more, the errors that
  matter most clinically.
- **severe recall** is recall on level 3, the hardest and most important class.

Confidence intervals are bootstrapped by **participant cluster**, not by example,
because items from one participant are not independent.
