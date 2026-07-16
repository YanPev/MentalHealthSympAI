# R2 semantic multi-query method

**Stage B of the R2 corrective experiment.** Source:
`src/retrieval/semantic_queries.py`; artifact:
`outputs/r2_systematic/semantic_queries/semantic_queries.json`; tests:
`tests/test_semantic_queries.py` (7 passing).

## The defect being corrected

The previous full-expansion semantic variant (V3 `exphyb_bothq`) built the
semantic query by **concatenating the whole tiered lexicon into one string** and
encoding that keyword dump with MiniLM. Sentence encoders are trained on natural
language, not comma-lists; a 60-term dump has no coherent meaning to embed, and
it collapses the two opposing poles of bipolar items (insomnia+hypersomnia,
poor-appetite+overeating, retardation+agitation) into a single blurred vector.

## The R2 representation

For each PHQ-8 item, a small set of **natural-language sentences**:

- **Q0_ITEM** — the item meaning as a concise sentence (always present).
- **Q1_LAY** — one sentence in everyday language (accepted lay expressions).
- **Q2_CLINICAL** — one sentence naming the direct clinical construct and
  accepted subtypes, using **only audited clinical terms** (a test asserts no
  removed term leaks in).

**Bipolar items get one query per pole** instead of a merged query:

| item | poles |
|---|---|
| Sleep | insomnia / hypersomnia |
| Appetite | poor_appetite / overeating |
| Moving | retardation / agitation |

So Sleep/Appetite/Moving carry 2 queries per kind (6 total); the five unipolar
items carry 1 per kind (3 total).

## Scoring

For each transcript window the embedding is computed **once**; cosine similarity
is taken against every active query; the retriever retains the **per-query
similarity, the winning query, and the maximum**:

```
SemanticScore(window, config) = max_i cosine(query_i, window)
```

The **maximum** (not the average) is the primary score — averaging would dilute a
strong single-pole match (a clear insomnia window would be penalised by the
unrelated hypersomnia query). Per-query scores are still stored for diagnostics.

## Which queries are active

Active queries depend on the lexicon config under evaluation, so the semantic
side stays aligned with the lexicon side:

| config | active query kinds | Sleep example (# queries) |
|---|---|---|
| L0_CORE | item | 2 (insomnia, hypersomnia) |
| L1_CORE_LAY | item + lay | 4 |
| L2_CORE_CLINICAL | item + clinical | 4 |
| L3_FULL | item + lay + clinical | 6 |

## Tests (brief §4)

`tests/test_semantic_queries.py` asserts: every item has Q0 (per pole for
bipolar); no empty query; no duplicated query text; clinical queries contain only
audited clinical terms (no removed term appears; every used term ∈ curated set);
no raw list dump (each query has ≥3 function words, is phrased as a sentence, and
is ≥8 tokens); opposing poles represented separately for the three bipolar items
and single-pole for the rest; and the config→active-query selection is correct.

## Leakage statement

Queries are written from the PHQ-8 item wording, the reviewed lay tier, and the
audited clinical tier — never from DAIC-WOZ transcripts, labels, or predictions.
One global query set, fold-independent by construction.
