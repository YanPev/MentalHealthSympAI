# R2 clinical lexicon — provenance audit method

**Stage A.3 of the R2 corrective experiment.** Source of truth:
`src/retrieval/clinical_lexicon_curated.py`. Audit table:
`outputs/r2_systematic/lexicons/clinical_audit.csv`. Unified manifest across
tiers: `outputs/r2_systematic/lexicons/lexicon_manifest.csv`.

## Why re-audit

The previous R1 clinical tier (`symptom_lexicon.CLINICAL`) was hand-curated from
DSM-5 constructs **and then augmented by an offline Qwen brainstorm pass merged
with only light review**. The R2 brief requires that an LLM origin is *not*
sufficient provenance, and named seven suspicious terms to check: *imposter
syndrome, inferiority complex, cynicism, hyperactivity, bradykinesia,
underarousal, passivity*. All seven are removed by this audit (reasons below).

## Rule

A clinical term is **kept** only if it is *independently* documented in a
recognized source **and** is reasonably specific to the PHQ-8 item's DSM-5
Criterion-A construct. An LLM suggestion is not provenance; but a term an LLM
happened to suggest is kept if it is *independently* a standard clinical term
(e.g. *somnolence*, *asthenia*). Terms are classified by `relation`:

- `direct_construct` — the criterion wording itself or a canonical synonym
- `subtype` — a recognized clinical subtype of the construct
- `manifestation` — a documented clinical sign/manifestation
- `related_nonspecific` — related but not specific to *this* construct (another
  disorder/domain, or a vague/non-standard paraphrase)

`related_nonspecific` terms are **excluded from the primary clinical query**
(`curated_query`), even if recorded in the audit. `curated_terms(item)` returns
`keep == True AND relation != related_nonspecific`.

## Provenance sources

DSM-5 MDD Criterion A (A1–A9) construct wording; standard psychopathology
terminology; MeSH descriptor **names** (e.g. *Anhedonia, Apathy, Hypersomnia,
Psychomotor Agitation, Asthenia, Lethargy, Anorexia, Shame, Craving*); APA
Dictionary of Psychology. These are stable canonical references; a live
UMLS/SNOMED/MeSH terminology server is **not available on the offline cluster**,
so the audit cites the canonical construct/descriptor name rather than a numeric
CUI/DUI. Each item's construct:

| Item | DSM-5 A-criterion |
|---|---|
| NoInterest | A2 diminished interest or pleasure (anhedonia) |
| Depressed | A1 depressed mood |
| Sleep | A4 insomnia or hypersomnia |
| Tired | A6 fatigue or loss of energy |
| Appetite | A3 appetite / weight change |
| Failure | A7 worthlessness or excessive/inappropriate guilt |
| Concentrating | A8 diminished ability to think/concentrate; indecisiveness |
| Moving | A5 psychomotor agitation or retardation |

## Result

**146 candidate terms audited → 101 kept, 45 removed (31% removed).** Per item
(kept/candidates): NoInterest 13/18, Depressed 10/19, Sleep 12/15, Tired 11/18,
Appetite 15/17, Failure 17/24, Concentrating 12/19, Moving 11/16.

### The seven flagged terms — all removed
- **imposter syndrome** (Failure) — pop-psychology; "impostor phenomenon" is not
  a DSM worthlessness/guilt construct → `related_nonspecific`.
- **inferiority complex** (Failure) — Adlerian/historical, not an MDD construct.
- **cynicism** (Depressed) — burnout/personality attitude, not depressed mood.
- **passivity** (Depressed) — "passivity phenomena" is a psychosis term; non-specific.
- **hyperactivity** (Moving) — ADHD terminology, not psychomotor agitation.
- **bradykinesia** (Moving) — neurological (Parkinsonism) motor slowing, not the
  psychiatric psychomotor-retardation construct.
- **underarousal** (NoInterest) — arousal is a distinct construct from anhedonia;
  no clinical provenance for this usage.

### Other notable removals (by pattern)
- **Different disorder/domain:** flat affect, emotional numbness (Depressed);
  parasomnia (Sleep); hypokinesia, akathisia (Moving); cognitive disorganization
  (Concentrating); malaise, weakness, vigilance impairment (Tired).
- **Cross-item overlap:** cognitive fatigue (→ Tired), chronic fatigue (CFS
  confound).
- **Lay register (moved out of clinical):** brain fog, impaired mental clarity,
  poor mental sharpness (Concentrating).
- **Non-standard LLM paraphrases:** interest deficit, valued activities
  diminished, depressive thinking, reduced vitality, internalized criticism,
  self-defeatism, reactive overeating, impulsive eating, motor inhibition, etc.
- **Anxiety/other-construct:** self-doubt, self-pity, self-punishment (Failure).

### Gaps
Every item retains ≥10 provenanced clinical terms, so **no item required an
open-ended LLM brainstorm** to fill a gap (brief §3.4). No new unrestricted
term generation was run.

## Leakage statement

The clinical tier derives only from PHQ-8/DSM-5 symptom constructs and standard
terminology — never from DAIC-WOZ labels, transcripts, predictions, or error
cases. It is a single global, fold-independent artifact; term lists are never
tuned against out-of-fold PHQ metrics.
