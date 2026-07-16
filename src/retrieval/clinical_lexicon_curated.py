"""
Curated clinical lexicon (R2 corrective experiment, Stage A.3).

The previous `symptom_lexicon.CLINICAL` tier mixed genuine clinical constructs
with terms produced by an offline Qwen brainstorm pass and merged with only
light review. Per the R2 brief, an LLM origin is NOT sufficient provenance: a
clinical term is retained only if it is *independently* documented in a
recognized source AND is reasonably specific to the PHQ-8 item's construct.

Each PHQ-8 item maps to a DSM-5 Major Depressive Disorder Criterion-A symptom:
    NoInterest    -> A2  diminished interest or pleasure (anhedonia)
    Depressed     -> A1  depressed mood
    Sleep         -> A4  insomnia or hypersomnia
    Tired         -> A6  fatigue or loss of energy
    Appetite      -> A3  appetite / weight change
    Failure       -> A7  worthlessness or excessive/inappropriate guilt
    Concentrating -> A8  diminished ability to think/concentrate; indecisiveness
    Moving        -> A5  psychomotor agitation or retardation

`relation` classifies each candidate against that construct:
    direct_construct     -- the criterion wording itself or a canonical synonym
    subtype              -- a recognized clinical subtype of the construct
    manifestation        -- a documented clinical sign/manifestation of it
    related_nonspecific  -- related but not specific to THIS construct (belongs
                            to another disorder/domain, or is a vague paraphrase)

`source` cites the provenance used for the KEEP decision (the canonical
construct name); these are standard, stable references (DSM-5 criteria, MeSH
descriptor names, APA Dictionary of Psychology, established psychopathology
terminology) rather than a live terminology-server query, which is unavailable
on the offline cluster. Terms whose only "source" was the LLM brainstorm, or
that belong to a different construct/domain, are REMOVED.

RULES ENFORCED
  * Terms with relation == "related_nonspecific" are EXCLUDED from the primary
    clinical query (`curated_query`) even when kept for the audit record.
  * `curated_terms(item)` returns only keep == True AND relation != related_nonspecific.

    python -m src.retrieval.clinical_lexicon_curated       # audit summary + CSV
"""

from pathlib import Path
import csv

PR = Path(__file__).resolve().parents[2]
AUDIT_CSV = PR / "outputs" / "r2_systematic" / "lexicons" / "clinical_audit.csv"

# DSM-5 MDD Criterion-A construct each item screens (for provenance citations).
DSM5_CONSTRUCT = {
    "NoInterest": "DSM-5 MDD Criterion A2 (markedly diminished interest or pleasure; anhedonia)",
    "Depressed": "DSM-5 MDD Criterion A1 (depressed mood)",
    "Sleep": "DSM-5 MDD Criterion A4 (insomnia or hypersomnia)",
    "Tired": "DSM-5 MDD Criterion A6 (fatigue or loss of energy)",
    "Appetite": "DSM-5 MDD Criterion A3 (decrease/increase in appetite or weight)",
    "Failure": "DSM-5 MDD Criterion A7 (worthlessness or excessive/inappropriate guilt)",
    "Concentrating": "DSM-5 MDD Criterion A8 (diminished ability to think or concentrate; indecisiveness)",
    "Moving": "DSM-5 MDD Criterion A5 (psychomotor agitation or retardation)",
}

# Full audit of the previous CLINICAL tier.
# (term, relation, keep, source, reason)
AUDIT = {
    "NoInterest": [
        ("anhedonia", "direct_construct", True, "MeSH descriptor 'Anhedonia'; DSM-5 A2", "canonical construct for loss of interest/pleasure"),
        ("loss of interest", "direct_construct", True, "DSM-5 A2 wording", "criterion wording"),
        ("diminished interest", "direct_construct", True, "DSM-5 A2 verbatim", "criterion wording"),
        ("diminished pleasure", "direct_construct", True, "DSM-5 A2 ('or pleasure')", "criterion wording"),
        ("loss of pleasure", "direct_construct", True, "DSM-5 A2; anhedonia", "criterion wording"),
        ("diminished ability to experience pleasure", "direct_construct", True, "DSM-5 A2 paraphrase (anhedonia)", "canonical paraphrase of anhedonia"),
        ("hedonic capacity impairment", "direct_construct", True, "anhedonia research terminology (hedonic capacity)", "documented anhedonia phrasing"),
        ("loss of interest in usual pleasures", "direct_construct", True, "DSM-5 A2 paraphrase", "criterion paraphrase"),
        ("avolition", "manifestation", True, "APA Dictionary; psychopathology (negative symptom)", "documented lack of motivation to initiate/persist"),
        ("amotivation", "manifestation", True, "clinical terminology (amotivational state)", "documented motivational-deficit term tied to interest"),
        ("apathy", "manifestation", True, "MeSH descriptor 'Apathy'; APA Dictionary", "recognized diminished-motivation/interest syndrome"),
        ("social withdrawal", "manifestation", True, "psychopathology (behavioural sign of anhedonia)", "documented behavioural manifestation"),
        ("reduced libido", "manifestation", True, "DSM-5 A2 examples include loss of sexual interest", "sexual-interest loss is part of A2"),
        # REMOVED
        ("underarousal", "related_nonspecific", False, "LLM brainstorm only", "arousal is a distinct construct, not anhedonia; no clinical provenance for this usage"),
        ("interest deficit", "related_nonspecific", False, "LLM brainstorm only", "non-standard paraphrase; not an established term"),
        ("valued activities diminished", "related_nonspecific", False, "LLM brainstorm only", "non-standard paraphrase; not an established term"),
        ("emotional blunting", "related_nonspecific", False, "MeSH-adjacent (affective blunting)", "reduced emotional expression is a different construct (schizophrenia/SSRI); not specific to A2"),
        ("disengagement", "related_nonspecific", False, "LLM brainstorm only", "vague; not an established clinical term for anhedonia"),
    ],
    "Depressed": [
        ("depressed mood", "direct_construct", True, "DSM-5 A1 verbatim", "criterion wording"),
        ("low mood", "direct_construct", True, "clinical synonym for depressed mood", "canonical synonym"),
        ("dysphoria", "direct_construct", True, "APA Dictionary; psychopathology", "canonical mood construct"),
        ("dysphoric", "direct_construct", True, "APA Dictionary (adjective form of dysphoria)", "canonical mood construct"),
        ("hopelessness", "manifestation", True, "DSM-5 A1 example ('hopeless'); Beck Hopelessness", "criterion example / documented construct"),
        ("tearfulness", "manifestation", True, "DSM-5 A1 note ('appears tearful')", "criterion-linked sign"),
        ("despondent", "manifestation", True, "APA Dictionary (dejected/low spirits)", "documented affective descriptor of low mood"),
        ("melancholy", "subtype", True, "DSM-5 'with melancholic features' specifier", "recognized depressive subtype descriptor"),
        ("melancholic", "subtype", True, "DSM-5 'with melancholic features' specifier", "recognized depressive subtype"),
        ("dysthymia", "subtype", True, "DSM-5 persistent depressive disorder (dysthymia)", "chronic depressed-mood construct"),
        # REMOVED
        ("cynicism", "related_nonspecific", False, "LLM brainstorm; burnout/personality construct", "attitude construct (burnout/personality), not depressed mood"),
        ("passivity", "related_nonspecific", False, "LLM brainstorm; 'passivity phenomena' is a psychosis term", "non-specific / different construct (psychosis)"),
        ("flat affect", "related_nonspecific", False, "MeSH-adjacent (blunted/flat affect)", "negative-symptom (schizophrenia) construct, not depressed mood"),
        ("emotional numbness", "related_nonspecific", False, "trauma/dissociation literature", "not specific to A1 (also PTSD/dissociation)"),
        ("negative affect", "related_nonspecific", False, "PANAS dimensional construct", "broad dimensional trait, not the A1 mood construct"),
        ("depressive thinking", "related_nonspecific", False, "LLM brainstorm only", "non-standard; conflates cognition with mood"),
        ("resignation", "related_nonspecific", False, "LLM brainstorm only", "non-specific emotional descriptor"),
        ("unhappiness", "related_nonspecific", False, "lay register", "lay synonym (Core already covers 'unhappy'); not a distinct clinical term"),
        ("wretchedness", "related_nonspecific", False, "archaic/literary", "not clinical terminology"),
    ],
    "Sleep": [
        ("hypersomnia", "direct_construct", True, "MeSH descriptor 'Hypersomnia'; DSM-5 A4", "criterion construct (oversleep pole)"),
        ("sleep disturbance", "direct_construct", True, "MeSH 'Sleep Wake Disorders'; DSM-5 A4", "criterion construct"),
        ("non-restorative sleep", "manifestation", True, "clinical sleep terminology; DSM insomnia note", "documented sleep-quality manifestation"),
        ("sleep onset", "subtype", True, "sleep medicine (sleep-onset / initial insomnia)", "recognized insomnia subtype"),
        ("sleep latency", "manifestation", True, "sleep medicine (increased sleep latency)", "documented insomnia sign"),
        ("early morning awakening", "subtype", True, "clinical (terminal insomnia)", "recognized insomnia subtype"),
        ("fragmented sleep", "subtype", True, "clinical (sleep-maintenance/middle insomnia)", "recognized insomnia subtype"),
        ("nocturnal awakening", "subtype", True, "clinical (middle insomnia)", "recognized insomnia subtype"),
        ("middle insomnia", "subtype", True, "clinical insomnia subtype terminology", "recognized insomnia subtype"),
        ("terminal insomnia", "subtype", True, "clinical insomnia subtype terminology", "recognized insomnia subtype"),
        ("excessive daytime sleepiness", "manifestation", True, "sleep medicine (EDS)", "documented hypersomnia manifestation"),
        ("somnolence", "manifestation", True, "MeSH 'Disorders of Excessive Somnolence'", "documented hypersomnia manifestation"),
        # REMOVED
        ("circadian", "related_nonspecific", False, "bare term; circadian-rhythm disorders", "too generic alone; not the insomnia/hypersomnia construct"),
        ("sleep quality", "related_nonspecific", False, "generic descriptor", "non-specific; not a symptom construct by itself"),
        ("parasomnia", "related_nonspecific", False, "MeSH 'Parasomnias' (sleepwalking, terrors)", "distinct sleep-disorder class, not MDD insomnia/hypersomnia"),
    ],
    "Tired": [
        ("loss of energy", "direct_construct", True, "DSM-5 A6 verbatim", "criterion wording"),
        ("low energy", "direct_construct", True, "DSM-5 A6", "criterion wording"),
        ("lack of energy", "direct_construct", True, "DSM-5 A6 paraphrase", "criterion paraphrase"),
        ("decreased energy", "direct_construct", True, "DSM-5 A6 paraphrase", "criterion paraphrase"),
        ("diminished energy", "direct_construct", True, "DSM-5 A6 paraphrase", "criterion paraphrase"),
        ("anergia", "direct_construct", True, "psychopathology (anergia = lack of energy)", "canonical low-energy construct"),
        ("asthenia", "manifestation", True, "MeSH descriptor 'Asthenia'", "documented weakness/lack-of-energy construct"),
        ("fatigability", "manifestation", True, "clinical terminology (easy fatigability)", "documented fatigue manifestation"),
        ("lethargy", "manifestation", True, "MeSH descriptor 'Lethargy'", "documented low-energy/drowsiness manifestation"),
        ("exhaustion", "manifestation", True, "clinical descriptor of severe fatigue", "documented fatigue manifestation"),
        ("vital exhaustion", "manifestation", True, "Maastricht Questionnaire construct (vital exhaustion)", "documented fatigue-related construct"),
        # REMOVED
        ("chronic fatigue", "related_nonspecific", False, "risk of Chronic Fatigue Syndrome confound", "'chronic fatigue' strongly evokes CFS, a distinct disorder"),
        ("malaise", "related_nonspecific", False, "MeSH-adjacent; general unwellness", "non-specific to depressive fatigue (any illness)"),
        ("debilitation", "related_nonspecific", False, "LLM brainstorm; non-specific", "generic enfeeblement, not the A6 construct"),
        ("prostration", "related_nonspecific", False, "archaic", "not standard psychiatric terminology"),
        ("reduced vitality", "related_nonspecific", False, "LLM brainstorm only", "non-standard phrasing"),
        ("weakness", "related_nonspecific", False, "MeSH 'Muscle Weakness'", "physical-weakness term; non-specific (pulls somatic content)"),
        ("vigilance impairment", "related_nonspecific", False, "vigilance = sustained attention", "belongs to the concentration domain, not fatigue"),
    ],
    "Appetite": [
        ("decreased appetite", "direct_construct", True, "DSM-5 A3 ('decrease in appetite')", "criterion wording (reduced pole)"),
        ("increased appetite", "direct_construct", True, "DSM-5 A3 ('increase in appetite')", "criterion wording (overeating pole)"),
        ("appetite loss", "direct_construct", True, "DSM-5 A3; anorexia (symptom)", "criterion construct (reduced pole)"),
        ("appetite disturbance", "direct_construct", True, "DSM-5 A3", "criterion construct"),
        ("appetite change", "direct_construct", True, "DSM-5 A3", "criterion construct"),
        ("weight change", "direct_construct", True, "DSM-5 A3 (weight loss or gain)", "criterion construct"),
        ("anorexia", "direct_construct", True, "MeSH descriptor 'Anorexia' (symptom: loss of appetite)", "canonical reduced-appetite construct"),
        ("hyperphagia", "direct_construct", True, "clinical terminology (hyperphagia = overeating)", "canonical overeating construct"),
        ("hyporexia", "subtype", True, "clinical terminology (reduced appetite)", "documented reduced-appetite term"),
        ("polyphagia", "subtype", True, "clinical terminology (excessive eating/hunger)", "documented overeating term"),
        ("binge eating", "manifestation", True, "MeSH 'Binge-Eating Disorder'", "documented overeating manifestation"),
        ("psychogenic overeating", "manifestation", True, "ICD 'overeating associated with psychological disturbances'", "documented psychiatric overeating manifestation"),
        ("food craving", "manifestation", True, "MeSH descriptor 'Craving'", "documented increased-appetite manifestation"),
        ("emotional eating", "manifestation", True, "eating-behaviour literature", "documented overeating manifestation"),
        ("compulsive eating", "manifestation", True, "eating-behaviour literature", "documented overeating manifestation"),
        # REMOVED
        ("impulsive eating", "related_nonspecific", False, "LLM brainstorm; non-standard", "non-standard phrasing; duplicative of binge/compulsive"),
        ("reactive overeating", "related_nonspecific", False, "LLM brainstorm; non-standard", "non-standard phrasing"),
    ],
    "Failure": [
        ("worthlessness", "direct_construct", True, "DSM-5 A7 verbatim ('feelings of worthlessness')", "criterion wording"),
        ("excessive guilt", "direct_construct", True, "DSM-5 A7 ('excessive or inappropriate guilt')", "criterion wording"),
        ("inappropriate guilt", "direct_construct", True, "DSM-5 A7 verbatim", "criterion wording"),
        ("low self-worth", "direct_construct", True, "DSM-5 A7 (worthlessness)", "canonical worthlessness synonym"),
        ("self-blame", "manifestation", True, "clinical (guilt cognition)", "documented guilt manifestation"),
        ("self-criticism", "manifestation", True, "clinical (self-critical cognition)", "documented worthlessness manifestation"),
        ("self-reproach", "manifestation", True, "classic psychopathology (self-reproach)", "documented guilt manifestation"),
        ("negative self-evaluation", "manifestation", True, "Beck cognitive model (negative view of self)", "documented cognitive manifestation"),
        ("negative self-schema", "manifestation", True, "Beck cognitive theory of depression", "documented cognitive manifestation"),
        ("inadequacy", "manifestation", True, "clinical (feelings of inadequacy)", "documented worthlessness manifestation"),
        ("shame", "manifestation", True, "MeSH descriptor 'Shame'", "documented self-conscious affect (worthlessness-adjacent)"),
        ("guilty rumination", "manifestation", True, "clinical (ruminative guilt)", "documented guilt manifestation"),
        ("self-deprecation", "manifestation", True, "clinical descriptor (self-deprecating)", "documented low-self-worth manifestation"),
        ("self-hatred", "manifestation", True, "clinical descriptor", "documented severe-worthlessness manifestation"),
        ("self-loathing", "manifestation", True, "clinical descriptor", "documented severe-worthlessness manifestation"),
        ("self-contempt", "manifestation", True, "self-conscious-affect literature", "documented worthlessness manifestation"),
        ("self-disgust", "manifestation", True, "self-disgust in depression literature (has a scale)", "documented worthlessness manifestation"),
        # REMOVED
        ("imposter syndrome", "related_nonspecific", False, "pop-psychology; 'impostor phenomenon' is not a DSM construct", "not an MDD worthlessness/guilt construct"),
        ("inferiority complex", "related_nonspecific", False, "Adlerian/historical", "not an MDD construct"),
        ("internalized criticism", "related_nonspecific", False, "LLM brainstorm; non-standard", "non-standard phrasing"),
        ("self-defeatism", "related_nonspecific", False, "LLM brainstorm; non-standard", "non-standard ('self-defeating' is a personality term)"),
        ("self-doubt", "related_nonspecific", False, "anxiety/uncertainty construct", "not specific to worthlessness/guilt (also anxiety)"),
        ("self-pity", "related_nonspecific", False, "self-focused-sorrow construct", "distinct from worthlessness/guilt"),
        ("self-punishment", "related_nonspecific", False, "ambiguous with self-harm", "ambiguous; risks pulling self-harm content"),
    ],
    "Concentrating": [
        ("impaired concentration", "direct_construct", True, "DSM-5 A8", "criterion wording"),
        ("poor concentration", "direct_construct", True, "DSM-5 A8", "criterion wording"),
        ("diminished ability to think", "direct_construct", True, "DSM-5 A8 verbatim", "criterion wording"),
        ("indecisiveness", "direct_construct", True, "DSM-5 A8 verbatim ('or indecisiveness')", "criterion wording"),
        ("cognitive impairment", "direct_construct", True, "MeSH 'Cognitive Dysfunction'; DSM-5 A8", "criterion-linked construct"),
        ("distractibility", "manifestation", True, "clinical/DSM terminology (distractibility)", "documented attention manifestation"),
        ("inattention", "manifestation", True, "clinical terminology (inattention)", "documented attention manifestation"),
        ("difficulty sustaining attention", "manifestation", True, "attention terminology (sustained attention)", "documented attention manifestation"),
        ("attentional difficulties", "manifestation", True, "clinical descriptor of attention problems", "documented attention manifestation"),
        ("memory complaints", "manifestation", True, "clinical (subjective memory complaints)", "documented cognitive manifestation"),
        ("cognitive slowing", "manifestation", True, "clinical (bradyphrenia / cognitive slowing)", "documented cognitive manifestation"),
        ("executive dysfunction", "manifestation", True, "neuropsychology (executive function)", "documented cognitive manifestation"),
        # REMOVED
        ("brain fog", "related_nonspecific", False, "lay/informal register", "belongs in the lay tier, not clinical"),
        ("cognitive disorganization", "related_nonspecific", False, "schizotypy/psychosis term", "different construct (thought disorder)"),
        ("cognitive fatigue", "related_nonspecific", False, "overlaps fatigue (Tired)", "cross-domain with A6 fatigue"),
        ("cognitive interference", "related_nonspecific", False, "experimental psych (Stroop) term", "academic; non-specific to A8"),
        ("cognitive processing speed reduction", "related_nonspecific", False, "verbose paraphrase of cognitive slowing", "non-standard; duplicative of 'cognitive slowing'"),
        ("impaired mental clarity", "related_nonspecific", False, "LLM brainstorm; ~brain fog", "non-standard/lay phrasing"),
        ("poor mental sharpness", "related_nonspecific", False, "LLM brainstorm; lay", "non-standard/lay phrasing"),
    ],
    "Moving": [
        ("psychomotor retardation", "direct_construct", True, "DSM-5 A5 verbatim", "criterion wording (retardation pole)"),
        ("psychomotor agitation", "direct_construct", True, "MeSH 'Psychomotor Agitation'; DSM-5 A5", "criterion wording (agitation pole)"),
        ("psychomotor slowing", "direct_construct", True, "clinical synonym of psychomotor retardation", "canonical retardation synonym"),
        ("agitation", "direct_construct", True, "MeSH 'Psychomotor Agitation'; DSM-5 A5", "criterion construct (agitation pole)"),
        ("motor retardation", "manifestation", True, "clinical synonym (motor slowing)", "documented retardation manifestation"),
        ("restlessness", "manifestation", True, "DSM-5 A5 ('fidgety or restless')", "criterion-linked agitation manifestation"),
        ("motor restlessness", "manifestation", True, "clinical (motor restlessness)", "documented agitation manifestation"),
        ("slowed speech", "manifestation", True, "DSM-5 A5 ('speaking so slowly')", "criterion-linked retardation manifestation"),
        ("speech latency", "manifestation", True, "clinical (increased speech latency)", "documented retardation sign"),
        ("decreased motor activity", "manifestation", True, "clinical descriptor of the retardation pole", "documented retardation manifestation"),
        ("excessive motor activity", "manifestation", True, "DSM-5 A5 ('moving around a lot more than usual')", "criterion-linked agitation manifestation"),
        # REMOVED
        ("bradykinesia", "related_nonspecific", False, "MeSH 'Bradykinesia'; neurological (Parkinsonism)", "neurological motor slowing, not psychiatric psychomotor retardation"),
        ("hypokinesia", "related_nonspecific", False, "MeSH 'Hypokinesia'; neurological", "neurological reduced movement, not the A5 construct"),
        ("hyperactivity", "related_nonspecific", False, "ADHD terminology", "non-specific (ADHD), not psychomotor agitation"),
        ("akathisia", "related_nonspecific", False, "MeSH 'Akathisia, Drug-Induced'", "drug-induced movement disorder; risks medication-side-effect content"),
        ("motor inhibition", "related_nonspecific", False, "LLM brainstorm; ambiguous", "non-standard; 'inhibition' is ambiguous"),
    ],
}

ITEMS = tuple(AUDIT.keys())


def curated_terms(item_name):
    """Kept clinical terms for the PRIMARY clinical query.

    keep == True AND relation != related_nonspecific.
    """
    return [t for (t, rel, keep, _src, _why) in AUDIT[item_name]
            if keep and rel != "related_nonspecific"]


def all_kept(item_name):
    """All keep == True terms (currently identical to curated_terms, since every
    kept term has a specific relation; kept here as an explicit API)."""
    return [t for (t, _rel, keep, _s, _w) in AUDIT[item_name] if keep]


def curated_query(item_name):
    """Space-joined primary clinical query string for an item."""
    return " ".join(curated_terms(item_name))


def removed_terms(item_name):
    return [t for (t, _rel, keep, _s, _w) in AUDIT[item_name] if not keep]


def write_audit_csv(path=AUDIT_CSV):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "term", "dsm5_construct", "source", "relation",
                    "decision", "in_primary_query", "reason"])
        for item in ITEMS:
            for (term, rel, keep, src, why) in AUDIT[item]:
                in_primary = keep and rel != "related_nonspecific"
                w.writerow([item, term, DSM5_CONSTRUCT[item], src, rel,
                            "keep" if keep else "remove",
                            "yes" if in_primary else "no", why])
    return path


def _summary():
    total = kept = removed = 0
    print(f"{'item':14s} {'candidates':>10} {'kept':>5} {'removed':>7}  kept terms")
    for item in ITEMS:
        cand = AUDIT[item]
        k = curated_terms(item)
        total += len(cand); kept += len(k); removed += len(removed_terms(item))
        print(f"{item:14s} {len(cand):10d} {len(k):5d} {len(removed_terms(item)):7d}  "
              f"{', '.join(k[:4])}...")
    print(f"\nTOTAL candidates={total}  kept={kept}  removed={removed}")
    p = write_audit_csv()
    print(f"Wrote audit: {p}")
    # sanity: every kept term has a real relation; every removed has a reason
    for item in ITEMS:
        for (t, rel, keep, src, why) in AUDIT[item]:
            assert rel in ("direct_construct", "subtype", "manifestation", "related_nonspecific")
            assert src and why, f"{item}/{t}: missing provenance/reason"
            if keep:
                assert rel != "related_nonspecific", f"{item}/{t}: kept but related_nonspecific"
    print("clinical audit self-check OK")


if __name__ == "__main__":
    _summary()
