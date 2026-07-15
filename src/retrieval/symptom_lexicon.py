"""
Tiered per-item symptom lexicon: the expanded retrieval dictionary.

Extends the original hand-curated glossary (symptom_glossary.EXPAND, kept
byte-identical as the frozen baseline vocabulary) with two additional tiers:

  * core      -- EXPAND, imported (never copied): the terms behind the existing
                 item-aware retrieval variants and the keyword-hit-rate metric.
  * synonyms  -- general synonyms, paraphrases and multi-word lay expressions
                 for how DAIC-WOZ participants actually talk about the symptom.
  * clinical  -- professional clinical register (DSM-5 construct terminology
                 for the MDD criteria the PHQ-8 items screen).

Provenance: tiers are curated from the PHQ-8 item definitions and the DSM-5
MDD criterion constructs (plus a reviewed offline-LLM brainstorm pass, see
src/llm/brainstorm_lexicon.py). They are derived from the symptom constructs
only -- never from DAIC-WOZ labels or transcripts -- so the lexicon is
label-free and fold-independent by construction (no leakage).

Consumers:
  * expanded_query()  -> query-expansion string appended to the item text
                         (build_expanded_prod_hybrid.py), analogous to how
                         EXPAND is appended in build_itemaware_prod_hybrid.py.
  * flat_terms()      -> lowercase match terms (multi-word expressions kept
                         intact) for keyword-hit-rate measurement
                         (retrieval_window_analysis.py --lexicon expanded).

    python -m src.retrieval.symptom_lexicon      # smoke check
"""

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.retrieval.bm25_retriever import tokenize
from src.retrieval.symptom_glossary import EXPAND

TIERS = ("core", "synonyms", "clinical")

# General synonyms / paraphrases / multi-word lay expressions.
SYNONYMS = {
    "NoInterest": [
        "lost interest", "nothing sounds fun", "don't enjoy", "not enjoying",
        "can't be bothered", "no motivation", "don't feel like doing anything",
        "stopped caring", "couldn't care less", "gave up", "no point",
        "apathetic", "indifferent", "withdrawn", "keep to myself",
        "don't go out anymore", "not fun anymore", "lost my passion",
        "nothing interests me", "don't look forward to anything",
        # reviewed brainstorm additions (outputs/cot/lexicon_brainstorm.json)
        "activities seem pointless", "bored with life", "doesn't care much",
        "doesn't seek out activities", "doesn't get excited about anything",
        "feel flat", "feels detached", "no enthusiasm",
        "just going through the motions", "no longer interested in friends",
        "nothing feels good", "stop participating", "uninterested in sex",
    ],
    "Depressed": [
        "feeling low", "feel like crying", "tearful", "teary", "gloomy",
        "dark place", "down in the dumps", "heavy hearted", "numb",
        "empty inside", "no hope", "what's the point", "given up hope",
        "can't see things getting better", "feel awful", "feel terrible",
        "hurting inside", "broken", "want to give up", "really down",
        # reviewed brainstorm additions (outputs/cot/lexicon_brainstorm.json)
        "can't find joy", "dejected", "down on life", "downcast",
        "downer mood", "downhearted", "dreary", "feeling gloomy",
        "feeling lost", "forlorn", "morose", "negative outlook", "sorrowful",
        "sullen", "unhappy inside",
    ],
    "Sleep": [
        "can't sleep", "couldn't sleep", "up all night", "wide awake",
        "lie awake", "lay awake", "toss and turn", "barely slept",
        "only slept", "wake up in the middle of the night", "trouble sleeping",
        "hard time sleeping", "oversleep", "sleep all day", "nightmares",
        "bad dreams", "sleeping pills", "melatonin", "sleep schedule",
        "hard to get up",
        # reviewed brainstorm additions (outputs/cot/lexicon_brainstorm.json)
        "can't stay asleep", "clock watching", "day sleeper",
        "hit the hay early", "hit the hay late", "night owl",
        "restless nights", "stay in bed", "worn out from lack of sleep",
        "yawn a lot",
    ],
    "Tired": [
        "worn out", "wiped out", "burned out", "burnt out", "no energy",
        "run down", "no get up and go", "can barely move",
        "everything is an effort", "everything takes effort", "no strength",
        "always sleepy", "sleepy all the time", "can't get out of bed",
        "dragging", "heavy limbs", "no stamina", "out of it",
        # reviewed brainstorm additions (outputs/cot/lexicon_brainstorm.json)
        "always tired", "can't seem to stay awake", "dead tired",
        "exhausted easily", "feel heavy", "just can't muster up",
        "lack of pep", "low on fuel", "need a lot of rest",
        "physically drained", "sleepy most of the time", "slow start",
        "tired even after sleep", "trouble getting going", "weak feeling",
    ],
    "Appetite": [
        "not hungry", "no appetite", "don't feel like eating",
        "forget to eat", "skip meals", "barely eat", "eating less",
        "eating more", "food doesn't taste", "lost weight", "gained weight",
        "put on weight", "comfort eating", "stress eating", "binge",
        "junk food", "no interest in food", "force myself to eat",
        # reviewed brainstorm additions (outputs/cot/lexicon_brainstorm.json)
        "can't stop eating", "comfort food overload", "eat everything in sight",
        "fill up quickly", "food coma", "gorging", "grazing all day",
        "lose interest in cooking", "munchies", "nibbling", "pig out",
        "snack constantly", "stuff my face",
    ],
    "Failure": [
        "hate myself", "not good enough", "blame myself", "beat myself up",
        "screwed up", "messed up", "let everyone down", "letting people down",
        "burden", "no good at anything", "hard on myself", "low self-esteem",
        "disappointed in myself", "embarrassed", "regret", "feel like a loser",
        "everything i do is wrong", "never do anything right",
        # reviewed brainstorm additions (outputs/cot/lexicon_brainstorm.json)
        "always feeling inadequate", "can't live up to expectations",
        "can't shake off guilt", "don't measure up", "feel inadequate",
        "feel like a failure", "feel like crap", "feeling worthless",
        "feeling like a fraud", "feeling unlovable", "incompetent",
        "my own worst critic", "poor self-image", "worthless feeling",
    ],
    "Concentrating": [
        "can't focus", "hard to focus", "can't concentrate", "mind goes blank",
        "zone out", "zoning out", "spaced out", "lose my train of thought",
        "scatterbrained", "keep forgetting", "memory problems", "bad memory",
        "hard to make decisions", "can't decide", "foggy",
        "all over the place", "can't pay attention", "lose track",
        # reviewed brainstorm additions (outputs/cot/lexicon_brainstorm.json)
        "lose interest quickly", "can't concentrate on books",
        "can't follow a story", "can't follow directions", "can't hold attention",
        "can't process info", "can't stay on topic", "can't think straight",
        "confused mind", "difficulty with mental tasks",
        "hard to concentrate on details", "hard to grasp ideas",
        "lose track of conversations", "mind feels jumbled",
    ],
    "Moving": [
        "slow motion", "everything takes longer", "can't sit still",
        "keyed up", "on edge", "jittery", "antsy", "pace around",
        "pacing back and forth", "wound up", "leg bouncing", "talking slower",
        "dragging my feet", "squirming", "twitchy", "moving slower",
        "slower than usual", "hard to get going",
        # reviewed brainstorm additions (outputs/cot/lexicon_brainstorm.json)
        "always on the go", "can't calm down", "can't relax",
        "can't seem to move", "trouble moving", "feeling hyper", "hyperactive",
        "less active", "moving like molasses", "moving less",
        "restless legs", "stiff movements", "twitchy hands",
    ],
}

# Professional clinical terminology (DSM-5 MDD criterion constructs).
CLINICAL = {
    "NoInterest": [
        "anhedonia", "loss of interest", "diminished interest",
        "diminished pleasure", "loss of pleasure", "avolition", "amotivation",
        "apathy", "social withdrawal", "disengagement", "reduced libido",
        # reviewed brainstorm additions (outputs/cot/lexicon_brainstorm.json)
        "diminished ability to experience pleasure", "emotional blunting",
        "hedonic capacity impairment", "interest deficit",
        "loss of interest in usual pleasures", "underarousal",
        "valued activities diminished",
    ],
    "Depressed": [
        "dysphoria", "dysphoric", "depressed mood", "low mood", "hopelessness",
        "despondent", "melancholy", "melancholic", "dysthymia", "tearfulness",
        "flat affect", "emotional numbness", "negative affect",
        # reviewed brainstorm additions (outputs/cot/lexicon_brainstorm.json)
        "cynicism", "depressive thinking", "passivity", "resignation",
        "unhappiness", "wretchedness",
    ],
    "Sleep": [
        "hypersomnia", "sleep disturbance", "sleep onset", "sleep latency",
        "early morning awakening", "fragmented sleep", "non-restorative sleep",
        "nocturnal awakening", "middle insomnia", "terminal insomnia",
        "circadian", "sleep quality",
        # reviewed brainstorm additions (outputs/cot/lexicon_brainstorm.json)
        "excessive daytime sleepiness", "somnolence", "parasomnia",
    ],
    "Tired": [
        "anergia", "lethargy", "exhaustion", "asthenia", "fatigability",
        "low energy", "loss of energy", "lack of energy", "decreased energy",
        "diminished energy",
        # reviewed brainstorm additions (outputs/cot/lexicon_brainstorm.json)
        "chronic fatigue", "debilitation", "malaise", "prostration",
        "reduced vitality", "vigilance impairment", "vital exhaustion",
        "weakness",
    ],
    "Appetite": [
        "anorexia", "hyperphagia", "hyporexia", "appetite loss",
        "decreased appetite", "increased appetite", "appetite disturbance",
        "appetite change", "weight change", "binge eating", "emotional eating",
        # reviewed brainstorm additions (outputs/cot/lexicon_brainstorm.json)
        "compulsive eating", "food craving", "impulsive eating", "polyphagia",
        "psychogenic overeating", "reactive overeating",
    ],
    "Failure": [
        "worthlessness", "excessive guilt", "inappropriate guilt",
        "self-blame", "self-criticism", "self-reproach", "low self-worth",
        "negative self-evaluation", "inadequacy", "shame", "guilty rumination",
        # reviewed brainstorm additions (outputs/cot/lexicon_brainstorm.json)
        "imposter syndrome", "inferiority complex", "internalized criticism",
        "negative self-schema", "self-contempt", "self-defeatism",
        "self-deprecation", "self-disgust", "self-doubt", "self-hatred",
        "self-loathing", "self-pity", "self-punishment",
    ],
    "Concentrating": [
        "impaired concentration", "poor concentration", "distractibility",
        "inattention", "indecisiveness", "cognitive impairment", "brain fog",
        "memory complaints", "diminished ability to think", "cognitive slowing",
        "executive dysfunction",
        # reviewed brainstorm additions (outputs/cot/lexicon_brainstorm.json)
        "attentional difficulties", "cognitive disorganization",
        "cognitive fatigue", "cognitive interference",
        "cognitive processing speed reduction", "difficulty sustaining attention",
        "impaired mental clarity", "poor mental sharpness",
    ],
    "Moving": [
        "psychomotor retardation", "psychomotor agitation",
        "psychomotor slowing", "motor retardation", "agitation",
        "restlessness", "akathisia", "slowed speech", "speech latency",
        "motor restlessness",
        # reviewed brainstorm additions (outputs/cot/lexicon_brainstorm.json)
        "bradykinesia", "decreased motor activity", "excessive motor activity",
        "hyperactivity", "hypokinesia", "motor inhibition",
    ],
}

_TIER_SOURCES = {"synonyms": SYNONYMS, "clinical": CLINICAL}


def core_terms(item_name: str) -> list[str]:
    """EXPAND tokenized exactly like retrieval_window_analysis.LEXICONS."""
    return sorted({t for t in tokenize(EXPAND[item_name])
                   if t not in ENGLISH_STOP_WORDS and len(t) > 2})


def expanded_query(item_name: str, tiers=TIERS) -> str:
    """Vocabulary string to append to the item text as the retrieval query."""
    parts = []
    if "core" in tiers:
        parts.append(EXPAND[item_name])
    for tier in ("synonyms", "clinical"):
        if tier in tiers:
            parts.append(" ".join(_TIER_SOURCES[tier][item_name]))
    return " ".join(parts)


def flat_terms(item_name: str, tiers=TIERS) -> list[str]:
    """Lowercase match terms for hit-rate measurement (MWEs kept intact)."""
    terms = set()
    if "core" in tiers:
        terms.update(core_terms(item_name))
    for tier in ("synonyms", "clinical"):
        if tier in tiers:
            for t in _TIER_SOURCES[tier][item_name]:
                t = t.lower().strip()
                if len(t) > 2 and (" " in t or t not in ENGLISH_STOP_WORDS):
                    terms.add(t)
    return sorted(terms)


def _smoke_check():
    assert set(SYNONYMS) == set(CLINICAL) == set(EXPAND), "tier keys must match EXPAND"
    for name in sorted(EXPAND):
        core = set(core_terms(name))
        syn = {t.lower() for t in SYNONYMS[name]}
        cli = {t.lower() for t in CLINICAL[name]}
        assert core and syn and cli, f"{name}: empty tier"
        for a, b, la, lb in [(core, syn, "core", "synonyms"),
                             (core, cli, "core", "clinical"),
                             (syn, cli, "synonyms", "clinical")]:
            dup = a & b
            assert not dup, f"{name}: duplicate terms across {la}/{lb}: {dup}"
        q = expanded_query(name)
        assert q.startswith(EXPAND[name]) and len(q) > len(EXPAND[name])
        print(f"{name:14s} core={len(core):3d}  synonyms={len(syn):3d}  "
              f"clinical={len(cli):3d}  flat_terms={len(flat_terms(name)):3d}  "
              f"query_chars={len(q)}")
    print("symptom_lexicon smoke check OK")


if __name__ == "__main__":
    _smoke_check()
