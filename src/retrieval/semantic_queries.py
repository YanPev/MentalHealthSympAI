"""
Corrected semantic multi-query representation (R2 corrective experiment, Stage B).

The previous full-expansion semantic variant concatenated many lexicon terms
into ONE sequence and encoded that keyword dump as the semantic query. That is
not how sentence encoders are meant to be used. R2 replaces it with a small set
of **natural-language queries** per PHQ-8 item:

    Q0_ITEM     -- the PHQ-8 item meaning as a concise natural sentence (always).
    Q1_LAY      -- one natural sentence describing the accepted lay expressions.
    Q2_CLINICAL -- one natural sentence describing the direct clinical construct
                   and accepted subtypes (using ONLY audited clinical terms).

For the three **bipolar** items (Sleep, Appetite, Moving) the opposing
manifestations are separate queries (poles) rather than collapsed into one:
    Sleep    : insomnia   vs hypersomnia
    Appetite : poor_appetite vs overeating
    Moving   : retardation vs agitation

Scoring (used by semantic_multiquery_retriever):
    SemanticScore(window, config) = max over the ACTIVE queries of
                                    cosine_similarity(query, window)
The per-query scores and the winning query are retained (never averaged).

Which queries are active depends on the lexicon config being evaluated:
    L0_CORE          -> item queries only
    L1_CORE_LAY      -> item + lay
    L2_CORE_CLINICAL -> item + clinical
    L3_FULL          -> item + lay + clinical

    python -m src.retrieval.semantic_queries      # smoke check + write JSON
"""

from pathlib import Path
import json

from src.retrieval.clinical_lexicon_curated import curated_terms, removed_terms

PR = Path(__file__).resolve().parents[2]
QUERIES_JSON = PR / "outputs" / "r2_systematic" / "semantic_queries" / "semantic_queries.json"

# kind produced by each lexicon tier
TIER_KIND = {"core": "item", "lay": "lay", "clinical": "clinical"}
CONFIG_TIERS = {
    "L0_CORE": ("core",),
    "L1_CORE_LAY": ("core", "lay"),
    "L2_CORE_CLINICAL": ("core", "clinical"),
    "L3_FULL": ("core", "lay", "clinical"),
}

# Natural-language queries. Each entry: (id, kind, pole, text).
# Bipolar items carry two poles; unipolar items use pole "default".
SEMANTIC_QUERIES = {
    "NoInterest": [
        ("Q0_ITEM", "item", "default",
         "The person has little interest or pleasure in doing things, and has "
         "lost enjoyment in activities and hobbies they used to like."),
        ("Q1_LAY", "lay", "default",
         "The person doesn't feel like doing anything, has stopped caring about "
         "things, and finds that nothing sounds fun or interesting anymore."),
        ("Q2_CLINICAL", "clinical", "default",
         "The person shows anhedonia — a marked loss of interest or pleasure — "
         "with apathy, social withdrawal, and reduced motivation."),
    ],
    "Depressed": [
        ("Q0_ITEM", "item", "default",
         "The person feels down, depressed, sad, or hopeless, with a "
         "persistently low mood."),
        ("Q1_LAY", "lay", "default",
         "The person feels low and tearful, down in the dumps and empty inside, "
         "and like there is no hope that things will get better."),
        ("Q2_CLINICAL", "clinical", "default",
         "The person has a depressed, dysphoric mood with hopelessness and "
         "tearfulness, possibly with melancholic features."),
    ],
    "Sleep": [
        ("Q0_ITEM", "item", "insomnia",
         "The person has trouble falling asleep or staying asleep, lying awake "
         "at night with disturbed, restless sleep."),
        ("Q0_ITEM", "item", "hypersomnia",
         "The person is sleeping too much, oversleeping and spending excessive "
         "time asleep during the day."),
        ("Q1_LAY", "lay", "insomnia",
         "The person can't sleep, lies awake tossing and turning, and wakes up "
         "in the middle of the night."),
        ("Q1_LAY", "lay", "hypersomnia",
         "The person sleeps all day, oversleeps, and finds it hard to get out "
         "of bed."),
        ("Q2_CLINICAL", "clinical", "insomnia",
         "The person has insomnia with prolonged sleep latency, early morning "
         "awakening, and fragmented, non-restorative sleep."),
        ("Q2_CLINICAL", "clinical", "hypersomnia",
         "The person has hypersomnia with excessive daytime sleepiness and "
         "somnolence."),
    ],
    "Tired": [
        ("Q0_ITEM", "item", "default",
         "The person feels tired and has little energy, feeling fatigued and "
         "worn out."),
        ("Q1_LAY", "lay", "default",
         "The person feels worn out and drained, with no energy, like "
         "everything is an effort and they can barely get going."),
        ("Q2_CLINICAL", "clinical", "default",
         "The person has fatigue and loss of energy, with anergia, lethargy, "
         "asthenia, and easy fatigability."),
    ],
    "Appetite": [
        ("Q0_ITEM", "item", "poor_appetite",
         "The person has a poor appetite, is eating less, and has lost interest "
         "in food."),
        ("Q0_ITEM", "item", "overeating",
         "The person is overeating and eating more than usual."),
        ("Q1_LAY", "lay", "poor_appetite",
         "The person isn't hungry, skips meals or forgets to eat, and has to "
         "force themselves to eat."),
        ("Q1_LAY", "lay", "overeating",
         "The person can't stop eating, snacks constantly, and eats for "
         "comfort."),
        ("Q2_CLINICAL", "clinical", "poor_appetite",
         "The person has decreased appetite with appetite loss and weight loss, "
         "showing anorexia."),
        ("Q2_CLINICAL", "clinical", "overeating",
         "The person has increased appetite with hyperphagia, food cravings, "
         "and binge or emotional eating."),
    ],
    "Failure": [
        ("Q0_ITEM", "item", "default",
         "The person feels bad about themselves, feels like a failure, or feels "
         "they have let themselves or their family down."),
        ("Q1_LAY", "lay", "default",
         "The person hates themselves, feels not good enough and worthless, "
         "blames themselves, and feels like a burden."),
        ("Q2_CLINICAL", "clinical", "default",
         "The person has feelings of worthlessness and excessive or "
         "inappropriate guilt, with self-blame, self-criticism, and low "
         "self-worth."),
    ],
    "Concentrating": [
        ("Q0_ITEM", "item", "default",
         "The person has trouble concentrating on things such as reading or "
         "watching television, and struggles to focus or make decisions."),
        ("Q1_LAY", "lay", "default",
         "The person can't focus, their mind wanders and goes blank, they keep "
         "forgetting things, and they find it hard to make decisions."),
        ("Q2_CLINICAL", "clinical", "default",
         "The person has impaired concentration and a diminished ability to "
         "think, with distractibility, inattention, indecisiveness, and "
         "cognitive slowing."),
    ],
    "Moving": [
        ("Q0_ITEM", "item", "retardation",
         "The person is moving or speaking so slowly that other people could "
         "have noticed."),
        ("Q0_ITEM", "item", "agitation",
         "The person is so fidgety or restless that they have been moving "
         "around a lot more than usual."),
        ("Q1_LAY", "lay", "retardation",
         "The person moves slowly, everything takes longer, and they talk "
         "slower and drag their feet."),
        ("Q1_LAY", "lay", "agitation",
         "The person can't sit still, feels antsy and on edge, and paces "
         "around fidgeting."),
        ("Q2_CLINICAL", "clinical", "retardation",
         "The person shows psychomotor retardation with motor and cognitive "
         "slowing, slowed speech, and increased speech latency."),
        ("Q2_CLINICAL", "clinical", "agitation",
         "The person shows psychomotor agitation with restlessness and "
         "excessive, restless motor activity."),
    ],
}

ITEMS = tuple(SEMANTIC_QUERIES.keys())
BIPOLAR = {"Sleep": ("insomnia", "hypersomnia"),
           "Appetite": ("poor_appetite", "overeating"),
           "Moving": ("retardation", "agitation")}


def poles(item_name):
    return list(BIPOLAR.get(item_name, ("default",)))


def queries_for(item_name):
    """All query dicts for an item."""
    return [{"id": qid, "kind": kind, "pole": pole, "text": text}
            for (qid, kind, pole, text) in SEMANTIC_QUERIES[item_name]]


def active_queries(item_name, config="L3_FULL"):
    """Queries active for a lexicon config (by kind)."""
    active_kinds = {TIER_KIND[t] for t in CONFIG_TIERS[config]}
    return [q for q in queries_for(item_name) if q["kind"] in active_kinds]


def clinical_terms_in(item_name, text):
    """Curated clinical terms that appear (as substrings) in a query text."""
    low = text.lower()
    return [t for t in curated_terms(item_name) if t.lower() in low]


def build_payload():
    payload = {}
    for item in ITEMS:
        qlist = []
        for q in queries_for(item):
            entry = dict(q)
            if q["kind"] == "clinical":
                entry["clinical_terms_used"] = clinical_terms_in(item, q["text"])
            qlist.append(entry)
        payload[item] = {
            "poles": poles(item),
            "bipolar": item in BIPOLAR,
            "n_queries": len(qlist),
            "active_by_config": {c: [f"{q['id']}:{q['pole']}"
                                     for q in active_queries(item, c)]
                                 for c in CONFIG_TIERS},
            "queries": qlist,
        }
    return payload


def write_json(path=QUERIES_JSON):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_payload(), indent=2, ensure_ascii=False))
    return path


def _summary():
    print(f"{'item':14s} {'bipolar':>7} {'poles':>5} {'#Q':>3}  "
          f"L0/L1/L2/L3 active")
    for item in ITEMS:
        counts = "/".join(str(len(active_queries(item, c))) for c in CONFIG_TIERS)
        print(f"{item:14s} {str(item in BIPOLAR):>7} {len(poles(item)):5d} "
              f"{len(queries_for(item)):3d}  {counts}")
    p = write_json()
    print(f"\nWrote: {p}")


if __name__ == "__main__":
    _summary()
