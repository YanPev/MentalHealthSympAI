"""
Lay lexicon tier (R2 corrective experiment, Stage A.2).

The lay tier is the everyday-language layer: spoken paraphrases, informal
symptom descriptions, short multi-word expressions, and indirect but
symptom-specific functional descriptions of each PHQ-8 item.

PROVENANCE: this tier REUSES the previously-reviewed `symptom_lexicon.SYNONYMS`
verbatim (imported, never re-typed). Those terms were curated from how people
describe the symptom in everyday language plus a reviewed offline-LLM lay
brainstorm; per the R2 brief, reused reviewed lay terms are allowed as long as
their manifest is made explicit. Crucially, NONE of the lay terms was derived
from DAIC-WOZ transcripts, labels, predictions, error cases, or test-set
inspection — they describe the symptom constructs only, so the tier is
label-free and fold-independent by construction.

Unlike the clinical tier, lay terms do NOT require documented clinical
provenance; the manifest instead records, per term, whether it is a
`direct_paraphrase` of the symptom or an `indirect_manifestation` (a downstream
functional/behavioural consequence), via the transparent heuristic below.

    python -m src.retrieval.lay_lexicon        # manifest summary
"""

from src.retrieval.symptom_lexicon import SYNONYMS

# Reuse the reviewed lay expressions (imported, not copied).
LAY = SYNONYMS

ITEMS = tuple(LAY.keys())

# Indirect-manifestation cue phrases: terms describing a *consequence* of the
# symptom (withdrawal, avoidance, functional impact) rather than naming the
# symptom itself. Transparent, documented heuristic used to fill the manifest's
# direct/indirect column; it never changes which terms are included.
_INDIRECT_CUES = (
    "don't go out", "keep to myself", "withdrawn", "stopped", "gave up",
    "no point", "can't get out of bed", "force myself", "skip meals",
    "forget to eat", "stay in bed", "hard to get up", "hard to get going",
    "everything is an effort", "everything takes effort", "no get up and go",
    "going through the motions", "no longer interested in friends",
    "stop participating", "don't look forward", "hit the hay", "clock watching",
    "dragging my feet", "pace around", "pacing", "leg bouncing", "lose track",
)


def classify_relation(term: str) -> str:
    """direct_paraphrase vs indirect_manifestation (documented heuristic)."""
    t = term.lower()
    if any(cue in t for cue in _INDIRECT_CUES):
        return "indirect_manifestation"
    return "direct_paraphrase"


def lay_terms(item_name):
    """The lay terms for an item (multi-word expressions kept intact)."""
    return list(LAY[item_name])


def lay_query(item_name):
    """Space-joined lay query string."""
    return " ".join(LAY[item_name])


def manifest_rows():
    """Yield (item, term, source_type, relation, decision, reason)."""
    for item in ITEMS:
        for term in LAY[item]:
            rel = classify_relation(term)
            is_mwe = " " in term
            src = "reviewed_lay_expression" + ("_mwe" if is_mwe else "")
            yield (item, term, src, rel, "accept",
                   "reviewed everyday expression, symptom-specific, "
                   "not transcript-derived")


def _summary():
    n = 0
    print(f"{'item':14s} {'lay_terms':>9} {'direct':>7} {'indirect':>9} {'mwe':>5}")
    for item in ITEMS:
        terms = LAY[item]
        direct = sum(1 for t in terms if classify_relation(t) == "direct_paraphrase")
        indirect = len(terms) - direct
        mwe = sum(1 for t in terms if " " in t)
        n += len(terms)
        print(f"{item:14s} {len(terms):9d} {direct:7d} {indirect:9d} {mwe:5d}")
    print(f"\nTOTAL lay terms = {n}")
    # sanity
    assert set(ITEMS) == set(SYNONYMS)
    for item in ITEMS:
        assert LAY[item], f"{item}: empty lay tier"
    print("lay_lexicon self-check OK")


if __name__ == "__main__":
    _summary()
