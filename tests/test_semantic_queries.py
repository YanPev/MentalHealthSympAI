"""
Tests for the corrected semantic multi-query representation (R2 Stage B, brief §4).

Run:  .venv/bin/python -m pytest tests/test_semantic_queries.py -q
 or:  .venv/bin/python tests/test_semantic_queries.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.retrieval.semantic_queries import (
    SEMANTIC_QUERIES, ITEMS, BIPOLAR, queries_for, active_queries,
    clinical_terms_in, CONFIG_TIERS)
from src.retrieval.clinical_lexicon_curated import curated_terms, removed_terms


def test_every_item_has_Q0_item_query():
    for item in ITEMS:
        q0 = [q for q in queries_for(item) if q["id"] == "Q0_ITEM"]
        assert q0, f"{item}: missing Q0_ITEM"
        # bipolar items must have a Q0 for each pole
        if item in BIPOLAR:
            poles = {q["pole"] for q in q0}
            assert poles == set(BIPOLAR[item]), f"{item}: Q0 missing a pole {poles}"


def test_no_empty_query():
    for item in ITEMS:
        for q in queries_for(item):
            assert q["text"].strip(), f"{item}/{q['id']}: empty query"


def test_no_duplicated_query_text():
    seen = {}
    for item in ITEMS:
        for q in queries_for(item):
            key = q["text"].strip().lower()
            assert key not in seen, f"duplicate query text: {item}/{q['id']} == {seen.get(key)}"
            seen[key] = f"{item}/{q['id']}"


def test_clinical_queries_use_only_audited_terms():
    """No REMOVED clinical term may appear in a clinical query; every listed
    clinical_term_used must be a curated (kept) term."""
    for item in ITEMS:
        removed = [t.lower() for t in removed_terms(item)]
        kept = set(curated_terms(item))
        for q in queries_for(item):
            if q["kind"] != "clinical":
                continue
            low = q["text"].lower()
            for bad in removed:
                assert bad not in low, f"{item}: removed clinical term '{bad}' leaked into clinical query"
            used = clinical_terms_in(item, q["text"])
            assert used, f"{item}: clinical query references no curated term"
            assert set(used) <= kept, f"{item}: clinical_terms_used not a subset of curated"


def test_no_raw_list_dump():
    """A query must read as a sentence, not a concatenated keyword list:
    it must contain several function words and not be a comma-separated dump of
    lexicon terms."""
    for item in ITEMS:
        for q in queries_for(item):
            toks = q["text"].lower().replace(",", " ").replace(".", " ").split()
            n_stop = sum(1 for t in toks if t in ENGLISH_STOP_WORDS)
            assert n_stop >= 3, f"{item}/{q['id']}: too few function words (looks like a list): {q['text']!r}"
            assert "the person" in q["text"].lower(), f"{item}/{q['id']}: not phrased as a natural sentence"
            # a comma-list dump would have most 'words' be content with commas between singles;
            # require the sentence to be meaningfully longer than a bare term join.
            assert len(toks) >= 8, f"{item}/{q['id']}: too short to be a descriptive sentence"


def test_opposing_poles_separate_where_relevant():
    for item in ITEMS:
        pset = {q["pole"] for q in queries_for(item)}
        if item in BIPOLAR:
            assert set(BIPOLAR[item]) <= pset and len(pset) == 2, \
                f"{item}: bipolar poles not represented separately: {pset}"
        else:
            assert pset == {"default"}, f"{item}: unexpected poles {pset}"


def test_active_query_selection_by_config():
    for item in ITEMS:
        n_poles = len(BIPOLAR.get(item, ("default",)))
        # L0 = item only -> exactly n_poles queries; L3 = all kinds present
        assert len(active_queries(item, "L0_CORE")) == n_poles
        kinds_L3 = {q["kind"] for q in active_queries(item, "L3_FULL")}
        assert kinds_L3 == {"item", "lay", "clinical"}
        # L2 excludes lay
        assert all(q["kind"] != "lay" for q in active_queries(item, "L2_CORE_CLINICAL"))
        # L1 excludes clinical
        assert all(q["kind"] != "clinical" for q in active_queries(item, "L1_CORE_LAY"))


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} semantic-query tests passed.")


if __name__ == "__main__":
    _run()
