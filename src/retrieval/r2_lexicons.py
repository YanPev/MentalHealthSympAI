"""
R2 tiered lexicon configurations L0-L3 (Stage A).

Assembles four explicit lexicon configurations from three provenance-separated
tiers, so the contribution of the lay and clinical layers can be measured
independently:

    L0_CORE           core only              (= frozen symptom_glossary.EXPAND)
    L1_CORE_LAY       core + lay             (reviewed everyday expressions)
    L2_CORE_CLINICAL  core + clinical        (provenance-audited clinical terms)
    L3_FULL           core + lay + clinical

Tiers:
    core     -- symptom_glossary.EXPAND, imported (never copied) -> byte-identical
                to R0, so the frozen keyword-hit-rate metric stays comparable.
    lay      -- lay_lexicon.LAY (reused reviewed synonyms; label-free).
    clinical -- clinical_lexicon_curated.curated_terms (primary clinical query;
                related_nonspecific terms already excluded).

APIs mirror the previous `symptom_lexicon`:
    expanded_query(item, config) -> query-expansion string (BM25/semantic query)
    flat_terms(item, config)     -> lowercase match terms for keyword hit-rate

    python -m src.retrieval.r2_lexicons          # smoke check + write manifest
"""

from pathlib import Path
import csv

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.retrieval.bm25_retriever import tokenize
from src.retrieval.symptom_glossary import EXPAND
from src.retrieval.lay_lexicon import LAY, manifest_rows as lay_manifest_rows
from src.retrieval.clinical_lexicon_curated import (
    curated_terms, AUDIT, DSM5_CONSTRUCT)

PR = Path(__file__).resolve().parents[2]
MANIFEST_CSV = PR / "outputs" / "r2_systematic" / "lexicons" / "lexicon_manifest.csv"

CONFIGS = {
    "L0_CORE": ("core",),
    "L1_CORE_LAY": ("core", "lay"),
    "L2_CORE_CLINICAL": ("core", "clinical"),
    "L3_FULL": ("core", "lay", "clinical"),
}
ITEMS = tuple(EXPAND.keys())


def _tier_terms(item_name, tier):
    if tier == "core":
        return [EXPAND[item_name]]           # kept as the raw glossary string
    if tier == "lay":
        return list(LAY[item_name])
    if tier == "clinical":
        return curated_terms(item_name)
    raise ValueError(f"unknown tier: {tier}")


def core_terms(item_name):
    """EXPAND tokenized exactly like the frozen keyword-hit-rate metric."""
    return sorted({t for t in tokenize(EXPAND[item_name])
                   if t not in ENGLISH_STOP_WORDS and len(t) > 2})


def expanded_query(item_name, config="L3_FULL"):
    """Query-expansion string for an item under a lexicon config.

    Core contributes its raw glossary string; lay/clinical contribute their
    space-joined term lists. (The item_text itself is prepended by the caller,
    exactly as build_expanded_prod_hybrid did.)
    """
    tiers = CONFIGS[config]
    parts = []
    if "core" in tiers:
        parts.append(EXPAND[item_name])
    if "lay" in tiers:
        parts.append(" ".join(LAY[item_name]))
    if "clinical" in tiers:
        parts.append(" ".join(curated_terms(item_name)))
    return " ".join(p for p in parts if p).strip()


def flat_terms(item_name, config="L3_FULL"):
    """Lowercase match terms for keyword-hit-rate (multi-word expressions kept)."""
    tiers = CONFIGS[config]
    terms = set()
    if "core" in tiers:
        terms.update(core_terms(item_name))
    for tier in ("lay", "clinical"):
        if tier in tiers:
            for t in _tier_terms(item_name, tier):
                t = t.lower().strip()
                if len(t) > 2 and (" " in t or t not in ENGLISH_STOP_WORDS):
                    terms.add(t)
    return sorted(terms)


def write_manifest(path=MANIFEST_CSV):
    """Unified manifest across all three tiers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "tier", "term", "source_type", "relation",
                    "decision", "in_primary_query", "reason"])
        # core
        for item in ITEMS:
            w.writerow([item, "core", EXPAND[item], "frozen_glossary(EXPAND)",
                        "direct_construct", "accept", "yes",
                        "byte-identical frozen R0 glossary; kept as-is"])
        # lay
        for (item, term, src, rel, decision, reason) in lay_manifest_rows():
            w.writerow([item, "lay", term, src, rel, decision,
                        "yes" if decision == "accept" else "no", reason])
        # clinical (from the audit)
        for item in ITEMS:
            for (term, rel, keep, src, why) in AUDIT[item]:
                in_primary = keep and rel != "related_nonspecific"
                w.writerow([item, "clinical", term, src, rel,
                            "keep" if keep else "remove",
                            "yes" if in_primary else "no", why])
    return path


def _smoke_check():
    # tier keys align
    assert set(LAY) == set(EXPAND) == set(AUDIT), "tier keys must match EXPAND"
    print(f"{'item':14s} {'core':>5} {'lay':>4} {'clin':>5}  "
          f"{'L0':>4} {'L1':>4} {'L2':>4} {'L3':>4}   (flat_terms per config)")
    for item in ITEMS:
        c = len(core_terms(item)); lyy = len(LAY[item]); cl = len(curated_terms(item))
        sizes = {k: len(flat_terms(item, k)) for k in CONFIGS}
        # L0 must equal the frozen core term count exactly (byte-identical vocab)
        assert sizes["L0_CORE"] == c, f"{item}: L0 != core terms"
        # monotone: adding tiers never shrinks the term set
        assert sizes["L1_CORE_LAY"] >= sizes["L0_CORE"]
        assert sizes["L2_CORE_CLINICAL"] >= sizes["L0_CORE"]
        assert sizes["L3_FULL"] >= max(sizes["L1_CORE_LAY"], sizes["L2_CORE_CLINICAL"])
        # every config query starts with the item's core glossary
        for k in CONFIGS:
            q = expanded_query(item, k)
            assert q.startswith(EXPAND[item]), f"{item}/{k}: query lost core"
        print(f"{item:14s} {c:5d} {lyy:4d} {cl:5d}  "
              f"{sizes['L0_CORE']:4d} {sizes['L1_CORE_LAY']:4d} "
              f"{sizes['L2_CORE_CLINICAL']:4d} {sizes['L3_FULL']:4d}")
    p = write_manifest()
    print(f"\nWrote manifest: {p}")
    print("r2_lexicons smoke check OK")


if __name__ == "__main__":
    _smoke_check()
