"""
Label-free retrieval-variant selection.

Compares the frozen hybrid-w3 baseline (R0) against the expanded-dictionary
variants on two label-free signals:

  * LLM-judged informative rate  = P(status in {supports, against})
    from evidence_quality_<tag>.json (llm_evidence_quality.py)
  * keyword coverage under both lexicons
    from retrieval_window_analysis_<tag>_{glossary,expanded}.json

The winner (R1) is the expanded variant with the highest mean informative
rate. Neither signal uses gold labels, so selection introduces no leakage and
needs no nested CV. Sanity-anchors the R0 judge run against the earlier
binary-relevance audit (llm_evidence_relevance.json).

    python -m src.evaluation.select_retrieval_variant
"""

from pathlib import Path
import argparse
import json

import pandas as pd

PR = Path(__file__).resolve().parents[2]
COT = PR / "outputs" / "cot"
OUT_CSV = COT / "retrieval_variant_selection.csv"
OUT_JSON = COT / "retrieval_variant_selection.json"

VARIANTS = ["r0hybw3", "expbm25", "exphyb_bm25q", "exphyb_bothq"]
CANDIDATES = ["expbm25", "exphyb_bm25q", "exphyb_bothq"]   # R0 excluded from argmax


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def kw_rates(tag, lexicon):
    obj = load_json(COT / f"retrieval_window_analysis_{tag}_{lexicon}.json")
    if obj is None and tag == "r0hybw3" and lexicon == "glossary":
        obj = load_json(COT / "retrieval_window_analysis.json")   # frozen baseline
    if obj is None:
        return {}
    return {r["item"]: r["keyword_hit_rate"] for r in obj["per_item"]}


def main():
    ap = argparse.ArgumentParser(description="Label-free retrieval-variant selection")
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--candidates", nargs="+", default=CANDIDATES)
    args = ap.parse_args()

    rows = []
    per_variant = {}
    for tag in args.variants:
        evq = load_json(COT / f"evidence_quality_{tag}.json")
        if evq is None:
            print(f"[warn] evidence_quality_{tag}.json missing -- skipping {tag}")
            continue
        kw_gl = kw_rates(tag, "glossary")
        kw_ex = kw_rates(tag, "expanded")
        for it in evq["per_item"]:
            rows.append({
                "variant": tag, "item_id": it["item_id"], "item": it["item"],
                "informative_rate": it["informative_rate"],
                "none_rate": it["none_rate"],
                "supports": it["status_counts"]["supports"],
                "against": it["status_counts"]["against"],
                "ambiguous": it["status_counts"]["ambiguous"],
                "none": it["status_counts"]["none"],
                "kw_hit_glossary": kw_gl.get(it["item"]),
                "kw_hit_expanded": kw_ex.get(it["item"]),
            })
        per_variant[tag] = {
            "mean_informative_rate": float(pd.DataFrame(evq["per_item"])["informative_rate"].mean()),
            "mean_none_rate": float(pd.DataFrame(evq["per_item"])["none_rate"].mean()),
            "parse_rate": evq.get("parse_rate"),
        }

    if not rows:
        raise SystemExit("no evidence_quality_*.json found -- run the judge array first")

    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)

    candidates = {t: v for t, v in per_variant.items() if t in args.candidates}
    winner = (max(candidates, key=lambda t: candidates[t]["mean_informative_rate"])
              if candidates else None)

    # sanity anchor: R0 judge vs the earlier binary relevance audit
    anchor = {}
    rel = load_json(COT / "llm_evidence_relevance.json")
    if rel and "r0hybw3" in per_variant:
        evq_r0 = load_json(COT / "evidence_quality_r0hybw3.json")
        inf = {r["item"]: r["informative_rate"] for r in evq_r0["per_item"]}
        bin_ = {r["item"]: r["llm_hit_rate"] for r in rel["per_item"]}
        common = sorted(set(inf) & set(bin_))
        anchor = {"items": common,
                  "informative_rate_r0": [inf[i] for i in common],
                  "binary_llm_hit_rate": [bin_[i] for i in common]}

    OUT_JSON.write_text(json.dumps({
        "per_variant": per_variant,
        "selected_R1": winner,
        "selection_rule": "argmax mean informative_rate over expanded candidates (label-free)",
        "binary_relevance_anchor": anchor}, indent=2))

    print(json.dumps(per_variant, indent=2))
    print(f"\nSelected R1 = {winner}")
    print(f"Saved: {OUT_CSV}\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
