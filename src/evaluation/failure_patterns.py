"""Phase 6 — recurring failure-pattern analysis (quantitative + qualitative).

Consumes an OOF prediction file, the LLM evidence-quality judge output, and the
retrieval dataset (for evidence text), and reports the failure structure the
research plan asks for:

  * items with chronically low F1 / QWK
  * metrics conditioned on evidence_status (esp. the no-evidence slice)
  * severe-case misses (gold 3 predicted <=1) and under-calling asymmetry
  * recurrent false positives (gold 0 predicted >=2)
  * explicit vs indirect self-report, via the keyword x judge cross:
       supports & keyword-hit  -> explicitly phrased (lexical match)
       supports & no keyword-hit -> indirect / functional-consequence phrasing
    (this is exactly the gap the NoInterest 0.995-keyword vs 0.02-LLM audit flagged)

Writes a metrics JSON + a stratified qualitative-examples markdown (k examples
per failure bucket, with the retrieved evidence text and the judge's reason).

    python -m src.evaluation.failure_patterns          # defaults to R1 encoder
    python -m src.evaluation.failure_patterns --oof outputs/cv/oof_predictions_mil_exphybw3.csv
"""
from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd

from src.evaluation.model_comparison_eval import comprehensive
from src.retrieval.symptom_lexicon import flat_terms

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "outputs" / "cv"
COT = ROOT / "outputs" / "cot"
PROC = ROOT / "data" / "processed"
LABELS = [0, 1, 2, 3]

DEF_OOF = CV / "oof_predictions_ctxm_corn_exphybw3.csv"
DEF_JUDGE = COT / "evidence_quality_exphyb_bm25q.csv"
DEF_DATA = PROC / "phq8_item_dataset_exphyb_bm25q_w3.csv"
EVIDENCE_COL = "retrieved_context_windows_hybrid_pack"


def clean_evidence(val):
    """Flatten the retrieved-window pack string into readable text."""
    if not isinstance(val, str):
        return ""
    return re.sub(r"\s+", " ", val.replace("[WINDOW_SEP]", " /// ")
                  .replace("[TURN_SEP]", "; ")).strip()


def keyword_hit(item_name, evidence_text):
    """Does the retrieved evidence contain any expanded-lexicon term for the item?
    Multi-word terms matched as substrings, single tokens on word boundaries."""
    t = evidence_text.lower()
    for term in flat_terms(item_name):
        if " " in term:
            if term in t:
                return True
        elif re.search(rf"\b{re.escape(term)}\b", t):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Recurring failure-pattern analysis")
    ap.add_argument("--oof", default=str(DEF_OOF))
    ap.add_argument("--judge", default=str(DEF_JUDGE))
    ap.add_argument("--dataset", default=str(DEF_DATA))
    ap.add_argument("--evidence-column", default=EVIDENCE_COL)
    ap.add_argument("--k-examples", type=int, default=4)
    ap.add_argument("--out-tag", default="exphybw3")
    args = ap.parse_args()

    oof = pd.read_csv(args.oof)
    oof["participant_id"] = oof["participant_id"].astype(str)
    judge = pd.read_csv(args.judge)[["participant_id", "item_id", "evidence_status",
                                     "severity_hint", "reason"]]
    judge["participant_id"] = judge["participant_id"].astype(str)
    data = pd.read_csv(args.dataset)[["participant_id", "item_id", args.evidence_column]]
    data["participant_id"] = data["participant_id"].astype(str)

    df = oof.merge(judge, on=["participant_id", "item_id"], how="left") \
            .merge(data, on=["participant_id", "item_id"], how="left")
    df["evidence_text"] = df[args.evidence_column].map(clean_evidence)
    df["kw_hit"] = [keyword_hit(n, t) for n, t in zip(df["item_name"], df["evidence_text"])]
    df["err"] = df["prediction"] - df["label"]

    report = {"oof": Path(args.oof).name, "n": int(len(df))}

    # ---- 1. chronically weak items (low F1/QWK) ----
    comp = comprehensive(oof)
    per_item = sorted(comp["per_item"], key=lambda d: d["macro_f1"])
    report["weakest_items_by_macro_f1"] = [
        {"item_name": d["item_name"], "macro_f1": d["macro_f1"], "qwk": d["qwk"],
         "mae": d["mae"]} for d in per_item[:4]]

    # ---- 2. metrics conditioned on evidence_status ----
    from sklearn.metrics import f1_score, cohen_kappa_score, mean_absolute_error, accuracy_score
    by_status = {}
    for status, g in df.groupby("evidence_status"):
        y, p = g["label"].to_numpy(), g["prediction"].to_numpy()
        by_status[status] = {
            "n": int(len(g)),
            "accuracy": round(float(accuracy_score(y, p)), 4),
            "macro_f1": round(float(f1_score(y, p, average="macro", labels=LABELS, zero_division=0)), 4),
            "mae": round(float(mean_absolute_error(y, p)), 4),
            "far_off_rate": round(float(np.mean(np.abs(y - p) >= 2)), 4),
            "mean_gold": round(float(y.mean()), 3),
        }
    report["metrics_by_evidence_status"] = by_status

    # ---- 3. severe misses + under-calling asymmetry ----
    sev = df[df["label"] == 3]
    report["severe"] = {
        "n_severe": int(len(sev)),
        "missed_le1_rate": round(float((sev["prediction"] <= 1).mean()), 4) if len(sev) else None,
        "recall": round(float((sev["prediction"] == 3).mean()), 4) if len(sev) else None,
        "mean_signed_err": round(float(sev["err"].mean()), 4) if len(sev) else None,
    }
    report["signed_err_by_gold_class"] = {
        str(c): round(float(df[df["label"] == c]["err"].mean()), 4)
        for c in LABELS if (df["label"] == c).any()}

    # ---- 4. recurrent false positives (gold 0 predicted high) ----
    fp = df[(df["label"] == 0) & (df["prediction"] >= 2)]
    report["false_positives_gold0_pred_ge2"] = {
        "n": int(len(fp)),
        "rate_among_gold0": round(float(((df["label"] == 0) & (df["prediction"] >= 2)).sum()
                                        / max((df["label"] == 0).sum(), 1)), 4),
        "by_item": fp["item_name"].value_counts().to_dict(),
    }

    # ---- 5. explicit vs indirect (keyword x judge cross) ----
    supp = df[df["evidence_status"] == "supports"]
    expl = supp[supp["kw_hit"]]
    indir = supp[~supp["kw_hit"]]
    def acc(g):
        return round(float((g["prediction"] == g["label"]).mean()), 4) if len(g) else None
    report["explicit_vs_indirect"] = {
        "explicit_supports_kwhit": {"n": int(len(expl)), "accuracy": acc(expl),
                                    "mae": round(float((expl["err"].abs()).mean()), 4) if len(expl) else None},
        "indirect_supports_nokw": {"n": int(len(indir)), "accuracy": acc(indir),
                                   "mae": round(float((indir["err"].abs()).mean()), 4) if len(indir) else None},
        "note": "indirect = judge says supports but no lexical term matched "
                "(symptom expressed via functional consequences / paraphrase)",
    }

    outp = COT / f"failure_patterns_{args.out_tag}.json"
    outp.write_text(json.dumps(report, indent=2))
    print(f"wrote {outp}")

    # ---- qualitative examples markdown ----
    buckets = {
        "severe_missed (gold 3, pred <=1)": df[(df.label == 3) & (df.prediction <= 1)],
        "false_positive (gold 0, pred >=2)": df[(df.label == 0) & (df.prediction >= 2)],
        "indirect_evidence (supports, no keyword)": indir,
        "no_evidence_but_symptom (status none, gold >=2)":
            df[(df.evidence_status == "none") & (df.label >= 2)],
        "contradiction (status against, gold >=2)":
            df[(df.evidence_status == "against") & (df.label >= 2)],
    }
    lines = [f"# Failure-pattern qualitative examples ({Path(args.oof).name})", ""]
    rng = np.random.default_rng(42)
    for name, g in buckets.items():
        lines.append(f"## {name}  (n={len(g)})\n")
        if len(g) == 0:
            lines.append("_none_\n")
            continue
        take = g.iloc[rng.choice(len(g), min(args.k_examples, len(g)), replace=False)]
        for _, r in take.iterrows():
            ev = (r["evidence_text"] or "")[:400] or "(no evidence retrieved)"
            lines.append(
                f"- **{r['item_name']}** · participant `{r['participant_id']}` · "
                f"gold **{r['label']}**, pred **{r['prediction']}** · "
                f"judge `{r.get('evidence_status')}` "
                f"(hint {r.get('severity_hint')}): _{r.get('reason')}_\n"
                f"  - evidence: {ev}")
        lines.append("")
    mdp = COT / f"failure_patterns_{args.out_tag}.md"
    mdp.write_text("\n".join(lines))
    print(f"wrote {mdp}")

    # console summary
    print("\nweakest items:", [d["item_name"] for d in report["weakest_items_by_macro_f1"]])
    print("by evidence_status (macroF1):",
          {s: v["macro_f1"] for s, v in by_status.items()})
    print("severe missed<=1 rate:", report["severe"]["missed_le1_rate"])
    print("explicit vs indirect acc:",
          report["explicit_vs_indirect"]["explicit_supports_kwhit"]["accuracy"],
          "vs", report["explicit_vs_indirect"]["indirect_supports_nokw"]["accuracy"])


if __name__ == "__main__":
    main()
