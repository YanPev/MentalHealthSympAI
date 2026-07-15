"""Phase 7 — structured clinician-report schema from PHQ-8 model outputs.

Assembles, per participant, a leakage-free JSON record from out-of-fold
predictions + the LLM evidence-quality judge + the retrieved evidence + the
PHQ-8 questionnaire self-report (the gold labels, reframed as "self-report vs
interview-derived"). This is the machine-checkable substrate the prose report is
rendered from — every number the clinician sees traces to a field here.

Per item we record: predicted score (or null when abstained), a confidence
value + category, the evidence status from the judge, supporting snippets, the
questionnaire self-report, and four clinically-meaningful flags:

    contradiction   |self-report - predicted| >= 2, or judge says `against`
                    while the participant self-reported the symptom
    not_discussed   self-reported (>0) but judge found no interview evidence
    interview_only  not self-reported (0) but interview supports it
    followup        any of the above, or a high-confidence severe prediction

Confidence = the model's own max class probability (softmax margin for the
encoder; pooled vote-fraction for the LLM/cascade). Categories: high >=0.60,
moderate >=0.40, low otherwise.

    python -m src.reporting.report_schema --limit 12
"""
from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd

from src.data.phq8_items import PHQ8_ITEMS

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "outputs" / "cv"
COT = ROOT / "outputs" / "cot"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"

DEF_OOF = CV / "oof_predictions_ctxm_corn_exphybw3.csv"
DEF_JUDGE = COT / "evidence_quality_exphyb_bm25q.csv"
DEF_DATA = PROC / "phq8_item_dataset_exphyb_bm25q_w3.csv"
EVIDENCE_COL = "retrieved_context_windows_hybrid_pack"

ITEM_TEXT = {v["symptom_name"]: v["item_text"] for v in PHQ8_ITEMS.values()}
BANDS = [(0, 4, "minimal"), (5, 9, "mild"), (10, 14, "moderate"),
         (15, 19, "moderately severe"), (20, 24, "severe")]


def band(total):
    for lo, hi, name in BANDS:
        if lo <= total <= hi:
            return name
    return "severe"


def conf_category(p):
    return "high" if p >= 0.60 else "moderate" if p >= 0.40 else "low"


def clean_evidence(val, max_snips=3, max_len=240):
    if not isinstance(val, str) or not val.strip():
        return []
    parts = re.split(r"\[WINDOW_SEP\]|///", val)
    snips = []
    for p in parts:
        s = re.sub(r"\s+", " ", p.replace("[TURN_SEP]", "; ")).strip()
        if s:
            snips.append(s[:max_len])
        if len(snips) >= max_snips:
            break
    return snips


def build_schema(pdf, jdf, ddf, evidence_col, abstain_on_none=False):
    """One participant's structured record. `pdf`/`jdf`/`ddf` are that
    participant's slices of the OOF, judge, and dataset frames."""
    pid = str(pdf.iloc[0]["participant_id"])
    prob_cols = [f"prob_{c}" for c in range(4)]
    jmap = {int(r["item_id"]): r for _, r in jdf.iterrows()}
    dmap = {int(r["item_id"]): r for _, r in ddf.iterrows()}

    items, no_ev, followups = [], [], []
    pred_total = 0
    for _, r in pdf.sort_values("item_id").iterrows():
        iid = int(r["item_id"])
        name = r["item_name"]
        probs = r[prob_cols].to_numpy(dtype=float)
        conf = float(np.max(probs)) if probs.sum() > 0 else None
        j = jmap.get(iid)
        status = (j["evidence_status"] if j is not None else "none")
        reason = (j["reason"] if j is not None else None)
        snips = clean_evidence(dmap.get(iid, {}).get(evidence_col)
                               if iid in dmap else None)
        self_report = int(r["label"])   # PHQ-8 questionnaire answer

        abstain = abstain_on_none and status == "none"
        pred = None if abstain else int(r["prediction"])
        if pred is not None:
            pred_total += pred

        contradiction = bool(
            (pred is not None and abs(self_report - pred) >= 2) or
            (status == "against" and self_report >= 1))
        not_discussed = bool(self_report >= 1 and status == "none")
        interview_only = bool(self_report == 0 and status == "supports")
        followup = bool(contradiction or not_discussed or interview_only or
                        (pred is not None and pred == 3 and (conf or 0) >= 0.6))

        rec = {
            "item_id": iid, "name": name, "item_text": ITEM_TEXT.get(name, name),
            "predicted_score": pred,
            "confidence": round(conf, 3) if conf is not None else None,
            "confidence_category": conf_category(conf) if conf is not None else None,
            "evidence_status": status,
            "evidence_reason": reason,
            "evidence_snippets": snips,
            "questionnaire_self_report": self_report,
            "flags": {"contradiction": contradiction, "not_discussed": not_discussed,
                      "interview_only": interview_only, "followup": followup},
        }
        items.append(rec)
        if status == "none":
            no_ev.append(name)
        if followup:
            reasons = [k for k, v in rec["flags"].items() if v and k != "followup"]
            followups.append({"name": name, "why": reasons or ["severe high-confidence"]})

    scored = [it for it in items if it["predicted_score"] is not None]
    mean_conf = float(np.mean([it["confidence"] for it in scored
                               if it["confidence"] is not None])) if scored else 0.0
    self_total = int(pdf["label"].sum())
    return {
        "participant_id": pid,
        "n_items": len(items),
        "phq_total": {
            "predicted_score": pred_total,
            "band": band(pred_total),
            "confidence_category": conf_category(mean_conf),
            "mean_item_confidence": round(mean_conf, 3),
            "self_report_total": self_total,
            "self_report_band": band(self_total),
        },
        "items": items,
        "no_evidence_items": no_ev,
        "followup_items": followups,
    }


def main():
    ap = argparse.ArgumentParser(description="Build clinician-report JSON schema")
    ap.add_argument("--oof", default=str(DEF_OOF))
    ap.add_argument("--judge", default=str(DEF_JUDGE))
    ap.add_argument("--dataset", default=str(DEF_DATA))
    ap.add_argument("--evidence-column", default=EVIDENCE_COL)
    ap.add_argument("--abstain-on-none", action="store_true",
                    help="Emit predicted_score=null where the judge found no "
                         "evidence (the NaN policy), instead of the model's score.")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all participants; else first N (sorted).")
    ap.add_argument("--out", default=str(COT / "clinician_report_schema.json"))
    args = ap.parse_args()

    oof = pd.read_csv(args.oof); oof["participant_id"] = oof["participant_id"].astype(str)
    jdf = pd.read_csv(args.judge); jdf["participant_id"] = jdf["participant_id"].astype(str)
    ddf = pd.read_csv(args.dataset); ddf["participant_id"] = ddf["participant_id"].astype(str)

    pids = sorted(oof["participant_id"].unique())
    if args.limit:
        pids = pids[:args.limit]

    schemas = []
    for pid in pids:
        schemas.append(build_schema(
            oof[oof.participant_id == pid], jdf[jdf.participant_id == pid],
            ddf[ddf.participant_id == pid], args.evidence_column,
            abstain_on_none=args.abstain_on_none))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(schemas, indent=2))
    print(f"wrote {args.out}  ({len(schemas)} participants)")
    nflag = sum(len(s["followup_items"]) for s in schemas)
    ncontra = sum(1 for s in schemas for it in s["items"] if it["flags"]["contradiction"])
    print(f"  {nflag} follow-up flags, {ncontra} contradiction flags across {len(schemas)} participants")
    return schemas


if __name__ == "__main__":
    main()
