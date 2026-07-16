"""
Stage I evaluation — MentalBERT+CORN on R2: missing-policy audit, metrics, and
paired participant bootstrap; selects E_FINAL.

Terminology (corrected per brief §11):
  status_quo  (cross_validate --missing-policy zero) : all train rows keep their
              gold label (the label is NOT changed to 0).
  mask_none   (--missing-policy mask) : rows judged `none` stay in the batch but
              contribute zero loss (excluded from the loss + class weights).
  drop_none   (--missing-policy drop) : rows judged `none` removed from the
              training split and class-weight calc.

Two phases:
  --phase audit   : pre-training. Replicates the fold split + policy on the R2
                    evidence_status and writes missing_policy_audit.csv (+ warnings).
  --phase metrics : post-training. Reads the R0 baseline + R2 zero/mask/drop OOF
                    files, computes overall + per-item metrics, paired bootstraps,
                    and selects E_FINAL.

    .venv/bin/python -m src.evaluation.r2_encoder_eval --phase audit
    .venv/bin/python -m src.evaluation.r2_encoder_eval --phase metrics
"""

from pathlib import Path
import argparse
import glob
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

from src.evaluation.r2_metrics import overall, by_item, paired_bootstrap, align_oof

PR = Path(__file__).resolve().parents[2]
CV = PR / "outputs" / "cv"
ENC = PR / "outputs" / "r2_systematic" / "encoder"
RET = PR / "outputs" / "r2_systematic" / "retrieval"
JUD = PR / "outputs" / "r2_systematic" / "judge"
R0_OOF = CV / "oof_predictions_ctxm_corn_hybw3.csv"
R1_OOF = CV / "oof_predictions_ctxm_corn_exphybw3.csv"   # previous R1 encoder
LABELS = [0, 1, 2, 3]


def r2_config():
    sel = json.loads((RET / "r2_selection.json").read_text())["selected_R2"]
    return sel["config"], sel["retriever"], sel["prefix"]


def extract_r2_status(out=ENC / "r2_evidence_status.csv"):
    """evidence_status per (participant,item) for the selected R2 set (budget
    prefix status). Written in cross_validate's expected schema."""
    cfg, ret, pfx = r2_config()
    files = (sorted(glob.glob(str(JUD / "set_judgments_fold*.csv"))) +
             sorted(glob.glob(str(JUD / "set_judgments_hybrid_fold*.csv"))))
    st = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    sub = st[(st.config == cfg) & (st.retriever == ret) & (st.prefix == pfx)]
    sub = sub[["participant_id", "item_id", "status"]].rename(columns={"status": "evidence_status"})
    ENC.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    return out, (cfg, ret, pfx)


def audit():
    status_path, (cfg, ret, pfx) = extract_r2_status()
    status = pd.read_csv(status_path)
    status["participant_id"] = status["participant_id"].astype(str)
    smap = dict(zip(zip(status.participant_id, status.item_id.astype(int)), status.evidence_status))

    base = pd.read_csv(PR / "data" / "processed" / "phq8_item_dataset_r2_w3.csv",
                       usecols=["participant_id", "item_id", "item_name", "label"])
    base["participant_id"] = base["participant_id"].astype(str)
    base["evidence_status"] = [smap.get((p, int(i))) for p, i in
                               zip(base.participant_id, base.item_id)]
    # Replicate the exact fold split (seed 42, grouped by participant, stratified by label).
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y = base["label"].to_numpy(); groups = base["participant_id"].to_numpy()
    base = base.reset_index(drop=True)

    rows = []
    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(base, y, groups), 1):
        tr = base.iloc[tr_idx]
        for iid, g in tr.groupby("item_id"):
            none_mask = g["evidence_status"] == "none"
            gold = g["label"].to_numpy()
            gold_after = g.loc[~none_mask, "label"].to_numpy()
            cw_before = _cw(gold)
            cw_after = _cw(gold_after)
            n3 = int((gold == 3).sum()); n23 = int((gold >= 2).sum())
            none3 = int(((gold == 3) & none_mask.to_numpy()).sum())
            none23 = int(((gold >= 2) & none_mask.to_numpy()).sum())
            classes_after = set(gold_after.tolist())
            rows.append({
                "fold": fold, "item_id": int(iid), "item_name": g.item_name.iloc[0],
                "train_rows": len(g), "none_rows": int(none_mask.sum()),
                "masked_rows": int(none_mask.sum()), "dropped_rows": int(none_mask.sum()),
                "eff_train_rows": int((~none_mask).sum()),
                "pct_removed": round(float(none_mask.mean()), 3),
                "gold3_rows": n3, "gold3_removed": none3,
                "pct_gold3_removed": round(none3 / n3, 3) if n3 else 0.0,
                "pct_gold23_removed": round(none23 / n23, 3) if n23 else 0.0,
                "gold_dist_before": json.dumps({c: int((gold == c).sum()) for c in LABELS}),
                "gold_dist_after": json.dumps({c: int((gold_after == c).sum()) for c in LABELS}),
                "cw_before": json.dumps([round(x, 3) for x in cw_before]),
                "cw_after": json.dumps([round(x, 3) for x in cw_after]),
                "class_absent_after": json.dumps(sorted(set(LABELS) - classes_after)),
                "warn_gt40pct": bool(none_mask.mean() > 0.40),
                "warn_gt30pct_gold3": bool(n3 and none3 / n3 > 0.30),
                "warn_class_disappears": bool(set(LABELS) - classes_after),
            })
    df = pd.DataFrame(rows)
    ENC.mkdir(parents=True, exist_ok=True)
    df.to_csv(ENC / "missing_policy_audit.csv", index=False)
    warned = df[df[["warn_gt40pct", "warn_gt30pct_gold3", "warn_class_disappears"]].any(axis=1)]
    print(f"R2 = {cfg}/{ret}/{pfx}")
    print(f"missing_policy_audit.csv written ({len(df)} fold-item rows).")
    print(f"none-rows overall: {df.none_rows.sum()}/{df.train_rows.sum()} "
          f"({df.none_rows.sum()/df.train_rows.sum():.1%} of train)")
    print(f"WARNINGS: {len(warned)} fold-item cells flagged")
    if len(warned):
        print(warned[["fold", "item_name", "pct_removed", "pct_gold3_removed",
                      "warn_gt40pct", "warn_gt30pct_gold3", "warn_class_disappears"]].to_string(index=False))


def _cw(y):
    y = np.asarray(y)
    present = np.unique(y)
    if len(present) < 2:
        return [np.nan] * 4
    w = compute_class_weight("balanced", classes=present, y=y)
    full = {c: 1.0 for c in LABELS}
    full.update(dict(zip(present.tolist(), w.tolist())))
    return [full[c] for c in LABELS]


def _load(tag_or_path):
    p = Path(tag_or_path)
    if not p.exists():
        p = CV / f"oof_predictions_{tag_or_path}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p); d["participant_id"] = d["participant_id"].astype(str)
    return d


def metrics():
    runs = {"R0": R0_OOF, "R1_prev": R1_OOF,
            "R2_status_quo": "r2_ctxm_corn_zero",
            "R2_mask_none": "r2_ctxm_corn_mask",
            "R2_drop_none": "r2_ctxm_corn_drop"}
    frames = {k: _load(v) for k, v in runs.items()}
    present = {k: v for k, v in frames.items() if v is not None}
    if "R2_status_quo" not in present:
        raise SystemExit("R2 encoder OOFs not found -- run run_r2_encoder.sbatch first")

    ov_rows, item_rows = [], []
    for k, d in present.items():
        o = overall(d); o["run"] = k; ov_rows.append(o)
        bi = by_item(d); bi["run"] = k; item_rows.append(bi)
    ov = pd.DataFrame(ov_rows)
    ENC.mkdir(parents=True, exist_ok=True)
    ov.to_csv(ENC / "encoder_metrics_overall.csv", index=False)
    pd.concat(item_rows).to_csv(ENC / "encoder_metrics_by_item.csv", index=False)

    # paired bootstraps
    boots = {}
    pairs = [("R2_status_quo", "R0"), ("R2_mask_none", "R2_status_quo"),
             ("R2_drop_none", "R2_status_quo")]
    ci_rows = []
    for a, b in pairs:
        if a in present and b in present:
            m = align_oof(present[a], present[b])
            res = paired_bootstrap(m, "pred_0", "pred_1")
            boots[f"{a}_vs_{b}"] = res
            for metric, r in res.items():
                ci_rows.append({"comparison": f"{a}_vs_{b}", "metric": metric,
                                "delta": r["delta"], "ci_lo": r["ci95"][0],
                                "ci_hi": r["ci95"][1], "excludes_0": r["excludes_0"]})
    pd.DataFrame(ci_rows).to_csv(ENC / "encoder_paired_bootstrap_ci.csv", index=False)

    e_final = _select_efinal(ov, boots)
    (ENC / "e_final_selection.json").write_text(json.dumps(e_final, indent=2))
    print("=== encoder metrics (overall) ===")
    print(ov[["run", "macro_f1", "qwk", "mae", "f1_class3", "severe_recall",
              "false_severe_rate"]].round(4).to_string(index=False))
    print("\nE_FINAL:", json.dumps(e_final, indent=2))


def _select_efinal(ov, boots):
    """QWK then macro-F1 primary; constraints on MAE / severe / false-severe vs status_quo."""
    o = ov.set_index("run")
    if "R2_status_quo" not in o.index:
        return {}
    sq = o.loc["R2_status_quo"]
    cands = {}
    for run in ("R2_status_quo", "R2_mask_none", "R2_drop_none"):
        if run not in o.index:
            continue
        r = o.loc[run]
        ok = True
        if run != "R2_status_quo":
            if r["mae"] > sq["mae"] + 0.02 and r["qwk"] <= sq["qwk"]:
                ok = False
            if r["severe_recall"] < sq["severe_recall"] - 0.03:
                ok = False
            if r["false_severe_rate"] > sq["false_severe_rate"] + 0.015:
                ok = False
        cands[run] = {"qwk": float(r["qwk"]), "macro_f1": float(r["macro_f1"]),
                      "mae": float(r["mae"]), "severe_recall": float(r["severe_recall"]),
                      "false_severe_rate": float(r["false_severe_rate"]),
                      "passes_constraints": ok}
    eligible = {k: v for k, v in cands.items() if v["passes_constraints"]}
    winner = max(eligible or cands, key=lambda k: (cands[k]["qwk"], cands[k]["macro_f1"]))
    return {"E_FINAL": winner, "candidates": cands,
            "rule": "max QWK then macro-F1 among policies passing MAE/severe/false-severe constraints"}


def main():
    ap = argparse.ArgumentParser(description="Stage I encoder eval")
    ap.add_argument("--phase", choices=["audit", "metrics", "extract-status"], required=True)
    args = ap.parse_args()
    if args.phase == "extract-status":
        p, r2 = extract_r2_status(); print(f"R2={r2} status -> {p}")
    elif args.phase == "audit":
        audit()
    else:
        metrics()


if __name__ == "__main__":
    main()
