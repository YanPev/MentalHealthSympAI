"""Phase 6 — how the expanded retrieval (R0 -> R1) moves every metric.

For each of the project's model configurations we recompute the full diagnostic
suite (via model_comparison_eval.comprehensive) on the OLD retrieval (R0 = the
frozen Hybrid-W3 evidence) and the NEW retrieval (R1 = exphyb_bm25q, the
label-free-selected expanded variant), then report the delta on:

    macro-F1, QWK, MAE, accuracy, far-off rate (|err|>=2), class-3 F1,
    per-item macro-F1/QWK, signed severe under/over-calling, and the
    LLM-judged evidence-quality (informative rate) that drove the change.

Participant-cluster bootstrap 95% CIs are attached to the headline item-level
deltas (resample whole participants, the 8 item rows move together).

Configurations (a config is skipped, with a note, if its inputs are absent —
so this runs now on the encoder/policy arms and again once the R1 CoT folds
land for the LLM-only and cascade arms):

    encoder ctxm_corn     R0 ctxm_corn_hybw3        R1 ctxm_corn_exphybw3
    encoder MIL           R0 mil_hybw3              R1 mil_exphybw3
    LLM-only (pooled)     R0 folds_tolerant_sc5*    R1 folds_exphybw3
    cascade MIL+merged    R0 (R0 LLM + mil_hybw3)   R1 (R1 LLM + mil_exphybw3)

Plus the missing-evidence training-policy comparison on R1 (all three policies
share R1 retrieval and identical folds): zero / mask / drop.

    python -m src.evaluation.retrieval_effect_analysis
"""
from pathlib import Path
import glob
import json

import numpy as np
import pandas as pd

from src.evaluation.model_comparison_eval import comprehensive
from src.evaluation.cot_cascade import (
    load_llm, load_encoder, align, gate_mask, apply_cascade, NUM_LABELS,
)

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "outputs" / "cv"
COT = ROOT / "outputs" / "cot"
OUT = COT / "retrieval_effect_analysis.json"

R0_LLM = [str(COT / "folds_tolerant_sc5" / "*.csv"),
          str(COT / "folds_tolerant_sc5_dv" / "*.csv")]
R1_LLM = [str(COT / "folds_exphybw3" / "*.csv")]
GATE, TAU, DIFF = "merged", 0.8, 0.5          # frozen cascade gate (not re-tuned)
B, SEED = 2000, 42

# evidence-quality (LLM-judged informative rate) per retrieval, per item
EVQ = {"R0": COT / "evidence_quality_r0hybw3.json",
       "R1": COT / "evidence_quality_exphyb_bm25q.json"}


# --------------------------------------------------------------------------- #
# building a (participant_id, item_id, label, prediction, prob_*) frame per cfg
# --------------------------------------------------------------------------- #
def _oof_frame(path):
    if not Path(path).exists():
        return None
    return pd.read_csv(path)


def _llm_folds_complete(globs, enc_path):
    """A retrieval arm's LLM run must cover ALL CV folds before we compare it to a
    full-data R0 baseline. Otherwise align()'s inner-join silently restricts R1 to
    whatever folds finished, and the R1-vs-R0 delta is computed on mismatched
    participant sets (fold-1-only R1 vs full R0). Require the aligned LLM frame to
    span every encoder fold and cover >=95% of the encoder OOF rows."""
    if not any(glob.glob(g) for g in globs) or not Path(enc_path).exists():
        return False
    enc = load_encoder(str(enc_path))
    aligned = align(load_llm(globs), enc)
    return (aligned["fold"].nunique() == enc["fold"].nunique()
            and len(aligned) >= 0.95 * len(enc))


def _llm_frame(globs, enc_path):
    """LLM-alone predictions (pooled vote argmax), aligned to an encoder OOF so
    the row set + labels + folds match the other configs exactly."""
    if not _llm_folds_complete(globs, enc_path):
        return None
    df = align(load_llm(globs), load_encoder(str(enc_path)))
    P = df[[f"llm_{c}" for c in range(NUM_LABELS)]].to_numpy()
    out = df[["participant_id", "item_id", "label", "fold"]].copy()
    out["item_name"] = df["item_id"].map(ITEM_NAMES)
    out["prediction"] = P.argmax(1)
    for c in range(NUM_LABELS):
        out[f"prob_{c}"] = P[:, c]
    return out


def _cascade_frame(globs, enc_path):
    if not _llm_folds_complete(globs, enc_path):
        return None
    df = align(load_llm(globs), load_encoder(str(enc_path)))
    mask = gate_mask(df, GATE, TAU, DIFF)
    pred = apply_cascade(df, mask)
    out = df[["participant_id", "item_id", "label", "fold"]].copy()
    out["item_name"] = df["item_id"].map(ITEM_NAMES)
    out["prediction"] = pred
    for c in range(NUM_LABELS):        # one-hot to satisfy the OOF prob schema
        out[f"prob_{c}"] = (pred == c).astype(float)
    return out


ITEM_NAMES = {1: "NoInterest", 2: "Depressed", 3: "Sleep", 4: "Tired",
              5: "Appetite", 6: "Failure", 7: "Concentrating", 8: "Moving"}


# --------------------------------------------------------------------------- #
# metrics: reuse comprehensive() + a few severe-specific extras
# --------------------------------------------------------------------------- #
def severe_calling(df):
    """Under/over-calling of the severe class from the confusion structure."""
    y, p = df["label"].to_numpy(), df["prediction"].to_numpy()
    sev = y == 3
    nonsev = y < 3
    return {
        "n_severe": int(sev.sum()),
        "severe_recall": round(float((p[sev] == 3).mean()), 4) if sev.any() else None,
        "severe_undercall_rate": round(float((p[sev] < 3).mean()), 4) if sev.any() else None,
        "severe_mean_signed_err": round(float((p[sev] - y[sev]).mean()), 4) if sev.any() else None,
        "false_severe_rate": round(float((p[nonsev] == 3).mean()), 4) if nonsev.any() else None,
        "mean_signed_err": round(float((p - y).mean()), 4),
    }


def flat_metrics(df):
    """Flatten comprehensive() into the scalar row we compare across R0/R1."""
    c = comprehensive(df)
    i, t = c["item"], c["total"]
    row = {
        "n": i["n"],
        "accuracy": i["accuracy"], "macro_f1": i["macro_f1"],
        "mae": i["mae"], "qwk": i["qwk"],
        "f1_class3": i["f1_per_class"]["3"], "f1_class2": i["f1_per_class"]["2"],
        "top2_accuracy": i.get("top2_accuracy"),
        "off_by_one_rate": i.get("off_by_one_rate"),
        "far_off_rate": round(float((np.abs(df["label"] - df["prediction"]) >= 2).mean()), 4),
        "total_mae": t["total_mae"], "total_qwk": t["total_qwk"],
        "total_pearson_r": t["pearson_r"],
    }
    row.update(severe_calling(df))
    row["per_item"] = {pi["item_name"]: {"macro_f1": pi["macro_f1"], "qwk": pi["qwk"],
                                         "mae": pi["mae"]}
                       for pi in c["per_item"]}
    return row


def _boot_delta(df0, df1, keys=("macro_f1", "qwk", "mae", "accuracy", "far_off_rate", "f1_class3")):
    """Cluster-bootstrap 95% CI on R1-R0 for a few headline scalars. Both frames
    are aligned to the same participants, so we resample participants once and
    apply the same index to both."""
    m = df0.merge(df1, on=["participant_id", "item_id"], suffixes=("_0", "_1"))
    pid = m["participant_id"].to_numpy()
    parts = np.unique(pid)
    rows_by = {p: np.where(pid == p)[0] for p in parts}
    y = m["label_0"].to_numpy()
    p0, p1 = m["prediction_0"].to_numpy(), m["prediction_1"].to_numpy()

    def scal(yy, pp):
        from sklearn.metrics import f1_score, cohen_kappa_score, mean_absolute_error, accuracy_score
        return {
            "macro_f1": f1_score(yy, pp, average="macro", labels=[0, 1, 2, 3], zero_division=0),
            "qwk": cohen_kappa_score(yy, pp, weights="quadratic", labels=[0, 1, 2, 3]),
            "mae": mean_absolute_error(yy, pp), "accuracy": accuracy_score(yy, pp),
            "far_off_rate": float(np.mean(np.abs(yy - pp) >= 2)),
            "f1_class3": f1_score(yy, pp, average=None, labels=[0, 1, 2, 3], zero_division=0)[3],
        }
    rng = np.random.default_rng(SEED)
    deltas = {k: np.empty(B) for k in keys}
    for b in range(B):
        idx = np.concatenate([rows_by[p] for p in rng.choice(parts, len(parts), replace=True)])
        s0, s1 = scal(y[idx], p0[idx]), scal(y[idx], p1[idx])
        for k in keys:
            deltas[k][b] = s1[k] - s0[k]
    return {k: {"delta_median": round(float(np.median(v)), 4),
                "ci95": [round(float(np.percentile(v, 2.5)), 4),
                         round(float(np.percentile(v, 97.5)), 4)]}
            for k, v in deltas.items()}


def evidence_quality_delta():
    if not (EVQ["R0"].exists() and EVQ["R1"].exists()):
        return None
    def rates(path):
        d = json.loads(Path(path).read_text())["per_item"]
        return {pi["item"]: pi["informative_rate"] for pi in d}
    r0, r1 = rates(EVQ["R0"]), rates(EVQ["R1"])
    per = {n: {"R0": round(r0[n], 4), "R1": round(r1[n], 4),
               "delta": round(r1[n] - r0[n], 4)} for n in r0}
    return {"mean_informative_R0": round(float(np.mean(list(r0.values()))), 4),
            "mean_informative_R1": round(float(np.mean(list(r1.values()))), 4),
            "mean_delta": round(float(np.mean([per[n]["delta"] for n in per])), 4),
            "per_item": per}


# --------------------------------------------------------------------------- #
def main():
    configs = {
        "encoder_ctxm_corn": (_oof_frame(CV / "oof_predictions_ctxm_corn_hybw3.csv"),
                              _oof_frame(CV / "oof_predictions_ctxm_corn_exphybw3.csv")),
        "encoder_MIL": (_oof_frame(CV / "oof_predictions_mil_hybw3.csv"),
                        _oof_frame(CV / "oof_predictions_mil_exphybw3.csv")),
        "LLM_only": (_llm_frame(R0_LLM, CV / "oof_predictions_ctxm_corn_hybw3.csv"),
                     _llm_frame(R1_LLM, CV / "oof_predictions_ctxm_corn_exphybw3.csv")),
        "cascade_MIL_merged": (_cascade_frame(R0_LLM, CV / "oof_predictions_mil_hybw3.csv"),
                               _cascade_frame(R1_LLM, CV / "oof_predictions_mil_exphybw3.csv")),
    }

    result = {"retrieval": {"R0": "hybw3", "R1": "exphyb_bm25q"},
              "gate": {"type": GATE, "tau": TAU, "diff": DIFF},
              "evidence_quality": evidence_quality_delta(),
              "configs": {}, "skipped": []}

    for name, (d0, d1) in configs.items():
        if d0 is None or d1 is None:
            miss = "R0" if d0 is None else "" + ("R1" if d1 is None else "")
            result["skipped"].append({"config": name, "missing": miss or "R0/R1"})
            print(f"  [skip] {name}: inputs not ready ({miss or 'R0/R1'})")
            continue
        m0, m1 = flat_metrics(d0), flat_metrics(d1)
        delta = {k: round(m1[k] - m0[k], 4) for k in m0
                 if isinstance(m0[k], (int, float)) and isinstance(m1.get(k), (int, float))}
        result["configs"][name] = {
            "R0": m0, "R1": m1, "delta": delta,
            "bootstrap_delta_R1_minus_R0": _boot_delta(d0, d1),
        }
        print(f"  [ok]   {name}: macroF1 {m0['macro_f1']:.3f}->{m1['macro_f1']:.3f} "
              f"({delta['macro_f1']:+.3f})  QWK {m0['qwk']:.3f}->{m1['qwk']:.3f} "
              f"({delta['qwk']:+.3f})  MAE {m0['mae']:.3f}->{m1['mae']:.3f} ({delta['mae']:+.3f})")

    # ---- missing-evidence training-policy comparison on R1 ----
    pol = {}
    for policy, tag in [("zero", "ctxm_corn_exphybw3"),
                        ("mask", "ctxm_corn_exphybw3_mask"),
                        ("drop", "ctxm_corn_exphybw3_drop")]:
        d = _oof_frame(CV / f"oof_predictions_{tag}.csv")
        if d is not None:
            pol[policy] = flat_metrics(d)
    result["missing_evidence_policies_R1"] = pol
    if pol:
        print("\n  missing-evidence policies (R1, ctxm_corn):")
        for policy, m in pol.items():
            print(f"    {policy:5} macroF1 {m['macro_f1']:.3f}  QWK {m['qwk']:.3f}  "
                  f"MAE {m['mae']:.3f}  farOff {m['far_off_rate']:.3f}  "
                  f"sevRecall {m['severe_recall']}")

    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}")
    if result["evidence_quality"]:
        eq = result["evidence_quality"]
        print(f"evidence informative rate: R0 {eq['mean_informative_R0']:.3f} -> "
              f"R1 {eq['mean_informative_R1']:.3f} ({eq['mean_delta']:+.3f})")


if __name__ == "__main__":
    main()
