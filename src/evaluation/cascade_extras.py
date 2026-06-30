"""Extra analyses for the results summary: near-miss ordinal breakdown +
PHQ-8 total-score screening ROC/PR curves, for the top-3 configs + encoder.

Reuses cot_cascade loading/gating. Writes outputs/cot/cascade_extras.json.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from src.evaluation.cot_cascade import (
    load_llm, load_encoder, align, gate_mask, apply_cascade,
    DEFAULT_ENCODER_OOF, NUM_LABELS,
)

LLM_GLOBS = ["outputs/cot/folds_tolerant_sc5/*.csv",
             "outputs/cot/folds_tolerant_sc5_dv/*.csv"]
SCREEN_CUT = 10   # PHQ-8 >= 10 = moderate depression (standard screening cutoff)


def near_miss(yt, yp):
    e = np.abs(yt - yp)
    n = len(yt)
    return {"exact": float(np.mean(e == 0)),
            "off_by_1": float(np.mean(e == 1)),
            "within_1": float(np.mean(e <= 1)),
            "far_off": float(np.mean(e >= 2)),
            "n": int(n)}


def totals(df, pred):
    """Per-participant true & predicted PHQ-8 total (sum of 8 item severities)."""
    t = df[["participant_id", "label"]].copy()
    t["pred"] = pred
    g = t.groupby("participant_id").agg(true_total=("label", "sum"),
                                        pred_total=("pred", "sum"),
                                        n_items=("label", "size"))
    g = g[g["n_items"] == 8]  # only complete participants
    return g["true_total"].to_numpy(), g["pred_total"].to_numpy()


def curves(true_total, pred_total):
    y = (true_total >= SCREEN_CUT).astype(int)
    score = pred_total.astype(float)            # rank by predicted total
    fpr, tpr, _ = roc_curve(y, score)
    prec, rec, _ = precision_recall_curve(y, score)
    return {
        "prevalence": float(y.mean()),
        "n": int(len(y)), "n_pos": int(y.sum()),
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "roc": {"fpr": fpr.round(4).tolist(), "tpr": tpr.round(4).tolist()},
        "pr": {"recall": rec.round(4).tolist(), "precision": prec.round(4).tolist()},
    }


def main():
    llm = load_llm(LLM_GLOBS)
    enc = load_encoder(str(DEFAULT_ENCODER_OOF))
    df = align(llm, enc).reset_index(drop=True)
    yt = df["label"].to_numpy()

    llm_pred = df[[f"llm_{c}" for c in range(NUM_LABELS)]].to_numpy().argmax(1)
    preds = {
        "LLM alone (pooled 10-chain)": llm_pred,
        "cascade difficult>=0.8":      apply_cascade(df, gate_mask(df, "difficult", 0.8, 0.8)),
        "cascade merged(tau.8,>=.5)":  apply_cascade(df, gate_mask(df, "merged", 0.8, 0.5)),
        "encoder alone":               df["enc_pred"].to_numpy(),
    }

    out = {"screen_cut": SCREEN_CUT, "near_miss": {}, "screening": {}}
    for name, pred in preds.items():
        out["near_miss"][name] = near_miss(yt, pred)
        tt, pt = totals(df, pred)
        out["screening"][name] = curves(tt, pt)

    Path("outputs/cot").mkdir(parents=True, exist_ok=True)
    Path("outputs/cot/cascade_extras.json").write_text(json.dumps(out, indent=2))

    # console summary
    print("\n=== near-miss (ordinal closeness) ===")
    print(f"{'config':32s} {'exact':>7} {'off1':>7} {'<=1':>7} {'>=2':>7}")
    for n, d in out["near_miss"].items():
        print(f"{n:32s} {d['exact']:7.3f} {d['off_by_1']:7.3f} {d['within_1']:7.3f} {d['far_off']:7.3f}")
    print(f"\n=== PHQ-8 total screening (>= {SCREEN_CUT}) ===")
    s0 = next(iter(out['screening'].values()))
    print(f"participants={s0['n']}  positives(>= {SCREEN_CUT})={s0['n_pos']}  prevalence={s0['prevalence']:.3f}")
    print(f"{'config':32s} {'AUROC':>7} {'AUPRC':>7}")
    for n, d in out["screening"].items():
        print(f"{n:32s} {d['auroc']:7.3f} {d['auprc']:7.3f}")
    print("\nwrote outputs/cot/cascade_extras.json")


if __name__ == "__main__":
    main()
