"""
Stage L -- two evidence-aware cascades with LEAKAGE-SAFE routing (brief §14).

Unlike the frozen project cascade (whose tau/diff were picked on the OOF), every
routing parameter here is fit INSIDE the training folds and applied to the
held-out fold, so the reported OOF cascade metrics are unbiased w.r.t. routing.

Two directions, each using only the allowed knobs (one confidence threshold, one
vote-margin threshold, one item-group rule, one evidence-status rule):

  encoder-first : default = encoder; route to LLM if
                  (enc_conf < tau_e) OR (evidence in {ambiguous,none})
                  OR (item in LLM-better group).
  llm-first     : default = LLM; route to encoder if
                  (llm_margin < tau_l) OR (item in encoder-better group).

tau_e / tau_l and the item groups are chosen on the training folds to maximise
train QWK; applied to the held-out fold. Compared against E_FINAL, L_FINAL, and
the frozen project-best cascade (reference). Paired participant bootstrap +
acceptance rule decide whether a cascade is recommended.

    .venv/bin/python -m src.evaluation.r2_cascade \
        --encoder outputs/cv/oof_predictions_r2_ctxm_corn_drop.csv \
        --llm-folds "outputs/r2_systematic/llm/folds_L1_keep/*.csv"
"""

from pathlib import Path
import argparse
import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.evaluation.r2_metrics import item_metrics, paired_bootstrap
from src.evaluation.r2_complementarity import _pool_llm

PR = Path(__file__).resolve().parents[2]
CAS = PR / "outputs" / "r2_systematic" / "cascade"
ENC = PR / "outputs" / "r2_systematic" / "encoder"
FROZEN_BEST = {"name": "frozen_MIL_merged_R0", "qwk": 0.447, "macro_f1": 0.409,
               "mae": 0.606, "note": "reference: Attention-MIL merged cascade on R0 (frozen)"}
TAU_E_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TAU_L_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def qwk(y, p):
    if len(set(list(y) + list(p))) < 2:
        return 0.0
    return cohen_kappa_score(y, p, weights="quadratic", labels=[0, 1, 2, 3])


def load(encoder, llm_globs, status_csv):
    enc = pd.read_csv(encoder); enc["participant_id"] = enc["participant_id"].astype(str)
    enc["enc_conf"] = enc[["prob_0", "prob_1", "prob_2", "prob_3"]].max(1)
    enc = enc.rename(columns={"prediction": "enc_pred"})
    llm = _pool_llm(llm_globs)
    df = enc[["participant_id", "item_id", "item_name", "label", "fold", "enc_pred", "enc_conf"]].merge(
        llm[["participant_id", "item_id", "llm_pred", "llm_margin"]],
        on=["participant_id", "item_id"], how="inner")
    st = pd.read_csv(status_csv); st["participant_id"] = st["participant_id"].astype(str)
    df = df.merge(st, on=["participant_id", "item_id"], how="left")
    df["evidence_status"] = df["evidence_status"].fillna("none")
    return df


def item_group(train, better="llm"):
    """Items where one model has lower MAE on the training folds."""
    g = train.groupby("item_id").apply(
        lambda x: pd.Series({"enc_mae": np.abs(x.enc_pred - x.label).mean(),
                             "llm_mae": np.abs(x.llm_pred - x.label).mean()}))
    if better == "llm":
        return set(g.index[g.llm_mae < g.enc_mae - 0.02])
    return set(g.index[g.enc_mae < g.llm_mae - 0.02])


def enc_first_route(df, tau_e, llm_items):
    route = (df.enc_conf < tau_e) | (df.evidence_status.isin(["ambiguous", "none"])) | (df.item_id.isin(llm_items))
    return np.where(route, df.llm_pred, df.enc_pred)


def llm_first_route(df, tau_l, enc_items):
    route = (df.llm_margin < tau_l) | (df.item_id.isin(enc_items))
    return np.where(route, df.enc_pred, df.llm_pred)


def fit_and_apply(df, direction):
    """Nested: fit routing on train folds, apply to held-out fold -> OOF preds."""
    preds = np.full(len(df), -1)
    folds = sorted(df.fold.unique())
    chosen = []
    for f in folds:
        tr = df[df.fold != f]; te_idx = df.index[df.fold == f]
        te = df.loc[te_idx]
        if direction == "encoder_first":
            llm_items = item_group(tr, "llm")
            best_tau, best_q = TAU_E_GRID[0], -9
            for tau in TAU_E_GRID:
                q = qwk(tr.label, enc_first_route(tr, tau, llm_items))
                if q > best_q:
                    best_q, best_tau = q, tau
            preds[df.fold.values == f] = enc_first_route(te, best_tau, llm_items)
            chosen.append({"fold": int(f), "tau_e": best_tau, "llm_items": sorted(llm_items)})
        else:
            enc_items = item_group(tr, "enc")
            best_tau, best_q = TAU_L_GRID[0], -9
            for tau in TAU_L_GRID:
                q = qwk(tr.label, llm_first_route(tr, tau, enc_items))
                if q > best_q:
                    best_q, best_tau = q, tau
            preds[df.fold.values == f] = llm_first_route(te, best_tau, enc_items)
            chosen.append({"fold": int(f), "tau_l": best_tau, "enc_items": sorted(enc_items)})
    return preds, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--llm-folds", nargs="+", required=True)
    ap.add_argument("--evidence-status", default=str(ENC / "r2_evidence_status.csv"))
    args = ap.parse_args()

    df = load(args.encoder, args.llm_folds, args.evidence_status)
    ef, ef_params = fit_and_apply(df, "encoder_first")
    lf, lf_params = fit_and_apply(df, "llm_first")
    df["cascade_encfirst"] = ef
    df["cascade_llmfirst"] = lf

    y = df.label.to_numpy()
    models = {"E_FINAL": df.enc_pred, "L_FINAL": df.llm_pred,
              "cascade_encoder_first": df.cascade_encfirst,
              "cascade_llm_first": df.cascade_llmfirst}
    rows = []
    for name, p in models.items():
        m = item_metrics(y, p.to_numpy()); m["model"] = name; rows.append(m)
    rows.append({"model": FROZEN_BEST["name"], "qwk": FROZEN_BEST["qwk"],
                 "macro_f1": FROZEN_BEST["macro_f1"], "mae": FROZEN_BEST["mae"]})
    tab = pd.DataFrame(rows)
    CAS.mkdir(parents=True, exist_ok=True)
    tab.to_csv(CAS / "cascade_metrics.csv", index=False)

    # paired bootstrap: each cascade vs the better of its two components
    better = "E_FINAL" if item_metrics(y, df.enc_pred.to_numpy())["qwk"] >= \
        item_metrics(y, df.llm_pred.to_numpy())["qwk"] else "L_FINAL"
    bcol = "enc_pred" if better == "E_FINAL" else "llm_pred"
    boots = {}
    for name, col in [("cascade_encoder_first", "cascade_encfirst"),
                      ("cascade_llm_first", "cascade_llmfirst")]:
        boots[f"{name}_vs_{better}"] = paired_bootstrap(
            df.assign(a=df[col], b=df[bcol]), "a", "b")
    (CAS / "cascade_routing_params.json").write_text(json.dumps(
        {"encoder_first": ef_params, "llm_first": lf_params,
         "better_component": better}, indent=2))
    (CAS / "cascade_bootstrap.json").write_text(json.dumps(boots, indent=2, default=float))

    # acceptance (brief §14.4): QWK > better component; macroF1 down <=.005;
    # MAE up <=.02; severe recall not down; false-severe up <=.015.
    bm = item_metrics(y, df[bcol].to_numpy())
    verdict = {}
    for name, col in [("cascade_encoder_first", "cascade_encfirst"),
                      ("cascade_llm_first", "cascade_llmfirst")]:
        cm = item_metrics(y, df[col].to_numpy())
        ok = (cm["qwk"] > bm["qwk"] and cm["macro_f1"] >= bm["macro_f1"] - 0.005
              and cm["mae"] <= bm["mae"] + 0.02
              and (np.isnan(cm["severe_recall"]) or cm["severe_recall"] >= bm["severe_recall"] - 1e-9)
              and cm["false_severe_rate"] <= bm["false_severe_rate"] + 0.015)
        verdict[name] = {"accepted": bool(ok), "qwk": cm["qwk"], "vs_better_qwk": bm["qwk"]}
    (CAS / "cascade_acceptance.json").write_text(json.dumps(verdict, indent=2))

    print("=== cascade metrics ===")
    print(tab[["model", "macro_f1", "qwk", "mae", "severe_recall", "false_severe_rate"]].round(4).to_string(index=False))
    print(f"\nbetter component = {better}")
    print("acceptance:", json.dumps(verdict, indent=2))
    print(f"Wrote cascade_metrics.csv + params/bootstrap/acceptance to {CAS}")


if __name__ == "__main__":
    main()
