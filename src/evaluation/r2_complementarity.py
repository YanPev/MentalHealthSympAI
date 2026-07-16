"""
Stage K -- encoder (E_FINAL) vs LLM (L_FINAL) complementarity on identical OOF
rows, BEFORE building any cascade (brief §13). Determines whether/where routing
could help. Uses no gold label to *fit* anything -- this is descriptive analysis.

Outputs:
  analysis/complementarity_overall.csv
  analysis/complementarity_by_item.csv
  analysis/complementarity_by_evidence.csv
  analysis/complementarity_by_severity.csv

    .venv/bin/python -m src.evaluation.r2_complementarity \
        --encoder outputs/cv/oof_predictions_r2_ctxm_corn_drop.csv \
        --llm-folds "outputs/r2_systematic/llm/folds_L1_keep/*.csv"
"""

from pathlib import Path
import argparse
import glob
import json

import numpy as np
import pandas as pd

PR = Path(__file__).resolve().parents[2]
ANA = PR / "outputs" / "r2_systematic" / "analysis"
ENC = PR / "outputs" / "r2_systematic" / "encoder"


def _pool_llm(globs):
    files = []
    for g in globs:
        files += glob.glob(g)
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["participant_id"] = df["participant_id"].astype(str)
    probc = [c for c in df.columns if c.startswith("prob_")]
    g = (df.groupby(["participant_id", "item_id", "item_name", "label"])[probc]
           .mean().reset_index())
    g["llm_pred"] = g[["prob_0", "prob_1", "prob_2", "prob_3"]].to_numpy().argmax(1)
    order = np.sort(g[["prob_0", "prob_1", "prob_2", "prob_3"]].to_numpy(), axis=1)
    g["llm_margin"] = order[:, -1] - order[:, -2]
    return g[["participant_id", "item_id", "item_name", "label", "llm_pred", "llm_margin"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--llm-folds", nargs="+", required=True)
    ap.add_argument("--evidence-status", default=str(ENC / "r2_evidence_status.csv"))
    args = ap.parse_args()

    enc = pd.read_csv(args.encoder)
    enc["participant_id"] = enc["participant_id"].astype(str)
    enc["enc_conf"] = enc[["prob_0", "prob_1", "prob_2", "prob_3"]].max(1)
    enc = enc.rename(columns={"prediction": "enc_pred"})[
        ["participant_id", "item_id", "item_name", "label", "enc_pred", "enc_conf", "fold"]]
    llm = _pool_llm(args.llm_folds)
    df = enc.merge(llm[["participant_id", "item_id", "llm_pred", "llm_margin"]],
                   on=["participant_id", "item_id"], how="inner")
    st = pd.read_csv(args.evidence_status)
    st["participant_id"] = st["participant_id"].astype(str)
    df = df.merge(st, on=["participant_id", "item_id"], how="left")

    y = df.label.to_numpy(); e = df.enc_pred.to_numpy(); l = df.llm_pred.to_numpy()
    df["enc_correct"] = e == y
    df["llm_correct"] = l == y
    df["enc_abserr"] = np.abs(e - y)
    df["llm_abserr"] = np.abs(l - y)
    df["disagree"] = np.abs(e - l)
    df["severe"] = np.where(y == 3, "severe", "non_severe")
    df["enc_conf_bin"] = pd.cut(df.enc_conf, [0, 0.5, 0.7, 0.85, 1.0], include_lowest=True).astype(str)
    df["llm_margin_bin"] = pd.cut(df.llm_margin, [0, 0.2, 0.4, 0.6, 1.0], include_lowest=True).astype(str)

    def summarize(g):
        n = len(g)
        return pd.Series({
            "n": n,
            "both_correct": float((g.enc_correct & g.llm_correct).mean()),
            "enc_only_correct": float((g.enc_correct & ~g.llm_correct).mean()),
            "llm_only_correct": float((~g.enc_correct & g.llm_correct).mean()),
            "both_incorrect": float((~g.enc_correct & ~g.llm_correct).mean()),
            "enc_lower_err": float((g.enc_abserr < g.llm_abserr).mean()),
            "llm_lower_err": float((g.llm_abserr < g.enc_abserr).mean()),
            "disagree_ge1": float((g.disagree >= 1).mean()),
            "disagree_ge2": float((g.disagree >= 2).mean()),
        })

    ANA.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summarize(df)]).to_csv(ANA / "complementarity_overall.csv", index=False)
    df.groupby("item_name").apply(summarize).reset_index().to_csv(
        ANA / "complementarity_by_item.csv", index=False)
    df.groupby("evidence_status").apply(summarize).reset_index().to_csv(
        ANA / "complementarity_by_evidence.csv", index=False)
    (df.groupby("severe").apply(summarize).reset_index()
       .to_csv(ANA / "complementarity_by_severity.csv", index=False))
    # extra stratifications (gold class, encoder confidence, LLM margin)
    for col, name in [("label", "by_goldclass"), ("enc_conf_bin", "by_encconf"),
                      ("llm_margin_bin", "by_llmmargin")]:
        df.groupby(col).apply(summarize).reset_index().to_csv(
            ANA / f"complementarity_{name}.csv", index=False)

    ov = summarize(df)
    print("=== complementarity (overall) ===")
    print(ov.round(4).to_string())
    print("\nby evidence status (llm_only_correct):")
    print(df.groupby("evidence_status").apply(lambda g: (~g.enc_correct & g.llm_correct).mean()).round(3).to_string())
    print(f"\nWrote complementarity_*.csv to {ANA}")


if __name__ == "__main__":
    main()
