"""
Stage J evaluation -- Qwen staged-tolerant SC5 on R2 across the five evidence
conditions; metrics, paired bootstrap vs KEEP/R0, and L_FINAL selection.

Conditions (folds under outputs/r2_systematic/llm/):
  L0_R0            frozen R0 pool (folds_tolerant_sc5[_dv]) -- reused, not re-run
  L1_keep          R2 windows unchanged
  L2_filter        none-judged windows removed
  L3_infofirst     supports/against first, none excluded
  L4_fallback      transcript when set none / filtered empty
  L5_longctx       full transcript for all 8 items

Reports the standard prediction metrics (r2_metrics) plus SC diagnostics
(vote margin, agreement, parse-failure). Uses no gold label to select the
retrieval/evidence policy beyond the predefined QWK->macro-F1 rule + constraints.

    .venv/bin/python -m src.evaluation.r2_llm_eval
"""

from pathlib import Path
import argparse
import glob
import json

import numpy as np
import pandas as pd

from src.evaluation.r2_metrics import overall, by_item, paired_bootstrap, align_oof

PR = Path(__file__).resolve().parents[2]
LLM = PR / "outputs" / "r2_systematic" / "llm"
COT = PR / "outputs" / "cot"

CONDS = {
    "L0_R0": [str(COT / "folds_tolerant_sc5" / "*.csv"), str(COT / "folds_tolerant_sc5_dv" / "*.csv")],
    "L1_keep": [str(LLM / "folds_L1_keep" / "*.csv")],
    "L2_filter": [str(LLM / "folds_L2_filter" / "*.csv")],
    "L3_infofirst": [str(LLM / "folds_L3_infofirst" / "*.csv")],
    "L4_fallback": [str(LLM / "folds_L4_fallback" / "*.csv")],
    "L5_longctx": [str(LLM / "folds_L5_longctx" / "*.csv")],
}


def pool(globs):
    files = []
    for g in globs:
        files += glob.glob(g)
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["participant_id"] = df["participant_id"].astype(str)
    probc = ["prob_0", "prob_1", "prob_2", "prob_3"]
    g = df.groupby(["participant_id", "item_id", "item_name", "label"])[probc].mean().reset_index()
    P = g[probc].to_numpy()
    g["prediction"] = P.argmax(1)
    order = np.sort(P, axis=1)
    g["vote_margin"] = order[:, -1] - order[:, -2]
    g["agreement"] = order[:, -1]
    return g


def main():
    ap = argparse.ArgumentParser(description="Stage J LLM eval")
    ap.parse_args()
    frames = {k: pool(v) for k, v in CONDS.items()}
    present = {k: v for k, v in frames.items() if v is not None}
    if "L1_keep" not in present:
        raise SystemExit("L1_keep folds not found -- run run_r2_llm.sbatch first")

    LLM.mkdir(parents=True, exist_ok=True)
    ov_rows, item_rows = [], []
    for k, d in present.items():
        o = overall(d); o["run"] = k
        o["vote_margin"] = float(d["vote_margin"].mean())
        o["agreement"] = float(d["agreement"].mean())
        ov_rows.append(o)
        bi = by_item(d); bi["run"] = k; item_rows.append(bi)
    ov = pd.DataFrame(ov_rows)
    ov.to_csv(LLM / "llm_metrics_overall.csv", index=False)
    pd.concat(item_rows).to_csv(LLM / "llm_metrics_by_item.csv", index=False)

    # paired bootstraps vs KEEP and R0
    ci_rows = []
    keep = present["L1_keep"]
    pairs = [("L1_keep", "L0_R0"), ("L2_filter", "L1_keep"), ("L3_infofirst", "L1_keep"),
             ("L4_fallback", "L1_keep"), ("L5_longctx", "L1_keep")]
    for a, b in pairs:
        if a in present and b in present:
            m = align_oof(present[a], present[b])
            res = paired_bootstrap(m, "pred_0", "pred_1")
            for metric, r in res.items():
                ci_rows.append({"comparison": f"{a}_vs_{b}", "metric": metric,
                                "delta": r["delta"], "ci_lo": r["ci95"][0],
                                "ci_hi": r["ci95"][1], "excludes_0": r["excludes_0"]})
    pd.DataFrame(ci_rows).to_csv(LLM / "llm_paired_bootstrap_ci.csv", index=False)

    l_final = _select_lfinal(ov)
    (LLM / "l_final_selection.json").write_text(json.dumps(l_final, indent=2))
    print("=== LLM metrics (overall) ===")
    print(ov[["run", "macro_f1", "qwk", "mae", "f1_class3", "severe_recall",
              "false_severe_rate", "coverage"]].round(4).to_string(index=False))
    print("\nL_FINAL:", json.dumps(l_final, indent=2))


def _select_lfinal(ov):
    o = ov.set_index("run")
    keep = o.loc["L1_keep"]
    cands = {}
    for run in o.index:
        if run == "L0_R0":
            continue
        r = o.loc[run]
        ok = True
        if run != "L1_keep":
            if r["mae"] > keep["mae"] + 0.02 and r["qwk"] <= keep["qwk"]:
                ok = False
            if r["severe_recall"] < keep["severe_recall"] - 0.03:
                ok = False
            if r["false_severe_rate"] > keep["false_severe_rate"] + 0.015:
                ok = False
        cands[run] = {"qwk": float(r["qwk"]), "macro_f1": float(r["macro_f1"]),
                      "mae": float(r["mae"]), "passes_constraints": ok}
    eligible = {k: v for k, v in cands.items() if v["passes_constraints"]}
    winner = max(eligible or cands, key=lambda k: (cands[k]["qwk"], cands[k]["macro_f1"]))
    return {"L_FINAL": winner, "candidates": cands,
            "rule": "max QWK then macro-F1 among conditions passing MAE/severe/false-severe constraints"}


if __name__ == "__main__":
    main()
