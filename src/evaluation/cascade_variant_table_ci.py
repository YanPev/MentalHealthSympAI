"""95% cluster-bootstrap CIs for the 4-model headline comparison table.

Recreates the presentation table

    model                          Exact  Macro-F1  QWK    MAE   Err>=2
    Midterm CORN                   0.455  0.348     0.354  0.680 0.119
    Pooled LLM                     0.513  0.400     0.406  0.641 0.128
    Attention-MIL merged cascade   0.530  0.409     0.447  0.606 0.118
    Rerank merged cascade          0.511  0.394     0.411  0.627 0.115

...adding a participant-level (cluster) bootstrap 95% CI to every cell. Each
PHQ-8 participant contributes 8 correlated item rows, so we resample WHOLE
PARTICIPANTS with replacement (B times) and recompute each metric per resample
(2.5 / 97.5 percentiles). Same B / SEED / method as cascade_bootstrap_ci.py, so
the shared "Pooled LLM" row reproduces the CIs already in cascade_top3_ci.json.

The 4 rows map onto outputs/cot/cascade_retrieval_variants/summary.csv:
  * Midterm CORN               = baseline CORN, "encoder alone" (ctxm_corn_hybw3)
  * Pooled LLM                 = "LLM alone (10-chain SC-tolerant)" (argmax of pooled votes)
  * Attention-MIL merged casc. = mil_hybw3 encoder, merged-gate (tau=0.8, diff>=0.5)
  * Rerank merged cascade      = ctxm_corn_rerank_w3 encoder, merged-gate (tau=0.8, diff>=0.5)

All four share ONE participant slice and ONE set of bootstrap resamples, so the
CIs are directly comparable and support the paired deltas printed at the end.

    python -m src.evaluation.cascade_variant_table_ci
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             cohen_kappa_score)

from src.evaluation.cot_cascade import (
    load_llm, load_encoder, align, gate_mask, apply_cascade, NUM_LABELS, LABELS,
)

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "outputs" / "cv"
LLM_GLOBS = ["outputs/cot/folds_tolerant_sc5/*.csv",
             "outputs/cot/folds_tolerant_sc5_dv/*.csv"]

BASELINE_ENC = CV / "oof_predictions_ctxm_corn_hybw3.csv"   # Midterm CORN
MIL_ENC = CV / "oof_predictions_mil_hybw3.csv"              # Attention-MIL
RERANK_ENC = CV / "oof_predictions_ctxm_corn_rerank_w3.csv"  # Rerank + CORN

B = 2000          # bootstrap resamples (matches cascade_bootstrap_ci.py)
SEED = 42
TAU, DIFF = 0.8, 0.5
METRIC_KEYS = ["exact", "macroF1", "QWK", "MAE", "err>=2"]
# lower-is-better metrics (used only for the paired-delta direction label)
LOWER_BETTER = {"MAE", "err>=2"}


def metric_vec(yt, yp):
    return {
        "exact":   accuracy_score(yt, yp),
        "macroF1": f1_score(yt, yp, average="macro", labels=LABELS, zero_division=0),
        "QWK":     cohen_kappa_score(yt, yp, weights="quadratic", labels=LABELS),
        "MAE":     mean_absolute_error(yt, yp),
        "err>=2":  float(np.mean(np.abs(yt - yp) >= 2)),
    }


def aligned(llm, enc_path):
    """align() the pooled LLM with one encoder, sorted to a canonical (pid,item)
    order so every model's prediction vector shares the same row index."""
    df = align(llm, load_encoder(str(enc_path)))
    df = df.sort_values(["participant_id", "item_id"]).reset_index(drop=True)
    return df


def main():
    print(f"Loading pooled LLM ({len(LLM_GLOBS)} runs) + 3 encoders ...")
    llm = load_llm(LLM_GLOBS)

    base = aligned(llm, BASELINE_ENC)     # carries labels, folds, llm_* + baseline enc_*
    mil = aligned(llm, MIL_ENC)
    rer = aligned(llm, RERANK_ENC)

    # canonical labels / participant ids (identical across the three, by construction)
    yt = base["label"].to_numpy()
    pid = base["participant_id"].to_numpy()
    for other, nm in [(mil, "MIL"), (rer, "rerank")]:
        assert np.array_equal(other["label"].to_numpy(), yt), f"label mismatch vs {nm}"
        assert np.array_equal(other["participant_id"].to_numpy(), pid), f"pid mismatch vs {nm}"

    # merged-gate mask depends ONLY on the LLM votes + difficult_frac -> identical
    # across encoders. Compute once on the base frame.
    mask = gate_mask(base, "merged", TAU, DIFF)
    llm_pred = base[[f"llm_{c}" for c in range(NUM_LABELS)]].to_numpy().argmax(1)

    # the 4 table rows, as full-data prediction vectors on the canonical index
    preds = {
        "Midterm CORN":                 base["enc_pred"].to_numpy(),
        "Pooled LLM":                   llm_pred,
        "Attention-MIL merged cascade": apply_cascade(mil, mask),
        "Rerank merged cascade":        apply_cascade(rer, mask),
    }

    # ---- participant-level (cluster) bootstrap ----
    rng = np.random.default_rng(SEED)
    participants = np.unique(pid)
    rows_by_pid = {p: np.where(pid == p)[0] for p in participants}
    boot = {name: {k: np.empty(B) for k in METRIC_KEYS} for name in preds}

    for b in range(B):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        idx = np.concatenate([rows_by_pid[p] for p in sampled])
        ytb = yt[idx]
        for name, pred in preds.items():
            mv = metric_vec(ytb, pred[idx])
            for k in METRIC_KEYS:
                boot[name][k][b] = mv[k]

    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    # ---- table ----
    rows = []
    payload = {}
    for name, pred in preds.items():
        pt = metric_vec(yt, pred)
        row = {"model": name}
        payload[name] = {}
        for k in METRIC_KEYS:
            lo, hi = ci(boot[name][k])
            row[k] = f"{pt[k]:.3f} [{lo:.3f}, {hi:.3f}]"
            payload[name][k] = {"point": pt[k], "ci95": [lo, hi]}
        rows.append(row)

    out = pd.DataFrame(rows)
    n_part, n_item = len(participants), len(yt)
    print(f"\n=== Point estimate [95% cluster-bootstrap CI]  |  B={B} participant "
          f"resamples, n={n_part} participants / {n_item} items ===\n")
    print(out.to_string(index=False))

    # ---- paired deltas vs the two natural baselines (same resamples) ----
    def paired(name, ref):
        parts = []
        for k in ["macroF1", "QWK", "MAE", "err>=2"]:
            d = boot[name][k] - boot[ref][k]
            lo, hi = ci(d)
            sig = "*" if (lo > 0 or hi < 0) else " "
            parts.append(f"{k} {d.mean():+.3f}[{lo:+.3f},{hi:+.3f}]{sig}")
        print(f"  {name:30s} vs {ref:12s}  " + "  ".join(parts))

    print("\n=== Paired Δ (same resamples; * = 95% CI excludes 0; MAE/err>=2 lower is better) ===")
    paired("Attention-MIL merged cascade", "Midterm CORN")
    paired("Attention-MIL merged cascade", "Pooled LLM")
    paired("Rerank merged cascade", "Pooled LLM")
    paired("Attention-MIL merged cascade", "Rerank merged cascade")

    # ---- save ----
    out_csv = ROOT / "outputs" / "cot" / "cascade_variant_table_ci.csv"
    out.to_csv(out_csv, index=False)
    meta = {"B": B, "seed": SEED, "n_participants": n_part, "n_items": n_item,
            "tau": TAU, "diff_thresh": DIFF, "method": "participant-level cluster bootstrap",
            "metrics": payload}
    out_json = ROOT / "outputs" / "cot" / "cascade_variant_table_ci.json"
    out_json.write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out_csv}\nwrote {out_json}")


if __name__ == "__main__":
    main()
