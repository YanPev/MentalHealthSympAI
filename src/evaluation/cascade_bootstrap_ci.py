"""Cluster-bootstrap 95% CIs for the top cascade configurations.

Reuses the loading / gating / cascade logic in cot_cascade.py and adds a
participant-level (cluster) bootstrap: each PHQ-8 participant contributes 8
correlated item rows, so we resample WHOLE PARTICIPANTS with replacement
(B times), recompute every metric on each resample, and report the 2.5 / 50 /
97.5 percentiles. This respects the non-independence the per-row bootstrap would
ignore.

Top-3 configs (pooled 10-chain SC-tolerant LLM + MentalBERT-CORN-W3 encoder):
  1. LLM alone (pooled)          -- best standalone macro-F1
  2. cascade difficult>=0.8      -- best cascade macro-F1, routes ~6%
  3. cascade merged(tau.8,>=.5)  -- best QWK + MAE

    python -m src.evaluation.cascade_bootstrap_ci
"""
from pathlib import Path
import numpy as np
import pandas as pd

from src.evaluation.cot_cascade import (
    load_llm, load_encoder, align, gate_mask, apply_cascade, metrics,
    DEFAULT_ENCODER_OOF, NUM_LABELS, LABELS,
)
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             cohen_kappa_score)

LLM_GLOBS = ["outputs/cot/folds_tolerant_sc5/*.csv",
             "outputs/cot/folds_tolerant_sc5_dv/*.csv"]
B = 2000          # bootstrap resamples
SEED = 42
METRIC_KEYS = ["exact", "macroF1", "QWK", "MAE", "err>=2"]


def metric_vec(yt, yp):
    return {
        "exact":   accuracy_score(yt, yp),
        "macroF1": f1_score(yt, yp, average="macro", labels=LABELS, zero_division=0),
        "QWK":     cohen_kappa_score(yt, yp, weights="quadratic", labels=LABELS),
        "MAE":     mean_absolute_error(yt, yp),
        "err>=2":  float(np.mean(np.abs(yt - yp) >= 2)),
    }


def main():
    print(f"Loading pooled LLM ({len(LLM_GLOBS)} runs) + encoder ...")
    llm = load_llm(LLM_GLOBS)
    enc = load_encoder(str(DEFAULT_ENCODER_OOF))
    df = align(llm, enc).reset_index(drop=True)
    yt = df["label"].to_numpy()
    pid = df["participant_id"].to_numpy()

    # ---- predictions for each top-3 config (computed once on full data) ----
    llm_pred = df[[f"llm_{c}" for c in range(NUM_LABELS)]].to_numpy().argmax(1)
    m_diff = gate_mask(df, "difficult", 0.8, 0.8)
    m_merg = gate_mask(df, "merged", 0.8, 0.5)
    configs = {
        "LLM alone (pooled 10-chain)":      (llm_pred,                         None),
        "cascade  difficult>=0.8":          (apply_cascade(df, m_diff),        m_diff),
        "cascade  merged(tau.8, diff>=.5)": (apply_cascade(df, m_merg),        m_merg),
    }
    # encoder parent for reference
    configs_ref = {"encoder alone (reference)": (df["enc_pred"].to_numpy(), None)}

    # ---- participant-level (cluster) bootstrap ----
    rng = np.random.default_rng(SEED)
    participants = np.unique(pid)
    # precompute row-index lists per participant
    rows_by_pid = {p: np.where(pid == p)[0] for p in participants}

    boot = {name: {k: np.empty(B) for k in METRIC_KEYS}
            for name in list(configs) + list(configs_ref)}

    for b in range(B):
        sampled = rng.choice(participants, size=len(participants), replace=True)
        idx = np.concatenate([rows_by_pid[p] for p in sampled])
        ytb = yt[idx]
        for name, (pred, _) in {**configs, **configs_ref}.items():
            mv = metric_vec(ytb, pred[idx])
            for k in METRIC_KEYS:
                boot[name][k][b] = mv[k]

    # ---- point estimates + CIs ----
    def ci(arr):
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    rows = []
    for name, (pred, mask) in {**configs, **configs_ref}.items():
        pt = metric_vec(yt, pred)
        row = {"config": name,
               "routed%": ("-" if mask is None else f"{100*mask.sum()/len(yt):.1f}")}
        for k in METRIC_KEYS:
            lo, hi = ci(boot[name][k])
            row[k] = f"{pt[k]:.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(row)

    out = pd.DataFrame(rows)
    print(f"\n=== Point estimate [95% cluster-bootstrap CI], "
          f"B={B} participant resamples, n={len(participants)} participants / "
          f"{len(df)} items ===\n")
    print(out.to_string(index=False))

    # ---- paired bootstrap deltas vs encoder (key comparisons) ----
    print("\n=== Paired Δ vs encoder (same resamples; 95% CI; >0 favours the config) ===")
    enc_boot = boot["encoder alone (reference)"]
    for name in configs:
        deltas = {k: boot[name][k] - enc_boot[k] for k in ["macroF1", "QWK", "MAE", "err>=2"]}
        parts = []
        for k in ["macroF1", "QWK", "MAE", "err>=2"]:
            d = deltas[k]
            lo, hi = ci(d)
            sig = "*" if (lo > 0 or hi < 0) else " "
            parts.append(f"{k} {d.mean():+.3f}[{lo:+.3f},{hi:+.3f}]{sig}")
        print(f"  {name:34s} " + "  ".join(parts))
    print("  (* = 95% CI excludes 0. For MAE/err>=2 lower is better, so a "
          "negative Δ is the improvement.)")

    # save machine-readable
    out_json = Path("outputs/cot/cascade_top3_ci.json")
    payload = {}
    for name in {**configs, **configs_ref}:
        pred = {**configs, **configs_ref}[name][0]
        pt = metric_vec(yt, pred)
        payload[name] = {k: {"point": pt[k],
                             "ci95": list(ci(boot[name][k]))} for k in METRIC_KEYS}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(__import__("json").dumps(payload, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
