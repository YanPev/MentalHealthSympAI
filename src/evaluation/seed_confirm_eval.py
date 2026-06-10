"""
Aggregate the multi-seed confirmation of the top MentalBERT cells.

For each config (Hybrid W3 / W5, CORN) computes per-model mean +/- std over the
5 seeds and a PAIRED per-seed comparison (MentalBERT - BERT, same folds per
seed) with a paired t-statistic, for macro-F1, QWK, PHQ-8 total QWK, and MAE.

    python -m src.evaluation.seed_confirm_eval
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.evaluation.model_comparison_eval import comprehensive

ROOT = Path(__file__).resolve().parents[2]
CV = ROOT / "outputs" / "cv"
OUT = ROOT / "outputs"
SEEDS = [42, 7, 31, 123, 2024]
CONFIGS = [("hybw3", "Hybrid W3"), ("hybw5", "Hybrid W5")]
METRICS = [("macro_f1", "item"), ("qwk", "item"), ("total_qwk", "total"), ("mae", "item")]


def metric(res, key, where):
    return res[where][key]


def load(model, cfg, seed):
    f = CV / f"oof_predictions_seedconf_{model}_{cfg}_s{seed}.csv"
    return comprehensive(pd.read_csv(f)) if f.exists() else None


def fmt(a):
    a = np.array(a)
    return f"{a.mean():.3f} ± {a.std(ddof=1):.3f}" if len(a) > 1 else (f"{a[0]:.3f}" if len(a) else "—")


def paired_t(diffs):
    d = np.array(diffs)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))


def main():
    out = {}
    for cfg, clabel in CONFIGS:
        bert = {s: load("bert", cfg, s) for s in SEEDS}
        mbert = {s: load("mbert", cfg, s) for s in SEEDS}
        seeds_done = [s for s in SEEDS if bert[s] and mbert[s]]
        print("=" * 70)
        print(f"{clabel} · CORN — seeds done: {seeds_done}  (n={len(seeds_done)})")
        print("-" * 70)
        if not seeds_done:
            print("  (pending)")
            continue
        cfg_out = {"n_seeds": len(seeds_done), "seeds": seeds_done, "metrics": {}}
        for key, where in METRICS:
            bvals = [metric(bert[s], key, where) for s in seeds_done]
            mvals = [metric(mbert[s], key, where) for s in seeds_done]
            diffs = [m - b for b, m in zip(bvals, mvals)]
            wins = int(sum(1 for d in diffs if d > 0))
            t = paired_t(diffs)
            arrow = "↓" if key == "mae" else ""
            print(f"  {key+arrow:11} BERT {fmt(bvals):>16}   MentalBERT {fmt(mvals):>16}"
                  f"   Δ {np.mean(diffs):+.3f}  win {wins}/{len(seeds_done)}  t={t:+.2f}")
            cfg_out["metrics"][key] = {
                "bert_mean": float(np.mean(bvals)), "bert_std": float(np.std(bvals, ddof=1)) if len(bvals) > 1 else 0.0,
                "mbert_mean": float(np.mean(mvals)), "mbert_std": float(np.std(mvals, ddof=1)) if len(mvals) > 1 else 0.0,
                "mean_delta": float(np.mean(diffs)), "mbert_wins": wins, "paired_t": t,
                "per_seed": {str(s): {"bert": metric(bert[s], key, where), "mbert": metric(mbert[s], key, where)} for s in seeds_done},
            }
        out[cfg] = {"label": clabel, **cfg_out}

    (OUT / "seed_confirm_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT/'seed_confirm_results.json'}")
    # verdict
    for cfg, d in out.items():
        if "metrics" in d and d["n_seeds"] >= 2:
            mf = d["metrics"]["macro_f1"]
            verdict = ("MentalBERT significantly higher" if mf["paired_t"] > 2.78
                       else "MentalBERT higher (n.s.)" if mf["mean_delta"] > 0
                       else "no MentalBERT advantage")
            print(f"  {d['label']}: macro-F1 Δ {mf['mean_delta']:+.3f} (t={mf['paired_t']:+.2f}) -> {verdict}")


if __name__ == "__main__":
    main()
