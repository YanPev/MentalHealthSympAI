"""
Stage J — assemble a plot-ready, NON-SENSITIVE upload bundle for R2.

Collects exactly the artifacts a downstream plotting/analysis consumer needs and
NOTHING that contains raw interview text. Every table is either whitelisted as
text-free or sanitized here by dropping known free-text / transcript columns
(`reason`, `reasoning`, `window_text`, `turn_ids`, `generations`, `text`,
`item_text`, `window_texts`). A final guard scans every emitted CSV and aborts if
any banned column survives, so the bundle cannot leak transcript content.

The bundle is staged under outputs/r2_systematic/upload/ with a manifest
(sha256 + provenance + explicit sensitive-exclusion list). It is NOT pushed to
any external service — no upload destination is configured in this repo, and
publishing is a confirm-first action.

    .venv/bin/python -m src.evaluation.build_r2_upload_bundle
"""

from pathlib import Path
import glob
import hashlib
import json

import pandas as pd

PR = Path(__file__).resolve().parents[2]
R2 = PR / "outputs" / "r2_systematic"
CV = PR / "outputs" / "cv"
RET, JUD, ENC, LLM, ANA, CAS = (R2 / d for d in
    ("retrieval", "judge", "encoder", "llm", "analysis", "cascade"))
OUT = R2 / "upload"

# Any column that may quote or embed interview text. Guard is column-name based.
BANNED_COLS = {"reason", "reasoning", "window_text", "window_texts", "turn_ids",
               "generations", "text", "item_text", "windows", "excerpt", "prompt",
               "context", "reasoning_trace", "cot", "transcript"}

# Files known to embed transcript text — never copied into the bundle.
SENSITIVE_EXCLUDED = [
    "retrieval/retrieval_window_scores.parquet",
    "retrieval/retrieval_window_scores_hybrid.parquet",
    "retrieval/retrieval_window_scores_smoke.parquet",
    "judge/*_raw_*.jsonl",
    "judge/window_judgments_*.csv",
    "encoder/r2_evidence_status.csv (kept out: per-participant, not plot-ready)",
]

# Verbatim-copy whitelist: text-free metric tables.
COPY_VERBATIM = [
    RET / "all_candidates.csv",
    RET / "retrieval_metrics_overall.csv",
    RET / "retrieval_metrics_by_item.csv",
    RET / "retrieval_metrics_by_rank.csv",
    RET / "retrieval_redundancy.csv",
    RET / "retrieval_complementarity.csv",
    RET / "r2_selection.json",
    ENC / "encoder_metrics_overall.csv",
    ENC / "encoder_metrics_by_item.csv",
    ENC / "encoder_paired_bootstrap_ci.csv",
    ENC / "missing_policy_audit.csv",
    LLM / "llm_metrics_overall.csv",
    LLM / "llm_metrics_by_item.csv",
    LLM / "llm_paired_bootstrap_ci.csv",
    ANA / "complementarity_overall.csv",
    ANA / "complementarity_by_item.csv",
    ANA / "complementarity_by_evidence.csv",
    ANA / "complementarity_by_severity.csv",
    ANA / "complementarity_by_goldclass.csv",
    ANA / "complementarity_by_encconf.csv",
    ANA / "complementarity_by_llmmargin.csv",
    CAS / "cascade_metrics.csv",
    CAS / "cascade_routing_params.json",
    CAS / "cascade_acceptance.json",
    CAS / "cascade_bootstrap.json",
    R2 / "run_manifest.json",
    R2 / "results_report.html",
]

# Compact OOF prediction tables (encoder = already text-free; LLM = drop reasoning).
ENCODER_OOF = {
    "encoder_oof_R0.csv": CV / "oof_predictions_ctxm_corn_hybw3.csv",
    "encoder_oof_R1_prev.csv": CV / "oof_predictions_ctxm_corn_exphybw3.csv",
    "encoder_oof_R2_status_quo.csv": CV / "oof_predictions_r2_ctxm_corn_zero.csv",
    "encoder_oof_R2_mask_none.csv": CV / "oof_predictions_r2_ctxm_corn_mask.csv",
    "encoder_oof_R2_drop_none.csv": CV / "oof_predictions_r2_ctxm_corn_drop.csv",
}


def _sha256(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def _sanitize_csv(df):
    drop = [c for c in df.columns if c in BANNED_COLS]
    return df.drop(columns=drop), drop


def _write_csv(df, dest, provenance, records):
    df.to_csv(dest, index=False)
    records.append({"file": dest.name, "rows": int(len(df)),
                    "cols": list(df.columns), "sha256": _sha256(dest),
                    "provenance": provenance})


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    records, dropped_log = [], {}

    # 1) verbatim text-free artifacts
    for src in COPY_VERBATIM:
        if not src.exists():
            records.append({"file": src.name, "status": "MISSING", "provenance": str(src.relative_to(PR))})
            continue
        dest = OUT / src.name
        if src.suffix == ".csv":
            df, drop = _sanitize_csv(pd.read_csv(src))
            if drop:
                dropped_log[src.name] = drop
            _write_csv(df, dest, str(src.relative_to(PR)), records)
        else:
            dest.write_bytes(src.read_bytes())
            records.append({"file": dest.name, "sha256": _sha256(dest),
                            "provenance": str(src.relative_to(PR))})

    # 2) set-level evidence judgments WITHOUT transcript text (single + hybrid)
    set_files = (sorted(glob.glob(str(JUD / "set_judgments_fold*.csv"))) +
                 sorted(glob.glob(str(JUD / "set_judgments_hybrid_fold*.csv"))))
    if set_files:
        st = pd.concat([pd.read_csv(f) for f in set_files], ignore_index=True)
        st, drop = _sanitize_csv(st)
        if drop:
            dropped_log["set_judgments_all.csv"] = drop
        _write_csv(st, OUT / "set_judgments_all.csv",
                   "judge/set_judgments_fold*.csv + set_judgments_hybrid_fold*.csv (reason dropped)",
                   records)

    # 3) derived retrieval by-alpha + by-prefix slices
    ov = RET / "retrieval_metrics_overall.csv"
    if ov.exists():
        o = pd.read_csv(ov)
        by_prefix = (o.groupby(["config", "retriever", "prefix"], as_index=False)
                       .agg(informative_rate=("informative_rate", "mean"),
                            none_rate=("none_rate", "mean"),
                            ambiguous_rate=("ambiguous_rate", "mean")))
        _write_csv(by_prefix, OUT / "retrieval_metrics_by_prefix.csv",
                   "derived from retrieval_metrics_overall.csv", records)
        hyb = o[o.retriever.astype(str).str.startswith("hybrid")].copy()
        if len(hyb):
            hyb["alpha"] = hyb.retriever.map({"hybrid_a25": 0.25, "hybrid_a50": 0.50,
                                              "hybrid_a75": 0.75})
            _write_csv(hyb.sort_values(["prefix", "alpha"]),
                       OUT / "retrieval_metrics_by_alpha.csv",
                       "derived from retrieval_metrics_overall.csv (hybrid rows)", records)

    # 4) compact OOF prediction tables
    for name, src in ENCODER_OOF.items():
        if not src.exists():
            records.append({"file": name, "status": "MISSING", "provenance": str(src.relative_to(PR))})
            continue
        df, drop = _sanitize_csv(pd.read_csv(src))
        if drop:
            dropped_log[name] = drop
        _write_csv(df, OUT / name, str(src.relative_to(PR)), records)

    for cond_dir in sorted(glob.glob(str(LLM / "folds_L*"))):
        cond = Path(cond_dir).name.replace("folds_", "")
        files = sorted(glob.glob(str(Path(cond_dir) / "cot_fold*.csv")))
        if not files:
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        df, drop = _sanitize_csv(df)
        if drop:
            dropped_log[f"llm_oof_{cond}.csv"] = drop
        _write_csv(df, OUT / f"llm_oof_{cond}.csv",
                   f"llm/folds_{cond}/cot_fold*.csv (reasoning dropped)", records)

    # 5) GUARD — no banned column may survive in any emitted CSV
    leaks = []
    for csv in glob.glob(str(OUT / "*.csv")):
        cols = set(pd.read_csv(csv, nrows=0).columns)
        bad = cols & BANNED_COLS
        if bad:
            leaks.append({"file": Path(csv).name, "banned_cols": sorted(bad)})
    if leaks:
        raise SystemExit(f"ABORT: transcript-bearing columns leaked into bundle: {leaks}")

    manifest = {
        "bundle": "R2 plot-ready, non-sensitive artifacts",
        "sensitive_policy": "no interview transcript text; free-text/quote columns "
                            f"({sorted(BANNED_COLS)}) dropped and guarded",
        "sensitive_excluded": SENSITIVE_EXCLUDED,
        "columns_dropped_during_sanitize": dropped_log,
        "not_uploaded_externally": "No external destination configured; bundle is "
                                   "staged in-repo for the user to sync. Publishing "
                                   "is a confirm-first action.",
        "files": records,
    }
    (OUT / "upload_manifest.json").write_text(json.dumps(manifest, indent=2))
    present = [r for r in records if r.get("status") != "MISSING"]
    missing = [r for r in records if r.get("status") == "MISSING"]
    print(f"Bundle staged at {OUT.relative_to(PR)}: {len(present)} files, "
          f"{len(missing)} missing, guard PASSED (no transcript columns).")
    if missing:
        print("  missing:", ", ".join(r["file"] for r in missing))
    if dropped_log:
        print("  sanitized (dropped text cols):", json.dumps(dropped_log))
    return OUT


if __name__ == "__main__":
    build()
