"""
Build a manifest of every training run from the saved per-run metadata.

Scans the metrics JSON files written by ``train_transformer_classifier`` (single
train/val/test split runs, the MentalBERT sweep, the multi-seed runs) and by
``cross_validate`` (k-fold CV / OOF runs), pairs each with its prediction file,
recomputes the headline metrics, and emits one manifest record per run in the
requested schema. Writes ``outputs/run_manifest.json``.

    python -m src.evaluation.build_run_manifest
"""

from pathlib import Path
import json
import subprocess
from datetime import datetime, timezone

import pandas as pd
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
LABELS = [0, 1, 2, 3]

INPUT_FORMAT = "[CLS] item_text [SEP] evidence [SEP] (truncation=only_second; item_text kept in full)"
DEVICE = "cuda (NVIDIA RTX 3090, SLURM gpu partition)"
SPLIT_SINGLE = "data/raw/edaic/labels/labels/{train,dev,test}_split.csv (embedded as dataset 'split' column)"
SPLIT_CV = "participant-grouped StratifiedGroupKFold (k=5, shuffle, random_state=seed); no external split file"


def retrieval_method(dataset_path, evidence_column):
    """Make the evidence source explicit so baselines on different retrieval
    pipelines (TF-IDF vs BM25 utterances vs context windows) are never conflated."""
    ds = Path(dataset_path).name
    col = evidence_column
    if col == "transcript_text":
        return "full transcript (no retrieval)"
    if col == "baseline_utterances":
        return "first-K utterances (no retrieval)"
    if col == "retrieved_utterances":
        if "full_bm25" in ds:
            return "BM25 (utterance retrieval)"
        return "TF-IDF (utterance retrieval)"          # phq8_item_dataset_full.csv
    if "bm25_pack" in col:
        return "BM25 (participant-side context windows)"
    if "hybrid_pack" in col:
        return "Hybrid: BM25 + semantic (participant-side context windows)"
    return "unknown"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return ""


def iso_mtime(p: Path):
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()


def compute_metrics(pred_path: Path):
    df = pd.read_csv(pred_path)
    ycol = "true_label" if "true_label" in df.columns else "label"
    y, p = df[ycol].to_numpy(), df["prediction"].to_numpy()
    return {
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "macro_f1": round(float(f1_score(y, p, average="macro", labels=LABELS, zero_division=0)), 4),
        "mae": round(float(mean_absolute_error(y, p)), 4),
        "qwk": round(float(cohen_kappa_score(y, p, weights="quadratic", labels=LABELS)), 4),
    }


def record(run_id, ts, commit, a, pred_path, kind):
    loss = a.get("loss", "cross_entropy")
    cw = a.get("class_weights", "none")
    if loss == "corn":
        cw = "none"  # CORN ignores class weights
        cw_desc = "none (CORN handles ordering; class weights ignored)"
        loss_desc = "CORN (rank-consistent ordinal)"
    elif cw == "balanced":
        cw_desc = "balanced inverse-frequency (train split only, pooled over items)"
        loss_desc = "weighted cross-entropy"
    else:
        cw_desc = "none"
        loss_desc = "cross-entropy"

    pred_abs = pred_path if pred_path.is_absolute() else (ROOT / pred_path)
    metrics = compute_metrics(pred_abs) if pred_abs.exists() else {}

    if kind == "cv":
        split_file = SPLIT_CV
        ckpt = "none (final epoch per fold; OOF predictions pooled)"
        cmd = (f"python -m src.models.cross_validate "
               f"--model-name {a['model_name']} --dataset-path {a['dataset_path']} "
               f"--evidence-column {a['evidence_column']} --k-folds {a.get('k_folds',5)} "
               f"--num-epochs {a['num_epochs']} --batch-size {a['batch_size']} "
               f"--max-length {a['max_length']} --learning-rate {a['learning_rate']} "
               f"--seed {a['seed']} --loss {loss}"
               + (f" --class-weights {cw}" if loss == 'cross_entropy' else "")
               + f" --tag {a.get('tag','')}")
    else:
        split_file = SPLIT_SINGLE
        ckpt = "none (final epoch used)"
        cmd = (f"python -m src.models.train_transformer_classifier "
               f"--dataset-path {a['dataset_path']} --evidence-column {a['evidence_column']} "
               f"--model-name {a['model_name']} --num-epochs {a['num_epochs']} "
               f"--batch-size {a['batch_size']} --max-length {a['max_length']} "
               f"--learning-rate {a['learning_rate']} --eval-split {a.get('eval_split','validation')} "
               + (f"--class-weights {cw} " if cw == 'balanced' else "")
               + f"--seed {a['seed']} --output-name {a.get('output_name','')}")

    return {
        "run_id": run_id,
        "timestamp": ts,
        "git_commit": commit,
        "dataset_path": a["dataset_path"],
        "split_file": split_file,
        "evidence_column": a["evidence_column"],
        "retrieval_method": retrieval_method(a["dataset_path"], a["evidence_column"]),
        "model_name": a["model_name"],
        "input_format": INPUT_FORMAT,
        "max_length": a["max_length"],
        "batch_size": a["batch_size"],
        "learning_rate": a["learning_rate"],
        "optimizer": "AdamW",
        "loss": loss_desc,
        "class_weights": cw_desc,
        "epochs": a["num_epochs"],
        "early_stopping": "none",
        "checkpoint_selection_metric": ckpt,
        "seed": a["seed"],
        "device": DEVICE,
        "eval_split": a.get("eval_split", "5-fold CV (OOF)") if kind != "cv" else "5-fold CV (OOF)",
        "metrics": metrics,
        "prediction_file": str(pred_abs.relative_to(ROOT)) if pred_abs.exists() else "",
        "command": cmd,
    }


def main():
    commit = git_commit()
    runs = []

    # 1) single-split runs (train_transformer_classifier) — outputs/*_metrics.json
    for mf in sorted(OUT.glob("*_metrics.json")):
        a = json.loads(mf.read_text())["args"]
        pred = Path(a["output_dir"]) / a["output_name"]
        runs.append(record(mf.stem.replace("_metrics", ""), iso_mtime(mf), commit, a, pred, "single"))

    # 2) sweep + 3) seeds (also train_transformer_classifier, in subdirs)
    for sub in ["sweep_mentalbert", "seeds_weighted"]:
        for mf in sorted((OUT / sub).glob("*_metrics.json")):
            a = json.loads(mf.read_text())["args"]
            pred = Path(a["output_dir"]) / a["output_name"]
            runs.append(record(f"{sub}/{mf.stem.replace('_metrics','')}", iso_mtime(mf), commit, a, pred, "single"))

    # 4) cross-validation runs — outputs/cv/cv_metrics_<tag>.json
    for mf in sorted((OUT / "cv").glob("cv_metrics_*.json")):
        a = json.loads(mf.read_text())["args"]
        # The early CV runs had tag=None in args; the real tag is in the filename.
        tag = mf.stem[len("cv_metrics_"):]
        a["tag"] = tag
        pred = Path(a["output_dir"]) / f"oof_predictions_{tag}.csv"
        runs.append(record(f"cv/{tag}", iso_mtime(mf), commit, a, pred, "cv"))

    runs.sort(key=lambda r: r["timestamp"])
    out = OUT / "run_manifest.json"
    out.write_text(json.dumps(runs, indent=2))
    print(f"wrote {out}  ({len(runs)} runs)")
    # quick summary
    by_kind = {}
    for r in runs:
        k = "CV" if r["run_id"].startswith("cv/") else r["run_id"].split("/")[0] if "/" in r["run_id"] else "single"
        by_kind[k] = by_kind.get(k, 0) + 1
    print("by group:", by_kind)


if __name__ == "__main__":
    main()
