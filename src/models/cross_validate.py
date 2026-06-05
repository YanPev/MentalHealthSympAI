"""
Participant-level k-fold cross-validation for the PHQ-8 item classifier.

The single fixed test split (264 rows) is too small to tell what helps — severe-
class F1 swings wildly. This runs stratified, *grouped-by-participant* k-fold CV
so every participant is held out exactly once, giving out-of-fold (OOF)
predictions over the entire dataset and low-variance metric estimates.

Splitting is by ``participant_id`` (no item from a held-out participant ever
leaks into training) and stratified by item label so folds stay class-balanced.

It reuses the train / eval loop from ``train_transformer_classifier`` and the
same ``--class-weights balanced`` option, so the CV config matches the headline
experiment.

    python -m src.models.cross_validate \
        --model-name bert-base-uncased --evidence-column retrieved_utterances \
        --class-weights balanced --num-epochs 8 --k-folds 5
"""

from pathlib import Path
import argparse
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import load_item_dataset
from src.models.data_module import DEFAULT_DATASET
from src.models.phq8_torch_dataset import PHQ8TorchDataset
from src.models.train_transformer_classifier import (
    NUM_LABELS, set_seed, train_one_epoch, evaluate,
)


def build_loader(df, tokenizer, evidence_column, batch_size, max_length, shuffle):
    from torch.utils.data import DataLoader

    ds = PHQ8TorchDataset(df, tokenizer, evidence_column=evidence_column,
                          max_length=max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def collect_probabilities(model, loader, device, prob_fn=None):
    """Return an (N, NUM_LABELS) array of class probabilities in loader order.

    ``prob_fn`` maps a logits tensor to a probability tensor; defaults to softmax
    (for the cross-entropy head). Pass ``corn_probas`` for the ordinal head.
    """
    import torch
    import numpy as np
    from src.models.train_transformer_classifier import _model_inputs

    if prob_fn is None:
        prob_fn = lambda z: torch.softmax(z, dim=-1)
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            inputs = _model_inputs(batch, device)
            logits = model(**inputs).logits
            probs.append(prob_fn(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def make_loss_fn(train_labels, device):
    import numpy as np
    import torch
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.arange(NUM_LABELS)
    weights = compute_class_weight("balanced", classes=classes, y=train_labels)
    wt = torch.tensor(weights, dtype=torch.float, device=device)
    return torch.nn.CrossEntropyLoss(weight=wt)


def main():
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    )
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    p = argparse.ArgumentParser(description="k-fold CV for PHQ-8 item classifier")
    p.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    p.add_argument("--evidence-column", default="retrieved_utterances")
    p.add_argument("--model-name", default="bert-base-uncased")
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-epochs", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--class-weights", default="balanced", choices=["none", "balanced"])
    p.add_argument("--loss", default="cross_entropy",
                   choices=["cross_entropy", "corn"],
                   help="'corn' = rank-consistent ordinal loss (K-1 output head).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "cv"))
    p.add_argument("--tag", default=None, help="Filename tag; default from model name.")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or args.model_name.split("/")[-1].replace("-", "_")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_item_dataset(args.dataset_path).reset_index(drop=True)
    groups = df["participant_id"].to_numpy()
    y = df["label"].to_numpy()

    print("=" * 64)
    print("PHQ-8 item classifier — participant-level stratified k-fold CV")
    print(f"  model           : {args.model_name}")
    print(f"  evidence column : {args.evidence_column}")
    print(f"  device          : {device}")
    print(f"  folds           : {args.k_folds} (grouped by participant)")
    print(f"  loss            : {args.loss}")
    print(f"  class weights   : {args.class_weights}")
    print(f"  epochs/fold     : {args.num_epochs}  bs {args.batch_size}  lr {args.learning_rate}")
    print("=" * 64)

    # Head / loss / prediction selection.
    is_corn = args.loss == "corn"
    n_out = NUM_LABELS - 1 if is_corn else NUM_LABELS
    if is_corn:
        from src.models.ordinal import corn_loss, corn_probas, corn_predict
        predict_fn = lambda z: corn_predict(z).cpu().numpy()
        prob_fn = corn_probas
        if args.class_weights == "balanced":
            print("  [note] class weights ignored for CORN (ordinal loss handles ordering).")
    else:
        predict_fn = None
        prob_fn = None

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    sgkf = StratifiedGroupKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    oof = df[["participant_id", "item_id", "item_name", "label"]].copy()
    oof["prediction"] = -1
    oof["fold"] = -1
    for c in range(NUM_LABELS):
        oof[f"prob_{c}"] = np.nan
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(df, y, groups), start=1):
        t0 = time.time()
        tr_df, va_df = df.iloc[tr_idx], df.iloc[va_idx]
        # Leakage guard: no participant in both sides.
        assert not (set(tr_df["participant_id"]) & set(va_df["participant_id"]))

        train_loader = build_loader(tr_df, tokenizer, args.evidence_column,
                                    args.batch_size, args.max_length, shuffle=True)
        val_loader = build_loader(va_df, tokenizer, args.evidence_column,
                                  args.batch_size, args.max_length, shuffle=False)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=n_out).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        if is_corn:
            loss_fn = lambda logits, labels: corn_loss(logits, labels, NUM_LABELS)
        elif args.class_weights == "balanced":
            loss_fn = make_loss_fn(tr_df["label"].to_numpy(), device)
        else:
            loss_fn = None

        for epoch in range(1, args.num_epochs + 1):
            train_one_epoch(model, train_loader, optimizer, device, loss_fn=loss_fn)

        metrics, preds = evaluate(model, val_loader, device,
                                  loss_fn=loss_fn, predict_fn=predict_fn)
        proba = collect_probabilities(model, val_loader, device, prob_fn=prob_fn)  # (n_val, NUM_LABELS)
        oof.iloc[va_idx, oof.columns.get_loc("prediction")] = preds
        oof.iloc[va_idx, oof.columns.get_loc("fold")] = fold
        for c in range(NUM_LABELS):
            oof.iloc[va_idx, oof.columns.get_loc(f"prob_{c}")] = proba[:, c]

        dt = time.time() - t0
        fold_metrics.append({"fold": fold, "n": int(len(va_df)), **metrics, "seconds": dt})
        print(f"Fold {fold}/{args.k_folds} | n={len(va_df):4d} "
              f"acc {metrics['accuracy']:.3f} macroF1 {metrics['macro_f1']:.3f} "
              f"MAE {metrics['mae']:.3f} ({dt:.0f}s)")

        del model
        torch.cuda.empty_cache()

    # ---- aggregate ---------------------------------------------------------
    assert (oof["prediction"] >= 0).all(), "some rows never predicted"
    yt, yp = oof["label"].to_numpy(), oof["prediction"].to_numpy()

    def mstd(key):
        a = np.array([m[key] for m in fold_metrics])
        return float(a.mean()), float(a.std(ddof=1))

    pooled = {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(yt, yp)),
        "confusion_matrix": confusion_matrix(yt, yp, labels=list(range(NUM_LABELS))).tolist(),
    }
    per_fold = {k: mstd(k) for k in ("accuracy", "macro_f1", "mae")}

    print("-" * 64)
    print(f"Per-fold (mean ± std over {args.k_folds} folds):")
    for k in ("accuracy", "macro_f1", "mae"):
        print(f"  {k:9s}: {per_fold[k][0]:.3f} ± {per_fold[k][1]:.3f}")
    print(f"Pooled OOF (all {len(oof)} examples): "
          f"acc {pooled['accuracy']:.3f}  macroF1 {pooled['macro_f1']:.3f}  MAE {pooled['mae']:.3f}")

    oof_path = out_dir / f"oof_predictions_{tag}.csv"
    oof.to_csv(oof_path, index=False)
    metrics_path = out_dir / f"cv_metrics_{tag}.json"
    metrics_path.write_text(json.dumps({
        "args": vars(args),
        "per_fold_mean_std": per_fold,
        "pooled_oof": pooled,
        "folds": fold_metrics,
    }, indent=2))
    print(f"Saved OOF predictions : {oof_path}")
    print(f"Saved CV metrics      : {metrics_path}")


if __name__ == "__main__":
    main()
