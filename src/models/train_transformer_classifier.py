"""
Train a transformer classifier for PHQ-8 item severity (tasks B11 + B18).

A single, fully CLI-configurable entry point so experiments launch without code
changes (B18). It fine-tunes ``bert-base-uncased`` (or any HF model) on the
item-level dataset, evaluates each epoch, and saves per-example predictions plus
a metrics JSON.

The ``--evidence-column`` flag selects the condition (baseline vs. retrieval),
so the exact same script produces both the no-retrieval (B12) and retrieval
(B13) prediction files.

Examples
--------
# Baseline (no retrieval)
python -m src.models.train_transformer_classifier \
    --evidence-column baseline_utterances \
    --num-epochs 3 --output-name predictions_no_retrieval_bert.csv

# Retrieval
python -m src.models.train_transformer_classifier \
    --evidence-column retrieved_utterances \
    --num-epochs 3 --output-name predictions_retrieval_bert.csv

# Fast smoke test (a few batches only)
python -m src.models.train_transformer_classifier \
    --limit-train-batches 5 --limit-eval-batches 5 --max-length 128
"""

from pathlib import Path
import argparse
import json
import random
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.data_module import make_dataloaders, DEFAULT_DATASET

NUM_LABELS = 4  # PHQ-8 item severity 0..3


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _model_inputs(batch, device):
    """Pull the tensors BERT needs out of a collated batch."""
    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    if "token_type_ids" in batch:
        inputs["token_type_ids"] = batch["token_type_ids"].to(device)
    return inputs


def _forward_loss(model, inputs, labels, loss_fn):
    """Compute (output, loss).

    With ``loss_fn`` we run the model without ``labels`` (so it works for an
    ordinal head whose output dim != num classes) and apply the custom loss.
    Without ``loss_fn`` we fall back to the model's built-in cross-entropy.
    """
    if loss_fn is None:
        out = model(**inputs, labels=labels)
        return out, out.loss
    out = model(**inputs)
    return out, loss_fn(out.logits, labels)


def _default_predict(logits):
    return logits.argmax(dim=-1).cpu().numpy()


def train_one_epoch(model, loader, optimizer, device, limit=0, loss_fn=None):
    import torch

    model.train()
    total_loss, n = 0.0, 0
    for step, batch in enumerate(loader):
        if limit and step >= limit:
            break
        optimizer.zero_grad()
        inputs = _model_inputs(batch, device)
        labels = batch["label"].to(device)
        _, loss = _forward_loss(model, inputs, labels, loss_fn)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        if step % 20 == 0:
            print(f"    step {step:4d} | loss {loss.item():.4f}")
    return total_loss / max(n, 1)


def evaluate(model, loader, device, limit=0, loss_fn=None, predict_fn=None):
    import numpy as np
    import torch

    predict_fn = predict_fn or _default_predict
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if limit and step >= limit:
                break
            inputs = _model_inputs(batch, device)
            labels = batch["label"].to(device)
            out, loss = _forward_loss(model, inputs, labels, loss_fn)
            total_loss += loss.item() * labels.size(0)
            n += labels.size(0)
            preds = predict_fn(out.logits)
            all_preds.extend(np.asarray(preds).tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    metrics = {
        "loss": total_loss / max(n, 1),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(labels, preds)),
        "n": int(n),
    }
    return metrics, preds


def save_predictions(loader, preds, out_path):
    """Attach predictions to the (unshuffled) eval subset and write a CSV."""
    df = loader.dataset.df.copy()
    df = df.iloc[: len(preds)].copy()  # honour --limit-eval-batches
    cols = [c for c in ["participant_id", "item_id", "item_name", "split"] if c in df.columns]
    out = df[cols].copy()
    out["true_label"] = df["label"].values[: len(preds)]
    out["prediction"] = preds
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    parser = argparse.ArgumentParser(description="Train PHQ-8 item classifier")
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--evidence-column", default="retrieved_utterances")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--eval-split", default="validation",
                        choices=["validation", "test"])
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--output-name", default=None,
                        help="Predictions filename; default derived from evidence column.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weights", default="none",
                        choices=["none", "balanced"],
                        help="'balanced' = inverse-frequency CE weights from the train split.")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-eval-batches", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    out_name = args.output_name or f"predictions_{args.evidence_column}_bert.csv"

    print("=" * 64)
    print("PHQ-8 item classifier — training")
    print(f"  model           : {args.model_name}")
    print(f"  evidence column : {args.evidence_column}")
    print(f"  device          : {device}")
    print(f"  epochs          : {args.num_epochs}  batch_size {args.batch_size}  max_len {args.max_length}")
    print(f"  eval split      : {args.eval_split}")
    print(f"  predictions out : {output_dir / out_name}")
    print("=" * 64)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=NUM_LABELS
    ).to(device)

    loaders = make_dataloaders(
        tokenizer,
        dataset_path=args.dataset_path,
        evidence_column=args.evidence_column,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    if args.eval_split not in loaders:
        raise ValueError(f"eval split '{args.eval_split}' not available: {list(loaders)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Optional inverse-frequency class weights, computed from the TRAIN split only.
    loss_fn = None
    if args.class_weights == "balanced":
        import numpy as np
        from sklearn.utils.class_weight import compute_class_weight

        train_labels = loaders["train"].dataset.df["label"].to_numpy()
        classes = np.arange(NUM_LABELS)
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=train_labels
        )
        weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
        print(f"  class weights   : {dict(zip(classes.tolist(), [round(w, 3) for w in weights]))}")

    history = []
    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, device,
            limit=args.limit_train_batches, loss_fn=loss_fn,
        )
        metrics, preds = evaluate(
            model, loaders[args.eval_split], device,
            limit=args.limit_eval_batches, loss_fn=loss_fn,
        )
        dt = time.time() - t0
        print(f"  train_loss {train_loss:.4f} | "
              f"val_loss {metrics['loss']:.4f} acc {metrics['accuracy']:.3f} "
              f"macroF1 {metrics['macro_f1']:.3f} MAE {metrics['mae']:.3f} "
              f"({dt:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics, "seconds": dt})

    # Save final predictions + metrics.
    pred_path = save_predictions(loaders[args.eval_split], preds, output_dir / out_name)
    metrics_path = output_dir / (Path(out_name).stem + "_metrics.json")
    metrics_path.write_text(json.dumps({
        "args": vars(args),
        "final": history[-1],
        "history": history,
    }, indent=2))

    print(f"\nSaved predictions : {pred_path}")
    print(f"Saved metrics     : {metrics_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()
