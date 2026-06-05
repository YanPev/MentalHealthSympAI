"""
Batch smoke test (task B10).

Verifies a *real* batch from the DataLoader reaches BERT without shape or label
errors: load dataset -> build DataLoader -> load BERT -> forward pass on one
batch (with labels, so we also get a loss).

Run:
    python -m src.models.bert_batch_smoke_test
    python -m src.models.bert_batch_smoke_test --evidence-column retrieved_utterances --batch-size 8
"""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.data_module import make_dataloaders

NUM_LABELS = 4
DEFAULT_MODEL = "bert-base-uncased"


def run_batch_smoke_test(
    model_name=DEFAULT_MODEL,
    evidence_column="retrieved_utterances",
    batch_size=8,
    max_length=256,
):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"Tokenizer + model : {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS
    )
    model.eval()

    loaders = make_dataloaders(
        tokenizer,
        evidence_column=evidence_column,
        batch_size=batch_size,
        max_length=max_length,
    )
    print(f"Evidence column   : {evidence_column}")
    print(f"Train batches     : {len(loaders['train'])} (batch_size={batch_size})")

    batch = next(iter(loaders["train"]))
    print("\nBatch tensors:")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"  {k:16s}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k:16s}: {type(v).__name__} (len {len(v)})")

    # Labels must be valid class indices.
    labels = batch["label"]
    assert labels.min() >= 0 and labels.max() < NUM_LABELS, "label out of range"

    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": labels,
    }
    if "token_type_ids" in batch:
        model_inputs["token_type_ids"] = batch["token_type_ids"]

    with torch.no_grad():
        out = model(**model_inputs)

    print("\nForward pass OK")
    print(f"  logits shape : {tuple(out.logits.shape)}  (expected ({batch_size}, {NUM_LABELS}))")
    print(f"  loss         : {out.loss.item():.4f}")

    assert out.logits.shape == (labels.shape[0], NUM_LABELS), "unexpected logits shape"
    print("\nBatch smoke test passed: real batch reaches BERT, no shape/label errors.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--evidence-column", default="retrieved_utterances")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    run_batch_smoke_test(
        args.model_name, args.evidence_column, args.batch_size, args.max_length
    )


if __name__ == "__main__":
    main()
