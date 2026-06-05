"""
BERT smoke test (task B2).

Verifies the full single-example path works locally:
  load tokenizer -> load classifier -> format a sample item + utterances ->
  tokenize -> forward pass -> print logits.

Run:
    python -m src.models.bert_smoke_test
    python -m src.models.bert_smoke_test --model-name mental/mental-bert-base-uncased
"""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.input_formatting import format_model_input

NUM_LABELS = 4  # PHQ-8 item severity: 0, 1, 2, 3
ID2LABEL = {0: "0", 1: "1", 2: "2", 3: "3"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
DEFAULT_MODEL = "bert-base-uncased"


def run_smoke_test(model_name=DEFAULT_MODEL, max_length=256):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"Loading tokenizer + classifier: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID
    )
    model.eval()
    print("Model loaded. num_labels =", model.config.num_labels)

    # Sample PHQ item + sample utterances (the shape real data takes).
    item_text = "Feeling tired or having little energy"
    utterances = [
        "i have been really exhausted lately",
        "i can barely get out of bed in the morning",
        "no energy to do anything after work",
    ]

    enc = format_model_input(
        item_text, utterances, tokenizer=tokenizer,
        max_length=max_length, return_tensors="pt",
    )
    print("\nTokenized input:")
    for k, v in enc.items():
        print(f"  {k:16s}: {tuple(v.shape)}")
    print("  decoded:", tokenizer.decode(enc["input_ids"][0][: int(enc['attention_mask'].sum())]))

    with torch.no_grad():
        out = model(**enc)

    logits = out.logits
    probs = torch.softmax(logits, dim=-1)
    pred = int(logits.argmax(dim=-1))
    print("\nForward pass OK")
    print("  logits :", [round(x, 3) for x in logits[0].tolist()])
    print("  probs  :", [round(x, 3) for x in probs[0].tolist()])
    print(f"  pred   : {pred} (untrained model -> meaningless, shape check only)")

    assert logits.shape == (1, NUM_LABELS), f"unexpected logits shape {logits.shape}"
    print("\nBERT smoke test passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    run_smoke_test(args.model_name, args.max_length)


if __name__ == "__main__":
    main()
