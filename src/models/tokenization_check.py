"""
Tokenization check (task B5).

Goal: understand how long the tokenized inputs are and how often the evidence is
truncated, so we can pick a sensible ``max_length`` for training.

For a handful of examples it tokenizes the ``item_text`` + evidence pair
**without truncation** to measure the true length, then reports how many tokens
would be dropped at a given ``max_length``.

Run:
    python -m src.models.tokenization_check
    python -m src.models.tokenization_check --evidence-column baseline_utterances --n 5
"""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_loader import load_item_dataset
from src.models.input_formatting import format_model_input

DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "phq8_item_dataset_full.csv"
MODEL_NAME = "mental/mental-bert-base-uncased"
FALLBACK_MODEL = "bert-base-uncased"


def load_tokenizer():
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        return tok, MODEL_NAME
    except Exception as exc:  # gated / offline
        print(f"Could not load {MODEL_NAME}: {exc}\nFalling back to {FALLBACK_MODEL}")
        return AutoTokenizer.from_pretrained(FALLBACK_MODEL), FALLBACK_MODEL


def check_tokenization(dataset_path, evidence_column="transcript_text",
                       n=5, max_length=256):
    df = load_item_dataset(dataset_path)
    if evidence_column not in df.columns:
        raise ValueError(
            f"evidence_column '{evidence_column}' not in dataset. "
            f"Available: {list(df.columns)}"
        )

    tokenizer, used_model = load_tokenizer()
    print(f"Tokenizer       : {used_model}")
    print(f"Dataset         : {dataset_path}")
    print(f"Evidence column : {evidence_column}")
    print(f"max_length      : {max_length}")
    print(f"Examples        : {n}\n")

    rows = df.head(n)
    full_lengths = []
    n_truncated = 0

    for pos, (_, row) in enumerate(rows.iterrows(), 1):
        pair = format_model_input(row["item_text"], row[evidence_column])
        # True length: no truncation, no padding.
        full = tokenizer(pair["text"], pair["text_pair"],
                         truncation=False, padding=False)
        full_len = len(full["input_ids"])
        full_lengths.append(full_len)
        truncated = full_len > max_length
        n_truncated += int(truncated)
        dropped = max(0, full_len - max_length)

        print(f"[{pos}] participant={row['participant_id']} "
              f"item={row['item_id']} ({row['item_name']}) label={row['label']}")
        print(f"     full tokens = {full_len:5d}  | "
              f"{'TRUNCATED' if truncated else 'fits'} at {max_length} "
              f"(drops {dropped} tokens)")

    print("\nSummary")
    print(f"  full token length min/mean/max : "
          f"{min(full_lengths)} / {sum(full_lengths)//len(full_lengths)} / {max(full_lengths)}")
    print(f"  truncated at max_length={max_length}: "
          f"{n_truncated}/{n} examples")
    return {
        "model": used_model,
        "evidence_column": evidence_column,
        "max_length": max_length,
        "full_lengths": full_lengths,
        "n_truncated": n_truncated,
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET))
    parser.add_argument("--evidence-column", default="transcript_text")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    check_tokenization(args.dataset_path, args.evidence_column,
                       args.n, args.max_length)


if __name__ == "__main__":
    main()
