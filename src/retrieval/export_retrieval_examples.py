"""
Export qualitative retrieval examples to docs/retrieval_examples.md.

The output is intended for manual review.
"""

from pathlib import Path
import argparse
import random

import pandas as pd


UTT_SEP = " [UTT_SEP] "


PREFERRED_ITEMS = [
    "Sleep",
    "Appetite",
    "Concentrating",
    "Tired",
]


def split_utterance_string(text):
    if text is None or pd.isna(text):
        return []

    text = str(text).strip()

    if text == "":
        return []

    return [
        part.strip()
        for part in text.split(UTT_SEP)
        if part.strip() != ""
    ]


def select_examples(df, n_examples, seed):
    rng = random.Random(seed)

    selected_rows = []

    # Prefer clinically interesting / harder items first.
    for item_name in PREFERRED_ITEMS:
        item_df = df[df["item_name"] == item_name]

        if len(item_df) == 0:
            continue

        # Prefer validation/test examples for qualitative inspection.
        preferred_df = item_df[item_df["split"].isin(["validation", "test"])]

        if len(preferred_df) == 0:
            preferred_df = item_df

        sample_n = min(3, len(preferred_df))
        indices = list(preferred_df.index)
        rng.shuffle(indices)

        selected_rows.extend(indices[:sample_n])

    # Fill remaining examples randomly.
    if len(selected_rows) < n_examples:
        remaining_indices = [
            idx for idx in df.index
            if idx not in selected_rows
        ]

        rng.shuffle(remaining_indices)
        selected_rows.extend(remaining_indices[: n_examples - len(selected_rows)])

    selected_rows = selected_rows[:n_examples]

    return df.loc[selected_rows].copy()


def write_examples_markdown(df_examples, output_path):
    lines = []

    lines.append("# Retrieval Examples")
    lines.append("")
    lines.append("Manual qualitative review of retrieved utterances.")
    lines.append("")
    lines.append("Use the `comment` field to mark each example as `good`, `mixed`, or `bad`.")
    lines.append("")

    for i, (_, row) in enumerate(df_examples.iterrows(), start=1):
        retrieved = split_utterance_string(row["retrieved_utterances"])
        baseline = split_utterance_string(row["baseline_utterances"])

        lines.append(f"## Example {i}")
        lines.append("")
        lines.append(f"- participant_id: `{row['participant_id']}`")
        lines.append(f"- split: `{row['split']}`")
        lines.append(f"- item_id: `{row['item_id']}`")
        lines.append(f"- item_name: `{row['item_name']}`")
        lines.append(f"- label: `{row['label']}`")
        lines.append("")
        lines.append("### Item text")
        lines.append("")
        lines.append(str(row["item_text"]))
        lines.append("")
        lines.append("### Retrieved utterances")
        lines.append("")

        if retrieved:
            for utt in retrieved:
                lines.append(f"- {utt}")
        else:
            lines.append("- No retrieved utterances.")

        lines.append("")
        lines.append("### Baseline utterances")
        lines.append("")

        if baseline:
            for utt in baseline:
                lines.append(f"- {utt}")
        else:
            lines.append("- No baseline utterances.")

        lines.append("")
        lines.append("### Comment")
        lines.append("")
        lines.append("TODO: good / mixed / bad")
        lines.append("")
        lines.append("---")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print("Retrieval examples saved successfully.")
    print(f"Output path: {output_path}")
    print(f"Examples: {len(df_examples)}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/phq8_item_dataset_full.csv",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="docs/retrieval_examples.md",
    )

    parser.add_argument(
        "--n_examples",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    df = pd.read_csv(input_path)

    required_cols = [
        "participant_id",
        "split",
        "item_id",
        "item_name",
        "item_text",
        "label",
        "retrieved_utterances",
        "baseline_utterances",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_examples = select_examples(
        df=df,
        n_examples=args.n_examples,
        seed=args.seed,
    )

    write_examples_markdown(df_examples, output_path)


if __name__ == "__main__":
    main()