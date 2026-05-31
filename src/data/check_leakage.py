"""
Check participant-level data leakage across train/validation/test splits.
"""

from pathlib import Path
import argparse
import sys

import pandas as pd


VALID_SPLITS = {"train", "validation", "test"}


def check_leakage(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    df = pd.read_csv(path)

    required_cols = ["participant_id", "split"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df["participant_id"].isna().any():
        raise ValueError("participant_id contains missing values.")

    if df["split"].isna().any():
        raise ValueError("split contains missing values.")

    df["participant_id"] = df["participant_id"].astype(str)
    df["split"] = df["split"].astype(str)

    invalid_splits = set(df["split"].unique()) - VALID_SPLITS
    if invalid_splits:
        raise ValueError(f"Invalid split values detected: {invalid_splits}")

    participant_split_counts = df.groupby("participant_id")["split"].nunique()
    leaking_participants = participant_split_counts[participant_split_counts > 1]

    if not leaking_participants.empty:
        for participant_id in leaking_participants.index:
            splits = sorted(df.loc[df["participant_id"] == participant_id, "split"].unique())
            print(f"Leakage detected for participant_id: {participant_id}; splits={splits}")
        return False

    split_to_participants = {
        split: set(df.loc[df["split"] == split, "participant_id"].unique())
        for split in VALID_SPLITS
    }

    split_pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    leakage_found = False

    for split_a, split_b in split_pairs:
        overlap = split_to_participants[split_a] & split_to_participants[split_b]

        if overlap:
            leakage_found = True
            for participant_id in sorted(overlap):
                print(
                    f"Leakage detected for participant_id: {participant_id}; "
                    f"appears in both {split_a} and {split_b}"
                )

    if leakage_found:
        return False

    print("No participant-level leakage detected.")
    return True


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/phq8_item_dataset_with_splits.csv",
        help="Path to dataset with split column.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    ok = check_leakage(Path(args.input_path))

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()