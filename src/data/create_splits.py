"""
Create participant-level train/validation/test splits.

Critical rule:
The same participant_id must appear in one split only.
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset_loader import load_item_dataset


VALID_SPLITS = {"train", "validation", "test"}


def create_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    validation_frac: float = 0.15,
    seed: int = 42,
):
    if train_frac <= 0 or validation_frac <= 0:
        raise ValueError("train_frac and validation_frac must be positive.")

    if train_frac + validation_frac >= 1:
        raise ValueError("train_frac + validation_frac must be less than 1.")

    participants = sorted(df["participant_id"].astype(str).unique())

    if len(participants) < 3:
        raise ValueError("At least 3 participants are required to create train/validation/test splits.")

    rng = np.random.default_rng(seed)
    participants = np.array(participants)
    rng.shuffle(participants)

    n = len(participants)

    n_train = int(round(n * train_frac))
    n_validation = int(round(n * validation_frac))

    # Ensure all three splits are non-empty
    if n_train < 1:
        n_train = 1
    if n_validation < 1:
        n_validation = 1
    if n_train + n_validation >= n:
        n_train = n - 2
        n_validation = 1

    train_ids = participants[:n_train]
    validation_ids = participants[n_train:n_train + n_validation]
    test_ids = participants[n_train + n_validation:]

    split_map = {}

    for pid in train_ids:
        split_map[pid] = "train"

    for pid in validation_ids:
        split_map[pid] = "validation"

    for pid in test_ids:
        split_map[pid] = "test"

    df = df.copy()
    df["participant_id"] = df["participant_id"].astype(str)
    df["split"] = df["participant_id"].map(split_map)

    if df["split"].isna().any():
        missing = df.loc[df["split"].isna(), "participant_id"].unique().tolist()
        raise ValueError(f"Some participants were not assigned to a split: {missing[:20]}")

    validate_splits(df)

    return df


def validate_splits(df: pd.DataFrame):
    if "split" not in df.columns:
        raise ValueError("Missing split column.")

    invalid_splits = set(df["split"].unique()) - VALID_SPLITS
    if invalid_splits:
        raise ValueError(f"Invalid split values: {invalid_splits}")

    present_splits = set(df["split"].unique())
    missing_splits = VALID_SPLITS - present_splits
    if missing_splits:
        raise ValueError(f"Missing required splits: {missing_splits}")

    participant_split_counts = df.groupby("participant_id")["split"].nunique()
    leaking_participants = participant_split_counts[participant_split_counts > 1]

    if not leaking_participants.empty:
        raise ValueError(
            "Participant-level leakage detected. "
            f"Examples: {leaking_participants.head().to_dict()}"
        )

    return True


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/phq8_item_dataset.csv",
        help="Path to item-level dataset without splits.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/phq8_item_dataset_with_splits.csv",
        help="Output path for item-level dataset with splits.",
    )

    parser.add_argument(
        "--train_frac",
        type=float,
        default=0.70,
        help="Fraction of participants assigned to train.",
    )

    parser.add_argument(
        "--validation_frac",
        type=float,
        default=0.15,
        help="Fraction of participants assigned to validation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    df = load_item_dataset(input_path)

    df_with_splits = create_splits(
        df=df,
        train_frac=args.train_frac,
        validation_frac=args.validation_frac,
        seed=args.seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_splits.to_csv(output_path, index=False)

    print("Split dataset created successfully.")
    print(f"Output path: {output_path}")

    print("\nRows per split:")
    print(df_with_splits["split"].value_counts())

    print("\nParticipants per split:")
    print(df_with_splits.groupby("split")["participant_id"].nunique())


if __name__ == "__main__":
    main()