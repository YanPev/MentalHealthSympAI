"""
Unified loader for the processed PHQ-8 item-level dataset.

Person B should use this loader instead of reading the raw data directly.
"""

from pathlib import Path
import argparse

import pandas as pd


REQUIRED_COLUMNS = [
    "participant_id",
    "item_id",
    "item_name",
    "item_text",
    "label",
    "transcript_text",
]


def load_item_dataset(path):
    """
    Load and validate the processed item-level PHQ-8 dataset.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to phq8_item_dataset.csv.

    Returns
    -------
    pandas.DataFrame
        Validated item-level dataset.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    df = pd.read_csv(path)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df["participant_id"].isna().any():
        raise ValueError("participant_id contains missing values.")

    df["participant_id"] = df["participant_id"].astype(str)

    item_id_numeric = pd.to_numeric(df["item_id"], errors="coerce")
    if item_id_numeric.isna().any():
        raise ValueError("item_id contains non-numeric or missing values.")

    df["item_id"] = item_id_numeric.astype(int)

    if not df["item_id"].isin(range(1, 9)).all():
        bad_values = sorted(df.loc[~df["item_id"].isin(range(1, 9)), "item_id"].unique())
        raise ValueError(f"item_id must be in range 1-8. Bad values: {bad_values}")

    label_numeric = pd.to_numeric(df["label"], errors="coerce")
    if label_numeric.isna().any():
        raise ValueError("label contains non-numeric or missing values.")

    df["label"] = label_numeric.astype(int)

    if not df["label"].isin([0, 1, 2, 3]).all():
        bad_values = sorted(df.loc[~df["label"].isin([0, 1, 2, 3]), "label"].unique())
        raise ValueError(f"label must be 0, 1, 2, or 3. Bad values: {bad_values}")

    for col in ["item_name", "item_text", "transcript_text"]:
        empty_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if empty_mask.any():
            raise ValueError(f"{col} contains missing or empty values.")

    if "split" in df.columns:
        valid_splits = {"train", "validation", "test"}
        invalid_splits = set(df["split"].dropna().unique()) - valid_splits
        if invalid_splits:
            raise ValueError(f"Invalid split values detected: {invalid_splits}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to phq8_item_dataset.csv")
    args = parser.parse_args()

    df = load_item_dataset(args.path)
    print("Dataset loaded successfully.")
    print(f"Rows: {len(df)}")
    print(f"Participants: {df['participant_id'].nunique()}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()