"""
Build an item-level PHQ-8 dataset.

Input:
- Raw transcript CSV files
- Detailed_PHQ8_Labels.csv
- PHQ8_ITEMS mapping

Output:
- data/processed/phq8_item_dataset.csv

Each participant becomes up to 8 rows:
participant_id + item_id -> label
"""

from pathlib import Path
import argparse
import sys
import re

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.phq8_items import PHQ8_ITEMS, validate_phq8_items


PARTICIPANT_COL = "Participant_ID"

TEXT_COLUMN_CANDIDATES = [
    "Text",
    "text",
    "Value",
    "value",
    "Utterance",
    "utterance",
    "transcript_text",
]


REQUIRED_OUTPUT_COLUMNS = [
    "participant_id",
    "item_id",
    "item_name",
    "item_text",
    "label",
    "transcript_text",
]


def normalize_participant_id(value):
    """
    Normalize participant IDs so that numeric IDs like 300.0 become '300'.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()
    text = re.sub(r"\.0$", "", text)

    return text


def find_labels_file(labels_dir: Path) -> Path:
    candidates = sorted(labels_dir.rglob("Detailed_PHQ8_Labels.csv"))

    if not candidates:
        raise FileNotFoundError(
            f"Detailed_PHQ8_Labels.csv was not found under: {labels_dir}"
        )

    return candidates[0]


def detect_text_column(transcript_df: pd.DataFrame) -> str:
    for col in TEXT_COLUMN_CANDIDATES:
        if col in transcript_df.columns:
            return col

    raise ValueError(
        "Could not detect transcript text column. "
        f"Available columns: {list(transcript_df.columns)}. "
        f"Expected one of: {TEXT_COLUMN_CANDIDATES}"
    )


def read_transcript_text(transcript_path: Path) -> str:
    df = pd.read_csv(transcript_path)

    if df.empty:
        raise ValueError(f"Transcript file is empty: {transcript_path}")

    text_col = detect_text_column(df)

    texts = (
        df[text_col]
        .dropna()
        .astype(str)
        .map(str.strip)
    )

    texts = texts[texts != ""]

    transcript_text = "\n".join(texts.tolist()).strip()

    if transcript_text == "":
        raise ValueError(f"Transcript text is empty after cleaning: {transcript_path}")

    return transcript_text


def validate_label_value(value, participant_id, item_id):
    numeric_value = pd.to_numeric(value, errors="coerce")

    if pd.isna(numeric_value):
        raise ValueError(
            f"Missing or non-numeric label for participant_id={participant_id}, item_id={item_id}"
        )

    if int(numeric_value) != numeric_value:
        raise ValueError(
            f"Label must be an integer 0-3. "
            f"Got {value} for participant_id={participant_id}, item_id={item_id}"
        )

    label = int(numeric_value)

    if label not in {0, 1, 2, 3}:
        raise ValueError(
            f"Label out of range 0-3. "
            f"Got {label} for participant_id={participant_id}, item_id={item_id}"
        )

    return label


def validate_item_dataset(df: pd.DataFrame):
    missing_cols = [col for col in REQUIRED_OUTPUT_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Output dataset is missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("Output dataset is empty.")

    if df["participant_id"].isna().any():
        raise ValueError("participant_id contains missing values.")

    if df["item_id"].isna().any():
        raise ValueError("item_id contains missing values.")

    if not df["item_id"].isin(range(1, 9)).all():
        bad_values = sorted(df.loc[~df["item_id"].isin(range(1, 9)), "item_id"].unique())
        raise ValueError(f"item_id values must be in range 1-8. Bad values: {bad_values}")

    if df["label"].isna().any():
        raise ValueError("label contains missing values.")

    if not df["label"].isin([0, 1, 2, 3]).all():
        bad_values = sorted(df.loc[~df["label"].isin([0, 1, 2, 3]), "label"].unique())
        raise ValueError(f"label values must be 0, 1, 2, 3. Bad values: {bad_values}")

    for col in ["item_name", "item_text", "transcript_text"]:
        empty_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if empty_mask.any():
            raise ValueError(f"{col} contains empty values.")

    rows_per_participant = df.groupby("participant_id").size()
    too_many = rows_per_participant[rows_per_participant > 8]
    if not too_many.empty:
        raise ValueError(
            "Some participants have more than 8 rows: "
            f"{too_many.to_dict()}"
        )

    duplicated_items = df.duplicated(subset=["participant_id", "item_id"])
    if duplicated_items.any():
        examples = df.loc[duplicated_items, ["participant_id", "item_id"]].head()
        raise ValueError(
            "Duplicate participant_id + item_id rows detected. "
            f"Examples:\n{examples}"
        )

    return True


def build_item_dataset(project_dir: Path, labels_path: Path, transcripts_dir: Path, output_path: Path):
    validate_phq8_items()

    if labels_path is None:
        labels_path = find_labels_file(project_dir / "data" / "raw" / "edaic" / "labels")

    if transcripts_dir is None:
        transcripts_dir = project_dir / "data" / "raw" / "edaic" / "transcripts"

    if output_path is None:
        output_path = project_dir / "data" / "processed" / "phq8_item_dataset.csv"

    labels_df = pd.read_csv(labels_path)

    if PARTICIPANT_COL not in labels_df.columns:
        raise ValueError(
            f"Participant column was not found: {PARTICIPANT_COL}. "
            f"Available columns: {list(labels_df.columns)}"
        )

    phq8_columns = [item["column"] for item in PHQ8_ITEMS.values()]
    missing_phq_cols = [col for col in phq8_columns if col not in labels_df.columns]
    if missing_phq_cols:
        raise ValueError(f"Missing PHQ-8 item columns in labels file: {missing_phq_cols}")

    labels_df["_participant_id"] = labels_df[PARTICIPANT_COL].map(normalize_participant_id)

    if labels_df["_participant_id"].isna().any():
        raise ValueError("Some labels rows have missing participant IDs.")

    duplicated_label_ids = labels_df["_participant_id"].duplicated()
    if duplicated_label_ids.any():
        duplicate_ids = labels_df.loc[duplicated_label_ids, "_participant_id"].tolist()
        raise ValueError(f"Duplicate participant IDs in labels file: {duplicate_ids[:20]}")

    labels_by_id = labels_df.set_index("_participant_id")

    transcript_files = sorted(transcripts_dir.glob("*_Transcript.csv"))
    if not transcript_files:
        raise FileNotFoundError(f"No transcript files found in: {transcripts_dir}")

    rows = []
    skipped_no_labels = []

    for transcript_path in transcript_files:
        participant_id = transcript_path.name.replace("_Transcript.csv", "")
        participant_id = normalize_participant_id(participant_id)

        if participant_id not in labels_by_id.index:
            skipped_no_labels.append(participant_id)
            continue

        transcript_text = read_transcript_text(transcript_path)
        label_row = labels_by_id.loc[participant_id]

        for item_id, item in sorted(PHQ8_ITEMS.items()):
            label = validate_label_value(
                value=label_row[item["column"]],
                participant_id=participant_id,
                item_id=item_id,
            )

            rows.append(
                {
                    "participant_id": participant_id,
                    "item_id": item_id,
                    "item_name": item["symptom_name"],
                    "item_text": item["item_text"],
                    "label": label,
                    "transcript_text": transcript_text,
                }
            )

    item_df = pd.DataFrame(rows, columns=REQUIRED_OUTPUT_COLUMNS)
    validate_item_dataset(item_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    item_df.to_csv(output_path, index=False)

    n_participants = item_df["participant_id"].nunique()

    print("Item-level dataset created successfully.")
    print(f"Output path: {output_path}")
    print(f"Rows: {len(item_df)}")
    print(f"Participants: {n_participants}")
    print(f"Skipped transcript files without labels: {len(skipped_no_labels)}")

    return item_df


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project_dir",
        type=str,
        default=str(PROJECT_ROOT),
        help="Project root directory.",
    )

    parser.add_argument(
        "--labels_path",
        type=str,
        default=None,
        help="Optional path to Detailed_PHQ8_Labels.csv.",
    )

    parser.add_argument(
        "--transcripts_dir",
        type=str,
        default=None,
        help="Optional path to transcript CSV directory.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional output path for phq8_item_dataset.csv.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    project_dir = Path(args.project_dir)

    labels_path = Path(args.labels_path) if args.labels_path else None
    transcripts_dir = Path(args.transcripts_dir) if args.transcripts_dir else None
    output_path = Path(args.output_path) if args.output_path else None

    build_item_dataset(
        project_dir=project_dir,
        labels_path=labels_path,
        transcripts_dir=transcripts_dir,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()