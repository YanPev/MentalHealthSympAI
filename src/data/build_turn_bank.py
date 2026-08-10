"""
Build a participant-side turn bank from ordered E-DAIC transcript rows.

The available participant archives do not contain explicit speaker labels or
Ellie turns. This module therefore preserves each raw transcript row as a
participant-side turn without inferring an interviewer role.
"""

from pathlib import Path
import argparse
import json
import re

import pandas as pd


ROLE_SOURCE = "participant_archive_no_explicit_speaker_labels"
REQUIRED_TRANSCRIPT_COLUMNS = {
    "Start_Time",
    "End_Time",
    "Text",
    "Confidence",
}


def normalize_participant_id(value):
    if value is None or pd.isna(value):
        return None

    return re.sub(r"\.0$", "", str(value).strip())


def normalize_text(value):
    if value is None or pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def load_participant_ids(dataset_path):
    df = pd.read_csv(dataset_path, usecols=["participant_id"])
    participant_ids = (
        df["participant_id"]
        .map(normalize_participant_id)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not participant_ids:
        raise ValueError("No participant IDs were found in the dataset.")

    return participant_ids


def parse_numeric(value, column, participant_id, row_number):
    numeric = pd.to_numeric(value, errors="coerce")

    if pd.isna(numeric):
        raise ValueError(
            f"Invalid {column} for participant_id={participant_id}, "
            f"raw_row_number={row_number}: {value!r}"
        )

    return float(numeric)


def build_participant_turns(participant_id, transcript_path):
    transcript_df = pd.read_csv(transcript_path)
    missing_columns = REQUIRED_TRANSCRIPT_COLUMNS - set(transcript_df.columns)

    if missing_columns:
        raise ValueError(
            f"Transcript {transcript_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    turns = []

    for row_index, row in transcript_df.iterrows():
        raw_row_number = row_index + 2
        text = normalize_text(row["Text"])

        if text == "":
            raise ValueError(
                f"Empty Text for participant_id={participant_id}, "
                f"raw_row_number={raw_row_number}"
            )

        start_time = parse_numeric(
            row["Start_Time"],
            "Start_Time",
            participant_id,
            raw_row_number,
        )
        end_time = parse_numeric(
            row["End_Time"],
            "End_Time",
            participant_id,
            raw_row_number,
        )
        confidence = parse_numeric(
            row["Confidence"],
            "Confidence",
            participant_id,
            raw_row_number,
        )

        turn_index = len(turns)
        turns.append(
            {
                "participant_id": participant_id,
                "turn_id": f"{participant_id}_t{turn_index:04d}",
                "turn_index": turn_index,
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
                "confidence": confidence,
                "role_source": ROLE_SOURCE,
            }
        )

    if not turns:
        raise ValueError(f"Transcript contains no turns: {transcript_path}")

    return turns


def build_turn_bank(dataset_path, transcripts_dir):
    participant_ids = load_participant_ids(dataset_path)
    turn_bank = {}
    missing_transcripts = []

    for participant_id in participant_ids:
        transcript_path = transcripts_dir / f"{participant_id}_Transcript.csv"

        if not transcript_path.exists():
            missing_transcripts.append(participant_id)
            continue

        turn_bank[participant_id] = build_participant_turns(
            participant_id=participant_id,
            transcript_path=transcript_path,
        )

    if missing_transcripts:
        raise FileNotFoundError(
            "Missing raw transcript files for participant IDs: "
            f"{missing_transcripts[:20]} "
            f"(total={len(missing_transcripts)})"
        )

    if len(turn_bank) != len(participant_ids):
        raise ValueError(
            "Participant coverage mismatch: "
            f"dataset={len(participant_ids)}, turn_bank={len(turn_bank)}"
        )

    return turn_bank


def save_turn_bank(turn_bank, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(turn_bank, file, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=Path("data/processed/phq8_item_dataset_with_splits.csv"),
    )
    parser.add_argument(
        "--transcripts_dir",
        type=Path,
        default=Path("data/raw/edaic/transcripts"),
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/processed/turn_bank.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {args.dataset_path}")

    if not args.transcripts_dir.exists():
        raise FileNotFoundError(
            f"Transcripts directory does not exist: {args.transcripts_dir}"
        )

    turn_bank = build_turn_bank(
        dataset_path=args.dataset_path,
        transcripts_dir=args.transcripts_dir,
    )
    save_turn_bank(turn_bank, args.output_path)

    turn_counts = [len(turns) for turns in turn_bank.values()]
    print("Turn bank created successfully.")
    print(f"Output path: {args.output_path}")
    print(f"Participants: {len(turn_bank)}")
    print(f"Turns: {sum(turn_counts)}")
    print(f"Min turns per participant: {min(turn_counts)}")
    print(f"Max turns per participant: {max(turn_counts)}")
    print(f"Role source: {ROLE_SOURCE}")


if __name__ == "__main__":
    main()
