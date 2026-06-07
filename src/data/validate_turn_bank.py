"""
Validate a participant-side turn bank against its source transcripts.
"""

from pathlib import Path
import argparse
import json
import math
import re

import pandas as pd


ROLE_SOURCE = "participant_archive_no_explicit_speaker_labels"
REQUIRED_TURN_FIELDS = {
    "participant_id",
    "turn_id",
    "turn_index",
    "start_time",
    "end_time",
    "text",
    "confidence",
    "role_source",
}


def normalize_participant_id(value):
    if value is None or pd.isna(value):
        return None

    return re.sub(r"\.0$", "", str(value).strip())


def normalize_text(value):
    if value is None or pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def load_expected_participant_ids(dataset_path):
    df = pd.read_csv(dataset_path, usecols=["participant_id"])
    return set(
        df["participant_id"]
        .map(normalize_participant_id)
        .dropna()
        .drop_duplicates()
        .tolist()
    )


def load_turn_bank(turn_bank_path):
    with open(turn_bank_path, "r", encoding="utf-8") as file:
        turn_bank = json.load(file)

    if not isinstance(turn_bank, dict) or not turn_bank:
        raise ValueError("turn_bank must be a non-empty JSON object.")

    return turn_bank


def numeric_equal(left, right, tolerance=1e-9):
    return math.isclose(
        float(left),
        float(right),
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def validate_participant_turns(participant_id, turns, transcript_path):
    if not isinstance(turns, list) or not turns:
        raise ValueError(
            f"Turns must be a non-empty list for participant_id={participant_id}"
        )

    transcript_df = pd.read_csv(transcript_path)

    if len(turns) != len(transcript_df):
        raise ValueError(
            f"Row count mismatch for participant_id={participant_id}: "
            f"turn_bank={len(turns)}, raw={len(transcript_df)}"
        )

    previous_start_time = None
    seen_turn_ids = set()
    timestamp_inversions = []

    for turn_index, (turn, (_, raw_row)) in enumerate(
        zip(turns, transcript_df.iterrows())
    ):
        if not isinstance(turn, dict):
            raise ValueError(
                f"Turn is not an object for participant_id={participant_id}, "
                f"turn_index={turn_index}"
            )

        missing_fields = REQUIRED_TURN_FIELDS - set(turn)
        if missing_fields:
            raise ValueError(
                f"Missing fields for participant_id={participant_id}, "
                f"turn_index={turn_index}: {sorted(missing_fields)}"
            )

        expected_turn_id = f"{participant_id}_t{turn_index:04d}"

        if turn["participant_id"] != participant_id:
            raise ValueError(
                f"Participant mismatch in turn {turn.get('turn_id')}: "
                f"{turn['participant_id']} != {participant_id}"
            )

        if turn["turn_index"] != turn_index:
            raise ValueError(
                f"Non-contiguous turn_index for participant_id={participant_id}: "
                f"expected={turn_index}, actual={turn['turn_index']}"
            )

        if turn["turn_id"] != expected_turn_id:
            raise ValueError(
                f"Unexpected turn_id for participant_id={participant_id}: "
                f"expected={expected_turn_id}, actual={turn['turn_id']}"
            )

        if turn["turn_id"] in seen_turn_ids:
            raise ValueError(f"Duplicate turn_id: {turn['turn_id']}")
        seen_turn_ids.add(turn["turn_id"])

        if turn["role_source"] != ROLE_SOURCE:
            raise ValueError(
                f"Unexpected role_source in turn {turn['turn_id']}: "
                f"{turn['role_source']!r}"
            )

        start_time = float(turn["start_time"])
        end_time = float(turn["end_time"])
        confidence = float(turn["confidence"])

        if not all(math.isfinite(value) for value in (start_time, end_time, confidence)):
            raise ValueError(f"Non-finite numeric value in turn {turn['turn_id']}")

        if start_time > end_time:
            raise ValueError(f"start_time exceeds end_time in turn {turn['turn_id']}")

        if previous_start_time is not None and start_time < previous_start_time:
            timestamp_inversions.append(turn["turn_id"])
        previous_start_time = start_time

        if confidence < 0 or confidence > 1:
            raise ValueError(
                f"confidence is outside [0, 1] in turn {turn['turn_id']}"
            )

        raw_text = normalize_text(raw_row["Text"])
        if turn["text"] != raw_text:
            raise ValueError(f"Text mismatch in turn {turn['turn_id']}")

        comparisons = (
            ("start_time", raw_row["Start_Time"]),
            ("end_time", raw_row["End_Time"]),
            ("confidence", raw_row["Confidence"]),
        )
        for field, raw_value in comparisons:
            if not numeric_equal(turn[field], raw_value):
                raise ValueError(
                    f"{field} mismatch in turn {turn['turn_id']}: "
                    f"turn_bank={turn[field]}, raw={raw_value}"
                )

    return {
        "turns": len(turns),
        "timestamp_inversions": timestamp_inversions,
    }


def validate_turn_bank(turn_bank, dataset_path, transcripts_dir):
    expected_ids = load_expected_participant_ids(dataset_path)
    actual_ids = set(turn_bank)

    if actual_ids != expected_ids:
        raise ValueError(
            "Participant coverage mismatch. "
            f"missing={sorted(expected_ids - actual_ids)[:20]}, "
            f"unexpected={sorted(actual_ids - expected_ids)[:20]}"
        )

    total_turns = 0
    global_turn_ids = set()
    timestamp_inversions = []

    for participant_id in sorted(turn_bank):
        transcript_path = transcripts_dir / f"{participant_id}_Transcript.csv"
        if not transcript_path.exists():
            raise FileNotFoundError(f"Missing raw transcript: {transcript_path}")

        turns = turn_bank[participant_id]
        participant_summary = validate_participant_turns(
            participant_id=participant_id,
            turns=turns,
            transcript_path=transcript_path,
        )
        total_turns += participant_summary["turns"]
        timestamp_inversions.extend(participant_summary["timestamp_inversions"])

        for turn in turns:
            turn_id = turn["turn_id"]
            if turn_id in global_turn_ids:
                raise ValueError(f"Globally duplicate turn_id: {turn_id}")
            global_turn_ids.add(turn_id)

    return {
        "participants": len(turn_bank),
        "turns": total_turns,
        "unique_turn_ids": len(global_turn_ids),
        "timestamp_inversions": timestamp_inversions,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--turn_bank_path",
        type=Path,
        default=Path("data/processed/turn_bank.json"),
    )
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
    return parser.parse_args()


def main():
    args = parse_args()

    for path in (args.turn_bank_path, args.dataset_path):
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")

    if not args.transcripts_dir.exists():
        raise FileNotFoundError(
            f"Transcripts directory does not exist: {args.transcripts_dir}"
        )

    turn_bank = load_turn_bank(args.turn_bank_path)
    summary = validate_turn_bank(
        turn_bank=turn_bank,
        dataset_path=args.dataset_path,
        transcripts_dir=args.transcripts_dir,
    )

    print("Turn bank validation passed.")
    print(f"Participants: {summary['participants']}")
    print(f"Turns: {summary['turns']}")
    print(f"Unique turn IDs: {summary['unique_turn_ids']}")
    print(f"Role source: {ROLE_SOURCE}")
    print(f"Timestamp inversions preserved from raw rows: {len(summary['timestamp_inversions'])}")
    if summary["timestamp_inversions"]:
        print(
            "Warning: raw row order is not always chronological. "
            "Examples: "
            + ", ".join(summary["timestamp_inversions"][:10])
        )


if __name__ == "__main__":
    main()
