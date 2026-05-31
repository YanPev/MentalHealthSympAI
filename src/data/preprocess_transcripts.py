"""
Preprocess E-DAIC transcripts and create an utterance bank.

Main functions:
- clean_transcript(text)
- extract_participant_utterances(transcript)

CLI output:
- data/processed/utterance_bank.json

The script prefers raw transcript CSV files when available, because they may
contain speaker information. If speaker information is unavailable, it falls
back to using all transcript text lines.
"""

from pathlib import Path
import argparse
import json
import re
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


TEXT_COLUMN_CANDIDATES = [
    "Text",
    "text",
    "Utterance",
    "utterance",
    "Value",
    "value",
    "transcript_text",
]

SPEAKER_COLUMN_CANDIDATES = [
    "Speaker",
    "speaker",
    "Role",
    "role",
    "speaker_role",
    "Participant",
    "participant",
]


INTERVIEWER_MARKERS = {
    "ellie",
    "interviewer",
    "therapist",
    "clinician",
    "doctor",
    "agent",
    "system",
}

PARTICIPANT_MARKERS = {
    "participant",
    "subject",
    "patient",
    "client",
    "user",
}


def clean_transcript(text):
    """
    Clean a single transcript utterance or text segment.

    Parameters
    ----------
    text : str
        Raw utterance text.

    Returns
    -------
    str
        Cleaned utterance.
    """
    if text is None or pd.isna(text):
        return ""

    text = str(text)

    # Remove common non-verbal markers but keep meaningful text.
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\(.*?\)", " ", text)

    # Normalize whitespace.
    text = text.replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_text_column(df):
    for col in TEXT_COLUMN_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not detect text column. "
        f"Available columns: {list(df.columns)}. "
        f"Expected one of: {TEXT_COLUMN_CANDIDATES}"
    )


def detect_speaker_column(df):
    for col in SPEAKER_COLUMN_CANDIDATES:
        if col in df.columns:
            return col

    return None


def is_participant_speaker(value):
    """
    Decide whether a speaker value probably belongs to the participant.

    Returns
    -------
    True if participant, False if interviewer, None if unknown.
    """
    if value is None or pd.isna(value):
        return None

    speaker = str(value).strip().lower()

    if speaker == "":
        return None

    if any(marker in speaker for marker in INTERVIEWER_MARKERS):
        return False

    if any(marker in speaker for marker in PARTICIPANT_MARKERS):
        return True

    return None


def extract_participant_utterances(transcript):
    """
    Extract participant utterances from either:
    1. a pandas DataFrame loaded from a raw transcript CSV, or
    2. a string containing transcript text.

    Parameters
    ----------
    transcript : pandas.DataFrame or str
        Raw transcript table or transcript text.

    Returns
    -------
    list[str]
        Cleaned participant utterances.
    """
    utterances = []

    if isinstance(transcript, pd.DataFrame):
        df = transcript.copy()

        if df.empty:
            return []

        text_col = detect_text_column(df)
        speaker_col = detect_speaker_column(df)

        if speaker_col is not None:
            speaker_flags = df[speaker_col].map(is_participant_speaker)

            # If we can identify participant rows, keep only them.
            if (speaker_flags == True).any():
                df = df.loc[speaker_flags == True]

            # Otherwise, remove clear interviewer rows if possible.
            elif (speaker_flags == False).any():
                df = df.loc[speaker_flags != False]

        raw_texts = df[text_col].tolist()

    elif isinstance(transcript, str):
        # transcript_text from the processed dataset is newline-separated.
        raw_texts = transcript.split("\n")

    else:
        raise TypeError(
            "transcript must be either a pandas DataFrame or a string."
        )

    for text in raw_texts:
        cleaned = clean_transcript(text)
        if cleaned != "":
            utterances.append(cleaned)

    return utterances


def normalize_participant_id(value):
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    text = re.sub(r"\.0$", "", text)

    return text


def load_participant_ids_from_dataset(dataset_path):
    df = pd.read_csv(dataset_path)

    if "participant_id" not in df.columns:
        raise ValueError("Dataset must contain participant_id column.")

    participant_ids = (
        df["participant_id"]
        .dropna()
        .map(normalize_participant_id)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    return participant_ids, df


def build_utterance_bank_from_raw_transcripts(participant_ids, transcripts_dir):
    """
    Build utterance bank from raw transcript CSV files.
    """
    utterance_bank = {}
    missing_files = []

    for participant_id in participant_ids:
        transcript_path = transcripts_dir / f"{participant_id}_Transcript.csv"

        if not transcript_path.exists():
            missing_files.append(participant_id)
            continue

        transcript_df = pd.read_csv(transcript_path)
        utterances = extract_participant_utterances(transcript_df)

        if utterances:
            utterance_bank[participant_id] = utterances

    return utterance_bank, missing_files


def build_utterance_bank_from_processed_dataset(df):
    """
    Fallback option: build utterance bank from transcript_text in the processed dataset.
    """
    if "transcript_text" not in df.columns:
        raise ValueError("Dataset must contain transcript_text column.")

    utterance_bank = {}

    unique_df = (
        df[["participant_id", "transcript_text"]]
        .drop_duplicates(subset=["participant_id"])
        .copy()
    )

    for _, row in unique_df.iterrows():
        participant_id = normalize_participant_id(row["participant_id"])
        transcript_text = row["transcript_text"]

        utterances = extract_participant_utterances(transcript_text)

        if utterances:
            utterance_bank[participant_id] = utterances

    return utterance_bank


def validate_utterance_bank(utterance_bank):
    if not isinstance(utterance_bank, dict):
        raise ValueError("utterance_bank must be a dictionary.")

    if len(utterance_bank) == 0:
        raise ValueError("utterance_bank is empty.")

    empty_participants = []

    for participant_id, utterances in utterance_bank.items():
        if not isinstance(participant_id, str) or participant_id.strip() == "":
            raise ValueError("All participant IDs must be non-empty strings.")

        if not isinstance(utterances, list):
            raise ValueError(f"Utterances for participant {participant_id} must be a list.")

        cleaned_utterances = []

        for utt in utterances:
            cleaned = clean_transcript(utt)
            if cleaned != "":
                cleaned_utterances.append(cleaned)

        utterance_bank[participant_id] = cleaned_utterances

        if len(cleaned_utterances) == 0:
            empty_participants.append(participant_id)

    if empty_participants:
        raise ValueError(
            "Some participants have empty utterance lists. "
            f"Examples: {empty_participants[:20]}"
        )

    return True


def save_utterance_bank(utterance_bank, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(utterance_bank, f, ensure_ascii=False, indent=2)

    print("Utterance bank saved successfully.")
    print(f"Output path: {output_path}")
    print(f"Participants: {len(utterance_bank)}")

    total_utterances = sum(len(v) for v in utterance_bank.values())
    print(f"Total utterances: {total_utterances}")

    lengths = [len(v) for v in utterance_bank.values()]
    print(f"Min utterances per participant: {min(lengths)}")
    print(f"Max utterances per participant: {max(lengths)}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/processed/phq8_item_dataset_with_splits.csv",
        help="Path to processed item-level dataset with splits.",
    )

    parser.add_argument(
        "--transcripts_dir",
        type=str,
        default="data/raw/edaic/transcripts",
        help="Path to raw transcript CSV files.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/utterance_bank.json",
        help="Output path for utterance_bank.json.",
    )

    parser.add_argument(
        "--fallback_to_processed",
        action="store_true",
        help="Use transcript_text from processed dataset if raw transcript files are unavailable.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    transcripts_dir = Path(args.transcripts_dir)
    output_path = Path(args.output_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    participant_ids, df = load_participant_ids_from_dataset(dataset_path)

    if not transcripts_dir.exists():
        if args.fallback_to_processed:
            print("Raw transcripts directory not found. Falling back to processed transcript_text.")
            utterance_bank = build_utterance_bank_from_processed_dataset(df)
        else:
            raise FileNotFoundError(
                f"Raw transcripts directory does not exist: {transcripts_dir}. "
                "Use --fallback_to_processed to build from transcript_text instead."
            )
    else:
        utterance_bank, missing_files = build_utterance_bank_from_raw_transcripts(
            participant_ids=participant_ids,
            transcripts_dir=transcripts_dir,
        )

        print(f"Missing raw transcript files: {len(missing_files)}")

        if len(utterance_bank) == 0 and args.fallback_to_processed:
            print("No utterances extracted from raw transcripts. Falling back to processed transcript_text.")
            utterance_bank = build_utterance_bank_from_processed_dataset(df)

    validate_utterance_bank(utterance_bank)
    save_utterance_bank(utterance_bank, output_path)


if __name__ == "__main__":
    main()