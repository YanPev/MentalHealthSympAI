from pathlib import Path
import pandas as pd


PROJECT_DIR = Path("/content/drive/MyDrive/master/courses/AI_in_medicine/AI_MD_Project")

RAW_DIR = PROJECT_DIR / "data" / "raw" / "edaic"
LABELS_DIR = RAW_DIR / "labels"
TRANSCRIPTS_DIR = RAW_DIR / "transcripts"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
DOCS_DIR = PROJECT_DIR / "docs"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def load_phq8_labels():
    candidates = list(LABELS_DIR.rglob("Detailed_PHQ8_Labels.csv"))

    if not candidates:
        raise FileNotFoundError("Detailed_PHQ8_Labels.csv was not found.")

    path = candidates[0]
    df = pd.read_csv(path)

    return path, df


def inspect_transcripts():
    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*_Transcript.csv"))

    if not transcript_files:
        raise RuntimeError(f"No transcript files found in {TRANSCRIPTS_DIR}")

    sample_path = transcript_files[0]
    sample_df = pd.read_csv(sample_path)

    transcript_ids = sorted([
        p.name.replace("_Transcript.csv", "")
        for p in transcript_files
    ])

    return transcript_files, transcript_ids, sample_path, sample_df


def detect_columns(phq8_df):
    possible_id_cols = [
        col for col in phq8_df.columns
        if "id" in col.lower() or "participant" in col.lower()
    ]

    possible_phq_cols = [
        col for col in phq8_df.columns
        if "phq" in col.lower()
    ]

    print("\\nPossible participant ID columns:")
    for col in possible_id_cols:
        print("-", col)

    print("\\nPossible PHQ-related columns:")
    for col in possible_phq_cols:
        print("-", col)

    participant_col = "Participant_ID"

    phq8_item_cols = [
        "PHQ_8NoInterest",
        "PHQ_8Depressed",
        "PHQ_8Sleep",
        "PHQ_8Tired",
        "PHQ_8Appetite",
        "PHQ_8Failure",
        "PHQ_8Concentrating",
        "PHQ_8Moving",
    ]

    phq8_total_col = "PHQ_8Total"

    if participant_col not in phq8_df.columns:
        raise ValueError(f"Participant column not found: {participant_col}")

    missing_cols = [col for col in phq8_item_cols if col not in phq8_df.columns]
    if missing_cols:
        raise ValueError(f"Missing PHQ-8 item columns: {missing_cols}")

    if phq8_total_col in phq8_df.columns:
        print(f"\\nPHQ-8 total column found: {phq8_total_col}")
    else:
        print(f"\\nWarning: PHQ-8 total column not found: {phq8_total_col}")

    print("\\nConfirmed PHQ-8 item-level columns:")
    for col in phq8_item_cols:
        print("-", col)

    return participant_col, phq8_item_cols


def compute_overlap(transcript_ids, phq8_df, participant_col):
    label_ids = sorted(
        phq8_df[participant_col]
        .dropna()
        .astype(int)
        .astype(str)
        .unique()
    )

    transcript_id_set = set(transcript_ids)
    label_id_set = set(label_ids)

    both_ids = sorted(transcript_id_set & label_id_set)
    transcript_only = sorted(transcript_id_set - label_id_set)
    labels_only = sorted(label_id_set - transcript_id_set)

    summary_df = pd.DataFrame([
        {"check": "participants_with_transcript", "count": len(transcript_id_set)},
        {"check": "participants_with_phq8_labels", "count": len(label_id_set)},
        {"check": "participants_with_both", "count": len(both_ids)},
        {"check": "transcript_only", "count": len(transcript_only)},
        {"check": "labels_only", "count": len(labels_only)},
    ])

    summary_path = OUTPUTS_DIR / "data_availability_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    valid_ids_path = OUTPUTS_DIR / "valid_participant_ids.csv"
    pd.DataFrame({"Participant_ID": both_ids}).to_csv(valid_ids_path, index=False)

    return summary_df, summary_path, valid_ids_path, both_ids, transcript_only, labels_only


def inspect_phq8_items(phq8_df, phq8_item_cols):
    print("\\nPHQ-8 item value ranges:")
    for col in phq8_item_cols:
        values = sorted([int(v) for v in phq8_df[col].dropna().unique()])
        print(col, "->", values)

    missing_per_item = phq8_df[phq8_item_cols].isna().sum()

    print("\\nMissing values per PHQ-8 item:")
    print(missing_per_item)

    return missing_per_item


def write_report(
    phq8_path,
    phq8_df,
    participant_col,
    phq8_item_cols,
    transcript_files,
    sample_path,
    sample_df,
    summary_df,
    valid_ids_path,
):
    report_path = DOCS_DIR / "data_availability_report.md"

    report = f"""# Data Availability Report — Week 1

## Dataset location

Project directory:

`{PROJECT_DIR}`

Transcript directory:

`{TRANSCRIPTS_DIR}`

Labels directory:

`{LABELS_DIR}`

## Transcript availability

Number of transcript files:

{len(transcript_files)}

Example transcript file:

`{sample_path.name}`

Transcript columns from sample:

{list(sample_df.columns)}

## PHQ-8 label availability

Detailed PHQ-8 labels file:

`{phq8_path}`

Number of rows in labels file:

{len(phq8_df)}

Participant ID column:

`{participant_col}`

PHQ-8 item columns:

{phq8_item_cols}

## Participant overlap

{summary_df.to_markdown(index=False)}

Valid participant IDs file:

`{valid_ids_path}`

## Main conclusion

The dataset contains transcript files and PHQ-8 item-level labels. The valid participant pool for the next stage is the overlap between participants with transcripts and participants with PHQ-8 labels.

## Next steps

- Build PHQ-8 item-level dataset.
- Create participant-level train/dev/test split.
- Build utterance bank from the transcript `Text` column.
"""

    report_path.write_text(report, encoding="utf-8")
    return report_path


def main():
    phq8_path, phq8_df = load_phq8_labels()
    transcript_files, transcript_ids, sample_path, sample_df = inspect_transcripts()
    participant_col, phq8_item_cols = detect_columns(phq8_df)

    summary_df, summary_path, valid_ids_path, both_ids, transcript_only, labels_only = compute_overlap(
        transcript_ids,
        phq8_df,
        participant_col,
    )

    inspect_phq8_items(phq8_df, phq8_item_cols)

    report_path = write_report(
        phq8_path=phq8_path,
        phq8_df=phq8_df,
        participant_col=participant_col,
        phq8_item_cols=phq8_item_cols,
        transcript_files=transcript_files,
        sample_path=sample_path,
        sample_df=sample_df,
        summary_df=summary_df,
        valid_ids_path=valid_ids_path,
    )

    print("\\nWeek 1 data inspection complete.")
    print("\\nSummary:")
    print(summary_df)

    print("\\nSaved:")
    print(summary_path)
    print(valid_ids_path)
    print(report_path)


if __name__ == "__main__":
    main()
