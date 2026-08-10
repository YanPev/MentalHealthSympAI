"""
Build an item-level dataset with BM25 participant-side context evidence.

All input columns, labels, splits, and utterance baselines are preserved.
Context retrieval is restricted to windows belonging to the row participant.
"""

from pathlib import Path
import argparse
import json
import re
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.retrieval.prod_context.bm25_context_retriever import (
    load_context_window_bank,
    retrieve_top_k_context_windows_bm25,
    validate_retrieval_parameters,
)


WINDOW_SEPARATOR = " [WINDOW_SEP] "
REQUIRED_INPUT_COLUMNS = {
    "participant_id",
    "item_id",
    "item_name",
    "item_text",
    "label",
    "split",
    "retrieved_utterances",
    "baseline_utterances",
}
ADDED_COLUMNS = [
    "baseline_context_windows_pack",
    "retrieved_context_windows_bm25_pack",
    "retrieved_context_window_ids_bm25",
    "retrieved_context_bm25_scores",
]
HYBRID_COLUMNS = [
    "retrieved_context_windows_hybrid_pack",
    "retrieved_context_windows_hybrid_list",
    "retrieved_context_window_ids_hybrid",
    "retrieved_context_hybrid_scores",
]


def normalize_participant_id(value):
    if value is None or pd.isna(value):
        return None

    return re.sub(r"\.0$", "", str(value).strip())


def pack_windows(window_texts):
    return WINDOW_SEPARATOR.join(
        str(text).strip()
        for text in window_texts
        if text is not None and str(text).strip()
    )


def count_words(text):
    return len(str(text).split())


def select_windows_with_word_budget(windows, max_pack_words):
    if max_pack_words <= 0:
        raise ValueError("max_pack_words must be positive.")

    included = []
    used_words = 0

    for window in windows:
        window_words = count_words(window["window_text"])
        if not included:
            included.append(window)
            used_words = window_words
            continue

        separator_words = count_words(WINDOW_SEPARATOR)
        if (
            used_words
            + separator_words
            + window_words
            > max_pack_words
        ):
            break

        included.append(window)
        used_words += separator_words + window_words

    return included


def serialize_json_array(values):
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def load_hybrid_retrieval(path):
    with open(path, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list) or not records:
        raise ValueError("Hybrid retrieval must be a non-empty JSON list.")

    indexed = {}
    for record in records:
        key = (
            normalize_participant_id(record["participant_id"]),
            int(record["item_id"]),
        )
        if key in indexed:
            raise ValueError(f"Duplicate hybrid retrieval key: {key}")
        indexed[key] = record

    return indexed


def validate_input_dataset(df):
    missing = REQUIRED_INPUT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Input dataset is missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Input dataset is empty.")

    if df[["participant_id", "item_id"]].isna().any().any():
        raise ValueError("participant_id or item_id contains missing values.")

    if df.duplicated(subset=["participant_id", "item_id"]).any():
        raise ValueError("Duplicate participant_id + item_id rows detected.")

    for column in ("retrieved_utterances", "baseline_utterances"):
        if df[column].isna().any():
            raise ValueError(f"{column} contains missing values.")


def build_context_window_dataset(
    input_df,
    context_window_bank,
    k,
    max_turn_overlap_ratio,
    candidate_multiplier,
    max_pack_words,
):
    if k <= 0:
        raise ValueError("k must be positive.")

    validate_retrieval_parameters(
        max_turn_overlap_ratio=max_turn_overlap_ratio,
        candidate_multiplier=candidate_multiplier,
    )
    if max_pack_words <= 0:
        raise ValueError("max_pack_words must be positive.")

    validate_input_dataset(input_df)
    output_df = input_df.copy()

    baseline_packs = []
    retrieved_packs = []
    retrieved_ids_json = []
    retrieved_scores_json = []
    zero_score_fallbacks = 0

    missing_participants = set()

    for _, row in input_df.iterrows():
        participant_id = normalize_participant_id(row["participant_id"])
        context_windows = context_window_bank.get(participant_id)

        if context_windows is None:
            missing_participants.add(participant_id)
            continue

        for window in context_windows:
            if normalize_participant_id(window["participant_id"]) != participant_id:
                raise ValueError(
                    "Participant-local retrieval violation in context bank: "
                    f"row={participant_id}, "
                    f"window={window['participant_id']}"
                )

        baseline_windows = context_windows[: min(k, len(context_windows))]
        retrieved_windows = retrieve_top_k_context_windows_bm25(
            item_text=row["item_text"],
            context_windows=context_windows,
            k=k,
            max_turn_overlap_ratio=max_turn_overlap_ratio,
            candidate_multiplier=candidate_multiplier,
        )
        zero_score_fallbacks += int(
            bool(
                retrieved_windows
                and retrieved_windows[0]["zero_score_fallback"]
            )
        )

        for window in retrieved_windows:
            if normalize_participant_id(window["participant_id"]) != participant_id:
                raise ValueError(
                    "Participant-local retrieval violation: "
                    f"row={participant_id}, "
                    f"window={window['participant_id']}"
                )

        included_baseline_windows = select_windows_with_word_budget(
            baseline_windows,
            max_pack_words=max_pack_words,
        )
        included_retrieved_windows = select_windows_with_word_budget(
            retrieved_windows,
            max_pack_words=max_pack_words,
        )

        baseline_packs.append(
            pack_windows(
                window["window_text"]
                for window in included_baseline_windows
            )
        )
        retrieved_packs.append(
            pack_windows(
                window["window_text"]
                for window in included_retrieved_windows
            )
        )
        retrieved_ids_json.append(
            serialize_json_array(
                [
                    window["context_window_id"]
                    for window in included_retrieved_windows
                ]
            )
        )
        retrieved_scores_json.append(
            serialize_json_array(
                [
                    float(window["bm25_score"])
                    for window in included_retrieved_windows
                ]
            )
        )

    if missing_participants:
        raise ValueError(
            "Participants missing from context_window_bank: "
            f"{sorted(missing_participants)[:20]}"
        )

    expected_length = len(input_df)
    generated_lengths = {
        len(baseline_packs),
        len(retrieved_packs),
        len(retrieved_ids_json),
        len(retrieved_scores_json),
    }
    if generated_lengths != {expected_length}:
        raise ValueError(
            "Generated context evidence length mismatch: "
            f"expected={expected_length}, actual={sorted(generated_lengths)}"
        )

    output_df[ADDED_COLUMNS[0]] = baseline_packs
    output_df[ADDED_COLUMNS[1]] = retrieved_packs
    output_df[ADDED_COLUMNS[2]] = retrieved_ids_json
    output_df[ADDED_COLUMNS[3]] = retrieved_scores_json

    return output_df, zero_score_fallbacks


def add_hybrid_columns(
    output_df,
    context_window_bank,
    hybrid_records,
    max_pack_words,
):
    hybrid_packs = []
    hybrid_lists = []
    hybrid_ids = []
    hybrid_scores = []

    for _, row in output_df.iterrows():
        participant_id = normalize_participant_id(row["participant_id"])
        item_id = int(row["item_id"])
        record = hybrid_records.get((participant_id, item_id))
        if record is None:
            raise ValueError(
                "Missing hybrid retrieval record: "
                f"participant_id={participant_id}, item_id={item_id}"
            )

        ids = record["retrieved_context_window_ids_hybrid"]
        scores = record["retrieved_context_hybrid_scores"]
        if not isinstance(ids, list) or not isinstance(scores, list):
            raise ValueError(
                f"Invalid hybrid arrays for {(participant_id, item_id)}"
            )
        if len(ids) != len(scores) or not ids:
            raise ValueError(
                f"Hybrid ID/score mismatch for {(participant_id, item_id)}"
            )

        participant_windows = {
            window["context_window_id"]: window
            for window in context_window_bank[participant_id]
        }
        if any(window_id not in participant_windows for window_id in ids):
            raise ValueError(
                "Participant-local hybrid retrieval violation for "
                f"{(participant_id, item_id)}"
            )

        ranked_windows = [
            {
                **participant_windows[window_id],
                "hybrid_score": float(score),
            }
            for window_id, score in zip(ids, scores)
        ]
        included = select_windows_with_word_budget(
            ranked_windows,
            max_pack_words=max_pack_words,
        )

        hybrid_packs.append(
            pack_windows(window["window_text"] for window in included)
        )
        hybrid_lists.append(
            serialize_json_array(
                [window["window_text"] for window in included]
            )
        )
        hybrid_ids.append(
            serialize_json_array(
                [window["context_window_id"] for window in included]
            )
        )
        hybrid_scores.append(
            serialize_json_array(
                [window["hybrid_score"] for window in included]
            )
        )

    output_df = output_df.copy()
    output_df[HYBRID_COLUMNS[0]] = hybrid_packs
    output_df[HYBRID_COLUMNS[1]] = hybrid_lists
    output_df[HYBRID_COLUMNS[2]] = hybrid_ids
    output_df[HYBRID_COLUMNS[3]] = hybrid_scores
    return output_df


def validate_output_dataset(
    input_df,
    output_df,
    context_window_bank,
    k,
    max_pack_words,
):
    if len(output_df) != len(input_df):
        raise ValueError(
            f"Row count changed: input={len(input_df)}, output={len(output_df)}"
        )

    original_columns = list(input_df.columns)
    if list(output_df.columns[: len(original_columns)]) != original_columns:
        raise ValueError("Original input columns or column order changed.")

    if list(output_df.columns[len(original_columns) :]) != ADDED_COLUMNS:
        raise ValueError("Unexpected added columns or added-column order.")

    for column in original_columns:
        if not output_df[column].equals(input_df[column]):
            raise ValueError(f"Input column changed: {column}")

    for row_index, row in output_df.iterrows():
        participant_id = normalize_participant_id(row["participant_id"])
        window_ids = json.loads(row["retrieved_context_window_ids_bm25"])
        scores = json.loads(row["retrieved_context_bm25_scores"])

        if not isinstance(window_ids, list) or not isinstance(scores, list):
            raise ValueError(f"Invalid JSON arrays at row_index={row_index}")

        if not 1 <= len(window_ids) <= min(
            k,
            len(context_window_bank[participant_id]),
        ):
            raise ValueError(
                f"Retrieved array length mismatch at row_index={row_index}"
            )

        if len(window_ids) != len(scores):
            raise ValueError(
                f"ID/score array length mismatch at row_index={row_index}"
            )

        if len(window_ids) != len(set(window_ids)):
            raise ValueError(
                f"Duplicate retrieved window IDs at row_index={row_index}"
            )

        valid_ids = {
            window["context_window_id"]
            for window in context_window_bank[participant_id]
        }
        if not set(window_ids).issubset(valid_ids):
            raise ValueError(
                "Participant-local retrieval violation at "
                f"row_index={row_index}"
            )

        if not row["baseline_context_windows_pack"]:
            raise ValueError(
                f"Empty baseline context pack at row_index={row_index}"
            )

        if not row["retrieved_context_windows_bm25_pack"]:
            raise ValueError(
                f"Empty retrieved context pack at row_index={row_index}"
            )

        pack_word_count = count_words(
            row["retrieved_context_windows_bm25_pack"]
        )
        if pack_word_count > max_pack_words and len(window_ids) > 1:
            raise ValueError(
                f"Pack exceeds word budget at row_index={row_index}"
            )

    return True


def validate_hybrid_columns(output_df, context_window_bank, max_pack_words):
    for row_index, row in output_df.iterrows():
        participant_id = normalize_participant_id(row["participant_id"])
        texts = json.loads(row[HYBRID_COLUMNS[1]])
        window_ids = json.loads(row[HYBRID_COLUMNS[2]])
        scores = json.loads(row[HYBRID_COLUMNS[3]])

        if not all(
            isinstance(values, list)
            for values in (texts, window_ids, scores)
        ):
            raise ValueError(
                f"Invalid hybrid JSON arrays at row_index={row_index}"
            )
        if not texts or not (
            len(texts) == len(window_ids) == len(scores)
        ):
            raise ValueError(
                f"Hybrid array length mismatch at row_index={row_index}"
            )
        if len(window_ids) != len(set(window_ids)):
            raise ValueError(
                f"Duplicate hybrid IDs at row_index={row_index}"
            )

        valid_ids = {
            window["context_window_id"]
            for window in context_window_bank[participant_id]
        }
        if not set(window_ids).issubset(valid_ids):
            raise ValueError(
                "Participant-local hybrid violation at "
                f"row_index={row_index}"
            )

        expected_pack = pack_windows(texts)
        if row[HYBRID_COLUMNS[0]] != expected_pack:
            raise ValueError(
                f"Hybrid pack/list mismatch at row_index={row_index}"
            )

        pack_words = count_words(row[HYBRID_COLUMNS[0]])
        if pack_words > max_pack_words and len(window_ids) > 1:
            raise ValueError(
                f"Hybrid pack exceeds budget at row_index={row_index}"
            )

    return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path(
            "data/processed/phq8_item_dataset_full_bm25.csv"
        ),
    )
    parser.add_argument(
        "--context_window_bank_path",
        type=Path,
        default=Path("data/processed/context_window_bank.json"),
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path(
            "data/processed/"
            "phq8_item_dataset_context_windows_bm25.csv"
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max_turn_overlap_ratio",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--candidate_multiplier",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max_pack_words",
        type=int,
        default=350,
    )
    parser.add_argument(
        "--hybrid_retrieval_path",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for path in (args.input_path, args.context_window_bank_path):
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")

    input_df = pd.read_csv(args.input_path)
    context_window_bank = load_context_window_bank(
        args.context_window_bank_path
    )
    output_df, zero_score_fallbacks = build_context_window_dataset(
        input_df=input_df,
        context_window_bank=context_window_bank,
        k=args.k,
        max_turn_overlap_ratio=args.max_turn_overlap_ratio,
        candidate_multiplier=args.candidate_multiplier,
        max_pack_words=args.max_pack_words,
    )
    validate_output_dataset(
        input_df=input_df,
        output_df=output_df,
        context_window_bank=context_window_bank,
        k=args.k,
        max_pack_words=args.max_pack_words,
    )

    if args.hybrid_retrieval_path is not None:
        if not args.hybrid_retrieval_path.exists():
            raise FileNotFoundError(
                "Hybrid retrieval does not exist: "
                f"{args.hybrid_retrieval_path}"
            )
        output_df = add_hybrid_columns(
            output_df=output_df,
            context_window_bank=context_window_bank,
            hybrid_records=load_hybrid_retrieval(
                args.hybrid_retrieval_path
            ),
            max_pack_words=args.max_pack_words,
        )
        validate_hybrid_columns(
            output_df=output_df,
            context_window_bank=context_window_bank,
            max_pack_words=args.max_pack_words,
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_path, index=False)

    print("BM25 context window dataset created successfully.")
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print(f"Rows: {len(output_df)}")
    print(f"Participants: {output_df['participant_id'].nunique()}")
    print(f"k: {args.k}")
    print(f"Max turn overlap ratio: {args.max_turn_overlap_ratio}")
    print(f"Candidate multiplier: {args.candidate_multiplier}")
    print(f"Max pack words: {args.max_pack_words}")
    print(f"Zero-score fallbacks: {zero_score_fallbacks}")
    print("Labels preserved: True")
    print("Splits preserved: True")
    print("Utterance retrieval columns preserved: True")
    print("Participant-local retrieval: True")
    print(
        "Hybrid columns added: "
        f"{args.hybrid_retrieval_path is not None}"
    )


if __name__ == "__main__":
    main()
