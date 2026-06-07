"""
BM25 retrieval over participant-side context windows.

The reusable API ranks windows belonging to one participant. The CLI applies
that API to each participant-item row in an item-level dataset and writes JSON
records without modifying the existing utterance retrieval baseline.
"""

from pathlib import Path
import argparse
from collections import Counter
import json
import math
import re

import pandas as pd


REQUIRED_WINDOW_FIELDS = {
    "participant_id",
    "context_window_id",
    "turn_ids",
    "window_text",
}

ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "been",
    "being",
    "did",
    "do",
    "does",
    "for",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def normalize_participant_id(value):
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    return re.sub(r"\.0$", "", text)


def tokenize(text):
    if text is None:
        return []

    cleaned = str(text).lower()
    cleaned = re.sub(r"[^a-z0-9\s']", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []

    return [
        token
        for token in cleaned.split()
        if token not in ENGLISH_STOPWORDS
    ]


def validate_context_windows(context_windows, participant_id=None):
    if not isinstance(context_windows, list):
        raise ValueError("context_windows must be a list.")

    seen_ids = set()

    for window in context_windows:
        if not isinstance(window, dict):
            raise ValueError("Each context window must be an object.")

        missing = REQUIRED_WINDOW_FIELDS - set(window)
        if missing:
            raise ValueError(
                f"Context window is missing fields: {sorted(missing)}"
            )

        window_participant_id = normalize_participant_id(
            window["participant_id"]
        )
        if participant_id is not None and window_participant_id != participant_id:
            raise ValueError(
                "Context window belongs to a different participant: "
                f"expected={participant_id}, actual={window_participant_id}"
            )

        window_id = str(window["context_window_id"])
        if window_id in seen_ids:
            raise ValueError(f"Duplicate context_window_id: {window_id}")
        seen_ids.add(window_id)

        if str(window["window_text"]).strip() == "":
            raise ValueError(f"Empty window_text for {window_id}")


def retrieve_top_k_context_windows_bm25(
    item_text,
    context_windows,
    k=5,
    max_turn_overlap_ratio=0.5,
    candidate_multiplier=5,
):
    """
    Rank one participant's context windows against one PHQ-8 item.

    Returns records containing the original window fields plus `bm25_score`.
    Ties are resolved by original chronological/list order.
    """
    if k <= 0:
        return []

    validate_retrieval_parameters(
        max_turn_overlap_ratio=max_turn_overlap_ratio,
        candidate_multiplier=candidate_multiplier,
    )
    validate_context_windows(context_windows)
    if not context_windows:
        return []

    query_tokens = tokenize(item_text)
    tokenized_windows = [
        tokenize(window["window_text"]) for window in context_windows
    ]

    valid_indices = [
        index
        for index, tokens in enumerate(tokenized_windows)
        if tokens
    ]
    if not valid_indices:
        return []

    valid_documents = [
        tokenized_windows[index] for index in valid_indices
    ]
    scores = score_bm25_okapi(query_tokens, valid_documents)

    ranked_positions = sorted(
        range(len(valid_indices)),
        key=lambda position: (-float(scores[position]), valid_indices[position]),
    )

    candidate_count = min(
        len(ranked_positions),
        max(k, k * candidate_multiplier),
    )
    candidates = []
    for position in ranked_positions[:candidate_count]:
        source_index = valid_indices[position]
        result = dict(context_windows[source_index])
        result["bm25_score"] = float(scores[position])
        result["zero_score_fallback"] = False
        candidates.append(result)

    if candidates and all(
        candidate["bm25_score"] == 0.0 for candidate in candidates
    ):
        fallback_results = []
        for window in context_windows[: min(k, len(context_windows))]:
            result = dict(window)
            result["bm25_score"] = 0.0
            result["zero_score_fallback"] = True
            fallback_results.append(result)
        return fallback_results

    selected = select_diverse_windows(
        candidates=candidates,
        k=k,
        max_turn_overlap_ratio=max_turn_overlap_ratio,
    )
    return selected


def validate_retrieval_parameters(
    max_turn_overlap_ratio,
    candidate_multiplier,
):
    if not 0 <= max_turn_overlap_ratio <= 1:
        raise ValueError(
            "max_turn_overlap_ratio must be between 0 and 1."
        )

    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be positive.")


def turn_overlap_ratio(left_window, right_window):
    left_turns = set(left_window["turn_ids"])
    right_turns = set(right_window["turn_ids"])

    smaller_size = min(len(left_turns), len(right_turns))
    if smaller_size == 0:
        return 0.0

    return len(left_turns & right_turns) / smaller_size


def select_diverse_windows(
    candidates,
    k,
    max_turn_overlap_ratio,
):
    selected = []
    skipped = []

    for candidate in candidates:
        overlaps_too_much = any(
            turn_overlap_ratio(candidate, existing)
            > max_turn_overlap_ratio
            for existing in selected
        )
        if overlaps_too_much:
            skipped.append(candidate)
        else:
            selected.append(candidate)

        if len(selected) == k:
            return selected

    selected_ids = {
        window["context_window_id"] for window in selected
    }
    for candidate in candidates:
        if candidate["context_window_id"] in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate["context_window_id"])
        if len(selected) == k:
            break

    return selected


def score_bm25_okapi(query_tokens, documents, k1=1.5, b=0.75):
    if not documents or not query_tokens:
        return [0.0] * len(documents)

    document_frequencies = Counter()
    term_frequencies = []
    document_lengths = []

    for document in documents:
        frequencies = Counter(document)
        term_frequencies.append(frequencies)
        document_lengths.append(len(document))
        document_frequencies.update(frequencies.keys())

    n_documents = len(documents)
    average_length = sum(document_lengths) / n_documents
    scores = []

    for frequencies, document_length in zip(
        term_frequencies,
        document_lengths,
    ):
        score = 0.0
        length_normalization = k1 * (
            1 - b + b * document_length / average_length
        )

        for token in query_tokens:
            frequency = frequencies.get(token, 0)
            if frequency == 0:
                continue

            document_frequency = document_frequencies[token]
            inverse_document_frequency = math.log(
                1
                + (
                    n_documents
                    - document_frequency
                    + 0.5
                )
                / (document_frequency + 0.5)
            )
            score += inverse_document_frequency * (
                frequency * (k1 + 1)
                / (frequency + length_normalization)
            )

        scores.append(score)

    return scores


def load_context_window_bank(path):
    with open(path, "r", encoding="utf-8") as file:
        context_window_bank = json.load(file)

    if not isinstance(context_window_bank, dict) or not context_window_bank:
        raise ValueError(
            "context_window_bank must be a non-empty JSON object."
        )

    normalized_bank = {}
    for participant_id, windows in context_window_bank.items():
        normalized_id = normalize_participant_id(participant_id)
        validate_context_windows(windows, participant_id=normalized_id)
        normalized_bank[normalized_id] = windows

    return normalized_bank


def build_retrieval_records(
    dataset_path,
    context_window_bank,
    k,
    max_turn_overlap_ratio,
    candidate_multiplier,
):
    required_columns = {
        "participant_id",
        "item_id",
        "item_text",
    }
    df = pd.read_csv(dataset_path)
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    records = []
    missing_participants = set()

    for row_index, row in df.iterrows():
        participant_id = normalize_participant_id(row["participant_id"])
        windows = context_window_bank.get(participant_id)
        if windows is None:
            missing_participants.add(participant_id)
            continue

        ranked = retrieve_top_k_context_windows_bm25(
            item_text=row["item_text"],
            context_windows=windows,
            k=k,
            max_turn_overlap_ratio=max_turn_overlap_ratio,
            candidate_multiplier=candidate_multiplier,
        )
        records.append(
            {
                "row_index": int(row_index),
                "participant_id": participant_id,
                "item_id": int(row["item_id"]),
                "retrieved_context_window_ids_bm25": [
                    window["context_window_id"] for window in ranked
                ],
                "retrieved_context_bm25_scores": [
                    window["bm25_score"] for window in ranked
                ],
                "retrieved_context_windows_bm25": [
                    window["window_text"] for window in ranked
                ],
                "zero_score_fallback": bool(
                    ranked and ranked[0]["zero_score_fallback"]
                ),
            }
        )

    if missing_participants:
        raise ValueError(
            "Participants missing from context_window_bank: "
            f"{sorted(missing_participants)[:20]}"
        )

    if len(records) != len(df):
        raise ValueError(
            f"Retrieval coverage mismatch: dataset={len(df)}, "
            f"records={len(records)}"
        )

    return records


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--context_window_bank_path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
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
    return parser.parse_args()


def main():
    args = parse_args()

    if args.k <= 0:
        raise ValueError("k must be positive.")

    validate_retrieval_parameters(
        max_turn_overlap_ratio=args.max_turn_overlap_ratio,
        candidate_multiplier=args.candidate_multiplier,
    )
    for path in (args.dataset_path, args.context_window_bank_path):
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")

    context_window_bank = load_context_window_bank(
        args.context_window_bank_path
    )
    records = build_retrieval_records(
        dataset_path=args.dataset_path,
        context_window_bank=context_window_bank,
        k=args.k,
        max_turn_overlap_ratio=args.max_turn_overlap_ratio,
        candidate_multiplier=args.candidate_multiplier,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    retrieved_counts = [
        len(record["retrieved_context_window_ids_bm25"])
        for record in records
    ]
    print("BM25 context retrieval completed successfully.")
    print(f"Output path: {args.output_path}")
    print(f"Rows: {len(records)}")
    print(f"k: {args.k}")
    print(f"Max turn overlap ratio: {args.max_turn_overlap_ratio}")
    print(f"Candidate multiplier: {args.candidate_multiplier}")
    print(f"Min retrieved windows: {min(retrieved_counts)}")
    print(f"Max retrieved windows: {max(retrieved_counts)}")
    print(
        "Zero-score fallbacks: "
        f"{sum(record['zero_score_fallback'] for record in records)}"
    )


if __name__ == "__main__":
    main()
