"""
Semantic retrieval over participant-side context windows.

Embeddings are computed only from PHQ-8 item text and context-window text.
Retrieval is participant-local and uses cosine similarity with deterministic
tie-breaking and turn-overlap diversity control.
"""

from pathlib import Path
import argparse
import json
import re
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
REQUIRED_WINDOW_FIELDS = {
    "participant_id",
    "context_window_id",
    "turn_ids",
    "window_text",
}


def normalize_participant_id(value):
    if value is None or pd.isna(value):
        return None

    return re.sub(r"\.0$", "", str(value).strip())


def validate_parameters(k, max_turn_overlap_ratio, candidate_multiplier):
    if k <= 0:
        raise ValueError("k must be positive.")

    if not 0 <= max_turn_overlap_ratio <= 1:
        raise ValueError(
            "max_turn_overlap_ratio must be between 0 and 1."
        )

    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be positive.")


def validate_context_windows(context_windows, participant_id=None):
    if not isinstance(context_windows, list) or not context_windows:
        raise ValueError("context_windows must be a non-empty list.")

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

        if not isinstance(window["turn_ids"], list) or not window["turn_ids"]:
            raise ValueError(f"Invalid turn_ids for {window_id}")

        if not str(window["window_text"]).strip():
            raise ValueError(f"Empty window_text for {window_id}")


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

    for candidate in candidates:
        if all(
            turn_overlap_ratio(candidate, existing)
            <= max_turn_overlap_ratio
            for existing in selected
        ):
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


def retrieve_top_k_context_windows_semantic(
    item_text,
    context_windows,
    model,
    window_embeddings=None,
    k=5,
    max_turn_overlap_ratio=0.5,
    candidate_multiplier=5,
):
    validate_parameters(
        k=k,
        max_turn_overlap_ratio=max_turn_overlap_ratio,
        candidate_multiplier=candidate_multiplier,
    )
    validate_context_windows(context_windows)

    if window_embeddings is None:
        window_embeddings = model.encode(
            [window["window_text"] for window in context_windows],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    if len(window_embeddings) != len(context_windows):
        raise ValueError(
            "Window embedding count does not match context windows."
        )

    query_embedding = model.encode(
        [str(item_text)],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    scores = np.asarray(window_embeddings) @ query_embedding

    ranked_indices = sorted(
        range(len(context_windows)),
        key=lambda index: (-float(scores[index]), index),
    )
    candidate_count = min(
        len(ranked_indices),
        max(k, k * candidate_multiplier),
    )

    candidates = []
    for index in ranked_indices[:candidate_count]:
        result = dict(context_windows[index])
        result["semantic_score"] = float(scores[index])
        candidates.append(result)

    return select_diverse_windows(
        candidates=candidates,
        k=min(k, len(context_windows)),
        max_turn_overlap_ratio=max_turn_overlap_ratio,
    )


def select_dataset_participants(df, limit_participants):
    participant_ids = []
    seen = set()

    for value in df["participant_id"]:
        participant_id = normalize_participant_id(value)
        if participant_id not in seen:
            seen.add(participant_id)
            participant_ids.append(participant_id)

    if limit_participants is not None:
        if limit_participants <= 0:
            raise ValueError("limit_participants must be positive.")
        participant_ids = participant_ids[:limit_participants]

    return participant_ids


def build_semantic_retrieval_records(
    dataset_path,
    context_window_bank,
    model,
    k,
    max_turn_overlap_ratio,
    candidate_multiplier,
    limit_participants=None,
):
    df = pd.read_csv(dataset_path)
    required_columns = {"participant_id", "item_id", "item_text"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    participant_ids = select_dataset_participants(
        df=df,
        limit_participants=limit_participants,
    )
    participant_set = set(participant_ids)
    limited_df = df[
        df["participant_id"]
        .map(normalize_participant_id)
        .isin(participant_set)
    ]

    records = []
    for participant_id in participant_ids:
        context_windows = context_window_bank.get(participant_id)
        if context_windows is None:
            raise ValueError(
                "Participant missing from context_window_bank: "
                f"{participant_id}"
            )

        window_embeddings = model.encode(
            [window["window_text"] for window in context_windows],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        participant_rows = limited_df[
            limited_df["participant_id"].map(normalize_participant_id)
            == participant_id
        ]

        for row_index, row in participant_rows.iterrows():
            ranked = retrieve_top_k_context_windows_semantic(
                item_text=row["item_text"],
                context_windows=context_windows,
                model=model,
                window_embeddings=window_embeddings,
                k=k,
                max_turn_overlap_ratio=max_turn_overlap_ratio,
                candidate_multiplier=candidate_multiplier,
            )

            if any(
                normalize_participant_id(window["participant_id"])
                != participant_id
                for window in ranked
            ):
                raise ValueError(
                    "Participant-local semantic retrieval violation."
                )

            records.append(
                {
                    "row_index": int(row_index),
                    "participant_id": participant_id,
                    "item_id": int(row["item_id"]),
                    "retrieved_context_window_ids_semantic": [
                        window["context_window_id"] for window in ranked
                    ],
                    "retrieved_context_semantic_scores": [
                        window["semantic_score"] for window in ranked
                    ],
                    "retrieved_context_windows_semantic": [
                        window["window_text"] for window in ranked
                    ],
                }
            )

    expected_rows = len(limited_df)
    if len(records) != expected_rows:
        raise ValueError(
            f"Semantic retrieval coverage mismatch: "
            f"expected={expected_rows}, actual={len(records)}"
        )

    return records, participant_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument(
        "--context_window_bank_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument("--k", type=int, default=5)
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
        "--limit_participants",
        type=int,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    validate_parameters(
        k=args.k,
        max_turn_overlap_ratio=args.max_turn_overlap_ratio,
        candidate_multiplier=args.candidate_multiplier,
    )

    for path in (args.dataset_path, args.context_window_bank_path):
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")

    start_time = time.perf_counter()
    model = SentenceTransformer(args.model_name)
    context_window_bank = load_context_window_bank(
        args.context_window_bank_path
    )
    records, participant_ids = build_semantic_retrieval_records(
        dataset_path=args.dataset_path,
        context_window_bank=context_window_bank,
        model=model,
        k=args.k,
        max_turn_overlap_ratio=args.max_turn_overlap_ratio,
        candidate_multiplier=args.candidate_multiplier,
        limit_participants=args.limit_participants,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    runtime_seconds = time.perf_counter() - start_time
    counts = [
        len(record["retrieved_context_window_ids_semantic"])
        for record in records
    ]
    print("Semantic context retrieval completed successfully.")
    print(f"Model: {args.model_name}")
    print(f"Output path: {args.output_path}")
    print(f"Participants: {len(participant_ids)}")
    print(f"Rows: {len(records)}")
    print(f"k: {args.k}")
    print(f"Min retrieved windows: {min(counts)}")
    print(f"Max retrieved windows: {max(counts)}")
    print(f"Runtime seconds: {runtime_seconds:.3f}")


if __name__ == "__main__":
    main()
