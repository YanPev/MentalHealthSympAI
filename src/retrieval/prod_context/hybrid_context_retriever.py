"""
Hybrid BM25 and semantic retrieval over participant-side context windows.

The input retrieval files contain ranked candidate subsets. Candidates are
merged by participant_id, item_id, and context_window_id. Missing modality
scores are treated as zero before per-row min-max normalization.
"""

from pathlib import Path
import argparse
import json
import math


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, list) or not value:
        raise ValueError(f"Expected a non-empty JSON list: {path}")

    return value


def load_context_window_lookup(path):
    with open(path, "r", encoding="utf-8") as file:
        bank = json.load(file)

    if not isinstance(bank, dict) or not bank:
        raise ValueError("context_window_bank must be a non-empty object.")

    return {
        str(participant_id): {
            window["context_window_id"]: window
            for window in windows
        }
        for participant_id, windows in bank.items()
    }


def validate_parameters(alpha, k, max_turn_overlap_ratio, candidate_multiplier):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if not 0 <= max_turn_overlap_ratio <= 1:
        raise ValueError(
            "max_turn_overlap_ratio must be between 0 and 1."
        )
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be positive.")


def record_key(record):
    return (
        str(record["participant_id"]),
        int(record["item_id"]),
    )


def index_records(records, name):
    indexed = {}
    for record in records:
        key = record_key(record)
        if key in indexed:
            raise ValueError(f"Duplicate {name} record key: {key}")
        indexed[key] = record
    return indexed


def min_max_normalize(score_by_id):
    if not score_by_id:
        return {}

    values = list(score_by_id.values())
    minimum = min(values)
    maximum = max(values)

    if math.isclose(minimum, maximum):
        return {window_id: 0.0 for window_id in score_by_id}

    scale = maximum - minimum
    return {
        window_id: (score - minimum) / scale
        for window_id, score in score_by_id.items()
    }


def turn_overlap_ratio(left_window, right_window):
    left_turns = set(left_window["turn_ids"])
    right_turns = set(right_window["turn_ids"])
    smaller_size = min(len(left_turns), len(right_turns))

    if smaller_size == 0:
        return 0.0

    return len(left_turns & right_turns) / smaller_size


def select_diverse_candidates(candidates, k, max_turn_overlap_ratio):
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
        candidate["context_window_id"] for candidate in selected
    }
    for candidate in candidates:
        if candidate["context_window_id"] in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate["context_window_id"])
        if len(selected) == k:
            break

    return selected


def modality_scores(record, ids_key, scores_key):
    window_ids = record.get(ids_key)
    scores = record.get(scores_key)

    if not isinstance(window_ids, list) or not isinstance(scores, list):
        raise ValueError(f"Invalid arrays in record {record_key(record)}")
    if len(window_ids) != len(scores):
        raise ValueError(
            f"ID/score length mismatch in record {record_key(record)}"
        )
    if len(window_ids) != len(set(window_ids)):
        raise ValueError(
            f"Duplicate window IDs in record {record_key(record)}"
        )

    return {
        str(window_id): float(score)
        for window_id, score in zip(window_ids, scores)
    }


def build_hybrid_records(
    bm25_records,
    semantic_records,
    context_window_lookup,
    alpha,
    k,
    max_turn_overlap_ratio,
    candidate_multiplier,
):
    bm25_by_key = index_records(bm25_records, "BM25")
    semantic_by_key = index_records(semantic_records, "semantic")

    if set(bm25_by_key) != set(semantic_by_key):
        raise ValueError(
            "BM25 and semantic retrieval keys do not match. "
            f"missing_semantic={len(set(bm25_by_key) - set(semantic_by_key))}, "
            f"missing_bm25={len(set(semantic_by_key) - set(bm25_by_key))}"
        )

    output = []
    candidate_limit = k * candidate_multiplier

    for key, bm25_record in bm25_by_key.items():
        semantic_record = semantic_by_key[key]
        participant_id, item_id = key
        participant_windows = context_window_lookup.get(participant_id)
        if participant_windows is None:
            raise ValueError(
                f"Participant missing from context window bank: {participant_id}"
            )

        bm25_scores_present = modality_scores(
            bm25_record,
            "retrieved_context_window_ids_bm25",
            "retrieved_context_bm25_scores",
        )
        semantic_scores_present = modality_scores(
            semantic_record,
            "retrieved_context_window_ids_semantic",
            "retrieved_context_semantic_scores",
        )
        candidate_ids = list(
            dict.fromkeys(
                list(bm25_scores_present)
                + list(semantic_scores_present)
            )
        )

        for window_id in candidate_ids:
            if window_id not in participant_windows:
                raise ValueError(
                    "Participant-local candidate violation: "
                    f"participant_id={participant_id}, window_id={window_id}"
                )

        bm25_scores = {
            window_id: bm25_scores_present.get(window_id, 0.0)
            for window_id in candidate_ids
        }
        semantic_scores = {
            window_id: semantic_scores_present.get(window_id, 0.0)
            for window_id in candidate_ids
        }
        bm25_normalized = min_max_normalize(bm25_scores)
        semantic_normalized = min_max_normalize(semantic_scores)

        candidates = []
        for source_order, window_id in enumerate(candidate_ids):
            window = dict(participant_windows[window_id])
            window["bm25_score"] = bm25_scores[window_id]
            window["semantic_score"] = semantic_scores[window_id]
            window["bm25_normalized"] = bm25_normalized[window_id]
            window["semantic_normalized"] = semantic_normalized[window_id]
            window["hybrid_score"] = (
                alpha * bm25_normalized[window_id]
                + (1 - alpha) * semantic_normalized[window_id]
            )
            window["_source_order"] = source_order
            candidates.append(window)

        candidates.sort(
            key=lambda window: (
                -window["hybrid_score"],
                -window["semantic_normalized"],
                -window["bm25_normalized"],
                window["_source_order"],
                window["context_window_id"],
            )
        )
        candidates = candidates[: min(candidate_limit, len(candidates))]
        selected = select_diverse_candidates(
            candidates=candidates,
            k=min(k, len(candidates)),
            max_turn_overlap_ratio=max_turn_overlap_ratio,
        )

        output.append(
            {
                "row_index": int(bm25_record["row_index"]),
                "participant_id": participant_id,
                "item_id": item_id,
                "retrieved_context_window_ids_hybrid": [
                    window["context_window_id"] for window in selected
                ],
                "retrieved_context_hybrid_scores": [
                    window["hybrid_score"] for window in selected
                ],
                "retrieved_context_windows_hybrid": [
                    window["window_text"] for window in selected
                ],
                "retrieved_context_bm25_normalized": [
                    window["bm25_normalized"] for window in selected
                ],
                "retrieved_context_semantic_normalized": [
                    window["semantic_normalized"] for window in selected
                ],
                "candidate_count": len(candidate_ids),
            }
        )

    output.sort(key=lambda record: record["row_index"])
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25_path", type=Path, required=True)
    parser.add_argument("--semantic_path", type=Path, required=True)
    parser.add_argument(
        "--context_window_bank_path",
        type=Path,
        required=True,
    )
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
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
    return parser.parse_args()


def main():
    args = parse_args()
    validate_parameters(
        alpha=args.alpha,
        k=args.k,
        max_turn_overlap_ratio=args.max_turn_overlap_ratio,
        candidate_multiplier=args.candidate_multiplier,
    )

    for path in (
        args.bm25_path,
        args.semantic_path,
        args.context_window_bank_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")

    records = build_hybrid_records(
        bm25_records=load_json(args.bm25_path),
        semantic_records=load_json(args.semantic_path),
        context_window_lookup=load_context_window_lookup(
            args.context_window_bank_path
        ),
        alpha=args.alpha,
        k=args.k,
        max_turn_overlap_ratio=args.max_turn_overlap_ratio,
        candidate_multiplier=args.candidate_multiplier,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    selected_counts = [
        len(record["retrieved_context_window_ids_hybrid"])
        for record in records
    ]
    candidate_counts = [record["candidate_count"] for record in records]
    print("Hybrid context retrieval completed successfully.")
    print(f"Output path: {args.output_path}")
    print(f"Rows: {len(records)}")
    print(f"Participants: {len({r['participant_id'] for r in records})}")
    print(f"Alpha: {args.alpha}")
    print(f"k: {args.k}")
    print(f"Min selected windows: {min(selected_counts)}")
    print(f"Max selected windows: {max(selected_counts)}")
    print(f"Min candidates: {min(candidate_counts)}")
    print(f"Max candidates: {max(candidate_counts)}")


if __name__ == "__main__":
    main()
