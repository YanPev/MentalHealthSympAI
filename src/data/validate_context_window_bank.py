"""
Validate participant-side context windows against their source turn bank.
"""

from pathlib import Path
import argparse
import json
import math


ROLE_SOURCE = "participant_archive_no_explicit_speaker_labels"
TURN_SEPARATOR = " [TURN_SEP] "
REQUIRED_WINDOW_FIELDS = {
    "participant_id",
    "context_window_id",
    "center_turn_id",
    "center_turn_index",
    "turn_ids",
    "start_time",
    "end_time",
    "n_turns",
    "window_text",
    "role_source",
}


def load_json_object(path, name):
    with open(path, "r", encoding="utf-8") as file:
        value = json.load(file)

    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty JSON object.")

    return value


def validate_parameters(window_size, max_time_gap):
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")

    if not math.isfinite(max_time_gap) or max_time_gap < 0:
        raise ValueError("max_time_gap must be a finite non-negative number.")


def can_cross_gap(left_turn, right_turn, max_time_gap):
    left_start = float(left_turn["start_time"])
    left_end = float(left_turn["end_time"])
    right_start = float(right_turn["start_time"])

    if right_start < left_start:
        return False

    return right_start - left_end <= max_time_gap


def expected_window_turns(turns, center_index, window_size, max_time_gap):
    radius = window_size // 2
    left_index = center_index
    right_index = center_index

    for _ in range(radius):
        candidate_index = left_index - 1
        if candidate_index < 0:
            break
        if not can_cross_gap(
            turns[candidate_index],
            turns[left_index],
            max_time_gap,
        ):
            break
        left_index = candidate_index

    for _ in range(radius):
        candidate_index = right_index + 1
        if candidate_index >= len(turns):
            break
        if not can_cross_gap(
            turns[right_index],
            turns[candidate_index],
            max_time_gap,
        ):
            break
        right_index = candidate_index

    return turns[left_index : right_index + 1]


def numeric_equal(left, right, tolerance=1e-9):
    return math.isclose(
        float(left),
        float(right),
        rel_tol=tolerance,
        abs_tol=tolerance,
    )


def validate_participant_windows(
    participant_id,
    turns,
    windows,
    window_size,
    max_time_gap,
    global_window_ids,
):
    if not isinstance(turns, list) or not turns:
        raise ValueError(
            f"Turns must be a non-empty list for participant_id={participant_id}"
        )

    if not isinstance(windows, list) or not windows:
        raise ValueError(
            f"Windows must be a non-empty list for participant_id={participant_id}"
        )

    if len(windows) != len(turns):
        raise ValueError(
            f"Window coverage mismatch for participant_id={participant_id}: "
            f"turns={len(turns)}, windows={len(windows)}"
        )

    n_turns_counts = {}

    for center_index, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError(
                f"Window is not an object for participant_id={participant_id}, "
                f"center_index={center_index}"
            )

        missing_fields = REQUIRED_WINDOW_FIELDS - set(window)
        if missing_fields:
            raise ValueError(
                f"Missing fields for participant_id={participant_id}, "
                f"center_index={center_index}: {sorted(missing_fields)}"
            )

        expected_window_id = f"{participant_id}_cw{center_index:04d}"
        expected_center_turn_id = turns[center_index]["turn_id"]

        if window["participant_id"] != participant_id:
            raise ValueError(
                f"Participant mismatch in {window.get('context_window_id')}"
            )

        if window["context_window_id"] != expected_window_id:
            raise ValueError(
                f"Unexpected context_window_id: "
                f"expected={expected_window_id}, "
                f"actual={window['context_window_id']}"
            )

        if window["context_window_id"] in global_window_ids:
            raise ValueError(
                f"Duplicate context_window_id: {window['context_window_id']}"
            )
        global_window_ids.add(window["context_window_id"])

        if window["center_turn_index"] != center_index:
            raise ValueError(
                f"Unexpected center_turn_index in {window['context_window_id']}"
            )

        if window["center_turn_id"] != expected_center_turn_id:
            raise ValueError(
                f"Unexpected center_turn_id in {window['context_window_id']}"
            )

        if window["role_source"] != ROLE_SOURCE:
            raise ValueError(
                f"Unexpected role_source in {window['context_window_id']}"
            )

        expected_turns = expected_window_turns(
            turns=turns,
            center_index=center_index,
            window_size=window_size,
            max_time_gap=max_time_gap,
        )
        expected_turn_ids = [turn["turn_id"] for turn in expected_turns]
        actual_turn_ids = window["turn_ids"]

        if actual_turn_ids != expected_turn_ids:
            raise ValueError(
                f"turn_ids mismatch in {window['context_window_id']}: "
                f"expected={expected_turn_ids}, actual={actual_turn_ids}"
            )

        if len(actual_turn_ids) != len(set(actual_turn_ids)):
            raise ValueError(
                f"Duplicate turn IDs in {window['context_window_id']}"
            )

        if expected_center_turn_id not in actual_turn_ids:
            raise ValueError(
                f"Center turn missing from {window['context_window_id']}"
            )

        expected_n_turns = len(expected_turns)
        if window["n_turns"] != expected_n_turns:
            raise ValueError(
                f"n_turns mismatch in {window['context_window_id']}"
            )

        if expected_n_turns > window_size:
            raise ValueError(
                f"Window exceeds window_size: {window['context_window_id']}"
            )

        expected_text = TURN_SEPARATOR.join(
            turn["text"] for turn in expected_turns
        )
        if window["window_text"] != expected_text:
            raise ValueError(
                f"window_text mismatch in {window['context_window_id']}"
            )

        if not numeric_equal(
            window["start_time"],
            expected_turns[0]["start_time"],
        ):
            raise ValueError(
                f"start_time mismatch in {window['context_window_id']}"
            )

        if not numeric_equal(
            window["end_time"],
            expected_turns[-1]["end_time"],
        ):
            raise ValueError(
                f"end_time mismatch in {window['context_window_id']}"
            )

        n_turns_counts[expected_n_turns] = (
            n_turns_counts.get(expected_n_turns, 0) + 1
        )

    return n_turns_counts


def validate_context_window_bank(
    turn_bank,
    context_window_bank,
    window_size,
    max_time_gap,
):
    validate_parameters(window_size, max_time_gap)

    turn_participants = set(turn_bank)
    window_participants = set(context_window_bank)

    if turn_participants != window_participants:
        raise ValueError(
            "Participant coverage mismatch. "
            f"missing={sorted(turn_participants - window_participants)[:20]}, "
            f"unexpected={sorted(window_participants - turn_participants)[:20]}"
        )

    global_window_ids = set()
    total_windows = 0
    total_distribution = {}

    for participant_id in sorted(turn_bank):
        participant_distribution = validate_participant_windows(
            participant_id=participant_id,
            turns=turn_bank[participant_id],
            windows=context_window_bank[participant_id],
            window_size=window_size,
            max_time_gap=max_time_gap,
            global_window_ids=global_window_ids,
        )
        total_windows += len(context_window_bank[participant_id])

        for n_turns, count in participant_distribution.items():
            total_distribution[n_turns] = (
                total_distribution.get(n_turns, 0) + count
            )

    return {
        "participants": len(context_window_bank),
        "windows": total_windows,
        "unique_window_ids": len(global_window_ids),
        "n_turns_distribution": total_distribution,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--turn_bank_path",
        type=Path,
        default=Path("data/processed/turn_bank.json"),
    )
    parser.add_argument(
        "--context_window_bank_path",
        type=Path,
        default=Path("data/processed/context_window_bank.json"),
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max_time_gap",
        type=float,
        default=60.0,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for path in (args.turn_bank_path, args.context_window_bank_path):
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")

    turn_bank = load_json_object(args.turn_bank_path, "turn_bank")
    context_window_bank = load_json_object(
        args.context_window_bank_path,
        "context_window_bank",
    )
    summary = validate_context_window_bank(
        turn_bank=turn_bank,
        context_window_bank=context_window_bank,
        window_size=args.window_size,
        max_time_gap=args.max_time_gap,
    )

    print("Context window bank validation passed.")
    print(f"Participants: {summary['participants']}")
    print(f"Context windows: {summary['windows']}")
    print(f"Unique context window IDs: {summary['unique_window_ids']}")
    print(f"Window size: {args.window_size}")
    print(f"Max time gap: {args.max_time_gap}")
    print("n_turns distribution:")
    for n_turns in sorted(summary["n_turns_distribution"]):
        count = summary["n_turns_distribution"][n_turns]
        print(f"  {n_turns}: {count}")
    print(f"Role source: {ROLE_SOURCE}")


if __name__ == "__main__":
    main()
