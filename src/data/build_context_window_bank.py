"""
Build participant-side context windows from an ordered turn bank.

Windows are centered on one turn and clipped at transcript boundaries, large
time gaps, and raw timestamp inversions. No interviewer roles are inferred.
"""

from pathlib import Path
import argparse
import json
import math


ROLE_SOURCE = "participant_archive_no_explicit_speaker_labels"
TURN_SEPARATOR = " [TURN_SEP] "


def load_turn_bank(turn_bank_path):
    with open(turn_bank_path, "r", encoding="utf-8") as file:
        turn_bank = json.load(file)

    if not isinstance(turn_bank, dict) or not turn_bank:
        raise ValueError("turn_bank must be a non-empty JSON object.")

    return turn_bank


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


def find_window_bounds(turns, center_index, radius, max_time_gap):
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

    return left_index, right_index


def build_participant_windows(
    participant_id,
    turns,
    window_size,
    max_time_gap,
):
    if not isinstance(turns, list) or not turns:
        raise ValueError(
            f"Turns must be a non-empty list for participant_id={participant_id}"
        )

    radius = window_size // 2
    windows = []

    for center_index, center_turn in enumerate(turns):
        left_index, right_index = find_window_bounds(
            turns=turns,
            center_index=center_index,
            radius=radius,
            max_time_gap=max_time_gap,
        )
        selected_turns = turns[left_index : right_index + 1]

        context_window_id = f"{participant_id}_cw{center_index:04d}"
        windows.append(
            {
                "participant_id": participant_id,
                "context_window_id": context_window_id,
                "center_turn_id": center_turn["turn_id"],
                "center_turn_index": center_index,
                "turn_ids": [turn["turn_id"] for turn in selected_turns],
                "start_time": float(selected_turns[0]["start_time"]),
                "end_time": float(selected_turns[-1]["end_time"]),
                "n_turns": len(selected_turns),
                "window_text": TURN_SEPARATOR.join(
                    turn["text"] for turn in selected_turns
                ),
                "role_source": ROLE_SOURCE,
            }
        )

    return windows


def build_context_window_bank(turn_bank, window_size, max_time_gap):
    validate_parameters(window_size, max_time_gap)
    context_window_bank = {}

    for participant_id, turns in turn_bank.items():
        context_window_bank[participant_id] = build_participant_windows(
            participant_id=participant_id,
            turns=turns,
            window_size=window_size,
            max_time_gap=max_time_gap,
        )

    return context_window_bank


def save_context_window_bank(context_window_bank, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            context_window_bank,
            file,
            ensure_ascii=False,
            indent=2,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--turn_bank_path",
        type=Path,
        default=Path("data/processed/turn_bank.json"),
    )
    parser.add_argument(
        "--output_path",
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

    if not args.turn_bank_path.exists():
        raise FileNotFoundError(
            f"Turn bank does not exist: {args.turn_bank_path}"
        )

    turn_bank = load_turn_bank(args.turn_bank_path)
    context_window_bank = build_context_window_bank(
        turn_bank=turn_bank,
        window_size=args.window_size,
        max_time_gap=args.max_time_gap,
    )
    save_context_window_bank(context_window_bank, args.output_path)

    window_counts = [
        len(windows) for windows in context_window_bank.values()
    ]
    turn_counts = [
        window["n_turns"]
        for windows in context_window_bank.values()
        for window in windows
    ]

    print("Context window bank created successfully.")
    print(f"Output path: {args.output_path}")
    print(f"Participants: {len(context_window_bank)}")
    print(f"Context windows: {sum(window_counts)}")
    print(f"Window size: {args.window_size}")
    print(f"Max time gap: {args.max_time_gap}")
    print(f"Min turns per window: {min(turn_counts)}")
    print(f"Max turns per window: {max(turn_counts)}")
    print(f"Role source: {ROLE_SOURCE}")


if __name__ == "__main__":
    main()
