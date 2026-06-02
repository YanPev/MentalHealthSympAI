"""
Build retrieval-enhanced PHQ-8 item-level datasets.

Inputs:
- data/processed/phq8_item_dataset_with_splits.csv
- data/processed/utterance_bank.json

Outputs:
- data/processed/phq8_item_dataset_with_retrieval.csv
- data/processed/phq8_item_dataset_full.csv

New columns:
- retrieved_utterances
- baseline_utterances
- n_available_utterances
- n_retrieved_utterances
- n_baseline_utterances
"""

from pathlib import Path
import argparse
import hashlib
import json
import random
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset_loader import load_item_dataset
from src.retrieval.tfidf_retriever import retrieve_top_k_tfidf
from src.retrieval.bm25_retriever import retrieve_top_k_bm25


UTT_SEP = " [UTT_SEP] "


def normalize_participant_id(value):
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    return text


def load_utterance_bank(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"utterance_bank.json does not exist: {path}")

    with open(path, "r", encoding="utf-8") as f:
        utterance_bank = json.load(f)

    if not isinstance(utterance_bank, dict):
        raise ValueError("utterance_bank must be a dictionary.")

    normalized = {}

    for pid, utterances in utterance_bank.items():
        pid = normalize_participant_id(pid)

        if not isinstance(utterances, list):
            raise ValueError(f"Utterances for participant {pid} must be a list.")

        cleaned = [
            str(u).strip()
            for u in utterances
            if u is not None and str(u).strip() != ""
        ]

        normalized[pid] = cleaned

    return normalized


def choose_retrieved_utterances(item_text, utterances, method, k):
    if method == "tfidf":
        return retrieve_top_k_tfidf(item_text, utterances, k=k)

    if method == "bm25":
        return retrieve_top_k_bm25(item_text, utterances, k=k)

    raise ValueError(f"Unknown retrieval method: {method}")


def stable_random_sample(utterances, k, seed_text):
    if len(utterances) <= k:
        return utterances

    seed_hash = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
    seed_int = int(seed_hash[:8], 16)

    rng = random.Random(seed_int)
    indices = list(range(len(utterances)))
    rng.shuffle(indices)

    selected_indices = sorted(indices[:k])

    return [utterances[i] for i in selected_indices]


def choose_baseline_utterances(utterances, k, strategy, seed_text):
    if utterances is None:
        return []

    utterances = [
        str(u).strip()
        for u in utterances
        if u is not None and str(u).strip() != ""
    ]

    if len(utterances) == 0:
        return []

    if k <= 0:
        return []

    if strategy == "first":
        return utterances[:k]

    if strategy == "random":
        return stable_random_sample(utterances, k=k, seed_text=seed_text)

    raise ValueError(f"Unknown baseline strategy: {strategy}")


def join_utterances(utterances):
    utterances = [
        str(u).strip()
        for u in utterances
        if u is not None and str(u).strip() != ""
    ]

    return UTT_SEP.join(utterances)


def validate_retrieval_dataset(df):
    required_cols = [
        "participant_id",
        "item_id",
        "item_name",
        "item_text",
        "label",
        "transcript_text",
        "split",
        "retrieved_utterances",
        "baseline_utterances",
        "n_available_utterances",
        "n_retrieved_utterances",
        "n_baseline_utterances",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Retrieval dataset is empty.")

    invalid_retrieval = df[
        (df["n_available_utterances"] > 0) &
        (df["n_retrieved_utterances"] == 0)
    ]

    if len(invalid_retrieval) > 0:
        raise ValueError(
            "Some rows have available utterances but no retrieved utterances. "
            f"Examples:\n{invalid_retrieval[['participant_id', 'item_id']].head()}"
        )

    invalid_baseline = df[
        (df["n_available_utterances"] > 0) &
        (df["n_baseline_utterances"] == 0)
    ]

    if len(invalid_baseline) > 0:
        raise ValueError(
            "Some rows have available utterances but no baseline utterances. "
            f"Examples:\n{invalid_baseline[['participant_id', 'item_id']].head()}"
        )

    return True


def build_retrieval_dataset(
    dataset_path: Path,
    utterance_bank_path: Path,
    retrieval_output_path: Path,
    full_output_path: Path,
    method: str,
    k: int,
    baseline_strategy: str,
):
    df = load_item_dataset(dataset_path)
    utterance_bank = load_utterance_bank(utterance_bank_path)

    df = df.copy()
    df["participant_id"] = df["participant_id"].map(normalize_participant_id)

    retrieved_values = []
    baseline_values = []

    n_available = []
    n_retrieved = []
    n_baseline = []

    missing_participants = set()

    for _, row in df.iterrows():
        participant_id = normalize_participant_id(row["participant_id"])
        item_id = int(row["item_id"])
        item_text = row["item_text"]

        utterances = utterance_bank.get(participant_id, [])

        if len(utterances) == 0:
            missing_participants.add(participant_id)

        retrieved = choose_retrieved_utterances(
            item_text=item_text,
            utterances=utterances,
            method=method,
            k=k,
        )

        seed_text = f"{participant_id}_{item_id}_{baseline_strategy}"

        baseline = choose_baseline_utterances(
            utterances=utterances,
            k=k,
            strategy=baseline_strategy,
            seed_text=seed_text,
        )

        retrieved_values.append(join_utterances(retrieved))
        baseline_values.append(join_utterances(baseline))

        n_available.append(len(utterances))
        n_retrieved.append(len(retrieved))
        n_baseline.append(len(baseline))

    df["retrieved_utterances"] = retrieved_values
    df["baseline_utterances"] = baseline_values

    df["n_available_utterances"] = n_available
    df["n_retrieved_utterances"] = n_retrieved
    df["n_baseline_utterances"] = n_baseline

    validate_retrieval_dataset(df)

    retrieval_output_path.parent.mkdir(parents=True, exist_ok=True)
    full_output_path.parent.mkdir(parents=True, exist_ok=True)

    retrieval_cols = [
        col for col in df.columns
        if col != "baseline_utterances" and col != "n_baseline_utterances"
    ]

    df[retrieval_cols].to_csv(retrieval_output_path, index=False)
    df.to_csv(full_output_path, index=False)

    print("Retrieval datasets created successfully.")
    print(f"Retrieval method: {method}")
    print(f"k: {k}")
    print(f"Baseline strategy: {baseline_strategy}")

    print(f"\nSaved retrieval dataset:")
    print(retrieval_output_path)

    print(f"\nSaved full dataset:")
    print(full_output_path)

    print("\nDataset shape:")
    print(df.shape)

    print("\nRows per split:")
    print(df["split"].value_counts())

    print("\nParticipants missing from utterance_bank:")
    print(len(missing_participants))

    print("\nRetrieved utterance counts:")
    print(df["n_retrieved_utterances"].describe())

    print("\nBaseline utterance counts:")
    print(df["n_baseline_utterances"].describe())

    return df


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/processed/phq8_item_dataset_with_splits.csv",
    )

    parser.add_argument(
        "--utterance_bank_path",
        type=str,
        default="data/processed/utterance_bank.json",
    )

    parser.add_argument(
        "--retrieval_output_path",
        type=str,
        default="data/processed/phq8_item_dataset_with_retrieval.csv",
    )

    parser.add_argument(
        "--full_output_path",
        type=str,
        default="data/processed/phq8_item_dataset_full.csv",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="tfidf",
        choices=["tfidf", "bm25"],
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--baseline_strategy",
        type=str,
        default="first",
        choices=["first", "random"],
    )

    return parser.parse_args()


def main():
    args = parse_args()

    build_retrieval_dataset(
        dataset_path=Path(args.dataset_path),
        utterance_bank_path=Path(args.utterance_bank_path),
        retrieval_output_path=Path(args.retrieval_output_path),
        full_output_path=Path(args.full_output_path),
        method=args.method,
        k=args.k,
        baseline_strategy=args.baseline_strategy,
    )


if __name__ == "__main__":
    main()