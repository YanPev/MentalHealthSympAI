"""
DataLoader module for the PHQ-8 item classifier (task B7).

Loads the processed dataset, filters by ``split``, and builds train / validation
/ test DataLoaders over ``PHQ8TorchDataset``. The ``evidence_column`` argument is
threaded through so the same module serves the baseline and retrieval conditions.
"""

from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from src.data.dataset_loader import load_item_dataset
from src.models.phq8_torch_dataset import PHQ8TorchDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "phq8_item_dataset_full.csv"

# Map the canonical split names to friendlier loader keys.
SPLITS = {"train": "train", "validation": "validation", "test": "test"}


def make_dataloaders(
    tokenizer,
    dataset_path=DEFAULT_DATASET,
    evidence_column: str = "transcript_text",
    batch_size: int = 16,
    max_length: int = 256,
    num_workers: int = 0,
):
    """Build a dict of DataLoaders keyed by split.

    Returns
    -------
    dict
        ``{"train": DataLoader, "validation": DataLoader, "test": DataLoader}``.
        Only splits actually present in the data are included. ``train`` is
        shuffled; ``validation`` and ``test`` are not.
    """
    df = load_item_dataset(dataset_path)
    if "split" not in df.columns:
        raise ValueError(
            f"Dataset has no 'split' column; cannot build split loaders. "
            f"Use a dataset like phq8_item_dataset_full.csv. Columns: {list(df.columns)}"
        )

    loaders = {}
    for split in SPLITS:
        subset = df[df["split"] == split]
        if subset.empty:
            continue
        ds = PHQ8TorchDataset(
            subset, tokenizer, evidence_column=evidence_column, max_length=max_length
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
    return loaders


def describe_loaders(loaders) -> None:
    """Print a quick summary of the built loaders (rows + batches per split)."""
    for split, loader in loaders.items():
        print(
            f"  {split:11s}: {len(loader.dataset):5d} examples "
            f"-> {len(loader):4d} batches (batch_size={loader.batch_size})"
        )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception:
        tok = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")

    loaders = make_dataloaders(
        tok, evidence_column="retrieved_utterances", batch_size=16, max_length=256
    )
    print("DataLoaders built:")
    describe_loaders(loaders)

    # Inspect one training batch.
    batch = next(iter(loaders["train"]))
    print("\nFirst train batch:")
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"  {k:16s}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k:16s}: {type(v).__name__} (len {len(v)})")
    assert "train" in loaders and "validation" in loaders
    print("\ndata_module smoke check passed.")
