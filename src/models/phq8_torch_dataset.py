"""
PyTorch Dataset for the PHQ-8 item classifier (tasks B6 + B9).

Each item is presented to BERT as a sentence pair
``[CLS] item_text [SEP] evidence [SEP]`` built by ``format_model_input``.
The ``evidence_column`` argument switches between conditions without changing
any other code:

    transcript_text       -> full interview transcript (no-retrieval, long)
    baseline_utterances    -> first-K utterances (no-retrieval baseline)
    retrieved_utterances   -> top-K retrieved utterances (retrieval condition)

``dataset[i]`` returns a dict of tensors plus metadata:
    input_ids, attention_mask, token_type_ids, label (tensors)
    participant_id (str), item_id (int)
"""

from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.models.input_formatting import format_model_input


class PHQ8TorchDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        evidence_column: str = "transcript_text",
        max_length: int = 256,
    ):
        if evidence_column not in df.columns:
            raise ValueError(
                f"evidence_column '{evidence_column}' not in dataframe. "
                f"Available: {list(df.columns)}"
            )
        # Reset index so positional __getitem__ is stable after split filtering.
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.evidence_column = evidence_column
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        enc = format_model_input(
            row["item_text"],
            row[self.evidence_column],
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "participant_id": str(row["participant_id"]),
            "item_id": int(row["item_id"]),
        }
        # token_type_ids distinguishes the item segment from the evidence segment.
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item


if __name__ == "__main__":
    # Smoke check: dataset[0] returns valid tensors.
    from pathlib import Path
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
    from transformers import AutoTokenizer
    from src.data.dataset_loader import load_item_dataset

    df = load_item_dataset(PROJECT_ROOT / "data" / "processed" / "phq8_item_dataset_full.csv")
    try:
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    except Exception:
        tok = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")

    ds = PHQ8TorchDataset(df, tok, evidence_column="retrieved_utterances", max_length=256)
    print(f"len(dataset) = {len(ds)}")
    sample = ds[0]
    for k, v in sample.items():
        if torch.is_tensor(v):
            print(f"  {k:16s}: tensor {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k:16s}: {type(v).__name__} = {v}")
    assert sample["input_ids"].shape == (256,)
    assert sample["label"].dtype == torch.long
    print("PHQ8TorchDataset smoke check passed.")
