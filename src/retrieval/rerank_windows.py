"""Cross-encoder reranking of context windows (Bucket-B Job 2).

The hybrid retriever surfaces candidate windows by lexical+semantic similarity,
but similarity to the PHQ-8 *question* is not the same as being *diagnostic* of
severity (Fig 3: some items get on-topic-but-uninformative evidence). A cross-
encoder reranker re-scores every window in a participant's transcript against the
item question and keeps the top-K, building a sharper evidence pack.

For each (participant, item) we score all of that participant's context windows
with cross-encoder/ms-marco-MiniLM-L-6-v2, take the top-K, and write a new
``reranked_pack`` column. The downstream CV (cross_validate.py) then trains on
this column exactly as it does on the hybrid pack, so the comparison is clean.

    python -m src.retrieval.rerank_windows --top-k 5
"""

from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.dataset_loader import load_item_dataset

BANK = PROJECT_ROOT / "data" / "processed" / "context_window_bank.json"
WINDOW_SEP = " [WINDOW_SEP] "


def main():
    import torch
    from sentence_transformers import CrossEncoder

    ap = argparse.ArgumentParser(description="Cross-encoder window reranking")
    ap.add_argument("--dataset-path",
                    default="data/processed/phq8_item_dataset_context_windows_hybrid_w3.csv")
    ap.add_argument("--out-path",
                    default="data/processed/phq8_item_dataset_reranked_w3.csv")
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | reranker={args.model} | top_k={args.top_k}")

    df = load_item_dataset(args.dataset_path).reset_index(drop=True)
    df["participant_id"] = df["participant_id"].astype(str)
    bank = json.loads(BANK.read_text())
    bank = {str(k): [w["window_text"] for w in v] for k, v in bank.items()}

    ce = CrossEncoder(args.model, device=device, max_length=256)

    packs = [None] * len(df)
    n_scored = 0
    for pid, g in df.groupby("participant_id"):
        windows = bank.get(pid, [])
        if not windows:                          # fallback: keep existing hybrid pack
            for ridx in g.index:
                packs[ridx] = df.at[ridx, "retrieved_context_windows_hybrid_pack"]
            continue
        for ridx in g.index:
            query = df.at[ridx, "item_text"]
            pairs = [[query, w] for w in windows]
            scores = ce.predict(pairs, batch_size=args.batch_size,
                                show_progress_bar=False)
            top = sorted(range(len(windows)), key=lambda i: -scores[i])[:args.top_k]
            packs[ridx] = WINDOW_SEP.join(windows[i] for i in top)
            n_scored += len(windows)
        print(f"  participant {pid}: {len(g)} items x {len(windows)} windows", flush=True)

    df["reranked_pack"] = packs
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_path, index=False)
    print(f"scored {n_scored} (item,window) pairs; wrote {args.out_path}")


if __name__ == "__main__":
    main()
