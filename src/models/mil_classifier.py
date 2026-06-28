"""Attention-MIL over item-aware retrieved windows (Bucket-B Job 3).

The pack/concatenation baseline glues the retrieved context windows into one long
string and truncates (W5 lost 94% of its text). Multiple-instance learning treats
each window as a separate instance and lets a *gated attention* pool (Ilse, Tomczak
& Welling, 2018) decide which window matters for this PHQ-8 item — so nothing is
truncated away and we get a per-window attribution (which utterance drove the
score) for free. The bag embedding feeds the same rank-consistent CORN head.

Shares the participant-level 5-fold CV protocol with ``cross_validate.py`` and
writes OOF predictions (outputs/cv/oof_predictions_mil_hybw3.csv) plus per-item
attention weights (outputs/cv/mil_attention_hybw3.csv) for the attribution figure.

    python -m src.models.mil_classifier --model-name mental/mental-bert-base-uncased
"""

from pathlib import Path
import argparse
import json
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.data.dataset_loader import load_item_dataset
from src.models.input_formatting import format_model_input
from src.models.ordinal import corn_loss, corn_probas, corn_predict
from src.models.train_transformer_classifier import set_seed

NUM_LABELS = 4
LIST_COL = "retrieved_context_windows_hybrid_list"
MAX_WINDOWS = 5


class MILDataset(Dataset):
    """Each item -> a bag of <=MAX_WINDOWS tokenised [CLS] item [SEP] window pairs."""

    def __init__(self, df, tokenizer, max_length=128, max_windows=MAX_WINDOWS):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_length = max_length
        self.max_windows = max_windows

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        windows = json.loads(row[LIST_COL]) if isinstance(row[LIST_COL], str) else []
        windows = [w for w in windows if isinstance(w, str) and w.strip()][:self.max_windows]
        if not windows:
            windows = [""]
        ids, mask, ttype = [], [], []
        for w in windows:
            enc = format_model_input(row["item_text"], w, tokenizer=self.tok,
                                     max_length=self.max_length, padding="max_length",
                                     return_tensors="pt")
            ids.append(enc["input_ids"].squeeze(0))
            mask.append(enc["attention_mask"].squeeze(0))
            ttype.append(enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])).squeeze(0))
        n = len(windows)
        # pad the bag to max_windows
        while len(ids) < self.max_windows:
            ids.append(torch.zeros(self.max_length, dtype=torch.long))
            mask.append(torch.zeros(self.max_length, dtype=torch.long))
            ttype.append(torch.zeros(self.max_length, dtype=torch.long))
        return {
            "input_ids": torch.stack(ids),            # (W, L)
            "attention_mask": torch.stack(mask),
            "token_type_ids": torch.stack(ttype),
            "bag_mask": torch.tensor([1] * n + [0] * (self.max_windows - n), dtype=torch.float),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "participant_id": str(row["participant_id"]),
            "item_id": int(row["item_id"]),
        }


class GatedAttentionMIL(nn.Module):
    def __init__(self, model_name, num_out=NUM_LABELS - 1, attn_dim=128, dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        h = self.encoder.config.hidden_size
        self.attn_V = nn.Linear(h, attn_dim)
        self.attn_U = nn.Linear(h, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(h, num_out)

    def forward(self, input_ids, attention_mask, token_type_ids, bag_mask):
        B, W, L = input_ids.shape
        flat = lambda t: t.reshape(B * W, L)
        kw = dict(input_ids=flat(input_ids), attention_mask=flat(attention_mask))
        if token_type_ids is not None:
            kw["token_type_ids"] = flat(token_type_ids)
        out = self.encoder(**kw).last_hidden_state[:, 0]      # CLS, (B*W, H)
        h = out.reshape(B, W, -1)                              # (B, W, H)
        A = torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))
        scores = self.attn_w(A).squeeze(-1)                   # (B, W)
        scores = scores.masked_fill(bag_mask < 0.5, float("-inf"))
        alpha = torch.softmax(scores, dim=1)                  # (B, W)
        bag = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)     # (B, H)
        logits = self.head(self.drop(bag))                    # (B, K-1)
        return logits, alpha


def run_fold(model, train_loader, val_loader, device, epochs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for ep in range(epochs):
        model.train()
        for b in train_loader:
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                  b["token_type_ids"].to(device), b["bag_mask"].to(device))
                loss = corn_loss(logits, b["label"].to(device), NUM_LABELS)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
    # inference
    model.eval()
    preds, probs, alphas = [], [], []
    with torch.no_grad():
        for b in val_loader:
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits, alpha = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                                      b["token_type_ids"].to(device), b["bag_mask"].to(device))
            preds.append(corn_predict(logits.float()).cpu().numpy())
            probs.append(corn_probas(logits.float()).cpu().numpy())
            alphas.append(alpha.float().cpu().numpy())
    return (np.concatenate(preds), np.concatenate(probs, axis=0),
            np.concatenate(alphas, axis=0))


def main():
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser(description="Attention-MIL PHQ-8 item classifier")
    p.add_argument("--dataset-path",
                   default="data/processed/phq8_item_dataset_context_windows_hybrid_w3.csv")
    p.add_argument("--model-name", default="mental/mental-bert-base-uncased")
    p.add_argument("--k-folds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-epochs", type=int, default=8)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="mil_hybw3")
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "cv"))
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = load_item_dataset(args.dataset_path).reset_index(drop=True)
    if LIST_COL not in df.columns:
        raise SystemExit(f"missing column {LIST_COL}")
    groups, y = df["participant_id"].to_numpy(), df["label"].to_numpy()

    print("=" * 64)
    print(f"Attention-MIL CV | model {args.model_name} | device {device}")
    print(f"  bags<= {MAX_WINDOWS} windows x {args.max_length} tok | bs {args.batch_size} "
          f"| {args.num_epochs} ep | {args.k_folds} folds")
    print("=" * 64)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    sgkf = StratifiedGroupKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    oof = df[["participant_id", "item_id", "item_name", "label"]].copy()
    oof["prediction"] = -1; oof["fold"] = -1
    for c in range(NUM_LABELS):
        oof[f"prob_{c}"] = np.nan
    attn_rows = []
    fold_metrics = []

    for fold, (tr, va) in enumerate(sgkf.split(df, y, groups), start=1):
        t0 = time.time()
        assert not (set(df.iloc[tr].participant_id) & set(df.iloc[va].participant_id))
        tl = DataLoader(MILDataset(df.iloc[tr], tok, args.max_length),
                        batch_size=args.batch_size, shuffle=True, num_workers=2)
        vl = DataLoader(MILDataset(df.iloc[va], tok, args.max_length),
                        batch_size=args.batch_size, shuffle=False, num_workers=2)
        model = GatedAttentionMIL(args.model_name).to(device)
        preds, probs, alphas = run_fold(model, tl, vl, device, args.num_epochs, args.learning_rate)

        oof.iloc[va, oof.columns.get_loc("prediction")] = preds
        oof.iloc[va, oof.columns.get_loc("fold")] = fold
        for c in range(NUM_LABELS):
            oof.iloc[va, oof.columns.get_loc(f"prob_{c}")] = probs[:, c]
        for j, vi in enumerate(va):
            r = df.iloc[vi]
            attn_rows.append({"participant_id": r["participant_id"], "item_id": int(r["item_id"]),
                              "item_name": r["item_name"], "fold": fold,
                              **{f"attn_{w}": float(alphas[j, w]) for w in range(MAX_WINDOWS)}})

        m = {"accuracy": float(accuracy_score(y[va], preds)),
             "macro_f1": float(f1_score(y[va], preds, average="macro", zero_division=0)),
             "mae": float(mean_absolute_error(y[va], preds))}
        dt = time.time() - t0
        fold_metrics.append({"fold": fold, "n": int(len(va)), **m, "seconds": dt})
        print(f"Fold {fold}/{args.k_folds} | n={len(va):4d} acc {m['accuracy']:.3f} "
              f"macroF1 {m['macro_f1']:.3f} MAE {m['mae']:.3f} ({dt:.0f}s)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    yt, yp = oof["label"].to_numpy(), oof["prediction"].to_numpy()
    pooled = {"accuracy": float(accuracy_score(yt, yp)),
              "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
              "mae": float(mean_absolute_error(yt, yp)),
              "confusion_matrix": confusion_matrix(yt, yp, labels=list(range(NUM_LABELS))).tolist()}

    def mstd(k):
        a = np.array([m[k] for m in fold_metrics]); return float(a.mean()), float(a.std(ddof=1))
    per_fold = {k: mstd(k) for k in ("accuracy", "macro_f1", "mae")}
    print("-" * 64)
    print(f"Pooled OOF (n={len(oof)}): acc {pooled['accuracy']:.3f} "
          f"macroF1 {pooled['macro_f1']:.3f} MAE {pooled['mae']:.3f}")

    oof.to_csv(out_dir / f"oof_predictions_{args.tag}.csv", index=False)
    pd.DataFrame(attn_rows).to_csv(out_dir / f"mil_attention_{args.tag}.csv", index=False)
    (out_dir / f"cv_metrics_{args.tag}.json").write_text(json.dumps(
        {"args": vars(args), "per_fold_mean_std": per_fold, "pooled_oof": pooled,
         "folds": fold_metrics}, indent=2))
    print(f"Saved OOF + attention + metrics with tag '{args.tag}'")


if __name__ == "__main__":
    main()
