"""
Missing-evidence training policies (shared by cross_validate.py and
mil_classifier.py).

The LLM evidence judge (src/evaluation/llm_evidence_quality.py) labels each
(participant, item) row's retrieved evidence as supports / against /
ambiguous / none. Three training policies:

  zero  -- status quo: train on every row with its gold label (absent
           evidence implicitly treated as scoreable).
  mask  -- rows judged `none` stay in the training set but contribute no
           gradient: their label is set to MASKED_LABEL (-1) on the TRAINING
           COPY only, and wrap_loss_fn() filters them out of each batch's
           loss (numerically identical to per-row zero-weighting).
  drop  -- rows judged `none` are removed from the training subset entirely
           (and from class-weight computation).

Leakage design: folds are computed on the FULL dataset exactly as before; the
policy is applied AFTER the split, to the training portion only. Validation
rows are never dropped or masked, so fold assignment and OOF coverage are
identical across policies. The judge itself never sees gold labels.
"""

import pandas as pd

MASKED_LABEL = -1
POLICIES = ("zero", "mask", "drop")
NO_EVIDENCE_STATUS = "none"


def attach_evidence_status(df, status_csv):
    """Merge the judge's evidence_status onto df by (participant_id, item_id)."""
    st = pd.read_csv(status_csv)
    status = dict(zip(zip(st["participant_id"].astype(str), st["item_id"].astype(int)),
                      st["evidence_status"]))
    out = df.copy()
    out["evidence_status"] = [status.get((str(p), int(i)))
                              for p, i in zip(out["participant_id"], out["item_id"])]
    n_missing = int(out["evidence_status"].isna().sum())
    if n_missing:
        raise ValueError(f"{n_missing} rows have no evidence_status in {status_csv} "
                         "(dataset/judge mismatch)")
    return out


def apply_policy(tr_df, policy):
    """Apply a missing-evidence policy to the TRAINING subset only.

    Returns (train_df, stats). `mask` sets label=MASKED_LABEL on a copy;
    `drop` removes the rows; `zero` is a no-op.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown missing policy: {policy}")
    if policy == "zero":
        return tr_df, {"policy": policy, "n_train": len(tr_df), "n_no_evidence": 0}

    no_ev = (tr_df["evidence_status"] == NO_EVIDENCE_STATUS).to_numpy()
    stats = {"policy": policy, "n_train": int(len(tr_df)),
             "n_no_evidence": int(no_ev.sum())}
    if policy == "drop":
        out = tr_df.loc[~no_ev]
        stats["n_train_effective"] = int(len(out))
        return out, stats
    out = tr_df.copy()
    out.loc[no_ev, "label"] = MASKED_LABEL
    stats["n_train_effective"] = int((~no_ev).sum())
    return out, stats


def training_labels(tr_df):
    """Labels usable for class-weight computation (masked rows excluded)."""
    return tr_df.loc[tr_df["label"] != MASKED_LABEL, "label"].to_numpy()


def wrap_loss_fn(loss_fn):
    """Filter MASKED_LABEL rows out of each batch before the underlying loss.

    Numerically identical to zero-weighting masked rows; an all-masked batch
    contributes an exact zero loss (still connected to the graph).
    """
    def wrapped(logits, labels):
        valid = labels != MASKED_LABEL
        if bool(valid.all()):
            return loss_fn(logits, labels)
        if not bool(valid.any()):
            return logits.sum() * 0.0
        return loss_fn(logits[valid], labels[valid])
    return wrapped
