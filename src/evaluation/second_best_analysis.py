"""
Second-best / near-miss analysis: when the model's top choice is wrong, is the
truth its runner-up — or at least adjacent? (Is the model "almost right"?)

PHQ-8 severity is ordinal (0<1<2<3), so a wrong top-1 that is the model's 2nd
choice, or only off-by-one, is a near-miss rather than a real failure. Plain
4-class accuracy hides this. This reads the cross-validation out-of-fold
predictions *with class probabilities* (written by ``cross_validate`` after the
probability-saving update) and reports:

  * top-1 vs top-2 accuracy, and the error-recovery rate from the 2nd choice
  * the rank of the true label in the model's probability ordering
  * ordinal near-miss rate (|pred - true| == 1)
  * per-true-class top-1 vs top-2 recall

    python -m src.evaluation.second_best_analysis --tag bert_base_uncased
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CV_DIR = ROOT / "outputs" / "cv"
NUM_LABELS = 4
SEV_NAMES = ["none(0)", "mild(1)", "moderate(2)", "severe(3)"]


def load(tag):
    df = pd.read_csv(CV_DIR / f"oof_predictions_{tag}.csv")
    prob_cols = [f"prob_{c}" for c in range(NUM_LABELS)]
    if not all(c in df.columns for c in prob_cols):
        raise SystemExit(
            f"{tag} OOF file has no prob_* columns. Re-run cross_validate "
            "after the probability-saving update.")
    return df, prob_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="bert_base_uncased")
    args = ap.parse_args()

    df, prob_cols = load(args.tag)
    P = df[prob_cols].to_numpy()
    y = df["label"].to_numpy()
    n = len(y)

    # Rank classes per row by descending probability.
    order = np.argsort(-P, axis=1)          # order[:,0]=top1, order[:,1]=2nd, ...
    top1 = order[:, 0]
    top2 = order[:, 1]

    correct1 = top1 == y
    in_top2 = (top1 == y) | (top2 == y)
    # rank of the true label (0 = model's 1st choice, 1 = 2nd, ...)
    true_rank = (order == y[:, None]).argmax(axis=1)

    acc1 = correct1.mean()
    acc2 = in_top2.mean()
    errors = ~correct1
    n_err = int(errors.sum())
    recovered = int(((top2 == y) & errors).sum())   # truth was the runner-up
    adjacent_err = int((errors & (np.abs(top1 - y) == 1)).sum())  # off-by-one
    near_miss = int((errors & ((top2 == y) | (np.abs(top1 - y) == 1))).sum())

    print(f"Second-best / near-miss analysis — {args.tag}")
    print(f"(out-of-fold CV predictions, n={n})")
    print("=" * 60)
    print(f"Top-1 accuracy           : {acc1:.3f}")
    print(f"Top-2 accuracy           : {acc2:.3f}   (+{acc2-acc1:.3f})")
    print(f"Mean P(true label)       : {P[np.arange(n), y].mean():.3f}")
    print()
    print(f"Of {n_err} top-1 ERRORS:")
    print(f"  truth was the 2nd choice : {recovered:4d}  ({recovered/n_err:.1%})")
    print(f"  off-by-one (|pred-true|=1): {adjacent_err:4d}  ({adjacent_err/n_err:.1%})")
    print(f"  'near miss' (either)     : {near_miss:4d}  ({near_miss/n_err:.1%})")
    print(f"  genuinely far off        : {n_err-near_miss:4d}  ({(n_err-near_miss)/n_err:.1%})")
    print()
    print("Rank of the TRUE label in the model's probability ordering:")
    for r in range(NUM_LABELS):
        cnt = int((true_rank == r).sum())
        print(f"  rank {r+1}: {cnt:4d}  ({cnt/n:.1%})  cumulative {(true_rank<=r).mean():.1%}")
    print()
    print("Per-true-class recall — top-1 vs top-2:")
    print(f"  {'class':12} {'support':>7} {'top1':>7} {'top2':>7} {'gain':>7}")
    rows = []
    for c in range(NUM_LABELS):
        mask = y == c
        sup = int(mask.sum())
        r1 = correct1[mask].mean() if sup else 0.0
        r2 = in_top2[mask].mean() if sup else 0.0
        rows.append({"class": c, "name": SEV_NAMES[c], "support": sup,
                     "recall_top1": r1, "recall_top2": r2, "gain": r2 - r1})
        print(f"  {SEV_NAMES[c]:12} {sup:>7} {r1:>7.3f} {r2:>7.3f} {r2-r1:>+7.3f}")

    out = CV_DIR / f"second_best_{args.tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
