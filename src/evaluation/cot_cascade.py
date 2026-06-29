"""Confidence-gated CoT→encoder cascade.

Strategy (the "LLM first, encoder breaks the 2-candidate tie" design):

  1. The LLM (a self-consistency CoT with per-item vote-fraction probs) predicts
     each PHQ-8 item with a confidence.
  2. If the LLM is confident on an item, keep its pick.
  3. If not, take the LLM's TWO leading candidate severities and let the ENCODER
     decide which of the two it ranks higher.

"Not confident" can be gated three ways:
  * vote      — max LLM vote-fraction < --tau  (statistical: chains disagree)
  * difficult — the staged model itself flagged the item difficult
                (semantic: from the `difficult_frac` column produced by
                cot_joint.py self-consistency, i.e. the fraction of chains that
                placed the item in the DISJOINT difficult set = difficult−confident;
                falls back to parsing chain-0's reasoning if that column is absent)
  * merged    — union of the two (default; best QWK/MAE/exact in experiments)

Everything is computed from existing OOF prediction files — no GPU. Both inputs
must be the SAME eval slice (participant_id, item_id); the encoder OOF defines the
labels and folds.

    python -m src.evaluation.cot_cascade \
        --llm-folds "outputs/cot/folds_tolerant_sc5_dv/*.csv" \
        --gate merged --tau 0.8 --diff-thresh 0.5 --sweep \
        --out outputs/cot/cascade_merged_oof.csv
"""
from pathlib import Path
import argparse
import glob
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             cohen_kappa_score)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NUM_LABELS = 4
LABELS = [0, 1, 2, 3]
DEFAULT_ENCODER_OOF = (PROJECT_ROOT / "outputs" / "cv" /
                       "oof_predictions_ctxm_corn_hybw3.csv")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _disjoint_difficult_from_reasoning(text):
    """Fallback when no difficult_frac column: parse chain-0's staged JSON and
    return 1.0 if the item-bucket text marks any item difficult-but-not-confident.
    Returns a dict {item_id: 1.0/0.0}. (Only chain-0 is available in older runs.)"""
    m = re.search(r"\{.*\}", str(text), flags=re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}

    def ids(key):
        d = obj.get(key, {})
        return {int(re.sub(r"\D", "", str(k))) for k in d
                if re.sub(r"\D", "", str(k))} if isinstance(d, dict) else set()
    disj = ids("difficult") - ids("confident")
    return {iid: 1.0 for iid in disj}


def load_llm(pattern):
    """Concat LLM OOF folds; ensure a per-row `difficult_frac` (from the column if
    present, else derived from chain-0 reasoning)."""
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no LLM fold files match: {pattern}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["participant_id"] = df["participant_id"].astype(str)
    if "difficult_frac" not in df.columns:
        if "reasoning" not in df.columns:
            df["difficult_frac"] = 0.0
        else:  # derive from chain-0 per participant (same value broadcast to its 8 rows)
            frac = {}
            for pid, g in df.groupby("participant_id"):
                frac[pid] = _disjoint_difficult_from_reasoning(g.iloc[0]["reasoning"])
            df["difficult_frac"] = [frac.get(r.participant_id, {}).get(int(r.item_id), 0.0)
                                    for r in df.itertuples()]
        src = "chain-0 reasoning (fallback)"
    else:
        src = "difficult_frac column"
    print(f"  LLM: {len(df)} rows from {len(files)} folds | difficult signal: {src}")
    return df


def load_encoder(path):
    enc = pd.read_csv(path)
    enc["participant_id"] = enc["participant_id"].astype(str)
    need = {"prediction", "fold"} | {f"prob_{c}" for c in range(NUM_LABELS)}
    missing = need - set(enc.columns)
    if missing:
        sys.exit(f"encoder OOF missing columns: {sorted(missing)}")
    return enc


def align(llm, enc):
    """Inner-join on (participant_id, item_id). Encoder carries label + fold."""
    a = llm[["participant_id", "item_id", "difficult_frac"]
            + [f"prob_{c}" for c in range(NUM_LABELS)]].rename(
        columns={f"prob_{c}": f"llm_{c}" for c in range(NUM_LABELS)})
    b = enc[["participant_id", "item_id", "label", "fold", "prediction"]
            + [f"prob_{c}" for c in range(NUM_LABELS)]].rename(
        columns={"prediction": "enc_pred",
                 **{f"prob_{c}": f"enc_{c}" for c in range(NUM_LABELS)}})
    df = a.merge(b, on=["participant_id", "item_id"], how="inner")
    if len(df) != len(b):
        print(f"  WARNING: aligned {len(df)} of {len(b)} encoder rows")
    return df


# --------------------------------------------------------------------------- #
# cascade
# --------------------------------------------------------------------------- #
def gate_mask(df, gate, tau, diff_thresh):
    """Boolean mask: which items are sent to the encoder tiebreak."""
    llmP = df[[f"llm_{c}" for c in range(NUM_LABELS)]].to_numpy()
    maxp = llmP.max(1)
    stat = maxp < tau
    sem = df["difficult_frac"].to_numpy() >= diff_thresh
    if gate == "vote":
        return stat
    if gate == "difficult":
        return sem
    if gate == "merged":
        return stat | sem
    sys.exit(f"unknown gate: {gate}")


def apply_cascade(df, mask):
    """LLM argmax everywhere; on `mask` items the encoder picks the better of the
    LLM's top-2 candidate classes."""
    llmP = df[[f"llm_{c}" for c in range(NUM_LABELS)]].to_numpy()
    encP = df[[f"enc_{c}" for c in range(NUM_LABELS)]].to_numpy()
    pred = llmP.argmax(1)
    for i in np.where(mask)[0]:
        top2 = np.argsort(llmP[i])[::-1][:2]
        pred[i] = top2[np.argmax(encP[i][top2])]
    return pred


def metrics(tag, yt, yp, routed=None):
    out = {
        "config": tag,
        "exact": accuracy_score(yt, yp),
        "macroF1": f1_score(yt, yp, average="macro", labels=LABELS, zero_division=0),
        "QWK": cohen_kappa_score(yt, yp, weights="quadratic", labels=LABELS),
        "MAE": mean_absolute_error(yt, yp),
        "err>=2": float(np.mean(np.abs(yt - yp) >= 2)),
    }
    if routed is not None:
        out["routed%"] = round(100 * routed / len(yt), 1)
    return out


def _show(rows):
    df = pd.DataFrame(rows)
    fmt = {c: (lambda x: f"{x:.3f}") for c in
           ["exact", "macroF1", "QWK", "MAE", "err>=2"]}
    print(df.to_string(index=False, formatters=fmt))


def main():
    ap = argparse.ArgumentParser(description="Confidence-gated CoT→encoder cascade")
    ap.add_argument("--llm-folds", required=True,
                    help="glob of LLM-with-confidence OOF fold CSVs (prob_0..3 = "
                         "vote fractions; optional difficult_frac column).")
    ap.add_argument("--encoder-oof", default=str(DEFAULT_ENCODER_OOF))
    ap.add_argument("--gate", choices=["vote", "difficult", "merged"], default="merged")
    ap.add_argument("--tau", type=float, default=0.8,
                    help="route if max LLM vote-fraction < tau (vote/merged gates).")
    ap.add_argument("--diff-thresh", type=float, default=0.5,
                    help="route if difficult_frac >= this (difficult/merged gates).")
    ap.add_argument("--sweep", action="store_true",
                    help="also print a gate/threshold comparison table.")
    ap.add_argument("--out", default="",
                    help="optional path to write the routed predictions OOF CSV.")
    args = ap.parse_args()

    print(f"Cascade | LLM={args.llm_folds} | encoder={Path(args.encoder_oof).name}")
    llm = load_llm(args.llm_folds)
    enc = load_encoder(args.encoder_oof)
    df = align(llm, enc)
    yt = df["label"].to_numpy()
    llm_pred = df[[f"llm_{c}" for c in range(NUM_LABELS)]].to_numpy().argmax(1)

    print("\n=== parents ===")
    _show([metrics("LLM alone", yt, llm_pred),
           metrics("encoder alone", yt, df["enc_pred"].to_numpy())])

    if args.sweep:
        print("\n=== gate / threshold sweep ===")
        rows = []
        for tau in (0.6, 0.8, 1.0):
            m = gate_mask(df, "vote", tau, args.diff_thresh)
            rows.append(metrics(f"vote<{tau}", yt, apply_cascade(df, m), m.sum()))
        for dt in (0.2, 0.5, 0.8):
            m = gate_mask(df, "difficult", args.tau, dt)
            rows.append(metrics(f"difficult>={dt}", yt, apply_cascade(df, m), m.sum()))
        for dt in (0.2, 0.5):
            m = gate_mask(df, "merged", args.tau, dt)
            rows.append(metrics(f"merged(tau={args.tau},diff>={dt})", yt,
                                apply_cascade(df, m), m.sum()))
        _show(rows)

    # chosen config
    mask = gate_mask(df, args.gate, args.tau, args.diff_thresh)
    pred = apply_cascade(df, mask)
    print(f"\n=== chosen: gate={args.gate} tau={args.tau} diff-thresh={args.diff_thresh} ===")
    _show([metrics(f"cascade[{args.gate}]", yt, pred, mask.sum())])

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        res = df[["participant_id", "item_id", "label", "fold"]].copy()
        res["prediction"] = pred
        # one-hot the routed/kept prediction so the file matches the OOF schema
        for c in range(NUM_LABELS):
            res[f"prob_{c}"] = (pred == c).astype(float)
        res["routed_to_encoder"] = mask.astype(int)
        res.to_csv(out_path, index=False)
        print(f"\nwrote routed predictions: {out_path}  ({len(res)} rows, "
              f"{int(mask.sum())} routed)")


if __name__ == "__main__":
    main()
