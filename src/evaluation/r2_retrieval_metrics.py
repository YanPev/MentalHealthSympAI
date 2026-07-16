"""
Stages E-H — retrieval metrics, complementarity, and label-free R2 selection.

Consumes the Stage C rankings + the Stage D judge outputs and produces every
retrieval-selection artifact. NO gold PHQ labels are used anywhere here.

Inputs:
  outputs/r2_systematic/retrieval/retrieval_window_scores.parquet   (Stage C)
  outputs/r2_systematic/judge/window_judgments_fold*.csv            (Stage D)
  outputs/r2_systematic/judge/set_judgments_fold*.csv               (Stage D)

Outputs (brief §16):
  retrieval/all_candidates.csv          every (config,retriever,prefix) candidate
  retrieval/retrieval_metrics_overall.csv
  retrieval/retrieval_metrics_by_item.csv
  retrieval/retrieval_complementarity.csv
  retrieval/retrieval_redundancy.csv
  retrieval/r2_selection.json           the single selected R2 + label-free reason

Primary selection signal: set informative rate = P(status in {supports,against}).
Keyword hit-rate is NOT used here (diagnostic only, computed elsewhere).

    .venv/bin/python -m src.evaluation.r2_retrieval_metrics
"""

from pathlib import Path
import argparse
import glob
import json

import numpy as np
import pandas as pd

PR = Path(__file__).resolve().parents[2]
RET = PR / "outputs" / "r2_systematic" / "retrieval"
JUD = PR / "outputs" / "r2_systematic" / "judge"
RANKINGS = RET / "retrieval_window_scores.parquet"

INFORMATIVE = {"supports", "against"}
CONFIGS = ["L0_CORE", "L1_CORE_LAY", "L2_CORE_CLINICAL", "L3_FULL"]
TIE = 0.005          # informative-rate tie tolerance
ITEM_ORDER = ["NoInterest", "Depressed", "Sleep", "Tired", "Appetite",
              "Failure", "Concentrating", "Moving"]


def _concat(pattern):
    files = sorted(glob.glob(str(pattern)))
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_inputs():
    rk = pd.read_parquet(RANKINGS)
    rk["participant_id"] = rk["participant_id"].astype(str)
    win = _concat(JUD / "window_judgments_fold*.csv")
    st = _concat(JUD / "set_judgments_fold*.csv")
    for d in (win, st):
        if d is not None:
            d["participant_id"] = d["participant_id"].astype(str)
    return rk, win, st


# ----------------------------------------------------------------------------
# 7.2 set-level metrics + 7.1 per-window metrics
# ----------------------------------------------------------------------------
def set_metrics(st):
    """Overall + by-item set metrics per (config, retriever, prefix)."""
    st = st.copy()
    st["informative"] = st["status"].isin(INFORMATIVE)
    st["is_none"] = st["status"] == "none"
    st["is_amb"] = st["status"] == "ambiguous"
    st["is_supp"] = st["status"] == "supports"
    st["is_agst"] = st["status"] == "against"
    st["agreement"] = st["votes"].map(_vote_agreement)

    keys = ["config", "retriever", "prefix"]
    overall = (st.groupby(keys)
                 .agg(n=("informative", "size"),
                      informative_rate=("informative", "mean"),
                      supports_rate=("is_supp", "mean"),
                      against_rate=("is_agst", "mean"),
                      none_rate=("is_none", "mean"),
                      ambiguous_rate=("is_amb", "mean"),
                      judge_agreement=("agreement", "mean"))
                 .reset_index())
    by_item = (st.groupby(keys + ["item_name"])
                 .agg(n=("informative", "size"),
                      informative_rate=("informative", "mean"),
                      none_rate=("is_none", "mean"),
                      ambiguous_rate=("is_amb", "mean"))
                 .reset_index())
    return overall, by_item


def _vote_agreement(votes_json):
    try:
        c = json.loads(votes_json)
    except (json.JSONDecodeError, TypeError):
        return np.nan
    if not c:
        return np.nan
    tot = sum(c.values())
    return max(c.values()) / tot if tot else np.nan


def per_window_metrics(rk, win):
    """7.1 by-rank per-window metrics for BM25 and semantic, per config."""
    if win is None:
        return None, None
    status = {(r.participant_id, int(r.item_id), r.window_id): r.status
              for r in win.itertuples()}
    rk = rk.copy()
    rk["status"] = [status.get((p, int(i), w)) for p, i, w in
                    zip(rk.participant_id, rk.item_id, rk.window_id)]
    rk = rk[rk["status"].notna()]
    rk["informative"] = rk["status"].isin(INFORMATIVE)

    # by rank (overall per config/retriever)
    by_rank = (rk.groupby(["config", "retriever", "rank"])
                 .agg(n=("informative", "size"),
                      informative_rate=("informative", "mean"),
                      supports_rate=("status", lambda s: (s == "supports").mean()),
                      against_rate=("status", lambda s: (s == "against").mean()),
                      ambiguous_rate=("status", lambda s: (s == "ambiguous").mean()),
                      none_rate=("status", lambda s: (s == "none").mean()))
                 .reset_index())

    # rank of first informative + informative@k, per (config,retriever,participant,item)
    recs = []
    for (cfg, ret, pid, iid), g in rk.groupby(["config", "retriever", "participant_id", "item_id"]):
        g = g.sort_values("rank")
        inf_ranks = g.loc[g["informative"], "rank"].tolist()
        first = int(min(inf_ranks)) if inf_ranks else 0
        recs.append({"config": cfg, "retriever": ret, "participant_id": pid,
                     "item_id": iid, "item_name": g.item_name.iloc[0],
                     "first_informative_rank": first,
                     "reciprocal_rank": (1.0 / first) if first else 0.0,
                     "informative_at_1": int(any(r <= 1 for r in inf_ranks)),
                     "informative_at_3": int(any(r <= 3 for r in inf_ranks)),
                     "informative_at_5": int(any(r <= 5 for r in inf_ranks))})
    firstinf = pd.DataFrame(recs)
    agg = (firstinf.groupby(["config", "retriever"])
             .agg(mrr=("reciprocal_rank", "mean"),
                  informative_at_1=("informative_at_1", "mean"),
                  informative_at_3=("informative_at_3", "mean"),
                  informative_at_5=("informative_at_5", "mean"))
             .reset_index())
    return by_rank, agg


# ----------------------------------------------------------------------------
# 7.4 complementarity
# ----------------------------------------------------------------------------
def complementarity(st):
    """Per (config, prefix, item) BM25-vs-semantic joint-status classification."""
    def cls(row):
        b, s = row["bm25"], row["semantic"]
        bi, si = b in INFORMATIVE, s in INFORMATIVE
        if bi and si:
            return "both_informative"
        if bi and s == "none":
            return "bm25_inf_sem_none"
        if si and b == "none":
            return "sem_inf_bm25_none"
        if bi and s == "ambiguous":
            return "bm25_inf_sem_amb"
        if si and b == "ambiguous":
            return "sem_inf_bm25_amb"
        if b == "none" and s == "none":
            return "both_none"
        if b == "ambiguous" and s == "ambiguous":
            return "both_ambiguous"
        return "other"

    piv = st.pivot_table(index=["config", "prefix", "participant_id", "item_id", "item_name"],
                         columns="retriever", values="status", aggfunc="first").reset_index()
    if "bm25" not in piv or "semantic" not in piv:
        return pd.DataFrame()
    piv["joint"] = piv.apply(cls, axis=1)
    overall = (piv.groupby(["config", "prefix", "joint"]).size()
                 .rename("n").reset_index())
    by_item = (piv.groupby(["config", "prefix", "item_name", "joint"]).size()
                 .rename("n").reset_index())
    overall["scope"] = "overall"; by_item["scope"] = "by_item"
    return pd.concat([overall, by_item], ignore_index=True)


# ----------------------------------------------------------------------------
# 7.3 redundancy (turn-overlap + lexical near-duplicate; cheap, CPU)
# ----------------------------------------------------------------------------
def redundancy(rk, prefixes=("top3", "top5")):
    rows = []
    for (cfg, ret, pid, iid), g in rk.groupby(["config", "retriever", "participant_id", "item_id"]):
        g = g.sort_values("rank")
        for prefix, k in [("top3", 3), ("top5", 5)]:
            if prefix not in prefixes:
                continue
            sel = g.head(k)
            if len(sel) < 2:
                rows.append({"config": cfg, "retriever": ret, "participant_id": pid,
                             "item_id": iid, "prefix": prefix, "n": len(sel),
                             "mean_turn_overlap": 0.0, "near_dup_rate": 0.0,
                             "distinct_regions": len(sel), "transcript_span": 0.0})
                continue
            turnsets = [set(str(t).split("|")) for t in sel["turn_ids"]]
            centers = [c for c in sel["center_turn_index"] if pd.notna(c)]
            overlaps, ndup = [], 0
            for a in range(len(turnsets)):
                for b in range(a + 1, len(turnsets)):
                    j = _jacc(turnsets[a], turnsets[b])
                    overlaps.append(j)
                    if j >= 0.5:
                        ndup += 1
            npairs = len(overlaps)
            span = (max(centers) - min(centers)) if len(centers) >= 2 else 0.0
            rows.append({"config": cfg, "retriever": ret, "participant_id": pid,
                         "item_id": iid, "prefix": prefix, "n": len(sel),
                         "mean_turn_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
                         "near_dup_rate": ndup / npairs if npairs else 0.0,
                         "distinct_regions": _distinct_regions(centers),
                         "transcript_span": float(span)})
    df = pd.DataFrame(rows)
    overall = (df.groupby(["config", "retriever", "prefix"])
                 .agg(mean_turn_overlap=("mean_turn_overlap", "mean"),
                      near_dup_rate=("near_dup_rate", "mean"),
                      distinct_regions=("distinct_regions", "mean"),
                      transcript_span=("transcript_span", "mean"))
                 .reset_index())
    return overall


def _jacc(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _distinct_regions(centers, gap=10):
    if not centers:
        return 0
    cs = sorted(centers)
    regions = 1
    for i in range(1, len(cs)):
        if cs[i] - cs[i - 1] > gap:
            regions += 1
    return regions


# ----------------------------------------------------------------------------
# Stage F/G/H selection
# ----------------------------------------------------------------------------
def build_candidates(overall_set):
    """Flatten to one row per (config, retriever, prefix) candidate."""
    return overall_set.sort_values("informative_rate", ascending=False).reset_index(drop=True)


def shortlist_lexicons(overall_set, prefix="top5"):
    """Stage F: rank L0-L3 by Top-5 set informative rate (best over retrievers)."""
    sub = overall_set[overall_set.prefix == prefix].copy()
    best = (sub.sort_values("informative_rate", ascending=False)
              .groupby("config").head(1))
    best = best.sort_values("informative_rate", ascending=False)
    top = best.iloc[0]
    keep = best[best.informative_rate >= top.informative_rate - TIE]["config"].tolist()
    # always allow up to 2; ensure L0 baseline is analysed even if not shortlisted
    shortlist = keep[:2]
    return best, shortlist


def select_r2(overall_set, complementarity_df):
    """Stage H: pick exactly one R2 (label-free), with the defined tie-breakers.

    Considers single-retriever candidates here; hybrid candidates (if built) are
    appended to overall_set upstream before calling this.
    """
    cand = overall_set.copy()
    cand = cand.sort_values("informative_rate", ascending=False).reset_index(drop=True)
    top = cand.iloc[0]
    tied = cand[cand.informative_rate >= top.informative_rate - TIE].copy()
    # tie-breakers: lower none, lower ambiguous, fewer windows (prefix rank), simpler retriever
    prefix_rank = {"top3": 0, "top5": 1, "budget": 2}
    retr_rank = {"bm25": 0, "semantic": 1, "hybrid": 2}
    tied["_pfx"] = tied.prefix.map(prefix_rank).fillna(9)
    tied["_ret"] = tied.retriever.map(retr_rank).fillna(9)
    tied = tied.sort_values(
        ["none_rate", "ambiguous_rate", "_pfx", "_ret", "informative_rate"],
        ascending=[True, True, True, True, False])
    winner = tied.iloc[0]
    # Top-5 vs budget rule: if within TIE informative and <0.01 none, prefer top5
    return winner, tied


def main():
    ap = argparse.ArgumentParser(description="Stages E-H retrieval metrics + R2 selection")
    ap.add_argument("--no-redundancy", action="store_true")
    args = ap.parse_args()

    rk, win, st = load_inputs()
    if st is None:
        raise SystemExit("no set_judgments_fold*.csv found -- run the set judge first")

    overall_set, by_item_set = set_metrics(st)
    by_rank, mrr = per_window_metrics(rk, win)
    comp = complementarity(st)
    red = None if args.no_redundancy else redundancy(rk)

    RET.mkdir(parents=True, exist_ok=True)
    cand = build_candidates(overall_set)
    cand.to_csv(RET / "all_candidates.csv", index=False)
    # attach mrr/informative@k to the overall metrics where available
    if mrr is not None:
        overall_set = overall_set.merge(mrr, on=["config", "retriever"], how="left")
    overall_set.to_csv(RET / "retrieval_metrics_overall.csv", index=False)
    by_item_set.to_csv(RET / "retrieval_metrics_by_item.csv", index=False)
    if by_rank is not None:
        by_rank.to_csv(RET / "retrieval_metrics_by_rank.csv", index=False)
    comp.to_csv(RET / "retrieval_complementarity.csv", index=False)
    if red is not None:
        red.to_csv(RET / "retrieval_redundancy.csv", index=False)

    best_lex, shortlist = shortlist_lexicons(overall_set)
    winner, tied = select_r2(overall_set, comp)

    # complementarity summary for the hybrid decision (Top-5)
    hyb_signal = _hybrid_signal(comp, shortlist)

    selection = {
        "primary_criterion": "set informative rate = P(status in {supports, against})",
        "label_free": True,
        "tie_tolerance": TIE,
        "lexicon_shortlist": shortlist,
        "lexicon_ranking_top5": best_lex[["config", "retriever", "informative_rate",
                                          "none_rate", "ambiguous_rate"]].to_dict("records"),
        "hybrid_signal": hyb_signal,
        "selected_R2": {
            "config": winner["config"], "retriever": winner["retriever"],
            "prefix": winner["prefix"],
            "informative_rate": float(winner["informative_rate"]),
            "none_rate": float(winner["none_rate"]),
            "ambiguous_rate": float(winner["ambiguous_rate"]),
        },
        "runner_up_tied": tied.head(6)[["config", "retriever", "prefix",
                                        "informative_rate", "none_rate",
                                        "ambiguous_rate"]].to_dict("records"),
        "note": "Hybrid candidates are considered only if hybrid_signal indicates "
                "meaningful two-sided unique wins; see docs/r2_retrieval_selection.md.",
    }
    (RET / "r2_selection.json").write_text(json.dumps(selection, indent=2))

    print("=== R2 selection (provisional; single-retriever candidates) ===")
    print(json.dumps(selection["selected_R2"], indent=2))
    print("\nLexicon shortlist (Top-5 informative):", shortlist)
    print("Hybrid signal:", json.dumps(hyb_signal, indent=2))
    print(f"\nWrote candidates/metrics/complementarity/redundancy + r2_selection.json to {RET}")


def _hybrid_signal(comp, shortlist, prefix="top5"):
    """Two-sided unique-win counts at Top-5 for shortlisted lexicons."""
    if comp.empty:
        return {}
    sub = comp[(comp.scope == "overall") & (comp.prefix == prefix) &
               (comp.config.isin(shortlist))]
    out = {}
    for cfg, g in sub.groupby("config"):
        d = dict(zip(g.joint, g.n))
        bm_only = d.get("bm25_inf_sem_none", 0) + d.get("bm25_inf_sem_amb", 0)
        sem_only = d.get("sem_inf_bm25_none", 0) + d.get("sem_inf_bm25_amb", 0)
        both = d.get("both_informative", 0)
        out[cfg] = {"bm25_unique_wins": int(bm_only),
                    "semantic_unique_wins": int(sem_only),
                    "both_informative": int(both),
                    "hybrid_justified": bool(bm_only >= 20 and sem_only >= 20)}
    return out


if __name__ == "__main__":
    main()
