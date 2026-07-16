"""
Multi-query semantic scorer (R2 corrective experiment, Stage B/C).

Unlike the production single-query semantic retriever (which encodes one query
string per item), this scores each window by the **maximum** cosine similarity
over a set of natural-language queries (semantic_queries.active_queries), and
retains the per-query scores + the winning query. Window and query embeddings
are normalized, so a dot product is cosine similarity.

    SemanticScore(window) = max_i cosine(query_i, window)

Diversity control (turn-overlap) is identical to the production retriever so the
returned ranking matches how evidence is actually assembled downstream.
"""

import numpy as np

from src.retrieval.prod_context.semantic_context_retriever import (
    select_diverse_windows)


def encode_normalized(model, texts):
    return model.encode(list(texts), convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)


def score_windows_multiquery(window_embeddings, query_embeddings):
    """Return (max_scores (n_win,), per_query (n_win, n_q), argmax (n_win,)).

    window_embeddings: (n_win, d) normalized. query_embeddings: (n_q, d) normalized.
    """
    sims = np.asarray(window_embeddings) @ np.asarray(query_embeddings).T  # (n_win, n_q)
    if sims.ndim == 1:
        sims = sims[:, None]
    max_scores = sims.max(axis=1)
    argmax = sims.argmax(axis=1)
    return max_scores, sims, argmax


def rank_multiquery(context_windows, window_embeddings, query_texts,
                    query_keys, query_embeddings, k=20,
                    max_turn_overlap_ratio=0.5, candidate_multiplier=5):
    """Rank a participant's windows for one item/config by max-cosine.

    Returns a list of dicts (length <= k) with the window fields plus:
      semantic_score  -- max cosine over the active queries
      per_query       -- {query_key: cosine}
      winning_query_key, winning_query_text
    Diversity applied exactly like the production semantic retriever.
    """
    max_scores, sims, argmax = score_windows_multiquery(
        window_embeddings, query_embeddings)

    order = sorted(range(len(context_windows)),
                   key=lambda i: (-float(max_scores[i]), i))
    cand_count = min(len(order), max(k, k * candidate_multiplier))
    candidates = []
    for i in order[:cand_count]:
        w = dict(context_windows[i])
        w["semantic_score"] = float(max_scores[i])
        w["per_query"] = {query_keys[j]: float(sims[i, j])
                          for j in range(len(query_keys))}
        w["winning_query_key"] = query_keys[int(argmax[i])]
        w["winning_query_text"] = query_texts[int(argmax[i])]
        candidates.append(w)

    return select_diverse_windows(
        candidates=candidates, k=min(k, len(context_windows)),
        max_turn_overlap_ratio=max_turn_overlap_ratio)
