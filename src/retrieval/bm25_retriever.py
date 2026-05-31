"""
BM25 based utterance retriever.

Given a PHQ-8 item text and a participant's utterances, return the top-k
utterances that are most relevant according to BM25.
"""

from typing import List
import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    if text is None:
        return []

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text == "":
        return []

    return text.split()


def retrieve_top_k_bm25(
    item_text: str,
    utterances: List[str],
    k: int = 5,
) -> List[str]:
    """
    Retrieve top-k utterances using BM25.

    Parameters
    ----------
    item_text : str
        PHQ-8 item text.
    utterances : list[str]
        Candidate utterances for one participant.
    k : int
        Number of utterances to return.

    Returns
    -------
    list[str]
        Top-k utterances, ordered from most to least relevant.
    """
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

    query_tokens = tokenize(item_text)

    if len(query_tokens) == 0:
        return utterances[:k]

    tokenized_utterances = [tokenize(u) for u in utterances]

    valid_pairs = [
        (original, tokens)
        for original, tokens in zip(utterances, tokenized_utterances)
        if len(tokens) > 0
    ]

    if len(valid_pairs) == 0:
        return utterances[:k]

    valid_utterances = [p[0] for p in valid_pairs]
    valid_tokens = [p[1] for p in valid_pairs]

    bm25 = BM25Okapi(valid_tokens)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    top_indices = ranked_indices[: min(k, len(valid_utterances))]

    return [valid_utterances[i] for i in top_indices]


if __name__ == "__main__":
    item_text = "Trouble falling or staying asleep, or sleeping too much"

    utterances = [
        "I cannot fall asleep until 3 AM.",
        "I went to the grocery store yesterday.",
        "I wake up many times during the night.",
        "My appetite is normal.",
    ]

    print(retrieve_top_k_bm25(item_text, utterances, k=2))