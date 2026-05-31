"""
TF-IDF based utterance retriever.

Given a PHQ-8 item text and a participant's utterances, return the top-k
utterances that are most similar to the item text.
"""

from typing import List
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_for_retrieval(text: str) -> str:
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def retrieve_top_k_tfidf(
    item_text: str,
    utterances: List[str],
    k: int = 5,
) -> List[str]:
    """
    Retrieve top-k utterances using TF-IDF cosine similarity.

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

    if item_text is None or str(item_text).strip() == "":
        return utterances[:k]

    cleaned_query = clean_for_retrieval(item_text)
    cleaned_utterances = [clean_for_retrieval(u) for u in utterances]

    valid_pairs = [
        (original, cleaned)
        for original, cleaned in zip(utterances, cleaned_utterances)
        if cleaned != ""
    ]

    if len(valid_pairs) == 0:
        return utterances[:k]

    valid_utterances = [p[0] for p in valid_pairs]
    valid_cleaned = [p[1] for p in valid_pairs]

    documents = [cleaned_query] + valid_cleaned

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        tfidf = vectorizer.fit_transform(documents)

        query_vec = tfidf[0]
        utterance_vecs = tfidf[1:]

        scores = cosine_similarity(query_vec, utterance_vecs).flatten()

    except ValueError:
        # Happens if vocabulary is empty.
        return valid_utterances[:k]

    ranked_indices = scores.argsort()[::-1]
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

    print(retrieve_top_k_tfidf(item_text, utterances, k=2))