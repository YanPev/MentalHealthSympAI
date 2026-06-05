"""
Standardized model-input formatting for the PHQ-8 item classifier.

The model sees every example as a **sentence pair**:

    [CLS] item_text [SEP] evidence_text [SEP]

where ``item_text`` is the PHQ-8 symptom question and ``evidence_text`` is the
participant evidence (full transcript, baseline utterances, or retrieved
utterances). Keeping this in one place means the baseline and retrieval
conditions are formatted by the exact same code path (see task B8).

Utterance evidence is stored in the dataset as a single string with utterances
joined by ``" [UTT_SEP] "``. This module accepts either that raw string or an
already-split list of utterances and normalizes both to one evidence string.
"""

from typing import Iterable, Optional, Union

# Separator used in the processed dataset to join utterances into one cell.
UTT_SEP = " [UTT_SEP] "


def split_utterances(value: Union[str, Iterable[str], None]) -> list:
    """Normalize an evidence cell to a list of utterance strings.

    Accepts the raw ``[UTT_SEP]``-joined string, an iterable of utterances, or
    ``None``/NaN. Empty and whitespace-only utterances are dropped.
    """
    if value is None:
        return []
    # pandas NaN is a float; treat any non-string, non-iterable as empty.
    if isinstance(value, float):
        return []
    if isinstance(value, str):
        parts = value.split(UTT_SEP.strip())
    else:
        parts = list(value)
    return [str(p).strip() for p in parts if str(p).strip()]


def utterances_to_text(utterances: Union[str, Iterable[str], None],
                       sep: str = " ") -> str:
    """Join utterance evidence into a single evidence string for the model."""
    return sep.join(split_utterances(utterances))


def format_model_input(
    item_text: str,
    utterances: Union[str, Iterable[str], None],
    tokenizer=None,
    max_length: int = 256,
    padding: Union[bool, str] = "max_length",
    return_tensors: Optional[str] = None,
):
    """Build standardized model input from an item and its evidence.

    Parameters
    ----------
    item_text : str
        PHQ-8 symptom text (becomes the first segment / query).
    utterances : str | iterable[str] | None
        Evidence. Either a ``[UTT_SEP]``-joined string or a list of utterances.
    tokenizer : transformers tokenizer, optional
        If given, the pair is tokenized and the encoding is returned.
        ``truncation="only_second"`` keeps the (short) item intact and trims the
        (long) evidence.
    max_length, padding, return_tensors :
        Forwarded to the tokenizer when one is supplied.

    Returns
    -------
    dict
        If ``tokenizer is None``: ``{"text": item_text, "text_pair": evidence}`` —
        ready to splat into a tokenizer call (``tokenizer(**out)``).
        If a tokenizer is given: the tokenizer encoding
        (``input_ids``, ``attention_mask``, ``token_type_ids``).
    """
    if not item_text or not str(item_text).strip():
        raise ValueError("item_text must be a non-empty string.")

    evidence_text = utterances_to_text(utterances)

    if tokenizer is None:
        return {"text": str(item_text), "text_pair": evidence_text}

    return tokenizer(
        str(item_text),
        evidence_text,
        truncation="only_second",
        max_length=max_length,
        padding=padding,
        return_tensors=return_tensors,
    )


if __name__ == "__main__":
    # Tiny self-check that runs without a tokenizer.
    out = format_model_input(
        "Feeling tired or having little energy",
        "i am so tired [UTT_SEP] no energy at all [UTT_SEP]   ",
    )
    print("text      :", out["text"])
    print("text_pair :", out["text_pair"])
    assert out["text_pair"] == "i am so tired no energy at all"
    assert split_utterances(None) == [] and split_utterances(float("nan")) == []
    print("input_formatting self-check passed.")
