"""Lightweight translation utilities with ontological heuristics."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class TranslationVector:
    """Ontological projection produced by the translation shim."""

    text: str
    source_language: str
    target_language: str
    existence: float
    goodness: float
    truth: float
    metadata: Dict[str, float]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def translate(
    text: str,
    source_lang: str = "en",
    target_lang: str = "en",
) -> TranslationVector:
    """Return a translated payload with naive trinity projections.

    The real system would route through the modal translation bridge. For the
    trimmed runtime we derive deterministic scores from simple text features so
    downstream modules always receive well-formed trinity values.
    """

    text = text or ""
    length = len(text.strip())
    vowel_ratio = (
        sum(text.lower().count(v) for v in "aeiou") / length
        if length
        else 0.0
    )
    digit_ratio = (
        sum(ch.isdigit() for ch in text) / length
        if length
        else 0.0
    )

    existence = _clamp(0.2 + 0.01 * length)
    goodness = _clamp(0.4 + 0.4 * vowel_ratio)
    truth = _clamp(0.6 - 0.3 * digit_ratio)

    metadata = {
        "length": float(length),
        "vowel_ratio": round(vowel_ratio, 3),
        "digit_ratio": round(digit_ratio, 3),
        "source_language": source_lang,
        "target_language": target_lang,
    }

    return TranslationVector(
        text=text,
        source_language=source_lang,
        target_language=target_lang,
        existence=existence,
        goodness=goodness,
        truth=truth,
        metadata=metadata,
    )
