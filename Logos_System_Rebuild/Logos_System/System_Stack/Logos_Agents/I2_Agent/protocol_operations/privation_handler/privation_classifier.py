# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""
I2 Privation Handler — privation_classifier.py

Role:
- Classify inputs for "privation" (degraded / inverted / null / corrupt forms).
- This does NOT block. It tags + scores confidence for downstream handling.
- Intended to run twice:
  (a) pre-transform (raw inbound)
  (b) post-transform (after privation_transformer)

Inputs:
- raw text OR dict packet OR UIP meaning object (stringifiable)

Outputs:
- tags: list[str]
- domain: one of {"linguistic","epistemic","ontological","axiological","teleological","agentic","unknown"}
- confidence: float 0..1
- evidence: small, explainable signals used to justify the tag
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PrivationClassification:
    tags: List[str]
    domain: str
    confidence: float
    evidence: Dict[str, Any]
    notes: Optional[str] = None


# Minimal, robust lexical signals (expand as you refine the privation library taxonomy)
NEGATION_TOKENS = {"not", "no", "never", "none", "nothing", "nowhere", "cannot", "can't", "without"}
NULLITY_TOKENS = {"nothing", "nonbeing", "void", "null", "zero", "absence", "non-existence", "doesn't exist", "does not exist"}
CONTRADICTION_TOKENS = {"contradiction", "paradox", "inconsistent", "self-contradict", "both true and false"}
AXIOLOGY_TOKENS = {"evil", "hate", "malice", "cruel", "harm", "destroy", "worthless", "corrupt", "lie", "deceive"}
TELEOLOGY_TOKENS = {"meaningless", "pointless", "futile", "purposeless", "no purpose"}
AGENTIC_TOKENS = {"i will", "i want to", "i intend", "force", "coerce", "threaten"}
GIBBERISH_HEURISTIC_THRESHOLD = 0.35  # low-quality text heuristic cutoff


def _extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("text", "content", "utterance", "message", "input", "raw"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return str(payload)
    return str(payload)


def _simple_quality_score(text: str) -> float:
    """
    Cheap heuristic: returns 0..1 where lower = more degraded.
    """

    stripped = text.strip()
    if not stripped:
        return 0.0
    good = sum(1 for ch in stripped if ch.isalnum() or ch.isspace())
    return good / max(1, len(stripped))


def _contains_any(text_lower: str, tokens: set[str]) -> bool:
    return any(tok in text_lower for tok in tokens)


def classify(payload: Any) -> PrivationClassification:
    text = _extract_text(payload)
    text_lower = text.lower()

    evidence: Dict[str, Any] = {"length": len(text)}
    tags: List[str] = []
    domain = "unknown"
    confidence = 0.2

    quality = _simple_quality_score(text)
    evidence["quality_score"] = round(quality, 3)
    if quality < GIBBERISH_HEURISTIC_THRESHOLD:
        tags.append("linguistic_degradation")
        domain = "linguistic"
        confidence = max(confidence, 0.6)

    if _contains_any(text_lower, NEGATION_TOKENS):
        tags.append("negation_present")
        confidence = max(confidence, 0.45)
        if domain == "unknown":
            domain = "linguistic"

    if _contains_any(text_lower, NULLITY_TOKENS):
        tags.append("ontological_nullity")
        domain = "ontological"
        confidence = max(confidence, 0.7)

    if _contains_any(text_lower, CONTRADICTION_TOKENS):
        tags.append("explicit_contradiction")
        if domain in ("unknown", "linguistic"):
            domain = "epistemic"
        confidence = max(confidence, 0.75)

    if _contains_any(text_lower, AXIOLOGY_TOKENS):
        tags.append("axiological_degradation")
        domain = "axiological"
        confidence = max(confidence, 0.7)

    if _contains_any(text_lower, TELEOLOGY_TOKENS):
        tags.append("teleological_void")
        domain = "teleological"
        confidence = max(confidence, 0.65)

    if _contains_any(text_lower, AGENTIC_TOKENS):
        tags.append("agentic_intent_signal")
        if domain == "unknown":
            domain = "agentic"
        confidence = max(confidence, 0.45)

    if not tags:
        tags = ["non_privative_or_uncertain"]
        domain = "unknown"
        confidence = max(confidence, 0.3)

    evidence["tags"] = tags
    evidence["domain"] = domain

    return PrivationClassification(
        tags=tags,
        domain=domain,
        confidence=round(confidence, 3),
        evidence=evidence,
    )


def reclassify_after_transform(original: Any, transformed: Any) -> Dict[str, PrivationClassification]:
    return {"original": classify(original), "transformed": classify(transformed)}
