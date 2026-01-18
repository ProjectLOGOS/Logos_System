# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from .transform_types import TransformOutcome, TransformStep

TransformFn = Callable[[Any, Dict[str, Any]], Tuple[Any, TransformStep]]

class TransformRegistry:
    """
    Registry of safe transform functions.
    Each transform returns (new_payload, TransformStep).
    """
    def __init__(self):
        self._fns: Dict[str, TransformFn] = {}

    def register(self, name: str, fn: TransformFn) -> None:
        self._fns[name] = fn

    def names(self) -> List[str]:
        return list(self._fns.keys())

    def get(self, name: str) -> Optional[TransformFn]:
        return self._fns.get(name)


def default_registry() -> TransformRegistry:
    """
    Default safe transforms only:
      - normalize: trim whitespace, basic string normalization
      - reframe: wrap negation statements without rewriting content
      - decompose: split on obvious delimiters into fragments
      - annotate: attach a non-destructive annotation wrapper
    """
    reg = TransformRegistry()

    def normalize(payload: Any, ctx: Dict[str, Any]):
        if isinstance(payload, str):
            out = payload.strip()
            applied = out != payload
            step = TransformStep(
                name="normalize",
                applied=applied,
                notes="Trimmed whitespace." if applied else "No change.",
                delta={"length_before": len(payload), "length_after": len(out)},
            )
            return out, step
        return payload, TransformStep(name="normalize", applied=False, notes="Non-string payload; skipped.")

    def reframe(payload: Any, ctx: Dict[str, Any]):
        if isinstance(payload, str):
            low = payload.strip().lower()
            if low.startswith("i don't") or low.startswith("i do not"):
                out = f"[REPORTED_NEGATION] {payload.strip()}"
                step = TransformStep(
                    name="reframe",
                    applied=True,
                    notes="Wrapped explicit self-negation as reported negation.",
                    delta={"wrapper": "REPORTED_NEGATION"},
                )
                return out, step
        return payload, TransformStep(name="reframe", applied=False, notes="No eligible negation pattern; skipped.")

    def decompose(payload: Any, ctx: Dict[str, Any]):
        if isinstance(payload, str):
            s = payload.strip()
            for delim in [";", "\n"]:
                if delim in s:
                    parts = [p.strip() for p in s.split(delim) if p.strip()]
                    if len(parts) > 1:
                        out = {"fragments": parts}
                        step = TransformStep(
                            name="decompose",
                            applied=True,
                            notes=f"Decomposed into {len(parts)} fragments.",
                            delta={"fragments": len(parts)},
                        )
                        return out, step
        return payload, TransformStep(name="decompose", applied=False, notes="No safe decomposition delimiter; skipped.")

    def annotate(payload: Any, ctx: Dict[str, Any]):
        out = {"annotated": True, "payload": payload}
        step = TransformStep(
            name="annotate",
            applied=True,
            notes="Wrapped payload for safe handling downstream.",
            delta={"annotated": True},
        )
        return out, step

    reg.register("normalize", normalize)
    reg.register("reframe", reframe)
    reg.register("decompose", decompose)
    reg.register("annotate", annotate)
    return reg
