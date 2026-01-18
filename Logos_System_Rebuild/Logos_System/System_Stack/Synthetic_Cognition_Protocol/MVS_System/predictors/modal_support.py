# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""Helpers for loading modal verification components with safe fallbacks."""

from __future__ import annotations

from typing import Dict, Tuple, Type


def _fallback_status(e: float, g: float, t: float) -> Dict[str, float | str]:
    coherence = max(0.0, min(1.0, (e + g + t) / 3.0))
    if coherence >= 0.85:
        status = "necessary"
    elif coherence >= 0.65:
        status = "actual"
    elif coherence >= 0.45:
        status = "possible"
    else:
        status = "unstable"
    return {"status": status, "coherence": coherence}


class _FallbackThonocVerifier:
    """Minimal modal verifier used when canonical implementations are missing."""

    @staticmethod
    def calculate_status(e: float, g: float, t: float) -> Dict[str, float | str]:
        return _fallback_status(e, g, t)

    def trinity_to_modal_status(self, trinity: Tuple[float, float, float]) -> Dict[str, float | str]:
        e, g, t = trinity
        return _fallback_status(e, g, t)


def get_thonoc_verifier() -> Type[_FallbackThonocVerifier]:
    """Return the best-available Thonoc verifier implementation."""
    try:
        from CONSCIOUS_Modal_Inference_System import ThonocVerifier as external_verifier

        return external_verifier  # type: ignore[no-any-return]
    except Exception:  # pragma: no cover - fall through to legacy or fallback
        pass

    try:
        from modal_verifier import ThonocVerifier as legacy_verifier

        return legacy_verifier  # type: ignore[no-any-return]
    except Exception:  # pragma: no cover - fall through to fallback
        pass

    return _FallbackThonocVerifier
