"""Fast, non-authoritative privation gate for I2 privation handler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import time

from .privation_override import PrivationOverride


@dataclass(frozen=True)
class PrivationGateResult:
    is_privative: bool
    routed: bool
    override_applied: bool
    metadata: Dict[str, Any]


class PrivationGate:
    """
    Fast triage gate for privative content.
    This gate:
      - Never blocks by default
      - Never transforms
      - Never stores memory
      - Only routes and annotates
    """

    @staticmethod
    def evaluate(
        *,
        input_payload: Any,
        context: Dict[str, Any],
        override_requested: bool = False,
        override_token: Optional[Dict[str, Any]] = None,
    ) -> PrivationGateResult:
        """
        Parameters
        ----------
        input_payload : Any
            Raw inbound content (opaque to this gate).
        context : dict
            Read-only execution context.
        override_requested : bool
            Whether an override path is explicitly invoked.
        override_token : dict | None
            Authorization metadata if override is requested.

        Returns
        -------
        PrivationGateResult
        """

        metadata: Dict[str, Any] = {
            "stage": "privation_gate",
            "timestamp": time.time(),
        }

        is_privative = input_payload is None or (
            isinstance(input_payload, str) and not input_payload.strip()
        )
        metadata["heuristic_flag"] = is_privative

        override_decision = PrivationOverride.evaluate(
            override_requested=override_requested,
            override_token=override_token,
            context=context,
        )

        if override_decision.allowed:
            metadata["override"] = {
                "applied": True,
                "authority": override_decision.authority,
                "reason": override_decision.reason,
            }
            return PrivationGateResult(
                is_privative=is_privative,
                routed=True,
                override_applied=True,
                metadata=metadata,
            )

        metadata["override"] = {
            "applied": False,
            "reason": override_decision.reason,
        }

        return PrivationGateResult(
            is_privative=is_privative,
            routed=True,
            override_applied=False,
            metadata=metadata,
        )
