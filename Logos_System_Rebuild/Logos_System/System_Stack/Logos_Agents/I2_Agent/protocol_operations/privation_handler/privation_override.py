# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""Stateless privation override validator for I2 privation handler."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time


@dataclass(frozen=True)
class PrivationOverrideDecision:
    allowed: bool
    reason: str
    authority: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class PrivationOverride:
    """
    Rare, explicit bypass mechanism.
    This module NEVER infers authority.
    It only validates an explicitly provided override context.
    """

    @staticmethod
    def evaluate(
        *,
        override_requested: bool,
        override_token: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> PrivationOverrideDecision:
        """
        Parameters
        ----------
        override_requested : bool
            Explicit signal that a bypass is being requested.
        override_token : dict | None
            Must contain explicit authorization metadata.
        context : dict
            Runtime context (read-only).

        Returns
        -------
        PrivationOverrideDecision
        """

        if not override_requested:
            return PrivationOverrideDecision(
                allowed=False,
                reason="No override requested.",
            )

        if override_token is None:
            return PrivationOverrideDecision(
                allowed=False,
                reason="Override requested without authorization token.",
            )

        required_fields = {"authority", "justification", "scope"}
        missing = required_fields - set(override_token.keys())
        if missing:
            return PrivationOverrideDecision(
                allowed=False,
                reason=f"Override token missing required fields: {sorted(missing)}",
            )

        if override_token["scope"] not in {"local", "single_pass"}:
            return PrivationOverrideDecision(
                allowed=False,
                reason="Invalid override scope.",
            )

        return PrivationOverrideDecision(
            allowed=True,
            reason="Explicit override authorized.",
            authority=str(override_token.get("authority")),
        )
