"""
Principal operator for I1: Sign Principle.

Role: causal mechanism for symbol grounding and reference resolution.
Constraints:
- Deterministic
- No inference / no belief formation
- Pure lookup + trace emission
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from I1_Agent.diagnostics.errors import IntegrationError


@dataclass(frozen=True)
class SignResolution:
    token: str
    resolved: bool
    referent: str


class SignPrincipalOperator:
    """
    Deterministic sign grounding operator.
    Input: symbol_table (token -> referent string)
    Output: resolution traces and lightweight coherence checks.
    """

    def __init__(self, symbol_table: Dict[str, str]):
        if not isinstance(symbol_table, dict):
            raise IntegrationError("symbol_table must be a dict[str, str]")
        # normalize to strings only
        self.symbol_table: Dict[str, str] = {
            str(k): "" if v is None else str(v) for k, v in symbol_table.items()
        }

    def anchor(self, token: str) -> str:
        """Lookup-only: returns referent or '<unresolved>'."""
        token = str(token)
        ref = self.symbol_table.get(token, "")
        return ref if ref else "<unresolved>"

    def resolve_tokens(self, tokens: List[str]) -> List[SignResolution]:
        """Resolves a list of tokens into referents with trace."""
        out: List[SignResolution] = []
        for t in tokens:
            ref = self.anchor(t)
            out.append(SignResolution(token=t, resolved=(ref != "<unresolved>"), referent=ref))
        return out

    def coherence_check(self) -> Tuple[bool, List[str]]:
        """
        Lightweight, deterministic check:
        - no empty referents allowed for keys
        - flags duplicate referents if multiple tokens map to same non-empty referent (optional warning)
        """
        violations: List[str] = []

        for k, v in self.symbol_table.items():
            if not v.strip():
                violations.append(f"empty_referent:{k}")

        # duplicate referents warning (not fatal)
        rev: Dict[str, List[str]] = {}
        for k, v in self.symbol_table.items():
            if v.strip():
                rev.setdefault(v, []).append(k)
        for ref, keys in rev.items():
            if len(keys) > 1:
                violations.append(f"duplicate_referent:{ref}:{','.join(sorted(keys))}")

        ok = len([v for v in violations if v.startswith("empty_referent:")]) == 0
        return ok, violations

    def apply_to_packet(
        self,
        *,
        packet: Dict[str, Any],
        tokens_field: str = "tokens",
        out_field: str = "sign_trace",
    ) -> Dict[str, Any]:
        """
        Reads tokens from packet[tokens_field] if present and list-like,
        emits resolutions to packet[out_field]. Returns new dict.
        """
        if not isinstance(packet, dict):
            raise IntegrationError("packet must be a dict")

        tokens_val = packet.get(tokens_field, [])
        if tokens_val is None:
            tokens_val = []
        if not isinstance(tokens_val, list):
            raise IntegrationError(f"{tokens_field} must be a list")

        resolutions = self.resolve_tokens([str(x) for x in tokens_val])
        ok, violations = self.coherence_check()

        out = dict(packet)
        out[out_field] = {
            "resolved": [r.__dict__ for r in resolutions],
            "coherence_ok": ok,
            "violations": violations,
        }
        return out
