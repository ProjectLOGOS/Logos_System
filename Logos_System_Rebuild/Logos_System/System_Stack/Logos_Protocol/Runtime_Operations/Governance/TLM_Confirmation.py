from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Dict, Tuple

from Logos_System.System_Stack.Logos_Protocol.Runtime_Operations.Governance.Constraint_Result import ConstraintResult
from Logos_System.System_Stack.Logos_Protocol.Runtime_Operations.Governance.Constraint_Context import ConstraintContext


class TLM_Confirmation:
    """
    Application-neutral confirmation + signing layer.

    Intended use:
      - After constraints and (optional) optimization, bind the final payload to the current session
        using the session_id and/or tlm_token derived from proof commutation.
      - Emit a deterministic signature for audit and downstream verification.

    Note:
      - This is not cryptographic authentication against an external authority.
        It is a deterministic integrity binding within the LOGOS runtime.
      - Replace/extend with the full Transcendental Locking Mechanism (TLM) implementation after audit.
    """

    @staticmethod
    def confirm_and_sign(payload: Dict[str, Any], *, context: ConstraintContext) -> Tuple[ConstraintResult, Dict[str, Any]]:
        if payload is None or not isinstance(payload, dict):
            return ConstraintResult(False, "TLM: payload must be a dict", tags={"validator": "TLM_Confirmation"}), {}
        if not context.session_id:
            return ConstraintResult(False, "TLM: session_id is required", tags={"validator": "TLM_Confirmation"}), {}

        material = {
            "payload": payload,
            "session_id": context.session_id,
        }
        if context.tlm_token:
            material["tlm_token"] = context.tlm_token

        signature = TLM_Confirmation._deterministic_signature(material)
        signed_payload = deepcopy(payload)
        signed_payload["tlm_signature"] = signature

        tags = {
            "validator": "TLM_Confirmation",
            "agent_id": context.agent_id,
            "session_id": context.session_id,
            "has_tlm_token": bool(context.tlm_token),
            "signature": signature,
        }

        return ConstraintResult(
            ok=True,
            reason="TLM: payload confirmed and signed",
            tags=tags,
        ), signed_payload

    @staticmethod
    def _deterministic_signature(material: Dict[str, Any]) -> str:
        serialized = TLM_Confirmation._serialize(material)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize(obj: Any) -> str:
        try:
            return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        except Exception:
            return repr(obj)
