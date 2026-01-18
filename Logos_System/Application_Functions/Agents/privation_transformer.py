# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""Safe, non-destructive privation transformer for I2 privation handler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import hashlib
import time


@dataclass(frozen=True)
class TransformResult:
    new_payload: Any
    transform_report: Dict[str, Any]
    exit_metadata: Dict[str, Any]


class PrivationTransformer:
    """
    Applies limited, safe transformations to degraded input.
    Will clean, reframe, or decompose input, but NEVER rewrite or hallucinate.
    Always preserves original content hash.
    """

    @staticmethod
    def transform_input(
        *,
        payload: Any,
        context: Dict[str, Any],
        classification: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> TransformResult:

        timestamp = time.time()
        original_hash = hashlib.sha256(str(payload).encode("utf-8")).hexdigest()

        transform_report: Dict[str, Any] = {
            "attempted": [],
            "succeeded": [],
            "failed": [],
            "original_hash": original_hash,
            "timestamp": timestamp,
        }

        exit_metadata: Dict[str, Any] = {
            "stage": "privation_transformer",
            "original_preserved": True,
            "transform_status": "no_op",
            "rationale": "",
            "triadic_score_vector": None,
        }

        action = (analysis.get("recommended_action") or analysis.get("action") or "").lower()
        if action in {"quarantine", "escalate"}:
            exit_metadata["rationale"] = f"Transform skipped due to upstream directive: {action}."
            return TransformResult(payload, transform_report, exit_metadata)

        new_payload = payload
        modified = False

        transform_report["attempted"].append("normalize")
        if isinstance(payload, str):
            normalized = payload.strip()
            if normalized != payload:
                new_payload = normalized
                transform_report["succeeded"].append("normalize")
                modified = True
            else:
                transform_report["failed"].append("normalize")
        else:
            transform_report["failed"].append("normalize")

        transform_report["attempted"].append("reframe")
        if isinstance(new_payload, str) and new_payload.lower().startswith("i don't"):
            reframed = f"Subject expresses negation: '{new_payload}'"
            if reframed != new_payload:
                new_payload = reframed
                transform_report["succeeded"].append("reframe")
                modified = True
            else:
                transform_report["failed"].append("reframe")
        else:
            transform_report["failed"].append("reframe")

        transform_report["attempted"].append("decompose")
        if isinstance(new_payload, str) and ";" in new_payload:
            parts = [part.strip() for part in new_payload.split(";") if part.strip()]
            if len(parts) > 1:
                new_payload = parts
                transform_report["succeeded"].append("decompose")
                modified = True
            else:
                transform_report["failed"].append("decompose")
        else:
            transform_report["failed"].append("decompose")

        if modified:
            exit_metadata["transform_status"] = "success"
            exit_metadata["rationale"] = f"{transform_report['succeeded']} applied successfully."
        else:
            exit_metadata["transform_status"] = "no_op"
            exit_metadata["rationale"] = "No valid transforms applicable."

        return TransformResult(new_payload, transform_report, exit_metadata)
