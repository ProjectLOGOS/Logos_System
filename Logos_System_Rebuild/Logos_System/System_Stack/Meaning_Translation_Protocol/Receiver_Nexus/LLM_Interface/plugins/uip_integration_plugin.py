# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""Deterministic UIP integration plugin backed by sandbox connectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ._uip_connector_stubs import (
    ConnectorMetadata,
    ConnectorResponse,
    ConnectorValidationError,
    build_standard_connector,
)


@dataclass(frozen=True)
class _Readiness:
    ready: bool
    reason: str
    diagnostics: Dict[str, Any]


def _evaluate_readiness() -> _Readiness:
    diagnostics: Dict[str, Any] = {}
    try:
        connector = build_standard_connector()
        metadata: ConnectorMetadata = connector.handshake()
        diagnostics["metadata"] = {
            "name": metadata.name,
            "version": metadata.version,
            "capabilities": list(metadata.capabilities),
        }
        capabilities = set(metadata.capabilities)
        required = {"ping"}
        if not required.issubset(capabilities):
            missing = sorted(required.difference(capabilities))
            return _Readiness(
                ready=False,
                reason=f"missing required capabilities: {missing}",
                diagnostics=diagnostics,
            )

        probe_request = {"command": "ping"}
        response: ConnectorResponse = connector.execute(probe_request)
        diagnostics["self_check"] = {
            "request": probe_request,
            "status": response.status,
            "payload": dict(response.payload),
        }
        if response.status != "ok":
            return _Readiness(
                ready=False,
                reason="self-check response status not ok",
                diagnostics=diagnostics,
            )
        if "heartbeat" not in response.payload:
            return _Readiness(
                ready=False,
                reason="ping payload missing heartbeat token",
                diagnostics=diagnostics,
            )
        return _Readiness(ready=True, reason="", diagnostics=diagnostics)
    except ConnectorValidationError as error:
        return _Readiness(
            ready=False,
            reason=f"validation error: {error}",
            diagnostics=diagnostics,
        )
    except BaseException as error:
        return _Readiness(
            ready=False,
            reason=f"unexpected error: {error.__class__.__name__}: {error}",
            diagnostics=diagnostics,
        )


_READINESS = _evaluate_readiness()


def available() -> bool:
    return _READINESS.ready


def readiness_reason() -> str:
    return _READINESS.reason


def readiness_diagnostics() -> Dict[str, Any]:
    return dict(_READINESS.diagnostics)


def get_connector_factory():
    return build_standard_connector


def get_uip_integration_plugin() -> dict[str, object]:
    return {
        "available": available(),
        "reason": readiness_reason(),
        "diagnostics": readiness_diagnostics(),
        "connector_factory": get_connector_factory(),
    }


def initialize_uip_integration() -> dict[str, object]:
    return get_uip_integration_plugin()


def uip_prompt_choice(
    prompt: str,
    choices: List[str],
    assume_yes: bool = False,
) -> str | None:
    """Prompt user for choice via UIP. If assume_yes, select first choice."""
    if assume_yes:
        return choices[0] if choices else None

    # For now, minimal implementation: print and read from stdin
    print(prompt)
    for i, choice in enumerate(choices, 1):
        print(f"{i}. {choice}")
    try:
        selection = int(input("Enter choice number: ")) - 1
        if 0 <= selection < len(choices):
            return choices[selection]
    except ValueError:
        pass
    return None
