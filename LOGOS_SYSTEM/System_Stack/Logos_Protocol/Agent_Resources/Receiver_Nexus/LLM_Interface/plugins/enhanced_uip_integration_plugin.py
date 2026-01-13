"""Deterministic enhanced UIP integration plugin with telemetry checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ._uip_connector_stubs import (
    ConnectorMetadata,
    ConnectorResponse,
    ConnectorValidationError,
    StaticEnhancedUIPConnector,
    build_enhanced_connector,
)


@dataclass(frozen=True)
class _Readiness:
    ready: bool
    reason: str
    diagnostics: Dict[str, Any]


def _evaluate_readiness() -> _Readiness:
    diagnostics: Dict[str, Any] = {}
    try:
        connector: StaticEnhancedUIPConnector = build_enhanced_connector()
        metadata: ConnectorMetadata = connector.handshake()
        diagnostics["metadata"] = {
            "name": metadata.name,
            "version": metadata.version,
            "capabilities": list(metadata.capabilities),
        }
        capabilities = set(metadata.capabilities)
        required = {"ping", "collect_telemetry"}
        if not required.issubset(capabilities):
            missing = sorted(required.difference(capabilities))
            return _Readiness(
                ready=False,
                reason=f"missing required capabilities: {missing}",
                diagnostics=diagnostics,
            )

        probe_request = {"command": "ping"}
        response: ConnectorResponse = connector.execute(probe_request)
        telemetry_snapshot = connector.collect_telemetry()
        diagnostics["self_check"] = {
            "request": probe_request,
            "status": response.status,
            "payload": dict(response.payload),
            "telemetry": dict(telemetry_snapshot),
        }
        if response.status != "ok":
            return _Readiness(
                ready=False,
                reason="self-check response status not ok",
                diagnostics=diagnostics,
            )
        payload = response.payload
        if "telemetry" not in payload:
            return _Readiness(
                ready=False,
                reason="telemetry field missing from response payload",
                diagnostics=diagnostics,
            )
        if payload["telemetry"] != telemetry_snapshot:
            return _Readiness(
                ready=False,
                reason="telemetry snapshot mismatch",
                diagnostics=diagnostics,
            )
        return _Readiness(ready=True, reason="", diagnostics=diagnostics)
    except ConnectorValidationError as error:
        return _Readiness(
            ready=False,
            reason=f"validation error: {error}",
            diagnostics=diagnostics,
        )
    except (RuntimeError, ValueError, KeyError, TypeError) as error:
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
    return build_enhanced_connector


def get_enhanced_uip_integration_plugin() -> dict[str, object]:
    return {
        "available": available(),
        "reason": readiness_reason(),
        "diagnostics": readiness_diagnostics(),
        "connector_factory": get_connector_factory(),
    }


def initialize_enhanced_uip_integration() -> dict[str, object]:
    return get_enhanced_uip_integration_plugin()
