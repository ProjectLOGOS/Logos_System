from __future__ import annotations

from typing import Any, Dict


def core_healthcheck(*, agent: str, core_version: str = "testing") -> Dict[str, Any]:
    """Lightweight health report for the agent core kit."""
    return {
        "ok": True,
        "agent": agent,
        "core_version": core_version,
    }
