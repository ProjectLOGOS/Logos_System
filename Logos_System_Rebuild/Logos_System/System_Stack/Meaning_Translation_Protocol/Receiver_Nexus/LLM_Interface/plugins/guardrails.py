# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""Reusable guardrails tied to the active mission profile."""

from __future__ import annotations

import json
import os
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])
STATE_FILE = Path(__file__).resolve().parent / "state" / "mission_profile.json"

try:
    from Logos_System.System_Stack.System_Operations_Protocol.infrastructure.agent_system.agent_nexus import (
        get_mission_profile,
    )  # type: ignore[import-not-found]
except (ImportError, AttributeError):

    def get_mission_profile() -> Dict[str, Any]:
        """Load the persisted mission profile if the nexus is unavailable."""

        if not STATE_FILE.exists():
            return {"label": "UNSET", "safe_interfaces_only": True}
        try:
            return cast(
                Dict[str, Any], json.loads(STATE_FILE.read_text(encoding="utf-8"))
            )
        except Exception:  # pragma: no cover - corrupted state fallback
            return {"label": "UNSET", "safe_interfaces_only": True}


def require_safe_interfaces(func: F) -> F:
    """Decorator blocking risky calls while the mission profile demands safety."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        profile = get_mission_profile() or {}
        if not profile.get("safe_interfaces_only", True):
            return func(*args, **kwargs)
        raise RuntimeError("Blocked by mission profile: safe_interfaces_only")

    return cast(F, wrapper)


__all__ = ["get_mission_profile", "require_safe_interfaces"]


def restrict_writes_to(root: str | os.PathLike[str]) -> Callable[[F], F]:
    """Block write attempts that escape the provided root directory."""

    sandbox = Path(root).resolve()

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            def _validate(candidate: os.PathLike[str]) -> None:
                path = Path(candidate).expanduser().resolve()
                if sandbox == path:
                    return
                try:
                    path.relative_to(sandbox)
                except ValueError as exc:
                    raise PermissionError(
                        f"blocked write outside {sandbox}: {path}"
                    ) from exc

            for name in ("path", "outfile", "dest", "filename", "target"):
                value = kwargs.get(name)
                if isinstance(value, os.PathLike):
                    _validate(value)
            for value in args:
                if isinstance(value, os.PathLike):
                    _validate(value)
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


__all__.append("restrict_writes_to")
