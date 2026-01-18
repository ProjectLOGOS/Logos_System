# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""Deterministic local Synthetic Cognition Protocol transport."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict

from .boot_system import SystemRuntime


class LocalSCPTransport:
    """Minimal in-process SCP transport backed by the SOP scheduler."""

    def __init__(self, runtime: SystemRuntime) -> None:
        self.runtime = runtime
        self.mode = "local"
        self._connected = False
        self._lock = asyncio.Lock()

    async def connect(self) -> Dict[str, Any]:
        """Bring the transport online."""
        async with self._lock:
            self._connected = True
        return {"connected": True, "mode": self.mode}

    async def ensure_connected(self) -> None:
        if not self._connected:
            await self.connect()

    async def health(self) -> Dict[str, Any]:
        """Return current health information for monitoring."""
        await self.ensure_connected()
        snapshot = (
            self.runtime.monitor.latest()
            or self.runtime.manager.snapshot().to_dict()
        )
        return {
            "status": "ok",
            "connected": self._connected,
            "mode": self.mode,
            "resource_snapshot": snapshot,
        }

    async def submit_meta_cycle(
        self, agent_id: str, request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a deterministic SCP cycle via the SOP runtime scheduler."""
        await self.ensure_connected()
        runtime = self.runtime
        started = time.time()
        if not getattr(runtime, "scheduler", None):
            raise RuntimeError("Local SCP transport requires an active scheduler")

        async def job() -> Dict[str, Any]:
            cycle_started = time.time()
            await asyncio.sleep(min(0.25, 0.05 + 0.01 * len(str(request)) / 64))
            snapshot = runtime.manager.snapshot().to_dict()
            return {
                "connected": True,
                "mode": self.mode,
                "agent_id": agent_id,
                "request": request,
                "processing_id": str(uuid.uuid4()),
                "cycle_started_at": cycle_started,
                "cycle_ended_at": time.time(),
                "resource_snapshot": snapshot,
                "insights": [
                    {
                        "type": "local_deterministic_cycle",
                        "summary": (
                            "Local SCP transport processed payload "
                            "deterministically."
                        ),
                    }
                ],
            }

        result = await runtime.scheduler.run_immediate(
            label=f"SCP:{agent_id}",
            coro_factory=job,
            cpu_slots=1,
            memory_mb=64,
            disk_mb=0,
            priority=8,
        )
        result["latency_ms"] = round((time.time() - started) * 1000, 2)
        return result

    async def shutdown(self) -> None:
        """Mark the transport as offline."""
        async with self._lock:
            self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected
