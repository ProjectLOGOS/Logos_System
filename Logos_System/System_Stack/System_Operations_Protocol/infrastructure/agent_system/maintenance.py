# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""Background maintenance loops for the SOP runtime."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional

from . import shared_resources
from .shared_resources import ResourceManager, ResourceSnapshot


def _health_log() -> Path:
    return shared_resources.runtime_log_path("health_snapshots.jsonl")


@dataclass
class HealthMonitor:
    """Streams resource snapshots to disk and exposes the latest sample."""

    manager: ResourceManager
    interval: float = 5.0
    _task: Optional[asyncio.Task] = field(default=None, init=False)
    _running: bool = field(default=False, init=False)
    _latest: Optional[ResourceSnapshot] = field(default=None, init=False)

    async def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._run(), name="SOPHealthMonitor")

    async def stop(self) -> None:
        self._running = False
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:  # Expected on shutdown
            pass
        finally:
            self._task = None

    async def _run(self) -> None:
        while self._running:
            snapshot = self.manager.snapshot()
            self._latest = snapshot
            self._write_snapshot(snapshot)
            await asyncio.sleep(self.interval)

    def latest(self) -> Optional[Dict[str, object]]:
        return self._latest.to_dict() if self._latest else None

    def _write_snapshot(self, snapshot: ResourceSnapshot) -> None:
        payload = snapshot.to_dict()
        payload["log_version"] = 1
        with _health_log().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


class MaintenanceService:
    """Runs low-frequency upkeep tasks for the SOP runtime."""

    def __init__(self, *, interval: float = 60.0) -> None:
        self.interval = interval
        self._jobs: List[Callable[[], Awaitable[None]]] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def register_job(self, job: Callable[[], Awaitable[None]]) -> None:
        self._jobs.append(job)

    async def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="SOPMaintenance")

    async def stop(self) -> None:
        self._running = False
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _loop(self) -> None:
        while self._running:
            start = time.time()
            for job in list(self._jobs):
                results = await asyncio.gather(job(), return_exceptions=True)
                if not results:
                    continue
                outcome = results[0]
                if isinstance(
                    outcome,
                    (asyncio.CancelledError, KeyboardInterrupt, SystemExit),
                ):
                    raise outcome
                if isinstance(outcome, BaseException):
                    self._record_job_failure(job, outcome)
            elapsed = time.time() - start
            await asyncio.sleep(max(self.interval - elapsed, 0))

    def _record_job_failure(
        self,
        job: Callable[[], Awaitable[None]],
        error: BaseException,
    ) -> None:
        # Maintenance failures should not crash the runtime; they are logged
        # to the health log for later inspection.
        payload = {
            "timestamp": time.time(),
            "event": "maintenance_error",
            "job": getattr(job, "__name__", "anonymous"),
            "error": repr(error),
        }
        with _health_log().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


async def rotate_resource_log(max_bytes: int = 512_000) -> None:
    """Truncates the resource event log if it grows beyond ``max_bytes``."""

    resource_log = shared_resources.runtime_log_path("resource_events.jsonl")
    if not resource_log.exists():
        return
    if resource_log.stat().st_size <= max_bytes:
        return

    temp = resource_log.with_suffix(".tmp")
    keep_bytes = max_bytes // 2
    with resource_log.open("rb") as src:
        src.seek(max(-keep_bytes, -resource_log.stat().st_size), os.SEEK_END)
        data = src.read()
    with temp.open("wb") as dst:
        dst.write(data)
    temp.replace(resource_log)
