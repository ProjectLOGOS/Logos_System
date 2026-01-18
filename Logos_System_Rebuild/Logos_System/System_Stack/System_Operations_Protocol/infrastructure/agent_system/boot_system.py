# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""Bootstraps the SOP runtime stack with concrete services."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from .maintenance import HealthMonitor, MaintenanceService, rotate_resource_log
from . import shared_resources
from .shared_resources import ResourceManager


CONFIG_DEFAULT_PATH = (
    Path(__file__).resolve().parents[1]
    / "deployment"
    / "configuration"
    / "runtime_config.json"
)


def _scheduler_log() -> Path:
    return shared_resources.runtime_log_path("scheduler_history.jsonl")


@dataclass
class RuntimeConfig:
    monitor_interval: float = 5.0
    maintenance_interval: float = 300.0
    scheduler_concurrency: Optional[int] = None

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "RuntimeConfig":
        target = path or CONFIG_DEFAULT_PATH
        if not target.exists():
            return cls()
        try:
            raw = target.read_text(encoding="utf-8")
        except OSError:
            return cls()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return cls()
        if not isinstance(data, dict):
            return cls()
        try:
            monitor_interval = float(data.get("monitor_interval", 5.0))
            maintenance_interval = float(data.get("maintenance_interval", 300.0))
            scheduler_raw = data.get("scheduler_concurrency")
            scheduler_concurrency = (
                int(scheduler_raw) if scheduler_raw not in (None, "") else None
            )
        except (TypeError, ValueError):
            return cls()
        return cls(
            monitor_interval=monitor_interval,
            maintenance_interval=maintenance_interval,
            scheduler_concurrency=scheduler_concurrency,
        )


@dataclass
class ScheduledJob:
    job_id: str
    created_at: float
    label: str
    coroutine: Callable[[], Awaitable[Any]]
    cpu_slots: int
    memory_mb: int
    disk_mb: int
    priority: int
    future: asyncio.Future


class TaskScheduler:
    """Cooperative task scheduler that honours resource leases."""

    def __init__(
        self,
        manager: ResourceManager,
        *,
        concurrency: Optional[int] = None,
    ) -> None:
        self.manager = manager
        max_workers = manager.capacities["cpu_slots"].limit or 1
        self.concurrency = max(1, min(concurrency or max_workers, max_workers))
        self._queue: "asyncio.PriorityQueue[Tuple[int, float, ScheduledJob]]"
        self._queue = asyncio.PriorityQueue()
        self._workers: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        for index in range(self.concurrency):
            worker = asyncio.create_task(
                self._worker_loop(index),
                name=f"SOPScheduler-{index}",
            )
            self._workers.append(worker)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for _ in self._workers:
            await self._queue.put((0, time.time(), self._shutdown_sentinel()))
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def run_immediate(
        self,
        label: str,
        coro_factory: Callable[[], Awaitable[Any]],
        *,
        cpu_slots: int = 1,
        memory_mb: int = 64,
        disk_mb: int = 0,
        priority: int = 10,
    ) -> Any:
        job = await self.submit(
            label=label,
            coro_factory=coro_factory,
            cpu_slots=cpu_slots,
            memory_mb=memory_mb,
            disk_mb=disk_mb,
            priority=priority,
        )
        return await job.future

    async def submit(
        self,
        *,
        label: str,
        coro_factory: Callable[[], Awaitable[Any]],
        cpu_slots: int,
        memory_mb: int,
        disk_mb: int,
        priority: int,
    ) -> ScheduledJob:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        job = ScheduledJob(
            job_id=str(uuid.uuid4()),
            created_at=time.time(),
            label=label,
            coroutine=coro_factory,
            cpu_slots=cpu_slots,
            memory_mb=memory_mb,
            disk_mb=disk_mb,
            priority=priority,
            future=future,
        )
        await self._queue.put((priority, job.created_at, job))
        return job

    async def _worker_loop(self, worker_index: int) -> None:
        while self._running:
            priority, _, job = await self._queue.get()
            if job.label == "__shutdown__":
                break
            await self._execute_job(job, worker_index, priority)

    async def _execute_job(
        self, job: ScheduledJob, worker_index: int, priority: int
    ) -> None:
        async with await self.manager.lease(
            cpu_slots=job.cpu_slots,
            memory_mb=job.memory_mb,
            disk_mb=job.disk_mb,
            label=job.label,
        ):
            results = await asyncio.gather(
                job.coroutine(),
                return_exceptions=True,
            )
        value = results[0]
        if isinstance(value, Exception):
            if isinstance(value, asyncio.CancelledError):
                if not job.future.done():
                    job.future.cancel()
            else:
                if not job.future.done():
                    job.future.set_exception(value)
            error_text = str(value) or value.__class__.__name__
            self._log_completion(job, worker_index, priority, "failed", error_text)
            return
        if isinstance(value, BaseException):
            raise value
        if not job.future.done():
            job.future.set_result(value)
        self._log_completion(job, worker_index, priority, "completed")

    def _shutdown_sentinel(self) -> ScheduledJob:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future.set_result(None)
        return ScheduledJob(
            job_id="shutdown",
            created_at=time.time(),
            label="__shutdown__",
            coroutine=lambda: asyncio.sleep(0),
            cpu_slots=0,
            memory_mb=0,
            disk_mb=0,
            priority=0,
            future=future,
        )

    def _log_completion(
        self,
        job: ScheduledJob,
        worker_index: int,
        priority: int,
        outcome: str,
        error: Optional[str] = None,
    ) -> None:
        payload = {
            "timestamp": time.time(),
            "job_id": job.job_id,
            "label": job.label,
            "worker": worker_index,
            "priority": priority,
            "cpu_slots": job.cpu_slots,
            "memory_mb": job.memory_mb,
            "disk_mb": job.disk_mb,
            "outcome": outcome,
        }
        if error:
            payload["error"] = error
        with _scheduler_log().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


@dataclass
class SystemRuntime:
    manager: ResourceManager
    scheduler: TaskScheduler
    monitor: HealthMonitor
    maintenance: MaintenanceService

    async def start(self) -> None:
        await self.scheduler.start()
        await self.monitor.start()
        await self.maintenance.start()

    async def shutdown(self) -> None:
        await self.maintenance.stop()
        await self.monitor.stop()
        await self.scheduler.stop()

    def describe(self) -> Dict[str, Any]:
        return {
            "resources": self.manager.describe(),
            "scheduler_concurrency": self.scheduler.concurrency,
            "monitor_interval": self.monitor.interval,
        }


async def initialize_runtime(config_path: Optional[Path] = None) -> SystemRuntime:
    config = RuntimeConfig.load(config_path)
    manager = ResourceManager()
    scheduler = TaskScheduler(manager, concurrency=config.scheduler_concurrency)
    monitor = HealthMonitor(manager, interval=config.monitor_interval)
    maintenance = MaintenanceService(interval=config.maintenance_interval)
    maintenance.register_job(rotate_resource_log)

    runtime = SystemRuntime(
        manager=manager,
        scheduler=scheduler,
        monitor=monitor,
        maintenance=maintenance,
    )
    await runtime.start()
    return runtime

