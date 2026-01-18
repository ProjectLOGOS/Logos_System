# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""Runtime resource management primitives for the SOP infrastructure.

The historical LOGOS stack treated compute, memory, and IO capacity as
hand-waved invariants. The demo environment needs explicit accounting so the
alignment and protocol layers can make decisions against real telemetry.

This module exposes a lightweight resource manager that:

- Loads capacity caps from ``runtime_resources.json`` when present or derives
  conservative defaults from the host machine.
- Issues cooperative resource leases that are tracked in memory and persisted
  to an append-only JSONL log for audit purposes.
- Emits live snapshots covering load averages, memory pressure, and disk head
  room so monitors and dashboards can report authentic health checks.

Only standard-library modules are used so the surrounding orchestration keeps
its zero-dependency story intact.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional


SOP_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_BASE_DIR = SOP_ROOT / "data_storage" / "runtime"
RUNTIME_STATE_DIR = RUNTIME_BASE_DIR
RESOURCE_CONFIG = SOP_ROOT / "deployment" / "configuration" / "runtime_resources.json"


def configure_runtime_state_dir(target: Optional[Path] = None) -> Path:
    """Select the active runtime directory and ensure it exists."""

    directory = target or RUNTIME_BASE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    globals()["RUNTIME_STATE_DIR"] = directory
    return directory


def runtime_log_path(filename: str) -> Path:
    """Return the path to a runtime log file, creating parents as needed."""

    directory = configure_runtime_state_dir(RUNTIME_STATE_DIR)
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    return path


configure_runtime_state_dir(RUNTIME_STATE_DIR)


def _load_resource_config() -> Dict[str, int]:
    if RESOURCE_CONFIG.exists():
        try:
            data = json.loads(RESOURCE_CONFIG.read_text(encoding="utf-8"))
            return {str(k): int(v) for k, v in data.items()}
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            # Fall back to derived defaults if the file is malformed.
            pass

    cpu_slots = max(os.cpu_count() or 1, 1)
    mem_total = _read_meminfo().get("MemAvailable", 1024 * 1024)
    disk_margin = _read_disk_free()
    return {
        "cpu_slots": cpu_slots,
        "memory_mb": int(mem_total / 1024),
        "disk_mb": int(disk_margin / (1024 * 1024)),
    }


def _read_meminfo() -> Dict[str, int]:
    info: Dict[str, int] = {}
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return info

    for line in meminfo.read_text(encoding="utf-8").splitlines():
        parts = line.split(":")
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        value_str = parts[1].strip().split()[0]
        try:
            info[key] = int(value_str)
        except ValueError:
            continue
    return info


def _read_disk_free() -> int:
    try:
        stat = os.statvfs(str(SOP_ROOT))
        return stat.f_frsize * stat.f_bavail
    except OSError:
        return 0


def _load_average() -> Dict[str, float]:
    try:
        one, five, fifteen = os.getloadavg()
        return {"1m": round(one, 2), "5m": round(five, 2), "15m": round(fifteen, 2)}
    except (AttributeError, OSError):
        return {"1m": 0.0, "5m": 0.0, "15m": 0.0}


@dataclass
class ResourceSpec:
    name: str
    limit: int
    units: str


@dataclass
class ResourceSnapshot:
    timestamp: float
    capacities: Dict[str, ResourceSpec]
    allocations: Dict[str, int]
    load_avg: Dict[str, float]
    memory_state: Dict[str, int]
    disk_free_mb: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "allocations": self.allocations,
            "load_average": self.load_avg,
            "memory": self.memory_state,
            "disk_free_mb": self.disk_free_mb,
            "capacities": {
                name: {"limit": spec.limit, "units": spec.units}
                for name, spec in self.capacities.items()
            },
        }


@dataclass
class ResourceLease(AbstractAsyncContextManager):
    manager: "ResourceManager"
    cpu_slots: int = 0
    memory_mb: int = 0
    disk_mb: int = 0
    label: str = ""
    active: bool = field(default=False, init=False)

    async def __aenter__(self) -> "ResourceLease":
        await self.manager._allocate(self)
        self.active = True
        return self

    async def __aexit__(self, *_exc) -> Optional[bool]:
        await self.manager._release(self)
        self.active = False
        return None

    # Support synchronous context managers for utilities that are not async.
    def __enter__(self) -> "ResourceLease":
        raise RuntimeError("Use 'async with' to obtain resource leases")

    def __exit__(self, *_exc) -> None:  # pragma: no cover - defensive stub
        return None


class ResourceManager:
    """Tracks compute, memory, and disk allocations for SOP workers."""

    def __init__(self) -> None:
        config = _load_resource_config()
        self.capacities: Dict[str, ResourceSpec] = {
            "cpu_slots": ResourceSpec("cpu_slots", config.get("cpu_slots", 1), "slots"),
            "memory_mb": ResourceSpec("memory_mb", config.get("memory_mb", 512), "MB"),
            "disk_mb": ResourceSpec("disk_mb", config.get("disk_mb", 1024), "MB"),
        }
        self.allocations: Dict[str, int] = {key: 0 for key in self.capacities}
        self._condition = asyncio.Condition()
        self._event_log = runtime_log_path("resource_events.jsonl")

    async def lease(
        self,
        *,
        cpu_slots: int = 0,
        memory_mb: int = 0,
        disk_mb: int = 0,
        label: str = "",
    ) -> ResourceLease:
        lease = ResourceLease(
            manager=self,
            cpu_slots=cpu_slots,
            memory_mb=memory_mb,
            disk_mb=disk_mb,
            label=label,
        )
        await lease.__aenter__()
        return lease

    async def _allocate(self, lease: ResourceLease) -> None:
        self._validate_request(lease)
        async with self._condition:
            await self._block_until_capacity(lease)
            self.allocations["cpu_slots"] += lease.cpu_slots
            self.allocations["memory_mb"] += lease.memory_mb
            self.allocations["disk_mb"] += lease.disk_mb
            self._record_event("allocate", lease)

    async def _release(self, lease: ResourceLease) -> None:
        async with self._condition:
            self.allocations["cpu_slots"] = max(
                0, self.allocations["cpu_slots"] - lease.cpu_slots
            )
            self.allocations["memory_mb"] = max(
                0, self.allocations["memory_mb"] - lease.memory_mb
            )
            self.allocations["disk_mb"] = max(
                0, self.allocations["disk_mb"] - lease.disk_mb
            )
            self._record_event("release", lease)
            self._condition.notify_all()

    def _assert_capacity(self, name: str, delta: int) -> None:
        if delta <= 0:
            return
        spec = self.capacities[name]
        current = self.allocations[name]
        if current + delta > spec.limit:
            raise RuntimeError(
                f"Insufficient {name}: requested {delta}, "
                f"available {max(spec.limit - current, 0)}"
            )

    def _validate_request(self, lease: ResourceLease) -> None:
        for name, amount in (
            ("cpu_slots", lease.cpu_slots),
            ("memory_mb", lease.memory_mb),
            ("disk_mb", lease.disk_mb),
        ):
            if amount <= 0:
                continue
            spec = self.capacities[name]
            if amount > spec.limit:
                raise RuntimeError(
                    f"Requested {amount} {name} exceeds capacity {spec.limit}"
                )

    async def _block_until_capacity(self, lease: ResourceLease) -> None:
        while True:
            try:
                self._assert_capacity("cpu_slots", lease.cpu_slots)
                self._assert_capacity("memory_mb", lease.memory_mb)
                self._assert_capacity("disk_mb", lease.disk_mb)
                return
            except RuntimeError:
                await self._condition.wait()

    def utilisation(self, name: str) -> float:
        spec = self.capacities[name]
        if spec.limit == 0:
            return 0.0
        return round(self.allocations[name] / spec.limit, 4)

    def snapshot(self) -> ResourceSnapshot:
        meminfo = _read_meminfo()
        disk_free = _read_disk_free()
        return ResourceSnapshot(
            timestamp=time.time(),
            capacities=self.capacities.copy(),
            allocations=self.allocations.copy(),
            load_avg=_load_average(),
            memory_state={
                "MemTotal": meminfo.get("MemTotal", 0),
                "MemFree": meminfo.get("MemFree", 0),
                "MemAvailable": meminfo.get("MemAvailable", 0),
            },
            disk_free_mb=int(disk_free / (1024 * 1024)),
        )

    def _record_event(self, action: str, lease: ResourceLease) -> None:
        event = {
            "timestamp": time.time(),
            "action": action,
            "label": lease.label,
            "cpu_slots": lease.cpu_slots,
            "memory_mb": lease.memory_mb,
            "disk_mb": lease.disk_mb,
            "allocations": self.allocations.copy(),
        }
        with self._event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def describe(self) -> Dict[str, object]:
        snapshot = self.snapshot()
        return snapshot.to_dict()


def stream_snapshots(manager: ResourceManager) -> Iterable[Dict[str, object]]:
    """Yield successive resource snapshots for polling diagnostics."""

    while True:
        yield manager.snapshot().to_dict()