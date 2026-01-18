# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class Wrapper_Info:
    wrapper_id: str
    wraps: str
    role: str
    module_path: str
    factory: Callable[[], object]


_REGISTRY: Dict[str, Wrapper_Info] = {}


def register(info: Wrapper_Info) -> None:
    if info.wrapper_id in _REGISTRY:
        raise RuntimeError(f"Duplicate wrapper_id: {info.wrapper_id}")
    _REGISTRY[info.wrapper_id] = info


def list_wrappers() -> Dict[str, Wrapper_Info]:
    return dict(_REGISTRY)


def get_wrapper(wrapper_id: str) -> object:
    if wrapper_id not in _REGISTRY:
        raise KeyError(f"Unknown wrapper_id: {wrapper_id}")
    return _REGISTRY[wrapper_id].factory()
