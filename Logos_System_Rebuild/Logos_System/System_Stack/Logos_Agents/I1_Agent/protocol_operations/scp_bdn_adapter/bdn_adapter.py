# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from __future__ import annotations

from typing import Protocol

from .bdn_types import BDNRequest, BDNResult


class IBDNAdapter(Protocol):
    """
    Contract for BDN integration.
    Real implementations may call external/SCP engines, but SCP should depend only on this interface.
    """
    def analyze(self, req: BDNRequest) -> BDNResult:
        ...


class StubBDNAdapter:
    """
    Safe default: indicates BDN is not wired yet.
    """
    def analyze(self, req: BDNRequest) -> BDNResult:
        return BDNResult(
            available=False,
            summary="BDN adapter not configured; returning stub result.",
            stability_score=0.0,
            meta={"reason": "stub"},
        )
