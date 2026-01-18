# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

from __future__ import annotations

from typing import Protocol

from .mvs_types import MVSRequest, MVSResult


class IMVSAdapter(Protocol):
    """
    Contract for MVS integration.
    Real implementations may call external/ARP engines, but SCP should depend only on this interface.
    """
    def analyze(self, req: MVSRequest) -> MVSResult:
        ...


class StubMVSAdapter:
    """
    Safe default: indicates MVS is not wired yet.
    """
    def analyze(self, req: MVSRequest) -> MVSResult:
        return MVSResult(
            available=False,
            summary="MVS adapter not configured; returning stub result.",
            coherence_score=0.0,
            meta={"reason": "stub"},
        )
