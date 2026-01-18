# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

from __future__ import annotations

class LogosCoreError(Exception):
    """Base exception for Logos Agent core utilities."""

class SchemaError(LogosCoreError):
    """Raised when a packet/schema is malformed for expected use."""

class RoutingError(LogosCoreError):
    """Raised when routing directives are invalid or unsupported."""
