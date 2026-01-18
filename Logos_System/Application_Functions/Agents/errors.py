# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

from __future__ import annotations

class LogosCoreError(Exception):
    """Base exception for Logos Agent core utilities."""

class SchemaError(LogosCoreError):
    """Raised when a packet/schema is malformed for expected use."""

class RoutingError(LogosCoreError):
    """Raised when routing directives are invalid or unsupported."""
