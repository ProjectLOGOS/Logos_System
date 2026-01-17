from __future__ import annotations

class LogosCoreError(Exception):
    """Base exception for Logos Agent core utilities."""

class SchemaError(LogosCoreError):
    """Raised when a packet/schema is malformed for expected use."""

class RoutingError(LogosCoreError):
    """Raised when routing directives are invalid or unsupported."""

class IntegrationError(LogosCoreError):
    """Raised when integrations are misconfigured or receive invalid inputs."""
