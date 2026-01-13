"""Session identity helpers for LOGOS-GPT FastAPI surface."""

from __future__ import annotations

import os
import uuid
from typing import Optional

SESSION_COOKIE_NAME = "logos_session"
SESSION_SECRET_ENV = "LOGOS_SESSION_SECRET"
SESSION_COOKIE_MAX_AGE = int(os.getenv("LOGOS_SESSION_COOKIE_MAX_AGE", "86400"))
SESSION_COOKIE_SECURE = os.getenv("LOGOS_SESSION_COOKIE_SECURE", "0") == "1"


def _secret() -> bytes:
    env_secret = os.getenv(SESSION_SECRET_ENV)
    if env_secret:
        return env_secret.encode("utf-8")
    cache = getattr(_secret, "_cache", None)
    if cache is None:
        cache = uuid.uuid4().hex.encode("ascii")
        setattr(_secret, "_cache", cache)
    return cache


def generate_session_id() -> str:
    return str(uuid.uuid4())


def is_valid_session_id(value: Optional[str]) -> bool:
    if not value:
        return False
    try:
        candidate = uuid.UUID(str(value), version=4)
    except (ValueError, TypeError, AttributeError):
        return False
    return str(candidate) == str(value)