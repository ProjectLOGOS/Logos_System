# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""Analytics feature toggles for the LOGOS stack."""
from __future__ import annotations

import os
from functools import lru_cache

# Environment variable names
_ANALYTICS_FLAG = "LOGOS_ENABLE_ANALYTICS"
_STATSMODELS_FLAG = "LOGOS_ENABLE_STATSMODELS"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    return value in {"1", "true", "yes", "on"}


@lru_cache(maxsize=None)
def analytics_enabled() -> bool:
    """Return True when analytics features are globally enabled."""

    return _env_flag(_ANALYTICS_FLAG, True)


@lru_cache(maxsize=None)
def statsmodels_enabled() -> bool:
    """Return True when statsmodels-dependent helpers are permitted."""

    return _env_flag(_STATSMODELS_FLAG, analytics_enabled())


ENABLE_ANALYTICS = analytics_enabled()
ENABLE_STATS_MODELS = statsmodels_enabled()
