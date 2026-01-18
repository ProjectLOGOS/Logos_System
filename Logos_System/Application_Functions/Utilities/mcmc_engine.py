# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""
core_processing/mcmc_engine.py

Probabilistic sampling support for the UIP protocol. Wraps PyMC when it is
available; otherwise falls back to deterministic placeholder traces so the
pipeline can continue operating in reduced capability mode.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - heavy optional dependency
    import pymc as pm

    PYMC_AVAILABLE = True
except ImportError:  # pragma: no cover - executed in minimal environments
    PYMC_AVAILABLE = False

    class _DummyTrace:
        def __init__(self) -> None:
            self.posterior = {"mu": np.array([0.5]), "sigma": np.array([1.0])}

    class _DummyModel:
        def __enter__(self) -> "_DummyModel":
            return self

        def __exit__(self, *exc: Any) -> None:  # noqa: D401 - context manager signature
            return None

    class _DummyPM:
        Model = _DummyModel

        @staticmethod
        def Normal(*args: Any, **kwargs: Any) -> None:  # pragma: no cover
            return None

        @staticmethod
        def sample(*args: Any, **kwargs: Any) -> _DummyTrace:  # pragma: no cover
            return _DummyTrace()

    pm = _DummyPM()  # type: ignore


def run_mcmc_model(
    model_definition: Callable[[], Any],
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 2,
    cores: int = 1,
) -> Any:
    """Execute an MCMC model definition.

    Parameters mirror :func:`pymc.sample`. When PyMC is absent or sampling
    fails, a neutral placeholder trace is returned instead of raising so that
    downstream consumers can detect reduced capability mode.
    """

    LOGGER.info("Starting MCMC execution (PyMC available=%s)", PYMC_AVAILABLE)

    try:
        with model_definition() as model:  # type: ignore[assignment]
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                return_inferencedata=True,
            )
        LOGGER.info("MCMC execution complete")
        return trace
    except Exception as exc:  # pragma: no cover - defensive path
        if PYMC_AVAILABLE:
            LOGGER.exception("MCMC execution failed with PyMC available")
            raise
        LOGGER.warning("PyMC unavailable; returning fallback trace: %s", exc)
        return pm.sample()  # type: ignore[call-arg]


def example_model() -> Callable[[], Any]:
    """Return a callable that builds a simple normal model when invoked."""

    def _model():
        with pm.Model() as model:  # type: ignore[attr-defined]
            pm.Normal("mu", 0, 1)
            pm.Normal("obs", "mu", 1, observed=np.random.randn(100))
        return model

    return _model


__all__ = ["run_mcmc_model", "example_model"]
