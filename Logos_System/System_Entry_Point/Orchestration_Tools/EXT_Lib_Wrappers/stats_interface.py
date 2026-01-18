# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""Statsmodels convenience layer with pandas-friendly outputs.

The LOGOS stack uses pandas extensively for reporting. This module centralises
statsmodels access so protocols can ask for regression or correlation measures
without duplicating boilerplate or failing when statsmodels is absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

try:
    from LOGOS_AGI.settings import analytics as analytics_settings
except ImportError:  # pragma: no cover - fallback when running inside package
    from ....Runtime_Operations.Monitoring import analytics as analytics_settings  # type: ignore

try:  # pandas is required for the helper interface
    import pandas as pd
except ImportError:  # pragma: no cover - pandas intentionally optional
    pd = None  # type: ignore

if analytics_settings.ENABLE_STATS_MODELS:
    try:
        import statsmodels.api as sm
    except ImportError:  # pragma: no cover - statsmodels optional at runtime
        sm = None  # type: ignore
else:  # pragma: no cover - optional branch mirrors opt-out behaviour
    sm = None  # type: ignore


class StatsInterfaceUnavailable(RuntimeError):
    """Raised when statsmodels and pandas are not both available."""


def is_stats_available() -> bool:
    """Return True when pandas and statsmodels can be imported."""

    if not analytics_settings.ENABLE_ANALYTICS:
        return False

    return pd is not None and sm is not None


def ensure_dataframe(data: Any) -> "pd.DataFrame":
    """Coerce arbitrary tabular input to a pandas DataFrame."""

    if pd is None:
        raise StatsInterfaceUnavailable("pandas is required for analytics helpers")

    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, Mapping):
        return pd.DataFrame(data)

    if isinstance(data, Sequence):
        return pd.DataFrame(data)

    raise TypeError("Unsupported data container for analytics helpers")


def _require_stats() -> None:
    if not analytics_settings.ENABLE_ANALYTICS:
        raise StatsInterfaceUnavailable("analytics helpers disabled via settings")

    if not is_stats_available():
        raise StatsInterfaceUnavailable("statsmodels+pandas stack unavailable")


def _coerce_predictors(columns: Iterable[str]) -> List[str]:
    preds = [str(col) for col in columns]
    if not preds:
        raise ValueError("At least one predictor is required")
    return preds


def _add_constant(frame: "pd.DataFrame", add_constant: bool) -> "pd.DataFrame":
    if not add_constant:
        return frame
    # has_constant="add" ensures an intercept column even if one exists
    return sm.add_constant(frame, has_constant="add")


def linear_regression(
    data: Any,
    target: str,
    predictors: Iterable[str],
    *,
    add_constant: bool = True,
    dropna: bool = True,
) -> Dict[str, Any]:
    """Run an ordinary least squares regression and return serialisable stats."""

    _require_stats()
    frame = ensure_dataframe(data)
    columns = _coerce_predictors(predictors)

    subset_cols = [target, *columns]
    subset = frame[subset_cols]
    if dropna:
        subset = subset.dropna()

    X = _add_constant(subset[columns], add_constant)
    y = subset[target]

    model = sm.OLS(y, X, missing="drop").fit()

    return {
        "target": target,
        "predictors": columns,
        "coefficients": {k: float(v) for k, v in model.params.items()},
        "p_values": {k: float(v) for k, v in model.pvalues.items()},
        "r_squared": float(model.rsquared),
        "r_squared_adj": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "f_statistic": float(model.fvalue) if model.fvalue is not None else None,
        "f_p_value": float(model.f_pvalue) if model.f_pvalue is not None else None,
        "residual_std_error": float(getattr(model, "mse_resid", 0) ** 0.5),
        "n_obs": int(model.nobs),
        "summary": model.summary().as_text(),
    }


def logistic_regression(
    data: Any,
    target: str,
    predictors: Iterable[str],
    *,
    add_constant: bool = True,
    dropna: bool = True,
    method: str = "lbfgs",
    maxiter: int = 128,
) -> Dict[str, Any]:
    """Run a logistic regression using statsmodels.Logit."""

    _require_stats()
    frame = ensure_dataframe(data)
    columns = _coerce_predictors(predictors)

    subset_cols = [target, *columns]
    subset = frame[subset_cols]
    if dropna:
        subset = subset.dropna()

    X = _add_constant(subset[columns], add_constant)
    y = subset[target]

    model = sm.Logit(y, X)
    result = model.fit(disp=False, method=method, maxiter=maxiter)

    return {
        "target": target,
        "predictors": columns,
        "coefficients": {k: float(v) for k, v in result.params.items()},
        "p_values": {k: float(v) for k, v in result.pvalues.items()},
        "llf": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "pseudo_r_squared": float(result.prsquared),
        "n_obs": int(result.nobs),
        "converged": bool(result.mle_retvals.get("converged", True)),
        "summary": result.summary().as_text(),
    }


def partial_correlation(
    data: Any,
    x: str,
    y: str,
    controls: Iterable[str],
    *,
    dropna: bool = True,
) -> Dict[str, Any]:
    """Compute the partial correlation of x and y controlling for given variables."""

    _require_stats()
    frame = ensure_dataframe(data)
    control_cols = _coerce_predictors(controls)

    subset_cols = [x, y, *control_cols]
    subset = frame[subset_cols]
    if dropna:
        subset = subset.dropna()

    if subset.empty:
        raise ValueError("No rows available after dropping NaN values")

    def residuals(target: str) -> "pd.Series":
        predictors = _add_constant(subset[control_cols], add_constant=True)
        fit = sm.OLS(subset[target], predictors).fit()
        resid = subset[target] - fit.fittedvalues
        resid.name = target
        return resid

    resid_x = residuals(x)
    resid_y = residuals(y)
    correlation = float(resid_x.corr(resid_y))

    regression_frame = pd.DataFrame({x: resid_x, y: resid_y})
    predictors = _add_constant(regression_frame[[x]], True)
    result = sm.OLS(regression_frame[y], predictors).fit()

    return {
        "x": x,
        "y": y,
        "controls": control_cols,
        "partial_correlation": correlation,
        "beta": float(result.params[x]),
        "p_value": float(result.pvalues[x]),
        "r_squared": float(result.rsquared),
        "n_obs": int(result.nobs),
    }


@dataclass
class StatsCapabilities:
    pandas: bool
    statsmodels: bool

    @property
    def available(self) -> bool:
        return self.pandas and self.statsmodels


def capabilities() -> StatsCapabilities:
    """Expose current analytics dependency availability."""

    return StatsCapabilities(pandas=pd is not None, statsmodels=sm is not None)
