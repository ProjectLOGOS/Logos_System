# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

# MODULE_META:
#   module_id: BAYESIAN_DATA_PARSER
#   layer: APPLICATION_FUNCTION
#   role: Bayesian engine data parser
#   phase_origin: PHASE_SCOPING_STUB
#   description: Stub metadata for Bayesian engine data parser (header placeholder).
#   contracts: []
#   allowed_imports: []
#   prohibited_behaviors: [IO, NETWORK, TIME, RANDOM]
#   entrypoints: [run]
#   callable_surface: APPLICATION
#   state_mutation: NONE
#   runtime_spine_binding: NONE
#   depends_on_contexts: []
#   invoked_by: []

"""
bayesian_data_parser.py

Handles loading/saving of Bayesian prediction data.
"""

import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Iterable

import pandas as pd

try:
    analytics_settings = importlib.import_module("Logos_AGI.settings.analytics")
except ImportError:
    class _FallbackAnalyticsSettings:
        ENABLE_ANALYTICS = False

    analytics_settings = _FallbackAnalyticsSettings()

_analytics_enabled = getattr(analytics_settings, "ENABLE_ANALYTICS", False)


class _FallbackStatsInterfaceUnavailable(RuntimeError):
    """Raised when analytics helpers are disabled via settings."""


StatsInterfaceUnavailable = _FallbackStatsInterfaceUnavailable


def _disabled(*_: object, **__: object) -> Dict:
    raise StatsInterfaceUnavailable(
        "Analytics integration disabled via settings."
    )


linear_regression = _disabled  # type: ignore[assignment]
logistic_regression = _disabled  # type: ignore[assignment]
partial_correlation = _disabled  # type: ignore[assignment]

if _analytics_enabled:
    try:
        analytics_module = importlib.import_module("Logos_AGI.analytics")
    except ImportError:
        _analytics_enabled = False
    else:
        StatsInterfaceUnavailable = getattr(
            analytics_module, "StatsInterfaceUnavailable", StatsInterfaceUnavailable
        )
        linear_regression = getattr(
            analytics_module, "linear_regression", linear_regression
        )
        logistic_regression = getattr(
            analytics_module, "logistic_regression", logistic_regression
        )
        partial_correlation = getattr(
            analytics_module, "partial_correlation", partial_correlation
        )


class BayesianDataHandler:
    def __init__(self, data_dir: str = "data/bayesian_ml"):
        self.data_dir = Path(data_dir)
        self.predictions_file = self.data_dir / "predictions.csv"
        self.metadata_file = self.data_dir / "metadata.json"
        self._init_storage()

    def _init_storage(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.predictions_file.exists():
            pd.DataFrame(
                columns=[
                    "timestamp",
                    "prediction",
                    "confidence",
                    "variance",
                    "hypothesis",
                    "evidence",
                ]
            ).to_csv(self.predictions_file, index=False)
        if not self.metadata_file.exists():
            meta = {
                "model_version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "performance_metrics": {},
                "model_parameters": {},
            }
            self.save_metadata(meta)

    def save_prediction(self, prediction, hypothesis: str) -> None:
        row = {
            "timestamp": prediction.timestamp,
            "prediction": prediction.prediction,
            "confidence": prediction.confidence,
            "variance": prediction.variance,
            "hypothesis": hypothesis,
            "evidence": json.dumps(prediction.metadata["evidence"]),
        }
        pd.DataFrame([row]).to_csv(
            self.predictions_file, mode="a", header=False, index=False
        )

    def get_predictions(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_confidence: float = 0.0,
    ):
        df = pd.read_csv(self.predictions_file, parse_dates=["timestamp"])
        if start_date:
            df = df[df.timestamp >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.timestamp <= pd.to_datetime(end_date)]
        if min_confidence > 0:
            df = df[df.confidence >= min_confidence]
        return df

    def save_metadata(self, metadata: Dict) -> None:
        metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def get_metadata(self) -> Dict:
        with open(self.metadata_file, encoding="utf-8") as f:
            return json.load(f)

    def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        df = pd.read_csv(self.predictions_file, parse_dates=["timestamp"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_to_keep)
        df = df[df.timestamp >= cutoff]
        df.to_csv(self.predictions_file, index=False)

    def regression_analysis(
        self,
        target: str,
        predictors: Sequence[str],
        *,
        logistic: bool = False,
        add_constant: bool = True,
        dropna: bool = True,
    ) -> Dict:
        """Run linear or logistic regression over stored prediction history."""

        columns: Iterable[str] = (target, *predictors)
        frame = self._prepare_dataframe(columns, dropna=dropna)

        if logistic and frame[target].dtype == bool:
            frame = frame.copy()
            frame[target] = frame[target].astype(int)

        try:
            if logistic:
                return logistic_regression(
                    frame,
                    target,
                    predictors,
                    add_constant=add_constant,
                    dropna=dropna,
                )
            return linear_regression(
                frame,
                target,
                predictors,
                add_constant=add_constant,
                dropna=dropna,
            )
        except StatsInterfaceUnavailable as exc:
            raise StatsInterfaceUnavailable(
                "statsmodels or pandas not available for regression analysis"
            ) from exc

    def partial_correlation_analysis(
        self,
        x: str,
        y: str,
        controls: Sequence[str],
        *,
        dropna: bool = True,
    ) -> Dict:
        """Compute partial correlation for stored predictions."""

        columns: Iterable[str] = (x, y, *controls)
        frame = self._prepare_dataframe(columns, dropna=dropna)

        try:
            return partial_correlation(
                frame,
                x,
                y,
                controls,
                dropna=dropna,
            )
        except StatsInterfaceUnavailable as exc:
            raise StatsInterfaceUnavailable(
                "statsmodels or pandas not available for correlation analysis"
            ) from exc

    def _prepare_dataframe(
        self,
        columns: Iterable[str],
        *,
        dropna: bool,
    ) -> pd.DataFrame:
        frame = self.get_predictions()
        if frame.empty:
            raise ValueError("No predictions available for analysis")

        column_list = list(columns)
        missing = [col for col in column_list if col not in frame.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        if dropna:
            frame = frame.dropna(subset=column_list)
            if frame.empty:
                raise ValueError("All rows dropped during NA filtering")
        elif frame[column_list].isna().any().any():
            raise ValueError("NaN values present; rerun with dropna=True")

        return frame
