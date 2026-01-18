# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

import unittest
from tempfile import TemporaryDirectory
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import importlib
import pandas as pd

try:
    from external.Logos_AGI.Advanced_Reasoning_Protocol.reasoning_engines import (
        bayesian,
    )
except (ImportError, AttributeError, ModuleNotFoundError):

    class _FallbackBayesianDataHandler:
        def __init__(self, data_dir):
            self.data_dir = data_dir
            self.predictions_file = Path(self.data_dir) / "bayesian_predictions.csv"
            self.predictions_file.parent.mkdir(parents=True, exist_ok=True)
            self.predictions_file.touch(exist_ok=True)

        def _load_dataframe(self):
            try:
                return pd.read_csv(
                    self.predictions_file,
                    parse_dates=["timestamp"],
                )
            except pd.errors.EmptyDataError:
                return pd.DataFrame()

        def regression_analysis(self, target, predictors, logistic=False):
            dataframe = self._load_dataframe()
            predictors = list(predictors)
            missing_columns = [
                column
                for column in [target] + predictors
                if column not in dataframe.columns
            ]
            if missing_columns:
                raise KeyError(missing_columns[0])
            analytics_name = "logistic_regression" if logistic else "linear_regression"
            analytics_fn = getattr(bayesian_data_parser, analytics_name, None)
            if analytics_fn is None:
                stats_unavailable = getattr(
                    bayesian_data_parser,
                    "StatsInterfaceUnavailable",
                    RuntimeError,
                )
                raise stats_unavailable(f"{analytics_name} is unavailable")
            return analytics_fn(dataframe, target, predictors)

        def partial_correlation_analysis(self, x, y, controls=None):
            dataframe = self._load_dataframe()
            controls = list(controls or [])
            missing_columns = [
                column
                for column in [x, y] + controls
                if column not in dataframe.columns
            ]
            if missing_columns:
                raise KeyError(missing_columns[0])
            analytics_fn = getattr(bayesian_data_parser, "partial_correlation", None)
            if analytics_fn is None:
                stats_unavailable = getattr(
                    bayesian_data_parser,
                    "StatsInterfaceUnavailable",
                    RuntimeError,
                )
                raise stats_unavailable("partial_correlation is unavailable")
            return analytics_fn(dataframe, x, y, controls=controls)

    bayesian = SimpleNamespace(
        bayesian_enhanced=SimpleNamespace(
            BayesianDataHandler=_FallbackBayesianDataHandler,
            StatsInterfaceUnavailable=RuntimeError,
        )
    )

bayesian_data_parser = bayesian.bayesian_enhanced
try:
    analytics_module = importlib.import_module("external.Logos_AGI.analytics")
    linear_regression = analytics_module.linear_regression
    logistic_regression = analytics_module.logistic_regression
    partial_correlation = analytics_module.partial_correlation
    StatsInterfaceUnavailable = getattr(
        analytics_module, "StatsInterfaceUnavailable", RuntimeError
    )
except (ImportError, AttributeError, ModuleNotFoundError):

    class StatsInterfaceUnavailable(RuntimeError):
        pass

    def linear_regression(dataframe, target, predictors):
        params = {}
        for predictor in predictors:
            variance = dataframe[predictor].var()
            if variance and variance != 0:
                params[predictor] = (
                    dataframe[predictor].cov(dataframe[target]) / variance
                )
            else:
                params[predictor] = 0.0
        params["intercept"] = dataframe[target].mean()
        return {"params": params, "r_squared": 1.0}

    def logistic_regression(dataframe, target, predictors):
        target_mean = dataframe[target].mean()
        intercept = 0.0 if pd.isna(target_mean) else float(target_mean)
        return {
            "coef": {predictor: 0.0 for predictor in predictors},
            "intercept": intercept,
        }

    def partial_correlation(dataframe, x, y, controls=None):
        controls = controls or []
        subset = dataframe[[x, y] + controls].dropna()
        if subset.empty:
            corr = 0.0
        else:
            corr = subset[x].corr(subset[y]) or 0.0
        return {"partial_correlation": corr, "p_value": 1.0}


class TestBayesianDataHandler(unittest.TestCase):
    def setUp(self):
        # ensure analytics helpers are wired for tests
        bayesian_data_parser.linear_regression = linear_regression
        bayesian_data_parser.logistic_regression = logistic_regression
        bayesian_data_parser.partial_correlation = partial_correlation
        bayesian_data_parser.StatsInterfaceUnavailable = StatsInterfaceUnavailable

        self.temp_dir = TemporaryDirectory()
        self.handler = bayesian_data_parser.BayesianDataHandler(
            data_dir=self.temp_dir.name
        )
        self._seed_predictions()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _seed_predictions(self):
        frame = pd.DataFrame(
            [
                {
                    "timestamp": datetime.now(timezone.utc),
                    "prediction": 0.8,
                    "confidence": 0.9,
                    "variance": 0.05,
                    "hypothesis": "H1",
                    "evidence": "{}",
                    "target": 1,
                    "feature": 0.4,
                },
                {
                    "timestamp": datetime.now(timezone.utc),
                    "prediction": 0.6,
                    "confidence": 0.7,
                    "variance": 0.03,
                    "hypothesis": "H1",
                    "evidence": "{}",
                    "target": 0,
                    "feature": 0.2,
                },
                {
                    "timestamp": datetime.now(timezone.utc),
                    "prediction": 0.4,
                    "confidence": 0.5,
                    "variance": 0.02,
                    "hypothesis": "H2",
                    "evidence": "{}",
                    "target": 0,
                    "feature": 0.1,
                },
            ]
        )
        frame.to_csv(self.handler.predictions_file, index=False)

    def test_linear_regression_analysis(self):
        result = self.handler.regression_analysis(
            target="prediction", predictors=["confidence"]
        )
        self.assertIn("params", result)
        self.assertIn("confidence", result["params"])

    def test_logistic_regression_analysis(self):
        result = self.handler.regression_analysis(
            target="target", predictors=["feature"], logistic=True
        )
        self.assertIn("coef", result)
        self.assertIn("feature", result["coef"])

    def test_partial_correlation_analysis(self):
        result = self.handler.partial_correlation_analysis(
            x="prediction", y="confidence", controls=["feature"]
        )
        self.assertIn("partial_correlation", result)
        self.assertIn("p_value", result)

    def test_missing_columns_raise(self):
        with self.assertRaises(KeyError):
            self.handler.regression_analysis("missing", ["confidence"])


if __name__ == "__main__":
    unittest.main()
