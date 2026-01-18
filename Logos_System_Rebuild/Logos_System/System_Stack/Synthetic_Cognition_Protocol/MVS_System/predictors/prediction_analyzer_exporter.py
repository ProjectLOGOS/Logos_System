# HEADER_TYPE: CANONICAL_REBUILD_MODULE
# EXECUTION: CONTROLLED
# AUTHORITY: GOVERNED
# ORIGIN: SYSTEMATIC_REWRITE

"""
prediction_analyzer_exporter.py

"""
import argparse
import json
import uuid
from typing import Iterable, Sequence
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd

from LOGOS_AGI.settings import analytics as analytics_settings

if analytics_settings.ENABLE_ANALYTICS:
    from LOGOS_AGI.analytics import (
        StatsInterfaceUnavailable,
        capabilities,
        linear_regression,
        logistic_regression,
        partial_correlation,
    )
else:
    class StatsInterfaceUnavailable(RuntimeError):
        """Raised when analytics helpers are disabled via settings."""

    def _disabled(*_: object, **__: object) -> None:
        raise StatsInterfaceUnavailable("Analytics integration disabled via settings.")

    def capabilities():  # type: ignore[override]
        return SimpleNamespace(pandas=False, statsmodels=False, available=False)

    def linear_regression(*args, **kwargs):  # type: ignore[override]
        return _disabled(*args, **kwargs)

    def logistic_regression(*args, **kwargs):  # type: ignore[override]
        return _disabled(*args, **kwargs)

    def partial_correlation(*args, **kwargs):  # type: ignore[override]
        return _disabled(*args, **kwargs)

def load_predictions(path="prediction_log.jsonl"):
    """Load all prediction logs from a JSONL file."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

def summarize(preds):
    df = pd.DataFrame(preds)
    print(f"\nLoaded {len(df)} predictions.")
    print("Modal Counts:\n", df['modal_status'].value_counts())
    print(f"Average Coherence: {df['coherence'].mean():.3f}")
    return df

def plot_coherence(df):
    plt.figure()
    plt.hist(df['coherence'], bins=20)
    plt.title("Coherence Distribution")
    plt.xlabel("Coherence"); plt.ylabel("Count")
    plt.show()

def filter_predictions(df, modal=None, min_coherence=None):
    r = df.copy()
    if modal:          r = r[r['modal_status']==modal]
    if min_coherence:  r = r[r['coherence']>=min_coherence]
    return r

def export_predictions(df, out_file="filtered_predictions.csv", fmt="csv"):
    if fmt=="json":
        df.to_json(out_file, orient="records", indent=2)
    else:
        df.to_csv(out_file, index=False)
    print(f"[✔] Exported {len(df)} rows to {out_file}")


def _print_regression(result: dict, *, title: str) -> None:
    print(f"\n{title}")
    print("- Target:", result.get("target"))
    print("- Predictors:", ", ".join(result.get("predictors", [])))
    for key in ("coefficients", "p_values"):
        payload = result.get(key)
        if payload:
            print(f"  {key}:")
            for name, value in payload.items():
                print(f"    {name}: {value:.6f}")
    for metric in ("r_squared", "r_squared_adj", "pseudo_r_squared", "aic", "bic"):
        if metric in result and result[metric] is not None:
            print(f"- {metric}: {result[metric]:.6f}")
    if "converged" in result:
        print("- converged:", result["converged"])
    print("- n_obs:", result.get("n_obs"))


def run_ols(df: pd.DataFrame, target: str, predictors: Sequence[str]) -> None:
    if not analytics_settings.ENABLE_ANALYTICS:
        print("[!] Analytics disabled via settings; enable LOGOS_ENABLE_ANALYTICS to run OLS.")
        return
    try:
        result = linear_regression(df, target, predictors)
    except StatsInterfaceUnavailable:
        caps = capabilities()
        print(
            "[!] statsmodels unavailable: pandas="
            f"{caps.pandas} statsmodels={caps.statsmodels}"
        )
        return
    except Exception as exc:
        print(f"[!] OLS regression failed: {exc}")
        return
    _print_regression(result, title="OLS Regression Summary")


def run_logit(df: pd.DataFrame, target: str, predictors: Sequence[str]) -> None:
    if not analytics_settings.ENABLE_ANALYTICS:
        print("[!] Analytics disabled via settings; enable LOGOS_ENABLE_ANALYTICS to run logistic regression.")
        return
    try:
        result = logistic_regression(df, target, predictors)
    except StatsInterfaceUnavailable:
        caps = capabilities()
        print(
            "[!] statsmodels unavailable: pandas="
            f"{caps.pandas} statsmodels={caps.statsmodels}"
        )
        return
    except Exception as exc:
        print(f"[!] Logistic regression failed: {exc}")
        return
    _print_regression(result, title="Logistic Regression Summary")


def run_partial(df: pd.DataFrame, x: str, y: str, controls: Iterable[str]) -> None:
    if not analytics_settings.ENABLE_ANALYTICS:
        print("[!] Analytics disabled via settings; enable LOGOS_ENABLE_ANALYTICS to run partial correlation.")
        return
    try:
        result = partial_correlation(df, x, y, controls)
    except StatsInterfaceUnavailable:
        caps = capabilities()
        print(
            "[!] statsmodels unavailable: pandas="
            f"{caps.pandas} statsmodels={caps.statsmodels}"
        )
        return
    except Exception as exc:
        print(f"[!] Partial correlation failed: {exc}")
        return
    print("\nPartial Correlation Analysis")
    print("- x:", result.get("x"))
    print("- y:", result.get("y"))
    print("- controls:", ", ".join(result.get("controls", [])))
    print("- partial_correlation:", result.get("partial_correlation"))
    print("- beta:", result.get("beta"))
    print("- p_value:", result.get("p_value"))
    print("- r_squared:", result.get("r_squared"))
    print("- n_obs:", result.get("n_obs"))

class FractalKnowledgeStore:
    """Simple JSONL-backed knowledge store for THŌNOC."""
    def __init__(self, config: dict):
        self.path = config.get("storage_path", "knowledge_store.jsonl")
    def store_node(self, **kwargs) -> str:
        node_id = kwargs.get("query_id", str(uuid.uuid4()))
        with open(self.path, "a") as f:
            f.write(json.dumps({"id":node_id, **kwargs}) + "\n")
        return node_id
    def get_node(self, node_id: str):
        try:
            with open(self.path) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("id")==node_id:
                        return rec
        except FileNotFoundError:
            return None
        return None

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="prediction_log.jsonl")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--hist", action="store_true")
    parser.add_argument("--modal", choices=["necessary","actual","possible","impossible"])
    parser.add_argument("--min-coh", type=float)
    parser.add_argument("--export", choices=["csv","json"])
    parser.add_argument("--ols-target")
    parser.add_argument("--ols-predictors", nargs="+")
    parser.add_argument("--logit-target")
    parser.add_argument("--logit-predictors", nargs="+")
    parser.add_argument("--partial-x")
    parser.add_argument("--partial-y")
    parser.add_argument("--partial-controls", nargs="+")
    args = parser.parse_args()

    preds = load_predictions(args.file)
    df = summarize(preds) if args.summary else pd.DataFrame(preds)
    if args.hist:           plot_coherence(df)
    df2 = filter_predictions(df, args.modal, args.min_coh)
    if args.export:         export_predictions(df2, fmt=args.export)
    if args.ols_target and args.ols_predictors:
        run_ols(df2, args.ols_target, args.ols_predictors)
    if args.logit_target and args.logit_predictors:
        run_logit(df2, args.logit_target, args.logit_predictors)
    if args.partial_x and args.partial_y and args.partial_controls:
        run_partial(df2, args.partial_x, args.partial_y, args.partial_controls)
