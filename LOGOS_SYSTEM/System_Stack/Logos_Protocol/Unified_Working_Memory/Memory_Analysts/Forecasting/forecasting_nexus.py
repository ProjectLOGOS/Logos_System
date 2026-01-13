# forecasting_nexus.py
"""
ForecastingNexus: Toolkit-level orchestrator for the Forecasting toolkit.
Registering ARIMA, GARCH, Kalman, State-Space, and TS-Kalman models,
executing them in sequence/parallel and providing an ensemble forecast.
Place this in your forecasting/ directory alongside other modules.
"""
import traceback
import json

from Forecasting.arima_wrapper import fit_arima_model, forecast_arima
from Forecasting.garch_wrapper import fit_garch_model, forecast_garch
from Forecasting.kalman_filter import KalmanFilter
from Forecasting.state_space_utils import build_state_space_model
from Forecasting.ts_kalman_filter import TimeSeriesKalman

class ForecastingNexus:
    """Orchestrates multiple forecasting models and aggregates their results."""
    def __init__(self):
        pass

    def run_arima(self, series, order=(1,1,1), steps=5):
        try:
            model = fit_arima_model(series, order=order)
            fc = forecast_arima(model, steps=steps)
            return {'output': list(fc), 'error': None}
        except Exception:
            return {'output': None, 'error': traceback.format_exc()}

    def run_garch(self, series, p=1, q=1, horizon=5):
        try:
            model = fit_garch_model(series, p=p, q=q)
            var_fc = forecast_garch(model, horizon=horizon)
            return {'output': list(var_fc), 'error': None}
        except Exception:
            return {'output': None, 'error': traceback.format_exc()}

    def run_kalman(self, A, B, H, Q, R, x0, P0, observations=None):
        try:
            kf = KalmanFilter(A, B, H, Q, R, x0, P0)
            if observations is not None:
                for z in observations:
                    kf.predict()
                    kf.update([[z]])
            else:
                kf.predict()
            state = kf.current_state()
            # state = (x, P)
            return {'output': {'state': (state[0].tolist(), state[1].tolist())}, 'error': None}
        except Exception:
            return {'output': None, 'error': traceback.format_exc()}

    def run_state_space(self, n, process_var=1e-5, measurement_var=1e-1):
        try:
            A, B, H, Q, R, x0, P0 = build_state_space_model(n, process_var, measurement_var)
            return {'output': {
                'A': A.tolist(), 'B': B.tolist(), 'H': H.tolist(),
                'Q': Q.tolist(), 'R': R.tolist()
            }, 'error': None}
        except Exception:
            return {'output': None, 'error': traceback.format_exc()}

    def run_ts_kalman(self, observations, **kwargs):
        try:
            kf = TimeSeriesKalman(**kwargs) if kwargs else TimeSeriesKalman()
            means, covs = kf.fit(observations)
            return {'output': {'means': means.tolist(), 'covs': covs.tolist()}, 'error': None}
        except Exception:
            return {'output': None, 'error': traceback.format_exc()}

    def ensemble(self, outputs):
        # simple mean ensemble: average element-wise
        lists = [o for o in outputs if isinstance(o, (list, tuple))]
        if not lists:
            return None
        length = min(len(o) for o in lists)
        ens = []
        for i in range(length):
            vals = [o[i] for o in lists]
            ens.append(sum(vals) / len(vals))
        return ens

    def run_pipeline(self, series, horizon=5):
        report = []
        # ARIMA
        r1 = self.run_arima(series, steps=horizon)
        report.append({'stage': 'arima', **r1})
        # GARCH
        r2 = self.run_garch(series, horizon=horizon)
        report.append({'stage': 'garch', **r2})
        # State-space for Kalman
        n = len(series)
        r3 = self.run_state_space(n)
        report.append({'stage': 'state_space', **r3})
        # Kalman filter
        if r3['output']:
            A, B, H = r3['output']['A'], r3['output']['B'], r3['output']['H']
            Q = r3['output']['Q']; R = r3['output']['R']
            x0 = [0]*n; P0 = [[1]*n for _ in range(n)]
            r4 = self.run_kalman(A, B, H, Q, R, x0, P0, observations=series)
        else:
            r4 = {'stage': 'kalman', 'output': None, 'error': 'state_space failed'}
        report.append({'stage': 'kalman', **r4})
        # TS Kalman
        r5 = self.run_ts_kalman(series)
        report.append({'stage': 'ts_kalman', **r5})
        # Ensemble
        outs = [item['output'] for item in report if item['error'] is None]
        r6 = {'stage': 'ensemble', 'output': self.ensemble(outs), 'error': None}
        report.append(r6)
        return report

if __name__ == '__main__':
    import argparse, pprint
    parser = argparse.ArgumentParser(description='Forecasting Nexus')
    parser.add_argument('--series', nargs='+', type=float, required=True,
                        help='Time series values')
    parser.add_argument('--horizon', type=int, default=5, help='Forecast steps')
    args = parser.parse_args()

    nexus = ForecastingNexus()
    result = nexus.run_pipeline(args.series, horizon=args.horizon)
    pprint.pprint(result)
    with open('forecasting_nexus_report.json', 'w') as f:
        json.dump(result, f, indent=2)
