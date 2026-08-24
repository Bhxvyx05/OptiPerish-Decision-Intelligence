# src/conformal_engine.py
"""
Split Conformal Predictor (Inductive Conformal).

Usage:
  cp = SplitConformalPredictor(fitted_forecaster, X_cal, y_cal)
  lower, yhat, upper = cp.predict_interval(X_new, coverage=0.90)
  coverage, avg_width = cp.evaluate_coverage(X_test, y_test, coverage=0.90)
"""
import numpy as np
from typing import Tuple


class SplitConformalPredictor:
    def __init__(self, forecaster, X_cal, y_cal, eps=1e-9):
        """
        forecaster: fitted model with .predict(X) -> array
        X_cal, y_cal: calibration features and targets (arrays or DataFrames)
        """
        self.forecaster = forecaster
        self.X_cal = X_cal
        self.y_cal = np.array(y_cal)
        self.eps = eps
        self._compute_residuals()

    def _compute_residuals(self):
        yhat_cal = np.array(self.forecaster.predict(self._ensure_array(self.X_cal)))
        self.residuals = np.abs(self.y_cal - yhat_cal)

    @staticmethod
    def _ensure_array(X):
        # Accept DataFrame or numpy array
        if hasattr(X, "values"):
            return X.values
        return np.array(X)

    def _quantile_q(self, coverage: float):
        alpha = 1.0 - coverage
        n = len(self.residuals)
        if n == 0:
            return 0.0
        k = int(np.ceil((n + 1) * (1 - alpha)))
        k = max(1, min(k, n))
        q = np.sort(self.residuals)[k - 1]
        return float(q)

    def predict_interval(self, X_new, coverage: float = 0.90) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_arr = self._ensure_array(X_new)
        yhat = np.array(self.forecaster.predict(X_arr))
        q = self._quantile_q(coverage)
        lower = np.maximum(0.0, yhat - q)
        upper = yhat + q
        return lower, yhat, upper

    def evaluate_coverage(self, X_test, y_test, coverage: float = 0.90) -> Tuple[float, float]:
        lower, yhat, upper = self.predict_interval(X_test, coverage=coverage)
        y_test = np.array(y_test)
        inside = (y_test >= lower - self.eps) & (y_test <= upper + self.eps)
        coverage_empirical = float(np.mean(inside)) * 100.0
        avg_width = float(np.mean(upper - lower))
        return coverage_empirical, avg_width
