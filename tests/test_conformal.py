import numpy as np
import pytest
from src.conformal_engine import SplitConformalPredictor

class MockForecaster:
    def predict(self, X):
        return np.ones(len(X)) * 50.0

def test_conformal_coverage():
    X_cal = np.random.randn(200, 5)
    y_cal = np.random.normal(loc=50.0, scale=5.0, size=200)
    
    mock_model = MockForecaster()
    cp = SplitConformalPredictor(forecaster=mock_model, X_cal=X_cal, y_cal=y_cal)
    
    X_test = np.random.randn(500, 5)
    y_test = np.random.normal(loc=50.0, scale=5.0, size=500)
    
    coverage, avg_width = cp.evaluate_coverage(X_test, y_test, coverage=0.90)
    
    assert 84.0 <= coverage <= 96.0, f"Expected ~90% coverage, got {coverage}%"
    assert avg_width > 0, "Interval width must be positive"