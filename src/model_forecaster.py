# src/model_forecaster.py
"""
Model forecasters:
 - SeasonalNaiveForecaster: predicts t-7 value grouped by store_id & item_id
 - LightGBMForecaster: LGBMRegressor wrapper with Optuna tuning

Provides:
 - fit, predict
 - tune (Optuna)
 - evaluation metrics: RMSE, MAE, MAPE, WAPE
 - feature importance extraction
"""
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import optuna
import joblib

# -------------------------
# Metrics
# -------------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-9))) * 100.0) if mask.any() else np.nan


def wape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-9) * 100.0)


# -------------------------
# Seasonal Naive Forecaster
# -------------------------
class SeasonalNaiveForecaster(BaseEstimator):
    """
    Predicts the value from t-7 (same weekday last week) grouped by store_id & item_id.
    """

    def __init__(self, season_lag: int = 7):
        self.season_lag = season_lag
        self.train_df = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # store the training series for lookup
        df = X.copy()
        df = df.reset_index(drop=True)
        df["sales_units"] = y.values if y is not None else np.nan
        self.train_df = df
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # For each row, find the value at date - season_lag for same store/item if available
        X = X.copy()
        X["date_lag"] = X["date"] - pd.to_timedelta(self.season_lag, unit="D")
        # create lookup from (store,item,date) -> sales_units in training data
        if self.train_df is None:
            raise ValueError("Model not fitted.")
        lookup = self.train_df.set_index(["store_id", "item_id", "date"])["sales_units"].to_dict()
        preds = []
        for _, row in X.iterrows():
            key = (row["store_id"], row["item_id"], row["date_lag"])
            val = lookup.get(key, np.nan)
            # fallback: use group mean if missing
            if np.isnan(val):
                group = self.train_df[(self.train_df["store_id"] == row["store_id"]) & (self.train_df["item_id"] == row["item_id"])]
                val = group["sales_units"].mean() if not group.empty else 0.0
            preds.append(float(val))
        return np.array(preds)


# -------------------------
# LightGBM Forecaster
# -------------------------
class LightGBMForecaster:
    """
    Wrapper around LightGBM with Optuna hyperparameter tuning.
    Uses LGBMRegressor with Huber or Poisson objective for positive counts.
    """

    def __init__(self, objective: str = "huber", random_state: int = 42):
        self.objective = objective
        self.random_state = random_state
        self.model = None
        self.feature_names = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None, params: Optional[Dict[str, Any]] = None, num_boost_round: int = 500):
        self.feature_names = feature_names
        params = params or {
            "objective": self.objective,
            "random_state": self.random_state,
            "verbosity": -1,
            "metric": "rmse",
        }
        self.model = lgb.LGBMRegressor(**params, n_estimators=num_boost_round)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained.")
        preds = self.model.predict(X)
        # ensure non-negative
        preds = np.maximum(0.0, preds)
        return preds

    def tune_with_optuna(self, X_train, y_train, X_val, y_val, n_trials: int = 30, timeout: Optional[int] = None):
        def objective(trial):
            param = {
                "objective": self.objective,
                "random_state": self.random_state,
                "verbosity": -1,
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.2),
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "feature_fraction": trial.suggest_uniform("feature_fraction", 0.5, 1.0),
                "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
            }
            model = lgb.LGBMRegressor(**param, n_estimators=500)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
            preds = model.predict(X_val)
            return rmse(y_val, preds)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        best_params = study.best_trial.params
        # finalize model with best params
        final_params = {
            "objective": self.objective,
            "random_state": self.random_state,
            "verbosity": -1,
            **best_params,
        }
        self.model = lgb.LGBMRegressor(**final_params, n_estimators=1000)
        self.model.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
        return study

    def feature_importances(self, top_k: int = 30):
        if self.model is None:
            return None
        fi = getattr(self.model, "feature_importances_", None)
        if fi is None:
            return None
        names = self.feature_names or [f"f{i}" for i in range(len(fi))]
        df = pd.DataFrame({"feature": names, "importance": fi})
        return df.sort_values("importance", ascending=False).head(top_k)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
        return self
