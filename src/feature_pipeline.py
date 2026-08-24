# src/feature_pipeline.py
"""
Feature pipeline for time-series forecasting.

Reads: data/raw/grocery_sales.csv
Produces: processed DataFrame with features and chronological splits.

Functions:
 - build_features(df)
 - chronological_split(df, train_frac=0.7, cal_frac=0.15)
 - fit_transformers(X_train, categorical_cols)
 - transform_with_fitted(X, transformers)
 - get_splits_and_transformers(csv_path)
"""
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "grocery_sales.csv"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # ensure datetime
    df["date"] = pd.to_datetime(df["date"])
    # calendar features
    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # cyclical encodings
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    # sort for group operations
    df = df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    # Lag features grouped by store_id & item_id
    lag_cols = [1, 2, 7, 14, 21, 28]
    for lag in lag_cols:
        df[f"lag_{lag}"] = df.groupby(["store_id", "item_id"])["sales_units"].shift(lag)

    # Rolling window stats (shift by 1 to avoid leakage)
    df["sales_shift_1"] = df.groupby(["store_id", "item_id"])["sales_units"].shift(1)
    df["rolling_mean_7"] = df.groupby(["store_id", "item_id"])["sales_shift_1"].rolling(window=7, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    df["rolling_std_7"] = df.groupby(["store_id", "item_id"])["sales_shift_1"].rolling(window=7, min_periods=1).std().reset_index(level=[0,1], drop=True).fillna(0.0)
    df["rolling_mean_14"] = df.groupby(["store_id", "item_id"])["sales_shift_1"].rolling(window=14, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    df["rolling_std_14"] = df.groupby(["store_id", "item_id"])["sales_shift_1"].rolling(window=14, min_periods=1).std().reset_index(level=[0,1], drop=True).fillna(0.0)
    df["rolling_mean_30"] = df.groupby(["store_id", "item_id"])["sales_shift_1"].rolling(window=30, min_periods=1).mean().reset_index(level=[0,1], drop=True)

    # ratio feature
    df["ratio_mean7_mean30"] = df["rolling_mean_7"] / (df["rolling_mean_30"] + 1e-5)

    # Promotional features
    # promo_in_next_3days is forward-looking (planned promotions)
    df["promo_in_next_3days"] = df.groupby(["store_id", "item_id"])["on_promotion"].shift(-1).fillna(0).astype(int)
    df["promo_in_next_3days"] = df["promo_in_next_3days"] | df.groupby(["store_id", "item_id"])["on_promotion"].shift(-2).fillna(0).astype(int)
    df["promo_in_next_3days"] = df["promo_in_next_3days"].astype(int)

    # Drop helper column
    df = df.drop(columns=["sales_shift_1"])

    # Drop rows with NaNs created by lags (these are unavoidable)
    df = df.dropna().reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, train_frac: float = 0.7, cal_frac: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split strictly by date (no leakage). Returns train, calibration, test DataFrames.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    unique_dates = df["date"].sort_values().unique()
    n_dates = len(unique_dates)
    i1 = int(np.floor(n_dates * train_frac))
    i2 = int(np.floor(n_dates * (train_frac + cal_frac)))
    date_train_end = unique_dates[i1 - 1]
    date_cal_end = unique_dates[i2 - 1]

    train = df[df["date"] <= date_train_end].reset_index(drop=True)
    cal = df[(df["date"] > date_train_end) & (df["date"] <= date_cal_end)].reset_index(drop=True)
    test = df[df["date"] > date_cal_end].reset_index(drop=True)
    return train, cal, test


def fit_transformers(X_train: pd.DataFrame, categorical_cols=None):
    """
    Fit scikit-learn transformers for categorical variables and numeric scaling.
    Returns a dict with fitted ColumnTransformer and feature names.
    """
    categorical_cols = categorical_cols or ["store_id", "item_id", "category"]
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols + ["date", "sales_units", "on_promotion"]]

    # OneHot encode categoricals (handle_unknown to avoid errors on test)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    scaler = StandardScaler()

    col_transformer = ColumnTransformer(
        transformers=[
            ("ohe", ohe, categorical_cols),
            ("num", scaler, numeric_cols),
        ],
        remainder="drop",
    )
    col_transformer.fit(X_train)
    # Build feature names
    ohe_feature_names = []
    try:
        ohe_feature_names = col_transformer.named_transformers_["ohe"].get_feature_names_out(categorical_cols).tolist()
    except Exception:
        # fallback
        ohe_feature_names = categorical_cols
    numeric_feature_names = numeric_cols
    feature_names = ohe_feature_names + numeric_feature_names
    return {"column_transformer": col_transformer, "feature_names": feature_names, "categorical_cols": categorical_cols, "numeric_cols": numeric_cols}


def transform_with_fitted(X: pd.DataFrame, transformers: Dict):
    """
    Apply fitted transformers to X and return transformed array and feature names.
    """
    ct = transformers["column_transformer"]
    X_trans = ct.transform(X)
    return X_trans, transformers["feature_names"]


def get_splits_and_transformers(csv_path: str = None):
    """
    Convenience function to read raw CSV, build features, split chronologically,
    and fit transformers on training set.

    Returns:
      X_train, y_train, X_cal, y_cal, X_test, y_test, transformers
    where X_* are DataFrames (original columns + engineered features) and y_* are Series.
    """
    csv_path = csv_path or RAW_CSV
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df_feat = build_features(df)

    train, cal, test = chronological_split(df_feat)

    target_col = "sales_units"
    # X are all columns except target
    X_train = train.drop(columns=[target_col]).reset_index(drop=True)
    y_train = train[target_col].reset_index(drop=True)
    X_cal = cal.drop(columns=[target_col]).reset_index(drop=True)
    y_cal = cal[target_col].reset_index(drop=True)
    X_test = test.drop(columns=[target_col]).reset_index(drop=True)
    y_test = test[target_col].reset_index(drop=True)

    transformers = fit_transformers(X_train, categorical_cols=["store_id", "item_id", "category"])
    return X_train, y_train, X_cal, y_cal, X_test, y_test, transformers


# If run as script, produce processed CSVs for convenience
if __name__ == "__main__":
    import os
    OUT_DIR = ROOT / "data" / "processed"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Reading raw CSV and building features...")
    X_train, y_train, X_cal, y_cal, X_test, y_test, transformers = get_splits_and_transformers()
    # Save splits
    X_train.assign(sales_units=y_train).to_csv(OUT_DIR / "train.csv", index=False)
    X_cal.assign(sales_units=y_cal).to_csv(OUT_DIR / "calibration.csv", index=False)
    X_test.assign(sales_units=y_test).to_csv(OUT_DIR / "test.csv", index=False)
    print(f"Saved processed splits to {OUT_DIR}")
