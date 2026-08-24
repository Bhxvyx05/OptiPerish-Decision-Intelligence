# src/benchmark_runner.py
"""
Benchmark runner for OptiPerish: 4-tier ablation study.

Saves results to: data/processed/benchmark_results.csv
Prints a summary table to stdout.

Usage:
    python -m src.benchmark_runner
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import project modules
from src.feature_pipeline import get_splits_and_transformers
from src.model_forecaster import SeasonalNaiveForecaster, LightGBMForecaster, rmse, mae, wape
from src.conformal_engine import SplitConformalPredictor
from src.simulation_engine import StochasticLeadTimeSimulator
from src.inventory_optimizer import PerishableInventoryOptimizer

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_CSV = PROCESSED_DIR / "benchmark_results.csv"

# Strategy constants
FIXED_LEAD_TIME = 5  # days used by strategies A-C
STATIC_SAFETY_BUFFER = 0.20  # 20% buffer for Strategy A
Z_FOR_95 = 1.645  # one-sided z for ~95%? (used as example)
Z_FOR_90 = 1.282


def _compute_costs_from_outcome(demand, order_qty, current_inventory, unit_cost, holding_cost, spoilage_cost, stockout_cost):
    """
    Given realized demand and order decision, compute cost components.
    - current_inventory: inventory before order
    - order_qty: Q placed
    Returns: dict with stockout_units, spoilage_units, holding_units, total_cost
    """
    inventory_after = current_inventory + order_qty
    shortage = max(0, demand - inventory_after)
    ending = max(0, inventory_after - demand)
    # For evaluation, approximate spoilage as ending units (conservative)
    spoilage = ending
    holding = ending  # we count holding cost on leftover units
    cost = stockout_cost * shortage + spoilage_cost * spoilage + holding_cost * holding
    return {
        "shortage_units": float(shortage),
        "spoilage_units": float(spoilage),
        "holding_units": float(holding),
        "total_cost": float(cost),
    }


def _strategy_A(row, history_df):
    """
    Strategy A: Rolling 7-day average forecast + fixed lead time + static 20% safety buffer.
    """
    # rolling 7-day mean computed in features; fallback to lag_7 if missing
    if "rolling_mean_7" in row and not np.isnan(row["rolling_mean_7"]):
        point = float(row["rolling_mean_7"])
    elif "lag_7" in row and not np.isnan(row["lag_7"]):
        point = float(row["lag_7"])
    else:
        point = float(history_df["sales_units"].mean() if not history_df.empty else 0.0)
    # DDLT approx = point * lead_time
    ddlt_mean = point * FIXED_LEAD_TIME
    Q = max(0.0, ddlt_mean * (1.0 + STATIC_SAFETY_BUFFER))
    return point, Q, FIXED_LEAD_TIME


def _strategy_B(row, history_df, sigma_estimate=1.0):
    """
    Strategy B: LightGBM point forecast + fixed lead time + normal safety stock z * sigma * sqrt(L)
    Here sigma_estimate can be derived from rolling_std_7 or group std.
    """
    point = float(row.get("lgb_point", np.nan))
    if np.isnan(point):
        # fallback
        point = float(row.get("rolling_mean_7", 0.0))
    # estimate sigma from rolling_std_7 or group std
    sigma = float(row.get("rolling_std_7", np.nan))
    if np.isnan(sigma) or sigma <= 0:
        sigma = sigma_estimate
    L = FIXED_LEAD_TIME
    z = Z_FOR_90  # choose z for ~90% service (user can vary)
    safety = z * sigma * np.sqrt(L)
    ddlt_mean = point * L
    Q = max(0.0, ddlt_mean + safety)
    return point, Q, L


def _strategy_C(row):
    """
    Strategy C: LightGBM + Conformal upper bound (90%) + fixed lead time.
    Use the conformal upper bound per day as conservative daily demand.
    """
    # conformal upper bound for the day is stored in row['conformal_upper']
    upper = float(row.get("conformal_upper", np.nan))
    if np.isnan(upper):
        upper = float(row.get("lgb_point", 0.0))
    L = FIXED_LEAD_TIME
    Q = max(0.0, upper * L)
    return upper, Q, L


def _strategy_D(row, conformal_predictor, lead_time_metadata, num_sim=5000, current_inventory=0):
    """
    Strategy D: Full OptiPerish
    - Use conformal predictor to get lower/upper per day for horizon L sampled stochastically
    - Use StochasticLeadTimeSimulator + PerishableInventoryOptimizer to find Q*
    """
    # For per-day conformal interval we will use the single-day interval repeated for horizon.
    # In a full implementation you'd forecast multi-day horizon; here we approximate using daily interval.
    lgb_point = float(row.get("lgb_point", 0.0))
    lower, _, upper = conformal_predictor.predict_interval(pd.DataFrame([row]), coverage=0.90)
    lower = float(lower[0])
    upper = float(upper[0])
    # lead time distribution for this item
    item = row["item_id"]
    lt_dist = lead_time_metadata.get(item, None)
    if lt_dist is None:
        lt_dist = {FIXED_LEAD_TIME: 1.0}
    # Build arrays for horizon = max lead time
    max_L = max(int(k) for k in lt_dist.keys())
    lower_arr = np.array([lower] * max_L)
    upper_arr = np.array([upper] * max_L)
    simulator = StochasticLeadTimeSimulator(lead_time_probs=lt_dist, lower_bounds=lower_arr, upper_bounds=upper_arr, random_state=42)
    ddlt_samples, summary = simulator.simulate_ddlt(num_simulations=num_sim)
    # optimizer
    optimizer = PerishableInventoryOptimizer(
        unit_cost=row["unit_cost"],
        holding_cost=row["holding_cost_per_day"],
        spoilage_cost=row["spoilage_cost_per_unit"],
        stockout_cost=row["stockout_cost_per_unit"],
        shelf_life_days=int(row["shelf_life_days"]),
        current_inventory=current_inventory,
    )
    res = optimizer.optimize_order_quantity(ddlt_samples, min_service_level=0.95)
    return lgb_point, res["optimal_order_quantity"], summary, res



    cal_df["sales_units"] = y_cal.values
    test_df = X_test.copy()
    test_df["sales_units"] = y_test.values

    # Train or load models
    print("Training SeasonalNaive and LightGBM forecasters...")
    naive = SeasonalNaiveForecaster(season_lag=7)
    naive.fit(train_df, train_df["sales_units"])

    # Prepare arrays for LightGBM: transform X via transformers
    ct = transformers["column_transformer"]
    feature_names = transformers["feature_names"]

    X_train_arr = ct.transform(X_train)
    X_cal_arr = ct.transform(X_cal)
    X_test_arr = ct.transform(X_test)

    lgb = LightGBMForecaster(objective="huber")
    # quick train with default params; for production use tune_with_optuna
    lgb.fit(X_train_arr, y_train.values, feature_names=feature_names, num_boost_round=200)

    # Attach point predictions to dataframes for convenience
    print("Generating point forecasts for calibration and test sets...")
    cal_preds = lgb.predict(X_cal_arr)
    test_preds = lgb.predict(X_test_arr)
    X_cal = X_cal.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    X_cal["lgb_point"] = cal_preds
    X_test["lgb_point"] = test_preds

    # Build conformal predictor using calibration set
    conformal = SplitConformalPredictor(forecaster=lgb, X_cal=X_cal, y_cal=y_cal)
    # attach conformal upper bound to test rows
    lower_test, yhat_test, upper_test = conformal.predict_interval(X_test, coverage=0.90)
    X_test["conformal_lower"] = lower_test
    X_test["conformal_upper"] = upper_test
    X_test["lgb_point"] = yhat_test

    # Load lead time metadata (from data/raw/lead_time_metadata.json if exists)
    import json
    lt_path = ROOT / "data" / "raw" / "lead_time_metadata.json"
    if lt_path.exists():
        with open(lt_path, "r", encoding="utf-8") as f:
            lead_time_metadata = json.load(f)
    else:
        lead_time_metadata = {}

    # Iterate over test rows and evaluate each strategy
    results = []
    print("Running strategies over test horizon (this may take a few minutes)...")
    # For speed, evaluate per-row; in a production run you'd vectorize per (store,item,date)
    for idx, row in tqdm(X_test.reset_index(drop=True).iterrows(), total=len(X_test)):
        # get actual demand
        actual = float(test_df.loc[idx, "sales_units"])
        # history for group (store,item) up to this date (from train+cal)
        mask_hist = (train_df["store_id"] == row["store_id"]) & (train_df["item_id"] == row["item_id"])
        history_df = train_df[mask_hist]

        # Strategy A
        _, Q_A, L_A = _strategy_A(row, history_df)
        costs_A = _compute_costs_from_outcome(actual, Q_A, current_inventory=0, unit_cost=row["unit_cost"],
                                              holding_cost=row["holding_cost_per_day"], spoilage_cost=row["spoilage_cost_per_unit"],
                                              stockout_cost=row["stockout_cost_per_unit"])
        # Strategy B
        _, Q_B, L_B = _strategy_B(row, history_df, sigma_estimate=max(1.0, row.get("rolling_std_7", 1.0)))
        costs_B = _compute_costs_from_outcome(actual, Q_B, current_inventory=0, unit_cost=row["unit_cost"],
                                              holding_cost=row["holding_cost_per_day"], spoilage_cost=row["spoilage_cost_per_unit"],
                                              stockout_cost=row["stockout_cost_per_unit"])
        # Strategy C
        _, Q_C, L_C = _strategy_C(row)
        costs_C = _compute_costs_from_outcome(actual, Q_C, current_inventory=0, unit_cost=row["unit_cost"],
                                              holding_cost=row["holding_cost_per_day"], spoilage_cost=row["spoilage_cost_per_unit"],
                                              stockout_cost=row["stockout_cost_per_unit"])
        # Strategy D
        try:
            _, Q_D, sim_summary, opt_res = _strategy_D(row, conformal, lead_time_metadata, num_sim=2000, current_inventory=0)
            costs_D = _compute_costs_from_outcome(actual, Q_D, current_inventory=0, unit_cost=row["unit_cost"],
                                                  holding_cost=row["holding_cost_per_day"], spoilage_cost=row["spoilage_cost_per_unit"],
                                                  stockout_cost=row["stockout_cost_per_unit"])
        except Exception as e:
            # fallback: treat as C
            Q_D = Q_C
            sim_summary = {}
            opt_res = {}
            costs_D = costs_C

        # Collect per-row metrics
        results.append({
            "idx": idx,
            "date": row["date"],
            "store_id": row["store_id"],
            "item_id": row["item_id"],
            "actual": actual,
            "A_Q": float(Q_A), "A_cost": costs_A["total_cost"], "A_short": costs_A["shortage_units"], "A_spoil": costs_A["spoilage_units"],
            "B_Q": float(Q_B), "B_cost": costs_B["total_cost"], "B_short": costs_B["shortage_units"], "B_spoil": costs_B["spoilage_units"],
            "C_Q": float(Q_C), "C_cost": costs_C["total_cost"], "C_short": costs_C["shortage_units"], "C_spoil": costs_C["spoilage_units"],
            "D_Q": float(Q_D), "D_cost": costs_D["total_cost"], "D_short": costs_D["shortage_units"], "D_spoil": costs_D["spoilage_units"],
        })

    df_res = pd.DataFrame(results)

    # Aggregate metrics per strategy
    def summarize(prefix):
        total_cost = df_res[f"{prefix}_cost"].sum()
        stockout_rate = (df_res[f"{prefix}_short"] > 0).mean() * 100.0
        spoil_rate = (df_res[f"{prefix}_spoil"] > 0).mean() * 100.0
        achieved_service = 100.0 - stockout_rate
        # Forecast RMSE: use lgb_point vs actual for strategies that use LGB; for naive use lag7 vs actual
        if prefix == "A":
            # use rolling_mean_7 if available; approximate by comparing Q/lead_time to actual
            forecast_vals = df_res[f"{prefix}_Q"] / FIXED_LEAD_TIME
        elif prefix == "B":
            forecast_vals = X_test["lgb_point"].values
        elif prefix in ("C", "D"):
            forecast_vals = X_test["lgb_point"].values
        else:
            forecast_vals = X_test["lgb_point"].values
        rmse_val = rmse(test_df["sales_units"].values, forecast_vals)
        return {
            "Strategy": prefix,
            "Forecast RMSE": float(rmse_val),
            "Stockout Rate (%)": float(stockout_rate),
            "Spoilage/Waste Rate (%)": float(spoil_rate),
            "Achieved Service Level (%)": float(achieved_service),
            "Total Financial Cost (₹)": float(total_cost),
        }

    summary_rows = [summarize(p) for p in ["A", "B", "C", "D"]]
    summary_df = pd.DataFrame(summary_rows)

    # Save CSV
    if save_csv:
        summary_df.to_csv(BENCHMARK_CSV, index=False)
        print(f"Saved benchmark summary to {BENCHMARK_CSV}")

    # Print table
    print("\nBenchmark Summary (aggregated over test horizon):")
    print(summary_df.to_string(index=False, float_format="%.2f"))
    return summary_df, df_res


if __name__ == "__main__":
    run_benchmark(save_csv=True)
