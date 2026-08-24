# app/api.py
"""
FastAPI backend for OptiPerish.

Endpoints:
 - POST /api/v1/forecast
 - POST /api/v1/optimize
 - GET  /api/v1/benchmark

Run:
    uvicorn app.api:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import json

# Import project modules
from src.feature_pipeline import get_splits_and_transformers
from src.model_forecaster import LightGBMForecaster
from src.conformal_engine import SplitConformalPredictor
from src.simulation_engine import StochasticLeadTimeSimulator
from src.inventory_optimizer import PerishableInventoryOptimizer

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CSV = ROOT / "data" / "processed" / "benchmark_results.csv"
LEAD_TIME_META = ROOT / "data" / "raw" / "lead_time_metadata.json"

app = FastAPI(title="OptiPerish API", version="0.1")

# Load artifacts lazily
_artifacts = {"model": None, "conformal": None, "transformers": None, "X_test": None, "lead_time_meta": None}


class ForecastRequest(BaseModel):
    store_id: str
    item_id: str
    horizon_days: int = Field(7, gt=0, le=30)


class ForecastResponse(BaseModel):
    dates: List[str]
    point_forecast: List[float]
    lower: List[float]
    upper: List[float]


class OptimizeRequest(BaseModel):
    forecast_lower: List[float]
    forecast_upper: List[float]
    current_inventory: int = 0
    lead_time_probs: Dict[int, float]
    unit_cost: float
    holding_cost: float
    spoilage_cost: float
    stockout_cost: float
    shelf_life_days: int
    num_simulations: int = 5000


class OptimizeResponse(BaseModel):
    optimal_order_quantity: float
    expected_total_cost: float
    safety_stock: float
    stockout_risk_pct: float
    expected_spoilage_units: float
    achieved_service_level: float


@app.on_event("startup")
def load_artifacts():
    # Load transformers and test set via feature_pipeline
    try:
        X_train, y_train, X_cal, y_cal, X_test, y_test, transformers = get_splits_and_transformers()
        _artifacts["transformers"] = transformers
        _artifacts["X_test"] = X_test
        # Train a LightGBM model quickly for API demo (in production load a persisted model)
        from src.model_forecaster import LightGBMForecaster
        ct = transformers["column_transformer"]
        X_train_arr = ct.transform(X_train)
        X_cal_arr = ct.transform(X_cal)
        lgb = LightGBMForecaster(objective="huber")
        lgb.fit(X_train_arr, y_train.values, feature_names=transformers["feature_names"], num_boost_round=200)
        _artifacts["model"] = lgb
        # Conformal predictor
        X_cal = X_cal.reset_index(drop=True)
        _artifacts["conformal"] = SplitConformalPredictor(forecaster=lgb, X_cal=X_cal, y_cal=y_cal)
    except Exception as e:
        print("Warning: failed to load full artifacts at startup:", e)
    # lead time metadata
    if LEAD_TIME_META.exists():
        with open(LEAD_TIME_META, "r", encoding="utf-8") as f:
            _artifacts["lead_time_meta"] = json.load(f)
    else:
        _artifacts["lead_time_meta"] = {}


@app.post("/api/v1/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    """
    Return point forecast and conformal intervals for the next horizon_days.
    This simplified endpoint uses the single-day conformal interval repeated across horizon.
    """
    if _artifacts["model"] is None or _artifacts["conformal"] is None or _artifacts["X_test"] is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded yet.")
    # Find a representative row in X_test for this store/item (use latest date)
    df = _artifacts["X_test"]
    mask = (df["store_id"] == req.store_id) & (df["item_id"] == req.item_id)
    if not mask.any():
        raise HTTPException(status_code=404, detail="store_id/item_id not found in processed test set.")
    row = df[mask].sort_values("date").iloc[-1]
    # point forecast for the day
    point = float(row.get("lgb_point", 0.0))
    lower, _, upper = _artifacts["conformal"].predict_interval(pd.DataFrame([row]), coverage=0.90)
    lower = float(lower[0]); upper = float(upper[0])
    # repeat for horizon (simple approach)
    dates = pd.date_range(pd.to_datetime(row["date"]) + pd.Timedelta(days=1), periods=req.horizon_days).strftime("%Y-%m-%d").tolist()
    point_list = [point] * req.horizon_days
    lower_list = [lower] * req.horizon_days
    upper_list = [upper] * req.horizon_days
    return ForecastResponse(dates=dates, point_forecast=point_list, lower=lower_list, upper=upper_list)


@app.post("/api/v1/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    # Validate lead_time_probs sum
    probs = np.array(list(req.lead_time_probs.values()), dtype=float)
    if probs.sum() <= 0:
        raise HTTPException(status_code=400, detail="Invalid lead_time_probs.")
    # Build simulator
    lower = np.array(req.forecast_lower)
    upper = np.array(req.forecast_upper)
    simulator = StochasticLeadTimeSimulator(lead_time_probs=req.lead_time_probs, lower_bounds=lower, upper_bounds=upper, random_state=42)
    ddlt_samples, summary = simulator.simulate_ddlt(num_simulations=req.num_simulations)
    optimizer = PerishableInventoryOptimizer(
        unit_cost=req.unit_cost,
        holding_cost=req.holding_cost,
        spoilage_cost=req.spoilage_cost,
        stockout_cost=req.stockout_cost,
        shelf_life_days=req.shelf_life_days,
        current_inventory=req.current_inventory,
    )
    res = optimizer.optimize_order_quantity(ddlt_samples, min_service_level=0.95)
    return OptimizeResponse(
        optimal_order_quantity=float(res["optimal_order_quantity"]),
        expected_total_cost=float(res["expected_total_cost"]),
        safety_stock=float(res["safety_stock"]),
        stockout_risk_pct=float(res["stockout_risk_pct"]),
        expected_spoilage_units=float(res["expected_spoilage_units"]),
        achieved_service_level=float(res["achieved_service_level"]),
    )


@app.get("/api/v1/benchmark")
def get_benchmark():
    if not BENCHMARK_CSV.exists():
        raise HTTPException(status_code=404, detail="Benchmark results not found. Run benchmark_runner first.")
    df = pd.read_csv(BENCHMARK_CSV)
    return df.to_dict(orient="records")
