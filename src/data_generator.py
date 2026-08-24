# src/data_generator.py
"""
Generate a realistic multi-store, multi-item daily grocery sales dataset.

Saves:
 - data/raw/grocery_sales.csv
 - data/raw/lead_time_metadata.json

Run as:
    python src/data_generator.py
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------------
# Configuration / Catalogs
# -------------------------
START_DATE = "2024-01-01"
N_DAYS = 730  # 2 years
N_STORES = 8
ITEM_CATALOG = [
    # (item_id, category, base_price_mean, base_price_std)
    ("ITEM_DAIRY_01", "Dairy", 60.0, 5.0),
    ("ITEM_DAIRY_02", "Dairy", 45.0, 4.0),
    ("ITEM_PRODUCE_01", "Produce", 30.0, 6.0),
    ("ITEM_PRODUCE_02", "Produce", 25.0, 5.0),
    ("ITEM_MEAT_01", "Meat", 180.0, 15.0),
    ("ITEM_MEAT_02", "Meat", 120.0, 10.0),
    ("ITEM_BAKERY_01", "Bakery", 40.0, 6.0),
    ("ITEM_BAKERY_02", "Bakery", 35.0, 5.0),
]

SHELF_LIFE = {"Dairy": 7, "Produce": 4, "Meat": 5, "Bakery": 3}

# Example vendor lead-time empirical distribution per item (discrete)
DEFAULT_LEAD_TIME_DIST = {2: 0.15, 3: 0.50, 5: 0.25, 7: 0.10}

# Create output directories
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Helper functions
# -------------------------
def _store_ids(n):
    return [f"STORE_{i:02d}" for i in range(1, n + 1)]


def _generate_promotion_schedule(n_days, promo_prob=0.03, seed=None):
    """Random planned promotions; promotions are planned so forward-looking features are allowed."""
    rng = np.random.default_rng(seed)
    promo = (rng.random(n_days) < promo_prob).astype(int)
    # occasionally create multi-day promotions
    for i in range(1, n_days - 1):
        if promo[i] == 1 and rng.random() < 0.3:
            promo[i + 1] = 1
    return promo


def _festival_windows(dates):
    """Return a boolean array marking festival windows (higher demand)."""
    # Simple synthetic festival windows: assume some fixed date ranges each year
    is_festival = np.zeros(len(dates), dtype=int)
    for year in sorted({d.year for d in dates}):
        # Example festival windows (synthetic): late Mar (spring festival), Oct (autumn festival), Dec (year-end)
        windows = [
            (f"{year}-03-20", f"{year}-03-28"),
            (f"{year}-10-10", f"{year}-10-20"),
            (f"{year}-12-20", f"{year}-12-31"),
        ]
        for start, end in windows:
            mask = (dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))
            is_festival[mask] = 1
    return is_festival


def _month_end_lift(dates):
    """Return a small lift near month end."""
    return dates.is_month_end.astype(int)


# -------------------------
# Core generator
# -------------------------
def generate_grocery_sales(
    start_date=START_DATE,
    n_days=N_DAYS,
    n_stores=N_STORES,
    item_catalog=ITEM_CATALOG,
    seed=42,
):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    stores = _store_ids(n_stores)
    items = item_catalog

    # Precompute calendar effects
    dow = dates.weekday  # 0=Mon .. 6=Sun
    is_weekend = (dow >= 5).astype(int)
    festival = _festival_windows(dates)
    month_end = _month_end_lift(dates)

    rows = []
    lead_time_metadata = {}

    for item_id, category, price_mean, price_std in items:
        # item-level parameters
        shelf_life = SHELF_LIFE[category]
        # base price sampled once per item (small jitter)
        base_price = max(1.0, float(rng.normal(price_mean, price_std)))
        unit_cost = round(base_price * float(rng.uniform(0.50, 0.70)), 2)
        holding_cost_per_day = round(unit_cost * float(rng.uniform(0.02, 0.05)), 2)
        salvage_fee = round(float(rng.uniform(2.0, 8.0)), 2)
        spoilage_cost_per_unit = round(unit_cost + salvage_fee, 2)
        # stockout cost: lost margin + ₹50 penalty (we'll use INR symbol in comment only)
        avg_margin = max(0.0, base_price - unit_cost)
        stockout_cost_per_unit = round(avg_margin + 50.0, 2)

        # lead time metadata per item (could vary slightly per item)
        # perturb default distribution slightly
        lt_keys = np.array(list(DEFAULT_LEAD_TIME_DIST.keys()))
        lt_probs = np.array(list(DEFAULT_LEAD_TIME_DIST.values()))
        lt_probs = lt_probs + rng.normal(0, 0.02, size=lt_probs.shape)
        lt_probs = np.clip(lt_probs, 0.001, None)
        lt_probs = lt_probs / lt_probs.sum()
        lead_time_metadata[item_id] = {int(k): float(p) for k, p in zip(lt_keys, lt_probs)}

        # item-level seasonality amplitude and baseline demand
        baseline_daily = max(1.0, float(rng.normal(40 if category == "Dairy" else
                                                   30 if category == "Produce" else
                                                   20 if category == "Bakery" else 10, 5)))
        yearly_amp = float(rng.uniform(0.05, 0.25))  # relative amplitude
        weekday_profile = np.array([1.0, 1.0, 1.0, 1.0, 1.05, 1.4, 1.4])  # weekend lift

        # promotion schedule (planned) for this item across the horizon
        promo_schedule = _generate_promotion_schedule(n_days, promo_prob=0.03, seed=seed + hash(item_id) % 1000)

        for s in stores:
            # store-level multiplier (store size / footfall)
            store_multiplier = float(rng.normal(1.0, 0.12))
            # small store-specific price adjustment
            store_price_adj = float(rng.normal(1.0, 0.03))
            for i, date in enumerate(dates):
                # seasonal yearly effect (sinusoidal)
                day_of_year = date.timetuple().tm_yday
                yearly_factor = 1.0 + yearly_amp * np.sin(2 * np.pi * day_of_year / 365.25)

                # month-end lift and festival
                month_end_lift = 1.12 if month_end[i] else 1.0
                festival_lift = 1.25 if festival[i] else 1.0

                # weekend lift
                weekend_lift = 1.40 if is_weekend[i] else 1.0

                # promotional effect (planned)
                on_promo = int(promo_schedule[i])
                promo_lift = float(rng.uniform(1.30, 1.60)) if on_promo else 1.0

                # random promotional spikes (unplanned) - occasional
                spike = 0.0
                if rng.random() < 0.005:
                    spike = float(rng.integers(10, 80))

                # noise
                noise = float(rng.normal(0, baseline_daily * 0.12))

                # compute expected demand (units)
                expected = baseline_daily * store_multiplier * weekday_profile[dow[i]]
                expected *= yearly_factor * month_end_lift * festival_lift * weekend_lift * promo_lift
                expected = max(0.0, expected + spike + noise)

                # realized sales units (Poisson-ish rounding)
                sales_units = int(max(0, np.round(rng.poisson(lam=max(0.1, expected)))))

                # final base price and unit cost per store-item-day (small jitter)
                final_base_price = round(base_price * store_price_adj * float(rng.normal(1.0, 0.01)), 2)
                final_unit_cost = round(unit_cost * float(rng.normal(1.0, 0.01)), 2)
                holding_cost = round(holding_cost_per_day * float(rng.normal(1.0, 0.02)), 2)
                spoilage_cost = round(spoilage_cost_per_unit * float(rng.normal(1.0, 0.02)), 2)
                stockout_cost = round(stockout_cost_per_unit * float(rng.normal(1.0, 0.02)), 2)

                rows.append(
                    {
                        "date": date,
                        "store_id": s,
                        "item_id": item_id,
                        "category": category,
                        "sales_units": int(sales_units),
                        "on_promotion": int(on_promo),
                        "base_price": float(final_base_price),
                        "unit_cost": float(final_unit_cost),
                        "shelf_life_days": int(shelf_life),
                        "holding_cost_per_day": float(holding_cost),
                        "spoilage_cost_per_unit": float(spoilage_cost),
                        "stockout_cost_per_unit": float(stockout_cost),
                    }
                )

    df = pd.DataFrame(rows)
    # shuffle rows deterministically for storage (not necessary)
    df = df.sort_values(["date", "store_id", "item_id"]).reset_index(drop=True)
    return df, lead_time_metadata


def save_outputs(df, lead_time_metadata, csv_path=RAW_DIR / "grocery_sales.csv", metadata_path=RAW_DIR / "lead_time_metadata.json"):
    df.to_csv(csv_path, index=False)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(lead_time_metadata, f, indent=2)
    print(f"Saved grocery sales to {csv_path}")
    print(f"Saved lead time metadata to {metadata_path}")


# -------------------------
# CLI entry point
# -------------------------
def main():
    print("Generating grocery sales dataset (2 years, multi-store, multi-item)...")
    df, metadata = generate_grocery_sales()
    save_outputs(df, metadata)


if __name__ == "__main__":
    main()
