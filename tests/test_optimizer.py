import numpy as np
from src.inventory_optimizer import PerishableInventoryOptimizer

def test_optimizer_cost_minimization():
    ddlt_samples = np.random.normal(loc=100.0, scale=15.0, size=2000)
    ddlt_samples = np.clip(ddlt_samples, 0, None)
    
    optimizer = PerishableInventoryOptimizer(
        unit_cost=50.0,
        holding_cost=2.0,
        spoilage_cost=60.0,
        stockout_cost=150.0,
        shelf_life_days=4,
        current_inventory=10
    )
    
    res = optimizer.optimize_order_quantity(ddlt_samples, min_service_level=0.95)
    
    assert res["optimal_order_quantity"] > 0
    assert 0.0 <= res["stockout_risk_pct"] <= 15.0
    assert res["achieved_service_level"] >= 0.85