# src/inventory_optimizer.py

from typing import Dict, Any

import numpy as np
from scipy.optimize import minimize_scalar


class PerishableInventoryOptimizer:
    """
    Cost-aware inventory optimizer for perishable grocery items.

    The optimizer uses simulated Demand During Lead Time (DDLT)
    samples and finds an order quantity that minimizes expected
    inventory-related loss while targeting a minimum service level.
    """

    def __init__(
        self,
        unit_cost: float,
        holding_cost: float,
        spoilage_cost: float,
        stockout_cost: float,
        shelf_life_days: int,
        current_inventory: int = 0,
    ):
        self.unit_cost = float(unit_cost)
        self.holding_cost = float(holding_cost)
        self.spoilage_cost = float(spoilage_cost)
        self.stockout_cost = float(stockout_cost)
        self.shelf_life_days = int(shelf_life_days)
        self.current_inventory = int(current_inventory)

    def evaluate_cost(
        self,
        Q: float,
        ddlt_samples: np.ndarray,
    ) -> float:
        """
        Calculate expected inventory cost for a candidate order quantity Q.
        """

        Q = float(max(0.0, Q))

        inventory_after_order = self.current_inventory + Q

        shortage = np.maximum(
            0.0,
            ddlt_samples - inventory_after_order,
        )

        ending = np.maximum(
            0.0,
            inventory_after_order - ddlt_samples,
        )

        # Spoilage increases for shorter shelf life.
        spoilage_rate = max(
            0.05,
            1.0 / max(1, self.shelf_life_days),
        )

        spoilage = ending * spoilage_rate

        cost = (
            self.stockout_cost * shortage
            + self.spoilage_cost * spoilage
            + self.holding_cost * ending
        )

        return float(np.mean(cost))

    def _cost_wrapper(
        self,
        Q: float,
        ddlt_samples: np.ndarray,
    ) -> float:
        return self.evaluate_cost(Q, ddlt_samples)

    def optimize_order_quantity(
        self,
        ddlt_samples: np.ndarray,
        min_service_level: float = 0.95,
        q_bounds: tuple = (0.0, None),
    ) -> Dict[str, Any]:
        """
        Find the order quantity Q* that minimizes expected cost.

        A minimum service level is enforced by ensuring that the
        inventory after ordering covers the requested percentile
        of simulated demand.
        """

        ddlt = np.asarray(ddlt_samples, dtype=float)

        if ddlt.size == 0:
            raise ValueError("ddlt_samples cannot be empty.")

        if not 0 < min_service_level <= 1:
            raise ValueError(
                "min_service_level must be between 0 and 1."
            )

        # Upper bound for optimization.
        ub = (
            float(np.percentile(ddlt, 99.5))
            + max(50.0, self.current_inventory)
        )

        if q_bounds[1] is not None:
            ub = min(ub, float(q_bounds[1]))

        # Minimum Q needed to achieve the requested service level.
        required_inventory = float(
            np.quantile(ddlt, min_service_level)
        )

        service_level_lower_bound = max(
            0.0,
            required_inventory - self.current_inventory,
        )

        lb = max(
            float(q_bounds[0]),
            service_level_lower_bound,
        )

        if lb > ub:
            raise ValueError(
                "Invalid optimization bounds. "
                "The requested service level cannot be achieved "
                "within the provided order quantity bounds."
            )

        # Optimize expected cost.
        try:
            result = minimize_scalar(
                lambda q: self._cost_wrapper(q, ddlt),
                bounds=(lb, ub),
                method="bounded",
                options={"xatol": 1.0},
            )

            Q_opt = float(max(lb, result.x))

        except Exception:
            # Safe fallback using grid search.
            grid = np.linspace(lb, ub, 201)

            costs = [
                self.evaluate_cost(q, ddlt)
                for q in grid
            ]

            Q_opt = float(
                grid[int(np.argmin(costs))]
            )

        # Diagnostics.
        inventory_after_order = (
            self.current_inventory + Q_opt
        )

        shortage = np.maximum(
            0.0,
            ddlt - inventory_after_order,
        )

        ending = np.maximum(
            0.0,
            inventory_after_order - ddlt,
        )

        service_level = float(
            np.mean(ddlt <= inventory_after_order)
        )

        expected_shortage = float(
            np.mean(shortage)
        )

        expected_leftover = float(
            np.mean(ending)
        )

        expected_total_cost = self.evaluate_cost(
            Q_opt,
            ddlt,
        )

        return {
            "optimal_order_quantity": Q_opt,
            "inventory_after_order": inventory_after_order,
            "expected_total_cost": expected_total_cost,
            "service_level": service_level,
            "expected_shortage_units": expected_shortage,
            "expected_leftover_units": expected_leftover,
            "minimum_service_level": float(
                min_service_level
            ),
        }