# src/simulation_engine.py
"""
StochasticLeadTimeSimulator

Given:
 - lead_time_probs: dict {L: prob}
 - lower_bounds, upper_bounds: arrays of length H (forecast horizon days)
Simulate DDLT samples vectorized and return summary stats.
"""
import numpy as np


class StochasticLeadTimeSimulator:
    def __init__(self, lead_time_probs: dict, lower_bounds: np.ndarray, upper_bounds: np.ndarray, random_state: int = 42):
        """
        lead_time_probs: dict mapping integer lead times to probabilities (must sum to 1)
        lower_bounds, upper_bounds: arrays (H,) for each day in forecast horizon
        """
        self.lead_time_values = np.array(sorted(list(lead_time_probs.keys())), dtype=int)
        probs = np.array([lead_time_probs[int(k)] for k in self.lead_time_values], dtype=float)
        probs = probs / probs.sum()
        self.lead_time_probs = probs
        self.lower = np.array(lower_bounds)
        self.upper = np.array(upper_bounds)
        self.horizon = len(self.lower)
        self.rng = np.random.default_rng(random_state)

    def simulate_ddlt(self, num_simulations: int = 10000):
        """
        Vectorized Monte Carlo:
         - sample lead times L for each simulation
         - for each simulation, sample daily demands uniformly from [lower, upper] for days 0..L-1
         - sum to get DDLT per simulation
        Returns:
         - ddlt_samples: np.ndarray shape (num_simulations,)
         - summary dict with mean, var, p90
        """
        N = num_simulations
        # sample lead times for each simulation
        L_samples = self.rng.choice(self.lead_time_values, size=N, p=self.lead_time_probs)

        # Pre-allocate ddlt array
        ddlt = np.zeros(N, dtype=float)

        # For efficiency, handle by unique lead times
        unique_L, inverse_idx = np.unique(L_samples, return_inverse=True)
        for L in unique_L:
            idx = np.where(L_samples == L)[0]
            if L <= 0:
                continue
            # For these simulations, sample L days per sim from uniform(lower[:L], upper[:L])
            # We'll sample shape (len(idx), L) and sum across axis=1
            low = self.lower[:L]
            high = self.upper[:L]
            # Broadcast to (len(idx), L)
            samples = self.rng.uniform(low=low, high=high, size=(len(idx), L))
            ddlt[idx] = samples.sum(axis=1)

        summary = {
            "mean": float(np.mean(ddlt)),
            "var": float(np.var(ddlt)),
            "p90": float(np.percentile(ddlt, 90)),
            "p50": float(np.percentile(ddlt, 50)),
        }
        return ddlt, summary
