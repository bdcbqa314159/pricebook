"""Payoff evaluators for the MC engine.

Each function returns a callable(paths, times) → values that plugs
into MCEngine.price().

    from pricebook.mc_payoffs import european_call, asian_arithmetic, barrier_ko

    result = engine.price(european_call(strike=100))
    result = engine.price(asian_arithmetic(strike=100))
    result = engine.price(barrier_ko(strike=100, barrier=120, barrier_type="up-and-out"))
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# European payoffs
# ---------------------------------------------------------------------------

def european_call(strike: float, log_space: bool = True) -> Callable:
    """European call: max(S_T - K, 0)."""
    def payoff(paths, times):
        terminal = paths[:, -1] if paths.ndim == 2 else paths[:, -1, 0]
        spot = np.exp(terminal) if log_space else terminal
        return np.maximum(spot - strike, 0.0)
    return payoff


def european_put(strike: float, log_space: bool = True) -> Callable:
    """European put: max(K - S_T, 0)."""
    def payoff(paths, times):
        terminal = paths[:, -1] if paths.ndim == 2 else paths[:, -1, 0]
        spot = np.exp(terminal) if log_space else terminal
        return np.maximum(strike - spot, 0.0)
    return payoff


def digital_call(strike: float, payout: float = 1.0, log_space: bool = True) -> Callable:
    """Digital call: payout if S_T > K, else 0."""
    def payoff(paths, times):
        terminal = paths[:, -1] if paths.ndim == 2 else paths[:, -1, 0]
        spot = np.exp(terminal) if log_space else terminal
        return np.where(spot > strike, payout, 0.0)
    return payoff


# ---------------------------------------------------------------------------
# Path-dependent payoffs
# ---------------------------------------------------------------------------

def asian_arithmetic(strike: float, log_space: bool = True) -> Callable:
    """Asian call (arithmetic average): max(avg(S) - K, 0)."""
    def payoff(paths, times):
        if paths.ndim == 3:
            p = paths[:, :, 0]
        else:
            p = paths
        spots = np.exp(p) if log_space else p
        avg = np.mean(spots[:, 1:], axis=1)  # exclude t=0
        return np.maximum(avg - strike, 0.0)
    return payoff


def asian_geometric(strike: float, log_space: bool = True) -> Callable:
    """Asian call (geometric average): max(geom_avg(S) - K, 0)."""
    def payoff(paths, times):
        if paths.ndim == 3:
            p = paths[:, :, 0]
        else:
            p = paths
        if log_space:
            log_avg = np.mean(p[:, 1:], axis=1)
            geo_avg = np.exp(log_avg)
        else:
            geo_avg = np.exp(np.mean(np.log(np.maximum(p[:, 1:], 1e-15)), axis=1))
        return np.maximum(geo_avg - strike, 0.0)
    return payoff


def lookback_call(log_space: bool = True) -> Callable:
    """Lookback call (floating strike): S_T - min(S)."""
    def payoff(paths, times):
        if paths.ndim == 3:
            p = paths[:, :, 0]
        else:
            p = paths
        spots = np.exp(p) if log_space else p
        return spots[:, -1] - np.min(spots, axis=1)
    return payoff


# ---------------------------------------------------------------------------
# Barrier payoffs
# ---------------------------------------------------------------------------

def barrier_knockout(
    strike: float,
    barrier: float,
    barrier_type: str = "up-and-out",
    log_space: bool = True,
) -> Callable:
    """Barrier knockout option.

    barrier_type: "up-and-out", "down-and-out".
    """
    def payoff(paths, times):
        if paths.ndim == 3:
            p = paths[:, :, 0]
        else:
            p = paths
        spots = np.exp(p) if log_space else p

        if barrier_type == "up-and-out":
            alive = np.all(spots < barrier, axis=1)
        elif barrier_type == "down-and-out":
            alive = np.all(spots > barrier, axis=1)
        else:
            raise ValueError(f"Unknown barrier_type: {barrier_type}")

        terminal = spots[:, -1]
        return np.where(alive, np.maximum(terminal - strike, 0.0), 0.0)
    return payoff


def barrier_knockin(
    strike: float,
    barrier: float,
    barrier_type: str = "up-and-in",
    log_space: bool = True,
) -> Callable:
    """Barrier knockin option.

    barrier_type: "up-and-in", "down-and-in".
    """
    def payoff(paths, times):
        if paths.ndim == 3:
            p = paths[:, :, 0]
        else:
            p = paths
        spots = np.exp(p) if log_space else p

        if barrier_type == "up-and-in":
            triggered = np.any(spots >= barrier, axis=1)
        elif barrier_type == "down-and-in":
            triggered = np.any(spots <= barrier, axis=1)
        else:
            raise ValueError(f"Unknown barrier_type: {barrier_type}")

        terminal = spots[:, -1]
        return np.where(triggered, np.maximum(terminal - strike, 0.0), 0.0)
    return payoff


# ---------------------------------------------------------------------------
# American payoffs (LSM)
# ---------------------------------------------------------------------------

def american_put(
    strike: float,
    log_space: bool = True,
    n_basis: int = 3,
) -> Callable:
    """American put via Longstaff-Schwartz (LSM).

    Returns a payoff callable that performs backward LSM regression
    on the paths to find the optimal exercise strategy.
    """
    def payoff(paths, times):
        if paths.ndim == 3:
            p = paths[:, :, 0]
        else:
            p = paths
        spots = np.exp(p) if log_space else p
        n_paths, n_steps_plus_1 = spots.shape

        # Intrinsic values at each step
        exercise = np.maximum(strike - spots, 0.0)

        # Start from terminal: value = intrinsic
        values = exercise[:, -1].copy()
        exercise_time = np.full(n_paths, n_steps_plus_1 - 1)

        # Backward induction
        dt = np.diff(times)
        for step in range(n_steps_plus_1 - 2, 0, -1):
            itm = exercise[:, step] > 0
            if not np.any(itm):
                continue

            # Discount continuation
            # Simple: use exp(-r*dt) but r is embedded in the process
            # For now, no discounting within LSM (payoff is undiscounted)
            continuation = values.copy()

            # Regression: E[continuation | S_t] via polynomial
            x = spots[itm, step]
            y = continuation[itm]
            if len(x) < n_basis + 1:
                continue

            # Polynomial regression
            coeffs = np.polyfit(x, y, min(n_basis, len(x) - 1))
            fitted = np.polyval(coeffs, x)

            # Exercise if intrinsic > continuation estimate
            exercise_now = exercise[itm, step] > fitted
            idx_itm = np.where(itm)[0]
            for j, ex in zip(idx_itm, exercise_now):
                if ex:
                    values[j] = exercise[j, step]
                    exercise_time[j] = step

        return values

    return payoff


# ---------------------------------------------------------------------------
# Multi-asset payoffs
# ---------------------------------------------------------------------------

def basket_call(strike: float, weights: list[float] | None = None,
                log_space: bool = True) -> Callable:
    """Basket call: max(weighted_avg(S_i,T) - K, 0)."""
    def payoff(paths, times):
        # paths shape: (n_paths, n_steps+1, n_factors)
        terminal = paths[:, -1, :]  # (n_paths, n_factors)
        spots = np.exp(terminal) if log_space else terminal
        n_assets = spots.shape[1]
        w = np.array(weights) if weights else np.ones(n_assets) / n_assets
        basket = spots @ w
        return np.maximum(basket - strike, 0.0)
    return payoff


def worst_of_put(strike: float, log_space: bool = True) -> Callable:
    """Worst-of put: max(K - min(S_i,T), 0)."""
    def payoff(paths, times):
        terminal = paths[:, -1, :]
        spots = np.exp(terminal) if log_space else terminal
        worst = np.min(spots, axis=1)
        return np.maximum(strike - worst, 0.0)
    return payoff
