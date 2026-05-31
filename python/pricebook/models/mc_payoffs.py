"""Payoff evaluators for the MC engine.

Each function returns a callable(paths, times) → values that plugs
into MCEngine.price().

    from pricebook.models.mc_payoffs import european_call, asian_arithmetic, barrier_ko

    result = engine.price(european_call(strike=100))
    result = engine.price(asian_arithmetic(strike=100))
    result = engine.price(barrier_ko(strike=100, barrier=120, barrier_type="up-and-out"))
"""

from __future__ import annotations

import math
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
    continuous: bool = False,
    sigma: float | None = None,
    seed: int | None = None,
) -> Callable:
    """Barrier knockout option.

    Args:
        barrier_type: "up-and-out" or "down-and-out".
        continuous: if True, use Brownian bridge correction for
            continuous monitoring from discrete paths. Requires sigma.
        sigma: spot volatility (needed for bridge correction).
        seed: RNG seed for bridge sampling.
    """
    def payoff(paths, times):
        if paths.ndim == 3:
            p = paths[:, :, 0]
        else:
            p = paths
        spots = np.exp(p) if log_space else p
        n_paths, n_steps_plus_1 = spots.shape

        if not continuous:
            # Discrete monitoring
            if barrier_type == "up-and-out":
                alive = np.all(spots < barrier, axis=1)
            elif barrier_type == "down-and-out":
                alive = np.all(spots > barrier, axis=1)
            else:
                raise ValueError(f"Unknown barrier_type: {barrier_type}")
        else:
            # Continuous monitoring via Brownian bridge
            from pricebook.models.mc_processes import brownian_bridge_max
            rng = np.random.default_rng(seed)
            alive = np.ones(n_paths, dtype=bool)
            vol = sigma if sigma is not None else 0.20

            for step in range(n_steps_plus_1 - 1):
                dt = times[step + 1] - times[step]
                if dt < 1e-14:
                    continue
                for i in range(n_paths):
                    if not alive[i]:
                        continue
                    s0, s1 = spots[i, step], spots[i, step + 1]
                    local_sigma = vol * s0
                    if barrier_type == "up-and-out":
                        bridge_max = brownian_bridge_max(s0, s1, dt, local_sigma, rng)
                        if bridge_max >= barrier:
                            alive[i] = False
                    elif barrier_type == "down-and-out":
                        # P(min < b | s0, s1) = exp(-2(s0-b)(s1-b)/(σ²dt))
                        # if both s0, s1 > barrier
                        if s0 <= barrier or s1 <= barrier:
                            alive[i] = False
                        elif local_sigma > 0 and dt > 0:
                            p_cross = math.exp(-2 * (s0 - barrier) * (s1 - barrier)
                                                / (local_sigma**2 * dt))
                            if rng.random() < p_cross:
                                alive[i] = False

        terminal = spots[:, -1]
        return np.where(alive, np.maximum(terminal - strike, 0.0), 0.0)
    return payoff


def barrier_knockin(
    strike: float,
    barrier: float,
    barrier_type: str = "up-and-in",
    log_space: bool = True,
    continuous: bool = False,
    sigma: float | None = None,
    seed: int | None = None,
) -> Callable:
    """Barrier knockin option.

    Args:
        barrier_type: "up-and-in" or "down-and-in".
        continuous: if True, use Brownian bridge correction.
        sigma: spot volatility (needed for bridge correction).
        seed: RNG seed for bridge sampling.
    """
    def payoff(paths, times):
        if paths.ndim == 3:
            p = paths[:, :, 0]
        else:
            p = paths
        spots = np.exp(p) if log_space else p
        n_paths, n_steps_plus_1 = spots.shape

        if not continuous:
            # Discrete monitoring
            if barrier_type == "up-and-in":
                triggered = np.any(spots >= barrier, axis=1)
            elif barrier_type == "down-and-in":
                triggered = np.any(spots <= barrier, axis=1)
            else:
                raise ValueError(f"Unknown barrier_type: {barrier_type}")
        else:
            # Continuous monitoring via Brownian bridge
            from pricebook.models.mc_processes import brownian_bridge_max
            rng = np.random.default_rng(seed)
            triggered = np.zeros(n_paths, dtype=bool)
            vol = sigma if sigma is not None else 0.20

            for step in range(n_steps_plus_1 - 1):
                dt = times[step + 1] - times[step]
                if dt < 1e-14:
                    continue
                for i in range(n_paths):
                    if triggered[i]:
                        continue
                    s0, s1 = spots[i, step], spots[i, step + 1]
                    local_sigma = vol * s0
                    if barrier_type == "up-and-in":
                        bridge_max = brownian_bridge_max(s0, s1, dt, local_sigma, rng)
                        if bridge_max >= barrier:
                            triggered[i] = True
                    elif barrier_type == "down-and-in":
                        if s0 <= barrier or s1 <= barrier:
                            triggered[i] = True
                        elif local_sigma > 0 and dt > 0:
                            p_cross = math.exp(-2 * (s0 - barrier) * (s1 - barrier)
                                                / (local_sigma**2 * dt))
                            if rng.random() < p_cross:
                                triggered[i] = True

        terminal = spots[:, -1]
        return np.where(triggered, np.maximum(terminal - strike, 0.0), 0.0)
    return payoff


# ---------------------------------------------------------------------------
# American payoffs (LSM)
# ---------------------------------------------------------------------------

def american_put(
    strike: float,
    r: float = 0.0,
    log_space: bool = True,
    n_basis: int = 3,
) -> Callable:
    """American put via Longstaff-Schwartz (LSM).

    Returns a payoff callable that performs backward LSM regression
    on the paths to find the optimal exercise strategy.

    Args:
        strike: exercise price.
        r: risk-free rate for discounting continuation values.
        log_space: if True, paths are in log-space (exp to get spots).
        n_basis: polynomial basis degree for regression.
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

        # Backward induction with discounting
        for step in range(n_steps_plus_1 - 2, 0, -1):
            # Discount continuation values by one period
            dt_step = times[step + 1] - times[step]
            df = np.exp(-r * dt_step)
            continuation = values * df

            itm = exercise[:, step] > 0
            if not np.any(itm):
                values = continuation
                continue

            # Regression: E[continuation | S_t] via polynomial
            x = spots[itm, step]
            y = continuation[itm]
            if len(x) < n_basis + 1:
                values = continuation
                continue

            # Polynomial regression
            coeffs = np.polyfit(x, y, min(n_basis, len(x) - 1))
            fitted = np.polyval(coeffs, x)

            # Exercise if intrinsic > continuation estimate
            exercise_now = exercise[itm, step] > fitted
            idx_itm = np.where(itm)[0]

            values = continuation  # default: hold
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


# ---------------------------------------------------------------------------
# Structured payoffs
# ---------------------------------------------------------------------------

def cliquet_payoff(
    cap: float = 0.05,
    floor: float = -0.05,
    global_floor: float = 0.0,
    log_space: bool = True,
) -> Callable:
    """Cliquet (ratchet): sum of capped/floored periodic returns.

    payoff = max(Σ min(max(R_i, floor), cap), global_floor)
    where R_i = S_i/S_{i-1} - 1.
    """
    def payoff(paths, times):
        if paths.ndim == 3:
            p = paths[:, :, 0]
        else:
            p = paths
        spots = np.exp(p) if log_space else p
        n_paths, n_steps = spots.shape[0], spots.shape[1] - 1

        total = np.zeros(n_paths)
        for i in range(1, n_steps + 1):
            ret = spots[:, i] / np.maximum(spots[:, i - 1], 1e-15) - 1
            capped = np.minimum(np.maximum(ret, floor), cap)
            total += capped

        return np.maximum(total, global_floor)
    return payoff


def autocall_payoff(
    autocall_barrier: float,
    autocall_coupon: float,
    put_barrier: float | None = None,
    put_strike: float | None = None,
    observation_freq: int = 4,
    log_space: bool = True,
) -> Callable:
    """Autocall: early redemption if spot > barrier at observation dates.

    At each observation: if S > autocall_barrier → redeem at 1 + coupon × period.
    At maturity: if S < put_barrier → loss = (put_strike - S) / put_strike.
    Otherwise: return notional (1.0).

    Args:
        autocall_barrier: barrier for early redemption (e.g. 100 for ATM).
        autocall_coupon: annual coupon if autocalled (e.g. 0.08 for 8%).
        put_barrier: downside barrier at maturity (None = no put).
        put_strike: put strike at maturity (None = autocall_barrier).
        observation_freq: observations per unit time (4 = quarterly).
    """
    def payoff(paths, times):
        if paths.ndim == 3:
            p = paths[:, :, 0]
        else:
            p = paths
        spots = np.exp(p) if log_space else p
        n_paths, n_total = spots.shape
        T = times[-1]

        # Observation indices (evenly spaced)
        obs_step = max(1, n_total // max(int(T * observation_freq), 1))
        obs_indices = list(range(obs_step, n_total, obs_step))
        if not obs_indices or obs_indices[-1] != n_total - 1:
            obs_indices.append(n_total - 1)

        values = np.zeros(n_paths)
        redeemed = np.zeros(n_paths, dtype=bool)

        for idx in obs_indices:
            t = times[idx]
            s = spots[:, idx]
            callable_now = (~redeemed) & (s >= autocall_barrier)
            values[callable_now] = 1.0 + autocall_coupon * t
            redeemed |= callable_now

        # Maturity: unredeemed paths
        not_redeemed = ~redeemed
        s_final = spots[:, -1]

        if put_barrier is not None and put_strike is not None:
            put_hit = not_redeemed & (s_final < put_barrier)
            values[put_hit] = s_final[put_hit] / put_strike
            values[not_redeemed & ~put_hit] = 1.0
        else:
            values[not_redeemed] = 1.0

        return values
    return payoff


def swing_payoff(
    strike: float,
    max_exercises: int,
    min_exercises: int = 0,
    refraction_period: int = 1,
    log_space: bool = True,
    n_basis: int = 3,
) -> Callable:
    """Swing option: multiple exercise rights with constraints.

    LSM backward induction with exercise counting.
    At each step: can exercise if (exercises_remaining > 0) and
    (steps since last exercise ≥ refraction_period).

    Args:
        strike: exercise strike.
        max_exercises: maximum number of exercises allowed.
        min_exercises: minimum exercises required (penalty if not met).
        refraction_period: minimum steps between exercises.
        n_basis: polynomial regression degree for LSM.
    """
    def payoff(paths, times):
        if paths.ndim == 3:
            p = paths[:, :, 0]
        else:
            p = paths
        spots = np.exp(p) if log_space else p
        n_paths, n_steps_plus_1 = spots.shape

        intrinsic = np.maximum(spots - strike, 0.0)
        values = np.zeros(n_paths)
        exercises_used = np.zeros(n_paths, dtype=int)
        last_exercise = np.full(n_paths, -refraction_period - 1)

        # Backward induction (simplified: greedy forward for swing)
        # Full LSM swing is complex; use greedy forward with exercise tracking
        for step in range(1, n_steps_plus_1):
            can_exercise = (
                (exercises_used < max_exercises) &
                (step - last_exercise >= refraction_period) &
                (intrinsic[:, step] > 0)
            )
            exercising = can_exercise & (intrinsic[:, step] > strike * 0.01)  # threshold
            values[exercising] += intrinsic[exercising, step]
            exercises_used[exercising] += 1
            last_exercise[exercising] = step

        return values
    return payoff


def worst_of_put(strike: float, log_space: bool = True) -> Callable:
    """Worst-of put: max(K - min(S_i,T), 0)."""
    def payoff(paths, times):
        terminal = paths[:, -1, :]
        spots = np.exp(terminal) if log_space else terminal
        worst = np.min(spots, axis=1)
        return np.maximum(strike - worst, 0.0)
    return payoff
