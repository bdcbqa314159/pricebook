"""Multi-asset exotic options: rainbow, knockout baskets, conditional barriers.

* :func:`rainbow_option` — best-of/worst-of N assets.
* :func:`knockout_basket` — barrier on one asset, payoff on another.
* :func:`conditional_barrier` — knock-in on asset A, knock-out on asset B.
* :func:`multi_asset_digital_range` — pays if ALL assets stay in range.

References:
    Johnson, *Options on Max or Min of Several Assets*, JFQA, 1987.
    De Weert, *Exotic Options Trading*, Wiley, 2008.
    Bouzoubaa & Osseiran, *Exotic Options and Hybrids*, Wiley, 2010.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _simulate_correlated(spots, rates, divs, vols, corr, T, n_paths, n_steps, seed):
    """Simulate N correlated GBM assets."""
    n = len(spots)
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(corr + 1e-8 * np.eye(n))

    spots_arr = np.array(spots, dtype=float)
    vols_arr = np.array(vols)
    divs_arr = np.array(divs)

    paths = np.zeros((n_paths, n_steps + 1, n))
    paths[:, 0, :] = spots_arr

    for step in range(n_steps):
        Z = rng.standard_normal((n_paths, n)) @ L.T
        drifts = (rates - divs_arr - 0.5 * vols_arr**2) * dt
        paths[:, step + 1, :] = paths[:, step, :] * np.exp(drifts + vols_arr * sqrt_dt * Z)

    return paths


@dataclass
class RainbowResult:
    """Rainbow option result."""
    price: float
    rainbow_type: str       # "best_of_call", "worst_of_call", "atlas", etc.
    n_assets: int


def rainbow_option(
    spots: list[float],
    strikes: list[float] | float,
    rate: float,
    dividend_yields: list[float],
    vols: list[float],
    correlations: np.ndarray,
    T: float,
    rainbow_type: str = "best_of_call",
    n_paths: int = 20_000,
    n_steps: int = 1,
    seed: int | None = 42,
) -> RainbowResult:
    """N-asset rainbow option.

    Types:
    - "best_of_call": max(S₁ − K₁, S₂ − K₂, ..., 0)
    - "worst_of_call": max(min(S₁ − K₁, S₂ − K₂, ...), 0)
    - "best_of_assets": max(S₁, S₂, ...) − K
    - "atlas": remove best + worst, average remainder − K

    Args:
        strikes: single strike or per-asset strikes.
    """
    n = len(spots)
    if isinstance(strikes, (int, float)):
        K = np.full(n, float(strikes))
    else:
        K = np.array(strikes)

    paths = _simulate_correlated(spots, rate, dividend_yields, vols,
                                  correlations, T, n_paths, n_steps, seed)
    S_T = paths[:, -1, :]
    df = math.exp(-rate * T)

    if rainbow_type == "best_of_call":
        # max over assets of (S_i - K_i), then max with 0
        per_asset = S_T - K
        payoff = np.maximum(per_asset.max(axis=1), 0.0)
    elif rainbow_type == "worst_of_call":
        per_asset = S_T - K
        payoff = np.maximum(per_asset.min(axis=1), 0.0)
    elif rainbow_type == "best_of_assets":
        payoff = np.maximum(S_T.max(axis=1) - K[0], 0.0)
    elif rainbow_type == "atlas":
        # Remove best + worst, average remainder
        if n < 3:
            payoff = np.maximum(S_T.mean(axis=1) - K[0], 0.0)
        else:
            sorted_S = np.sort(S_T, axis=1)
            middle = sorted_S[:, 1:-1]
            payoff = np.maximum(middle.mean(axis=1) - K[0], 0.0)
    else:
        raise ValueError(f"Unknown rainbow_type: {rainbow_type}")

    price = df * float(payoff.mean())
    return RainbowResult(price, rainbow_type, n)


@dataclass
class KnockoutBasketResult:
    """Knockout basket result."""
    price: float
    knockout_probability: float
    barrier_asset_idx: int
    payoff_asset_idx: int


def knockout_basket(
    spots: list[float],
    rate: float,
    dividend_yields: list[float],
    vols: list[float],
    correlations: np.ndarray,
    T: float,
    barrier_asset_idx: int,
    barrier_level: float,
    is_up_barrier: bool,
    payoff_asset_idx: int,
    strike: float,
    is_call: bool = True,
    n_paths: int = 20_000,
    n_steps: int = 100,
    seed: int | None = 42,
) -> KnockoutBasketResult:
    """Knockout basket: barrier on one asset, payoff on another.

    E.g. knock-out on FX rate, payoff on equity index.
    """
    paths = _simulate_correlated(spots, rate, dividend_yields, vols,
                                  correlations, T, n_paths, n_steps, seed)
    df = math.exp(-rate * T)

    barrier_paths = paths[:, :, barrier_asset_idx]
    payoff_paths = paths[:, -1, payoff_asset_idx]

    if is_up_barrier:
        alive = np.all(barrier_paths < barrier_level, axis=1)
    else:
        alive = np.all(barrier_paths > barrier_level, axis=1)

    if is_call:
        payoff = np.maximum(payoff_paths - strike, 0.0) * alive
    else:
        payoff = np.maximum(strike - payoff_paths, 0.0) * alive

    ko_prob = float(1 - alive.mean())
    price = df * float(payoff.mean())

    return KnockoutBasketResult(price, ko_prob, barrier_asset_idx, payoff_asset_idx)


@dataclass
class ConditionalBarrierResult:
    """Conditional barrier result."""
    price: float
    ki_probability: float
    ko_probability: float
    both_triggered: float


def conditional_barrier(
    spots: list[float],
    rate: float,
    dividend_yields: list[float],
    vols: list[float],
    correlations: np.ndarray,
    T: float,
    ki_asset_idx: int,
    ki_level: float,
    ki_is_up: bool,
    ko_asset_idx: int,
    ko_level: float,
    ko_is_up: bool,
    payoff_asset_idx: int,
    strike: float,
    is_call: bool = True,
    n_paths: int = 20_000,
    n_steps: int = 100,
    seed: int | None = 42,
) -> ConditionalBarrierResult:
    """Conditional barrier: knock-in on one asset, knock-out on another.

    Option only exists if knock-in triggered AND knock-out NOT triggered.
    """
    paths = _simulate_correlated(spots, rate, dividend_yields, vols,
                                  correlations, T, n_paths, n_steps, seed)
    df = math.exp(-rate * T)

    ki_paths = paths[:, :, ki_asset_idx]
    ko_paths = paths[:, :, ko_asset_idx]

    if ki_is_up:
        knocked_in = np.any(ki_paths >= ki_level, axis=1)
    else:
        knocked_in = np.any(ki_paths <= ki_level, axis=1)

    if ko_is_up:
        knocked_out = np.any(ko_paths >= ko_level, axis=1)
    else:
        knocked_out = np.any(ko_paths <= ko_level, axis=1)

    active = knocked_in & ~knocked_out

    S_T = paths[:, -1, payoff_asset_idx]
    if is_call:
        payoff = np.maximum(S_T - strike, 0.0) * active
    else:
        payoff = np.maximum(strike - S_T, 0.0) * active

    price = df * float(payoff.mean())

    return ConditionalBarrierResult(
        price=price,
        ki_probability=float(knocked_in.mean()),
        ko_probability=float(knocked_out.mean()),
        both_triggered=float((knocked_in & knocked_out).mean()),
    )


@dataclass
class MultiAssetDigitalRangeResult:
    """Multi-asset digital range result."""
    price: float
    all_in_range_probability: float
    n_assets: int


def multi_asset_digital_range(
    spots: list[float],
    rate: float,
    dividend_yields: list[float],
    vols: list[float],
    correlations: np.ndarray,
    T: float,
    range_lows: list[float],
    range_highs: list[float],
    payout: float = 1.0,
    n_paths: int = 20_000,
    n_steps: int = 100,
    seed: int | None = 42,
) -> MultiAssetDigitalRangeResult:
    """Digital range on multiple assets: pays if ALL stay in their range."""
    n = len(spots)
    paths = _simulate_correlated(spots, rate, dividend_yields, vols,
                                  correlations, T, n_paths, n_steps, seed)
    df = math.exp(-rate * T)

    all_in = np.ones(n_paths, dtype=bool)
    for i in range(n):
        asset_in = np.all(
            (paths[:, :, i] >= range_lows[i]) & (paths[:, :, i] <= range_highs[i]),
            axis=1,
        )
        all_in &= asset_in

    prob = float(all_in.mean())
    price = df * payout * prob

    return MultiAssetDigitalRangeResult(price, prob, n)
