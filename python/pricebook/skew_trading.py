"""Skew trading: RR strategies, skew mean-reversion, skew-carry.

* :func:`risk_reversal_strategy` — long OTM put, short OTM call.
* :func:`skew_mean_reversion_signal` — z-score of current skew vs history.
* :func:`skew_carry_trade` — earn theta from skew normalisation.
* :func:`cross_asset_skew_comparison` — compare skew across asset classes.

References:
    Gatheral, *The Volatility Surface*, Wiley, 2006.
    Bollen & Whaley, *Does Net Buying Pressure Affect the Shape of Implied Volatility Functions?*, JF, 2004.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class RiskReversalResult:
    rr_value: float             # 25D call vol − 25D put vol (negative = put skew)
    cost: float                 # net premium paid
    delta: float                # residual delta
    strategy: str

def risk_reversal_strategy(
    vol_25d_call: float, vol_25d_put: float, atm_vol: float,
    spot: float, rate: float, T: float,
) -> RiskReversalResult:
    rr = vol_25d_call - vol_25d_put
    # Approximate RR cost: proportional to skew
    cost = abs(rr) * spot * math.sqrt(T) * 0.4  # rough N'(d1) scaling
    delta = 0.5 - 0.5  # approximately delta-neutral
    strategy = "long_put_skew" if rr < 0 else "long_call_skew"
    return RiskReversalResult(float(rr), float(cost), float(delta), strategy)


@dataclass
class SkewMeanReversionSignal:
    current_skew: float
    historical_mean: float
    z_score: float
    signal: str                 # "buy_skew", "sell_skew", "neutral"

def skew_mean_reversion_signal(
    current_rr: float, historical_rrs: list[float],
) -> SkewMeanReversionSignal:
    arr = np.array(historical_rrs)
    mu = float(arr.mean()); sigma = float(arr.std())
    z = (current_rr - mu) / max(sigma, 1e-10)
    if z > 1.0: signal = "sell_skew"
    elif z < -1.0: signal = "buy_skew"
    else: signal = "neutral"
    return SkewMeanReversionSignal(current_rr, mu, float(z), signal)


@dataclass
class SkewCarryResult:
    theta_from_skew: float      # daily theta from skew position
    breakeven_days: int         # days to breakeven
    carry_ratio: float          # annualised carry / vol

def skew_carry_trade(
    rr_cost: float, daily_theta: float, skew_vol: float,
) -> SkewCarryResult:
    if abs(daily_theta) > 1e-10:
        breakeven = int(abs(rr_cost / daily_theta))
    else:
        breakeven = 999
    carry = daily_theta * 252
    ratio = carry / max(skew_vol, 1e-10)
    return SkewCarryResult(float(daily_theta), breakeven, float(ratio))


@dataclass
class CrossAssetSkewResult:
    rankings: list[tuple[str, float]]
    steepest_skew: str
    flattest_skew: str

def cross_asset_skew_comparison(
    skews: dict[str, float],
) -> CrossAssetSkewResult:
    entries = sorted(skews.items(), key=lambda x: x[1])
    return CrossAssetSkewResult(entries, entries[0][0], entries[-1][0])
