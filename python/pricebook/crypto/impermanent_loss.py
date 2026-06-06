"""Impermanent loss: calculation, breakeven, hedging.

* :func:`impermanent_loss` — IL for constant product AMM.
* :func:`il_vs_fees_breakeven` — fee rate needed to offset IL.
* :func:`il_v3_concentrated` — IL for concentrated liquidity.
* :func:`il_hedge_with_options` — hedge IL with put options.

References:
    Pintail, *Uniswap: A Good Deal for Liquidity Providers?*, 2019.
    Milionis et al., *Automated Market Making and Loss-Versus-Rebalancing*, 2022.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ILResult:
    """Impermanent loss result."""
    il_pct: float               # IL as % (negative = loss)
    il_factor: float            # LP_value / HODL_value
    price_change_pct: float     # price change that caused IL
    breakeven_fee_pct: float    # annual fee needed to offset

    def to_dict(self) -> dict:
        return vars(self)


def impermanent_loss(price_ratio: float) -> ILResult:
    """Impermanent loss for constant product AMM (v2).

    IL = 2√r / (1 + r) − 1

    where r = P_final / P_initial.

    IL is always ≤ 0 (you always lose vs holding).
    IL = 0 when r = 1 (no price change).
    IL = −100% when r → 0 or r → ∞.

    Args:
        price_ratio: final price / initial price.
    """
    r = max(price_ratio, 1e-10)
    il_factor = 2 * math.sqrt(r) / (1 + r)
    il_pct = (il_factor - 1) * 100
    price_change = (r - 1) * 100

    # Breakeven: annualised fee income needed to offset IL
    # Assumes fees accumulate linearly, IL is instantaneous
    breakeven = abs(il_pct)  # simplified: need this % in fees per year

    return ILResult(
        il_pct=il_pct,
        il_factor=il_factor,
        price_change_pct=price_change,
        breakeven_fee_pct=breakeven,
    )


def il_vs_fees_breakeven(
    daily_volume_to_tvl: float,
    fee_rate: float = 0.003,
    days: int = 365,
) -> float:
    """Maximum price move before IL exceeds accumulated fees.

    Given daily volume/TVL ratio and fee rate, compute the
    breakeven price ratio where IL = accumulated fees.

    Args:
        daily_volume_to_tvl: daily trading volume / pool TVL.
        fee_rate: swap fee (e.g. 0.003 for 0.3%).
        days: holding period.

    Returns:
        Breakeven price ratio (e.g. 1.5 means 50% price move).
    """
    total_fees_pct = daily_volume_to_tvl * fee_rate * days * 100

    # Binary search for r where |IL(r)| = total_fees_pct
    lo, hi = 1.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        il = abs(impermanent_loss(mid).il_pct)
        if il < total_fees_pct:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def il_v3_concentrated(
    price_ratio: float,
    range_lower: float,
    range_upper: float,
) -> ILResult:
    """IL for Uniswap v3 concentrated liquidity.

    Concentrated liquidity amplifies both fees AND IL.
    If price moves outside [range_lower, range_upper],
    the position becomes 100% one token (maximum IL).

    Amplification factor ≈ √(range_upper / range_lower).

    Args:
        price_ratio: final / initial price.
        range_lower: lower bound of LP range (as ratio to initial).
        range_upper: upper bound of LP range.
    """
    r = max(price_ratio, 1e-10)

    # If price stays in range: IL amplified by concentration
    if range_lower <= r <= range_upper:
        # Amplification factor
        amp = math.sqrt(range_upper / range_lower) if range_lower > 0 else 1
        base_il = impermanent_loss(r)
        il_pct = base_il.il_pct * amp
        il_factor = 1 + il_pct / 100
    else:
        # Price moved out of range: position is 100% one token
        if r < range_lower:
            # All token Y (the one that appreciated)
            il_factor = math.sqrt(r / range_lower) * 2 * math.sqrt(range_lower) / (1 + range_lower)
        else:
            # All token X (the one that depreciated)
            il_factor = math.sqrt(r / range_upper) * 2 * math.sqrt(range_upper) / (1 + range_upper)
        il_pct = (il_factor - 1) * 100

    return ILResult(
        il_pct=il_pct,
        il_factor=max(il_factor, 0),
        price_change_pct=(r - 1) * 100,
        breakeven_fee_pct=abs(il_pct),
    )


def il_table(
    price_changes: list[float] | None = None,
) -> list[dict]:
    """Generate IL table for various price changes.

    Default: ±10%, ±20%, ±30%, ±50%, ±75%, 2×, 3×, 5×.
    """
    if price_changes is None:
        price_changes = [0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0, 5.0]

    result = []
    for r in price_changes:
        il = impermanent_loss(r)
        result.append({
            "price_ratio": r,
            "price_change_pct": il.price_change_pct,
            "il_pct": round(il.il_pct, 4),
        })
    return result
