"""Liquidation cascade simulation: systemic risk from correlated liquidations.

* :func:`simulate_cascade` — agent-based liquidation cascade.
* :func:`cascade_risk_score` — systemic risk from leverage concentration.

References:
    Capponi & Jia, *The Adoption of Blockchain-Based DeFi*, 2021.
    Qin et al., *An Empirical Study of DeFi Liquidations*, 2021.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LeveragedPosition:
    """Single leveraged position in the market."""
    size_usd: float
    leverage: float
    liquidation_price: float
    side: str                   # "long" or "short"


@dataclass
class CascadeResult:
    """Liquidation cascade simulation result."""
    initial_price: float
    final_price: float
    price_impact_pct: float
    n_liquidations: int
    total_liquidated_usd: float
    cascade_depth: int          # rounds of cascading
    insurance_fund_impact: float
    max_single_round_usd: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def simulate_cascade(
    positions: list[LeveragedPosition],
    initial_price: float,
    price_shock_pct: float,
    market_depth_usd: float = 10_000_000.0,
    price_impact_per_usd: float = 0.00001,
    insurance_fund: float = 50_000_000.0,
    max_rounds: int = 20,
) -> CascadeResult:
    """Simulate a liquidation cascade.

    1. Apply initial price shock.
    2. Check which positions are liquidated.
    3. Liquidated positions cause market sell (longs) or buy (shorts).
    4. Price impact from liquidation volume triggers further liquidations.
    5. Repeat until no new liquidations or max rounds.

    Args:
        positions: list of leveraged positions.
        initial_price: starting price.
        price_shock_pct: initial shock (e.g. -0.10 for -10%).
        market_depth_usd: total liquidity available.
        price_impact_per_usd: price impact per USD of forced selling.
        insurance_fund: available insurance fund.
        max_rounds: maximum cascade rounds.
    """
    price = initial_price * (1 + price_shock_pct)
    remaining = list(positions)
    total_liquidated = 0.0
    total_rounds = 0
    max_round_vol = 0.0
    insurance_used = 0.0

    for round_num in range(max_rounds):
        # Find positions that are liquidated at current price
        to_liquidate = []
        still_alive = []

        for pos in remaining:
            if pos.side == "long" and price <= pos.liquidation_price:
                to_liquidate.append(pos)
            elif pos.side == "short" and price >= pos.liquidation_price:
                to_liquidate.append(pos)
            else:
                still_alive.append(pos)

        if not to_liquidate:
            break

        # Volume from liquidations
        liq_volume = sum(p.size_usd for p in to_liquidate)
        total_liquidated += liq_volume
        max_round_vol = max(max_round_vol, liq_volume)
        total_rounds = round_num + 1

        # Price impact: net selling (longs) or buying (shorts)
        net_sell = sum(p.size_usd for p in to_liquidate if p.side == "long")
        net_buy = sum(p.size_usd for p in to_liquidate if p.side == "short")
        net_pressure = net_sell - net_buy  # positive = downward pressure

        price_change = -net_pressure * price_impact_per_usd
        price += price_change

        # Insurance fund absorbs losses from failed liquidations
        failed_pct = max(0, liq_volume / market_depth_usd - 1) * 0.1
        insurance_used += liq_volume * failed_pct

        remaining = still_alive

    return CascadeResult(
        initial_price=initial_price,
        final_price=price,
        price_impact_pct=(price / initial_price - 1) * 100,
        n_liquidations=len(positions) - len(remaining),
        total_liquidated_usd=total_liquidated,
        cascade_depth=total_rounds,
        insurance_fund_impact=insurance_used,
        max_single_round_usd=max_round_vol,
    )


@dataclass
class CascadeRiskScore:
    """Systemic cascade risk score."""
    score: float                # 0–100
    leverage_concentration: float
    liquidation_cluster_pct: float
    insurance_fund_ratio: float
    risk_level: str

    def to_dict(self) -> dict:
        return dict(vars(self))


def cascade_risk_score(
    positions: list[LeveragedPosition],
    current_price: float,
    insurance_fund: float,
    total_open_interest: float,
) -> CascadeRiskScore:
    """Score systemic liquidation cascade risk.

    High risk when:
    - Many positions cluster near the same liquidation price.
    - Average leverage is high.
    - Insurance fund is small relative to open interest.

    Args:
        positions: all leveraged positions.
        current_price: current market price.
        insurance_fund: available insurance fund.
        total_open_interest: total open interest in USD.
    """
    if not positions:
        return CascadeRiskScore(0, 0, 0, 0, "low")

    # Leverage concentration (Herfindahl on leverage buckets)
    avg_leverage = sum(p.leverage for p in positions) / len(positions)

    # Liquidation clustering: what % liquidates within 5% of current price
    near_liq = sum(1 for p in positions
                   if abs(p.liquidation_price - current_price) / current_price < 0.05)
    cluster_pct = near_liq / len(positions) * 100

    # Insurance fund ratio
    if_ratio = insurance_fund / total_open_interest if total_open_interest > 0 else 1

    # Score
    leverage_score = min(avg_leverage / 20 * 30, 30)
    cluster_score = min(cluster_pct / 10 * 30, 30)
    if_score = max(0, 20 - if_ratio * 100)
    general = 10 if len(positions) > 1000 else 5

    total = leverage_score + cluster_score + if_score + general
    total = min(total, 100)

    level = "low" if total < 25 else "medium" if total < 50 else "high" if total < 75 else "critical"

    return CascadeRiskScore(
        score=total,
        leverage_concentration=avg_leverage,
        liquidation_cluster_pct=cluster_pct,
        insurance_fund_ratio=if_ratio,
        risk_level=level,
    )
