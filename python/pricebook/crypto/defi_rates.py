"""DeFi lending rates: Aave/Compound utilization-based models.

* :func:`aave_rate` — Aave-style kink rate model.
* :func:`compound_rate` — Compound-style rate model.
* :func:`liquidation_threshold` — LTV and liquidation levels.

References:
    Aave, *Protocol Governance — Interest Rate Strategy*.
    Compound, *Interest Rate Model Specification*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DeFiRateResult:
    """DeFi lending/borrowing rate result."""
    supply_apy: float           # annualised supply rate
    borrow_apy: float           # annualised borrow rate
    utilization: float          # current utilization (0–1)
    spread: float               # borrow − supply
    protocol: str

    def to_dict(self) -> dict:
        return vars(self)


def aave_rate(
    utilization: float,
    base_rate: float = 0.0,
    slope1: float = 0.04,
    slope2: float = 0.75,
    optimal_utilization: float = 0.80,
    reserve_factor: float = 0.10,
) -> DeFiRateResult:
    """Aave-style kink rate model.

    Below optimal: borrow_rate = base + slope1 × U/U_opt
    Above optimal: borrow_rate = base + slope1 + slope2 × (U − U_opt)/(1 − U_opt)

    Supply rate = borrow_rate × U × (1 − reserve_factor).

    Args:
        utilization: current pool utilization (0–1).
        base_rate: minimum borrow rate.
        slope1: rate slope below kink.
        slope2: rate slope above kink (steep).
        optimal_utilization: kink point.
        reserve_factor: protocol fee on interest.
    """
    U = max(0, min(utilization, 1))
    U_opt = optimal_utilization

    if U <= U_opt:
        borrow = base_rate + slope1 * U / U_opt
    else:
        borrow = base_rate + slope1 + slope2 * (U - U_opt) / (1 - U_opt)

    supply = borrow * U * (1 - reserve_factor)

    return DeFiRateResult(
        supply_apy=supply,
        borrow_apy=borrow,
        utilization=U,
        spread=borrow - supply,
        protocol="aave",
    )


def compound_rate(
    utilization: float,
    base_rate_per_block: float = 0.0,
    multiplier_per_block: float = 0.05,
    jump_multiplier_per_block: float = 1.09,
    kink: float = 0.80,
    blocks_per_year: int = 2_102_400,
    reserve_factor: float = 0.10,
) -> DeFiRateResult:
    """Compound-style jump rate model.

    Below kink: rate = base + multiplier × U
    Above kink: rate = base + multiplier × kink + jump × (U − kink)

    Args:
        utilization: pool utilization (0–1).
        base_rate_per_block: base rate per block.
        multiplier_per_block: slope per block below kink.
        jump_multiplier_per_block: steep slope above kink.
        kink: utilization kink point.
        blocks_per_year: for annualisation (ETH ~2.1M).
    """
    U = max(0, min(utilization, 1))

    if U <= kink:
        borrow_per_block = base_rate_per_block + multiplier_per_block * U
    else:
        normal = base_rate_per_block + multiplier_per_block * kink
        excess = jump_multiplier_per_block * (U - kink)
        borrow_per_block = normal + excess

    borrow_apy = borrow_per_block * blocks_per_year
    supply_apy = borrow_apy * U * (1 - reserve_factor)

    return DeFiRateResult(
        supply_apy=supply_apy,
        borrow_apy=borrow_apy,
        utilization=U,
        spread=borrow_apy - supply_apy,
        protocol="compound",
    )


@dataclass
class LiquidationThresholdResult:
    """Collateral and liquidation levels."""
    max_ltv: float              # maximum loan-to-value at borrow
    liquidation_ltv: float      # LTV at which liquidation triggers
    health_factor: float        # >1 safe, <1 liquidatable
    liquidation_price: float    # collateral price at liquidation
    current_ltv: float

    def to_dict(self) -> dict:
        return vars(self)


def liquidation_threshold(
    collateral_value: float,
    debt_value: float,
    collateral_price: float,
    max_ltv: float = 0.80,
    liquidation_ltv: float = 0.825,
) -> LiquidationThresholdResult:
    """DeFi lending position health and liquidation price.

    health_factor = (collateral × liquidation_ltv) / debt
    liquidation_price = debt / (collateral_qty × liquidation_ltv)

    Args:
        collateral_value: current collateral value in USD.
        debt_value: current debt value in USD.
        collateral_price: current price of collateral asset.
        max_ltv: maximum LTV allowed at borrow time.
        liquidation_ltv: LTV at which position is liquidated.
    """
    current_ltv = debt_value / collateral_value if collateral_value > 0 else float('inf')
    health = (collateral_value * liquidation_ltv) / debt_value if debt_value > 0 else float('inf')

    collateral_qty = collateral_value / collateral_price if collateral_price > 0 else 0
    liq_price = debt_value / (collateral_qty * liquidation_ltv) if collateral_qty > 0 and liquidation_ltv > 0 else 0

    return LiquidationThresholdResult(
        max_ltv=max_ltv,
        liquidation_ltv=liquidation_ltv,
        health_factor=health,
        liquidation_price=liq_price,
        current_ltv=current_ltv,
    )
