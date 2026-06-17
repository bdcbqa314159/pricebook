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
        return dict(vars(self))


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
        return dict(vars(self))


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


# ═══════════════════════════════════════════════════════════════
# CD6: Flash Loans, Yield Routing, Per-Collateral Risk
# ═══════════════════════════════════════════════════════════════

@dataclass
class FlashLoanResult:
    """Flash loan economics."""
    borrow_amount: float
    fee: float
    profit: float
    profitable: bool
    gas_cost: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def flash_loan_arb(
    borrow_amount: float,
    arb_profit_pct: float,
    flash_fee_pct: float = 0.0009,
    gas_cost_usd: float = 20.0,
) -> FlashLoanResult:
    """Flash loan arbitrage profitability.

    Flash loan: borrow, arb, repay in one transaction.
    Profit = arb_return − flash_fee − gas.

    Args:
        borrow_amount: flash loan size.
        arb_profit_pct: arbitrage return (decimal).
        flash_fee_pct: Aave flash loan fee (0.09%).
        gas_cost_usd: gas cost for the transaction.
    """
    gross = borrow_amount * arb_profit_pct
    fee = borrow_amount * flash_fee_pct
    net = gross - fee - gas_cost_usd
    return FlashLoanResult(borrow_amount, fee, net, net > 0, gas_cost_usd)


@dataclass
class YieldRouteResult:
    """Multi-protocol yield routing result."""
    net_apy: float
    steps: list[dict]
    total_gas_cost: float
    capital_efficiency: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def yield_route(
    deposit_amount: float,
    routes: list[dict],
) -> YieldRouteResult:
    """Multi-protocol yield routing optimisation.

    Each route: {"protocol", "action" (deposit/borrow/lend), "apy", "gas_usd"}.
    Net APY = Σ(apy_i × weight_i) − Σ(gas / deposit × annualise).

    Args:
        deposit_amount: initial capital.
        routes: ordered list of protocol steps.
    """
    total_apy = 0.0
    total_gas = 0.0
    steps = []

    for r in routes:
        apy = r.get("apy", 0)
        gas = r.get("gas_usd", 5)
        action = r.get("action", "deposit")
        protocol = r.get("protocol", "unknown")

        if action == "borrow":
            total_apy -= apy  # borrowing costs
        else:
            total_apy += apy

        total_gas += gas
        steps.append({"protocol": protocol, "action": action, "apy": apy})

    gas_drag = total_gas / deposit_amount * 365 if deposit_amount > 0 else 0
    net = total_apy - gas_drag
    efficiency = net / total_apy if total_apy > 0 else 0

    return YieldRouteResult(net * 100, steps, total_gas, efficiency)


@dataclass
class CollateralRiskParams:
    """Per-collateral risk parameters (Aave v3 style)."""
    asset: str
    max_ltv: float
    liquidation_ltv: float
    liquidation_bonus: float    # bonus for liquidators (e.g. 5%)
    supply_cap: float           # max supply in protocol
    borrow_cap: float           # max borrow
    is_isolated: bool           # isolated collateral mode
    e_mode_category: int        # efficiency mode (0 = none)

    def to_dict(self) -> dict:
        return dict(vars(self))


# Standard risk params for major assets (Aave v3 Ethereum)
AAVE_V3_RISK_PARAMS = {
    "ETH": CollateralRiskParams("ETH", 0.825, 0.86, 0.05, 0, 0, False, 1),
    "WBTC": CollateralRiskParams("WBTC", 0.73, 0.78, 0.065, 0, 0, False, 0),
    "USDC": CollateralRiskParams("USDC", 0.77, 0.80, 0.045, 0, 0, False, 2),
    "DAI": CollateralRiskParams("DAI", 0.67, 0.77, 0.045, 0, 0, False, 2),
    "LINK": CollateralRiskParams("LINK", 0.68, 0.74, 0.07, 0, 0, False, 0),
    "AAVE": CollateralRiskParams("AAVE", 0.66, 0.73, 0.075, 0, 0, False, 0),
    "stETH": CollateralRiskParams("stETH", 0.81, 0.84, 0.05, 0, 0, False, 1),
}
