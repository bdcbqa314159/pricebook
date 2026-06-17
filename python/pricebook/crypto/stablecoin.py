"""Stablecoin peg mechanics: reserve backing, depeg risk, arbitrage.

* :func:`peg_health` — stablecoin peg deviation and risk.
* :func:`reserve_backing` — collateral ratio analysis.
* :func:`depeg_risk_score` — composite depeg risk score.
* :func:`stablecoin_arb` — arbitrage from peg deviation.

References:
    Lyons & Viswanath-Natraj, *What Keeps Stablecoins Stable?*, JIF, 2023.
    Kozhan & Viswanath-Natraj, *Decentralized Stablecoins and Collateral Risk*, 2021.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class StablecoinType(Enum):
    FIAT_BACKED = "fiat_backed"         # USDC, USDT
    CRYPTO_BACKED = "crypto_backed"     # DAI, LUSD
    ALGORITHMIC = "algorithmic"         # FRAX, (former UST)
    COMMODITY_BACKED = "commodity_backed"  # PAXG


@dataclass
class PegHealthResult:
    """Stablecoin peg health assessment."""
    current_price: float
    peg_target: float
    deviation_bps: float        # basis points from peg
    premium: bool               # trading above peg
    discount: bool              # trading below peg
    within_band: bool           # within normal ±50bp band
    severity: str               # "normal", "stress", "depeg"

    def to_dict(self) -> dict:
        return dict(vars(self))


def peg_health(
    current_price: float,
    peg_target: float = 1.0,
    stress_threshold_bps: float = 50.0,
    depeg_threshold_bps: float = 200.0,
) -> PegHealthResult:
    """Assess stablecoin peg deviation.

    Normal: ±50bp. Stress: 50–200bp. Depeg: >200bp.

    Args:
        current_price: market price of stablecoin.
        peg_target: target price (typically 1.00).
    """
    dev = (current_price - peg_target) / peg_target * 10_000

    if abs(dev) < stress_threshold_bps:
        severity = "normal"
    elif abs(dev) < depeg_threshold_bps:
        severity = "stress"
    else:
        severity = "depeg"

    return PegHealthResult(
        current_price=current_price,
        peg_target=peg_target,
        deviation_bps=dev,
        premium=dev > 0,
        discount=dev < 0,
        within_band=abs(dev) < stress_threshold_bps,
        severity=severity,
    )


@dataclass
class ReserveBackingResult:
    """Reserve/collateral backing analysis."""
    total_supply: float
    total_reserves: float
    collateral_ratio: float     # reserves / supply (>1 = over-collateralised)
    reserve_composition: dict[str, float]
    liquid_reserves_pct: float  # % of reserves that are liquid
    shortfall: float            # supply − reserves (negative = excess)

    def to_dict(self) -> dict:
        return dict(vars(self))


def reserve_backing(
    total_supply: float,
    reserves: dict[str, float],
    liquid_assets: list[str] | None = None,
) -> ReserveBackingResult:
    """Analyse stablecoin reserve backing.

    Over-collateralised (ratio > 1): safe.
    Under-collateralised (ratio < 1): depeg risk.

    Args:
        total_supply: total stablecoin supply.
        reserves: {asset_type: value} of reserve assets.
        liquid_assets: which reserve assets are liquid.
    """
    total_res = sum(reserves.values())
    ratio = total_res / total_supply if total_supply > 0 else 0
    shortfall = total_supply - total_res

    liquid = liquid_assets or list(reserves.keys())
    liquid_val = sum(v for k, v in reserves.items() if k in liquid)
    liquid_pct = liquid_val / total_res * 100 if total_res > 0 else 0

    return ReserveBackingResult(
        total_supply=total_supply,
        total_reserves=total_res,
        collateral_ratio=ratio,
        reserve_composition=reserves,
        liquid_reserves_pct=liquid_pct,
        shortfall=shortfall,
    )


@dataclass
class DepegRiskScore:
    """Composite depeg risk score (0–100)."""
    score: float                # 0 = safest, 100 = highest risk
    components: dict[str, float]
    risk_level: str             # "low", "medium", "high", "critical"

    def to_dict(self) -> dict:
        return dict(vars(self))


def depeg_risk_score(
    stablecoin_type: StablecoinType,
    collateral_ratio: float = 1.0,
    peg_deviation_bps: float = 0.0,
    daily_volume: float = 0.0,
    total_supply: float = 0.0,
    age_days: int = 365,
    audited: bool = True,
) -> DepegRiskScore:
    """Composite depeg risk score.

    Factors:
    - Type (algo > crypto-backed > fiat-backed).
    - Collateral ratio (lower = riskier).
    - Current peg deviation.
    - Volume/supply ratio (low liquidity = riskier).
    - Age (newer = riskier).
    - Audit status.
    """
    # Type risk
    type_score = {
        StablecoinType.FIAT_BACKED: 10,
        StablecoinType.COMMODITY_BACKED: 15,
        StablecoinType.CRYPTO_BACKED: 30,
        StablecoinType.ALGORITHMIC: 60,
    }.get(stablecoin_type, 50)

    # Collateral
    if collateral_ratio >= 1.5:
        collateral_score = 0
    elif collateral_ratio >= 1.0:
        collateral_score = 20 * (1.5 - collateral_ratio) / 0.5
    else:
        collateral_score = 20 + 30 * (1.0 - collateral_ratio)

    # Peg deviation
    dev_score = min(abs(peg_deviation_bps) / 10, 30)

    # Liquidity
    vol_ratio = daily_volume / total_supply if total_supply > 0 else 0
    liq_score = max(0, 15 - vol_ratio * 100)

    # Age
    age_score = max(0, 10 - age_days / 365 * 5)

    # Audit
    audit_score = 0 if audited else 10

    total = type_score + collateral_score + dev_score + liq_score + age_score + audit_score
    total = min(total, 100)

    if total < 25:
        level = "low"
    elif total < 50:
        level = "medium"
    elif total < 75:
        level = "high"
    else:
        level = "critical"

    return DepegRiskScore(
        score=total,
        components={
            "type": type_score,
            "collateral": collateral_score,
            "peg_deviation": dev_score,
            "liquidity": liq_score,
            "age": age_score,
            "audit": audit_score,
        },
        risk_level=level,
    )


def stablecoin_arb(
    price: float,
    peg: float = 1.0,
    redemption_fee: float = 0.001,
    gas_cost_usd: float = 5.0,
    trade_size: float = 100_000.0,
) -> dict:
    """Arbitrage opportunity from peg deviation.

    If price < peg: buy stablecoin, redeem for $1.
    If price > peg: mint stablecoin at $1, sell at premium.

    Args:
        price: market price.
        peg: target price.
        redemption_fee: fee to redeem/mint.
        gas_cost_usd: gas cost for on-chain operation.
        trade_size: trade notional.
    """
    deviation = (price - peg) / peg
    gross_profit = abs(deviation) * trade_size
    costs = trade_size * redemption_fee + gas_cost_usd
    net_profit = gross_profit - costs

    return {
        "deviation_bps": deviation * 10_000,
        "gross_profit": gross_profit,
        "costs": costs,
        "net_profit": net_profit,
        "profitable": net_profit > 0,
        "direction": "buy_redeem" if price < peg else "mint_sell",
        "return_pct": net_profit / trade_size * 100 if trade_size > 0 else 0,
    }
