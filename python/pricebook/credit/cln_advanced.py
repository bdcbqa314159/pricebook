"""Advanced CLN mechanics: XVA, dynamic funding, wrong-way risk, haircut dynamics.

Extends bilateral CLN with:
B3: Spread-driven XVA (CVA where exposure depends on reference spread)
B4: Dynamic funding cost (CSA-threshold-aware, not flat)
B5: Wrong-way risk (issuer hazard rises when reference is distressed)
B6: Non-cash collateral haircut dynamics (bond collateral value drops under stress)

    from pricebook.credit.cln_advanced import (
        cln_xva_spread_driven, dynamic_funding_cost,
        wrong_way_risk_adjustment, collateral_haircut_stress,
    )

References:
    Gregory (2015). The xVA Challenge, Ch 12-14.
    Brigo, Morini & Pallavicini (2013). Counterparty Credit Risk, Ch 8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class XVAResult:
    """Extended XVA decomposition for CLN."""
    cva: float
    dva: float
    fva: float
    colva: float                # Collateral value adjustment
    kva: float                  # Capital value adjustment (simplified)
    total_xva: float
    wrong_way_adjustment: float
    haircut_stress_loss: float

    def to_dict(self) -> dict:
        return vars(self)


def cln_xva_spread_driven(
    notional: float,
    maturity_years: float,
    ref_hazard: float,
    issuer_hazard: float,
    ref_recovery: float = 0.40,
    issuer_recovery: float = 0.40,
    correlation: float = 0.30,
    risk_free_rate: float = 0.04,
    funding_spread: float = 0.005,
    spread_vol: float = 0.30,
    n_steps: int = 20,
) -> XVAResult:
    """B3: CVA where exposure is spread-dependent.

    When the reference entity's spread widens, the CLN's MtM drops,
    increasing the investor's exposure to issuer default (wrong-way risk).

    The exposure at time t is a function of the prevailing spread:
        E(t) ≈ notional × (1 - recovery) × spread_duration × Δspread

    Args:
        spread_vol: annualised volatility of the reference CDS spread.
    """
    dt = maturity_years / n_steps
    lgd_ref = 1.0 - ref_recovery
    lgd_iss = 1.0 - issuer_recovery

    cva = 0.0
    dva = 0.0
    fva = 0.0
    colva = 0.0

    for i in range(1, n_steps + 1):
        t = i * dt
        df = math.exp(-risk_free_rate * t)

        # Survival probabilities
        q_ref = math.exp(-ref_hazard * t)
        q_iss = math.exp(-issuer_hazard * t)

        # Default probabilities in period
        q_ref_prev = math.exp(-ref_hazard * (t - dt))
        q_iss_prev = math.exp(-issuer_hazard * (t - dt))
        pd_ref = q_ref_prev - q_ref
        pd_iss = q_iss_prev - q_iss

        # Expected positive exposure (spread-dependent)
        # When spreads widen, CLN value drops → exposure to issuer increases
        remaining_t = maturity_years - t
        spread_duration = remaining_t * lgd_ref * notional
        expected_spread_move = spread_vol * ref_hazard * math.sqrt(dt)
        epe = abs(spread_duration * expected_spread_move) * q_ref

        # CVA: issuer default × exposure
        cva += df * pd_iss * epe * lgd_iss

        # FVA: funding cost on expected exposure
        fva += df * epe * funding_spread * dt * q_ref * q_iss

    # Wrong-way risk: correlation adjustment
    # Higher correlation → higher CVA (both distressed simultaneously)
    wwr = cva * correlation * 0.5  # simplified factor
    cva += wwr

    # KVA: simplified as fraction of CVA
    kva = cva * 0.10  # 10% of CVA as capital charge

    total = cva + dva + fva + colva + kva

    return XVAResult(
        cva=cva, dva=dva, fva=fva, colva=colva, kva=kva,
        total_xva=total, wrong_way_adjustment=wwr,
        haircut_stress_loss=0.0,
    )


def dynamic_funding_cost(
    notional: float,
    maturity_years: float,
    exposure_profile: list[float],
    funding_curve: list[float],
    threshold: float = 0.0,
) -> float:
    """B4: CSA-threshold-aware dynamic funding cost.

    Instead of flat r×N, compute funding cost period-by-period based on
    the uncollateralised exposure above the CSA threshold.

    Args:
        exposure_profile: expected exposure at each period.
        funding_curve: funding spread at each period.
        threshold: CSA threshold (exposure below this is collateralised).
    """
    n = min(len(exposure_profile), len(funding_curve))
    dt = maturity_years / n if n > 0 else 1.0
    total = 0.0
    for i in range(n):
        uncoll = max(exposure_profile[i] - threshold, 0)
        total += uncoll * funding_curve[i] * dt
    return total


def wrong_way_risk_adjustment(
    base_cva: float,
    correlation: float,
    issuer_spread_sensitivity: float = 0.5,
) -> float:
    """B5: Second-order wrong-way risk.

    When the reference entity is distressed, the issuer's hazard rate
    also rises (if correlated). This amplifies CVA beyond the first-order
    Gaussian copula correlation.

    adjustment = base_CVA × β × ρ²

    where β captures the non-linear sensitivity.
    """
    return base_cva * issuer_spread_sensitivity * correlation ** 2


@dataclass
class HaircutStressResult:
    """Result of collateral haircut stress test."""
    base_haircut_pct: float
    stressed_haircut_pct: float
    additional_margin_call: float
    collateral_shortfall: float

    def to_dict(self) -> dict:
        return vars(self)


def collateral_haircut_stress(
    collateral_market_value: float,
    base_haircut_pct: float,
    spread_shock_bp: float,
    collateral_duration: float = 5.0,
    correlation_collateral_ref: float = 0.30,
) -> HaircutStressResult:
    """B6: Non-cash collateral haircut dynamics.

    Bond collateral value drops when spreads widen:
    ΔV ≈ -duration × Δspread × V

    The stressed haircut accounts for the price decline:
    stressed_haircut = base_haircut + spread_shock × duration

    Args:
        collateral_market_value: current MV of posted collateral.
        base_haircut_pct: normal-conditions haircut (e.g. 0.02 = 2%).
        spread_shock_bp: spread widening scenario (bp).
        collateral_duration: modified duration of collateral bonds.
        correlation_collateral_ref: correlation between collateral and reference.
    """
    spread_shock = spread_shock_bp / 10_000

    # Price decline of collateral
    price_decline_pct = collateral_duration * spread_shock
    stressed_value = collateral_market_value * (1 - price_decline_pct)

    # Stressed haircut
    stressed_haircut = base_haircut_pct + price_decline_pct

    # Additional margin call
    value_loss = collateral_market_value * price_decline_pct
    additional_margin = max(value_loss - collateral_market_value * base_haircut_pct, 0)

    # Shortfall if collateral insufficient
    shortfall = max(collateral_market_value * stressed_haircut - collateral_market_value * base_haircut_pct, 0)

    return HaircutStressResult(
        base_haircut_pct=base_haircut_pct,
        stressed_haircut_pct=min(stressed_haircut, 1.0),
        additional_margin_call=additional_margin,
        collateral_shortfall=shortfall,
    )
