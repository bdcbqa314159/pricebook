"""Corporate collateral & credit-repo integration.

Hazard rate integration for repo collateral: credit-adjusted haircuts,
collateral default loss, and all-in repo pricing with credit risk.

    from pricebook.fixed_income.repo_credit_collateral import (
        CreditCollateralSpec, credit_adjusted_haircut,
        repo_price_with_collateral_credit,
    )

References:
    BCBS (2013). Margin Requirements for Non-Centrally Cleared Derivatives.
    Lo (2016). Gap Risk in Secured Lending.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class CollateralAssetClass(Enum):
    """Collateral asset class for haircut determination."""
    SOVEREIGN = "sovereign"
    IG_CORPORATE = "ig_corporate"
    HY_CORPORATE = "hy_corporate"
    BANK_SENIOR = "bank_senior"
    BANK_AT1_T2 = "bank_at1_t2"
    STRUCTURED_IG = "structured_ig"     # IG CLO, RMBS
    STRUCTURED_HY = "structured_hy"     # HY CLO, subprime
    EQUITY = "equity"


# Base haircuts by asset class (regulatory standard, in %)
_BASE_HAIRCUTS: dict[CollateralAssetClass, float] = {
    CollateralAssetClass.SOVEREIGN: 0.02,
    CollateralAssetClass.IG_CORPORATE: 0.06,
    CollateralAssetClass.HY_CORPORATE: 0.15,
    CollateralAssetClass.BANK_SENIOR: 0.06,
    CollateralAssetClass.BANK_AT1_T2: 0.15,
    CollateralAssetClass.STRUCTURED_IG: 0.08,
    CollateralAssetClass.STRUCTURED_HY: 0.25,
    CollateralAssetClass.EQUITY: 0.25,
}


@dataclass
class CreditCollateralSpec:
    """Collateral bond with credit characteristics."""
    asset_class: CollateralAssetClass
    issuer_name: str
    rating: str                  # e.g. "BBB", "BB"
    cds_spread_bp: float         # issuer CDS spread
    hazard_rate: float           # annual hazard rate
    recovery: float              # expected recovery on default
    duration: float              # modified duration
    sector: str = ""             # e.g. "Financials", "Energy"
    country: str = ""            # e.g. "US", "IT"
    market_value: float = 100.0

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class CreditAdjustedHaircutResult:
    """Result of credit-adjusted haircut calculation."""
    base_haircut: float          # market/asset-class based
    credit_add_on: float         # PD-driven addition
    spread_add_on: float         # spread-vol-driven addition
    total_haircut: float
    collateral_asset_class: str

    def to_dict(self) -> dict:
        return vars(self)


def credit_adjusted_haircut(
    spec: CreditCollateralSpec,
    hold_period_days: int = 10,
    spread_vol: float = 0.30,
    confidence: float = 0.99,
) -> CreditAdjustedHaircutResult:
    """Compute credit-adjusted haircut for collateral.

    haircut = base_haircut + credit_add_on + spread_add_on

    credit_add_on = PD(hold_period) × LGD
    spread_add_on = duration × spread_vol × sqrt(hold_period/252) × z(confidence)

    Args:
        spec: collateral credit characteristics.
        hold_period_days: margin period of risk (MPOR).
        spread_vol: annualised vol of the credit spread.
        confidence: VaR confidence level.
    """
    from scipy.stats import norm

    base = _BASE_HAIRCUTS.get(spec.asset_class, 0.10)

    # Credit add-on: expected loss from collateral default during hold period
    pd_hold = 1.0 - math.exp(-spec.hazard_rate * hold_period_days / 365.0)
    lgd = 1.0 - spec.recovery
    credit_add_on = pd_hold * lgd

    # Spread add-on: VaR on spread widening → price decline
    z = norm.ppf(confidence)
    t_sqrt = math.sqrt(hold_period_days / 252.0)
    spread_var = spec.duration * spec.cds_spread_bp / 10_000 * spread_vol * t_sqrt * z
    spread_add_on = max(spread_var, 0.0)

    total = base + credit_add_on + spread_add_on

    return CreditAdjustedHaircutResult(
        base_haircut=base,
        credit_add_on=credit_add_on,
        spread_add_on=spread_add_on,
        total_haircut=min(total, 1.0),
        collateral_asset_class=spec.asset_class.value,
    )


def collateral_default_loss(
    spec: CreditCollateralSpec,
    repo_maturity_days: int,
    repo_notional: float,
) -> float:
    """Expected loss if collateral defaults before repo maturity.

    EL = notional × PD(T) × LGD × (1 - haircut)
    """
    pd = 1.0 - math.exp(-spec.hazard_rate * repo_maturity_days / 365.0)
    lgd = 1.0 - spec.recovery
    base_haircut = _BASE_HAIRCUTS.get(spec.asset_class, 0.10)
    unsecured_fraction = max(1.0 - base_haircut, 0.0)  # actually wrong: we're lending cash
    return repo_notional * pd * lgd


def repo_price_with_collateral_credit(
    repo_rate: float,
    repo_notional: float,
    repo_days: int,
    spec: CreditCollateralSpec,
    counterparty_hazard: float = 0.0,
    counterparty_recovery: float = 0.40,
    correlation: float = 0.0,
) -> dict:
    """All-in repo pricing with collateral credit risk.

    Components:
    1. Interest income: notional × rate × t
    2. Collateral credit charge: EL on collateral default
    3. Counterparty credit charge: EL on counterparty default (unsecured portion)
    4. Wrong-way risk add-on: correlation × charge amplification
    5. Gap risk: probability that collateral value drops below cash lent

    Args:
        repo_rate: agreed repo rate.
        repo_notional: cash amount lent.
        repo_days: term in days.
        spec: collateral credit specification.
        counterparty_hazard: counterparty annual hazard rate.
        counterparty_recovery: counterparty recovery.
        correlation: counterparty-collateral default correlation.
    """
    denom = 360.0
    t = repo_days / denom

    # 1. Interest
    interest = repo_notional * repo_rate * t

    # 2. Collateral credit charge
    coll_loss = collateral_default_loss(spec, repo_days, repo_notional)

    # 3. Counterparty credit charge (unsecured exposure after haircut)
    haircut_result = credit_adjusted_haircut(spec)
    unsecured = repo_notional * haircut_result.total_haircut  # overcollateralised portion protects
    cp_pd = 1.0 - math.exp(-counterparty_hazard * repo_days / 365.0)
    cp_lgd = 1.0 - counterparty_recovery
    cp_charge = unsecured * cp_pd * cp_lgd

    # 4. Wrong-way risk
    wwr = correlation * (coll_loss + cp_charge) * 0.5

    # 5. Gap risk (simplified: probability of > haircut decline)
    gap = _gap_risk(spec, repo_days, haircut_result.total_haircut)

    all_in_cost = coll_loss + cp_charge + wwr + gap
    net_income = interest - all_in_cost

    return {
        "interest_income": interest,
        "collateral_credit_charge": coll_loss,
        "counterparty_credit_charge": cp_charge,
        "wrong_way_risk": wwr,
        "gap_risk": gap,
        "all_in_credit_cost": all_in_cost,
        "net_income": net_income,
        "breakeven_rate": all_in_cost / (repo_notional * t) if repo_notional * t > 0 else 0.0,
    }


def hazard_to_haircut_mapping(
    hazard_rates: list[float],
    durations: list[float],
    base_asset_class: CollateralAssetClass = CollateralAssetClass.IG_CORPORATE,
) -> list[dict]:
    """Map hazard rates to haircut add-ons for a range of collateral."""
    results = []
    for h, d in zip(hazard_rates, durations):
        spec = CreditCollateralSpec(
            base_asset_class, "generic", "BBB",
            cds_spread_bp=h * (1 - 0.40) * 10_000,
            hazard_rate=h, recovery=0.40, duration=d,
        )
        hr = credit_adjusted_haircut(spec)
        results.append({
            "hazard_rate": h,
            "duration": d,
            "base_haircut": hr.base_haircut,
            "credit_add_on": hr.credit_add_on,
            "spread_add_on": hr.spread_add_on,
            "total_haircut": hr.total_haircut,
        })
    return results


def _gap_risk(spec, repo_days, haircut):
    """Simplified gap risk: probability that mark-to-market loss > haircut."""
    from scipy.stats import norm
    # Price vol ≈ duration × spread_vol × spread_level
    spread_level = spec.cds_spread_bp / 10_000
    price_vol = spec.duration * 0.30 * spread_level  # annual price vol
    t = math.sqrt(repo_days / 252.0)
    # P(loss > haircut) ≈ Φ(-haircut / (price_vol × √t))
    if price_vol * t <= 0:
        return 0.0
    p_gap = norm.cdf(-haircut / (price_vol * t))
    lgd_gap = haircut * 0.5  # expected loss given gap = half the haircut
    return spec.market_value * p_gap * lgd_gap
