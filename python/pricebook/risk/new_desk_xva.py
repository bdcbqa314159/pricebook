"""XVA for inflation, asset swap, and structured credit desks.

Extends the XVA framework to the recently-built desks that don't yet
have CVA/FVA/MVA/KVA coverage.

    from pricebook.risk.new_desk_xva import (
        inflation_analytic_cva, inflation_fva,
        asw_fva, asw_kva,
        structured_credit_cva,
    )

References:
    Green (2015). XVA: Credit, Funding and Capital Valuation Adjustments.
    Gregory (2020). The xVA Challenge. Wiley, 4th ed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve


# ---------------------------------------------------------------------------
# Inflation XVA
# ---------------------------------------------------------------------------

@dataclass
class InflationXVAResult:
    """XVA result for an inflation instrument."""
    cva: float
    dva: float
    fva: float
    mva: float
    kva: float
    total_xva: float

    def to_dict(self) -> dict:
        return {"cva": self.cva, "dva": self.dva, "fva": self.fva,
                "mva": self.mva, "kva": self.kva, "total": self.total_xva}


def inflation_analytic_xva(
    pv: float,
    notional: float,
    maturity_years: float,
    counterparty_spread: float = 0.01,
    own_spread: float = 0.005,
    funding_spread: float = 0.002,
    recovery: float = 0.40,
    ie01: float = 0.0,
    vol_breakeven: float = 0.005,
) -> InflationXVAResult:
    """Analytic XVA approximation for inflation instruments.

    Inflation exposure profile: EPE ≈ |IE01| × vol × √T for breakeven-sensitive,
    plus |PV| component for mark-to-market exposure.

    Args:
        pv: current PV of the instrument.
        notional: position notional.
        maturity_years: time to maturity.
        counterparty_spread: counterparty CDS spread (decimal).
        own_spread: own CDS spread (decimal).
        funding_spread: funding cost over risk-free (decimal).
        recovery: recovery rate.
        ie01: breakeven sensitivity (for exposure profile).
        vol_breakeven: breakeven volatility (annual, decimal).
    """
    T = maturity_years

    # Exposure profile: inflation instruments have breakeven-driven exposure
    # EPE ≈ max(PV, 0) + |IE01| × vol × √T × notional_scaling
    epe = max(pv, 0) + abs(ie01) * vol_breakeven * math.sqrt(max(T, 0.01)) * 100
    ene = max(-pv, 0) + abs(ie01) * vol_breakeven * math.sqrt(max(T, 0.01)) * 100

    # CVA = (1-R) × ∫ EPE(t) × h(t) dt ≈ (1-R) × EPE_avg × spread × T
    lgd = 1 - recovery
    cva = lgd * epe * counterparty_spread * T
    dva = lgd * ene * own_spread * T

    # FVA = funding_spread × EPE_avg × T (unsecured funding)
    fva = funding_spread * epe * T

    # MVA = IM × funding_spread × T (simplified: IM ≈ 1% of notional)
    im = notional * 0.01
    mva = im * funding_spread * T

    # KVA = capital × hurdle × T (simplified)
    sf = 0.005  # GIRR supervisory factor
    ead = 1.4 * (max(pv, 0) + notional * sf)
    capital = ead * 0.08
    kva = capital * 0.10 * T  # 10% hurdle rate

    total = cva - dva + fva + mva + kva

    return InflationXVAResult(cva=cva, dva=dva, fva=fva, mva=mva, kva=kva, total_xva=total)


# ---------------------------------------------------------------------------
# Asset Swap XVA
# ---------------------------------------------------------------------------

@dataclass
class ASWXVAResult:
    """XVA result for an asset swap."""
    fva: float          # funding cost of bond leg
    kva: float          # capital cost
    mva: float          # margin cost
    total_xva: float

    def to_dict(self) -> dict:
        return {"fva": self.fva, "kva": self.kva, "mva": self.mva,
                "total": self.total_xva}


def asw_xva(
    bond_dirty_price: float,
    notional: float,
    maturity_years: float,
    repo_rate: float = 0.04,
    funding_spread: float = 0.002,
    haircut: float = 0.05,
) -> ASWXVAResult:
    """XVA for an asset swap position.

    ASW is a funded position: the bond leg requires financing.
    FVA = funding cost above repo rate.
    KVA = capital cost of holding the bond.

    Args:
        bond_dirty_price: dirty price per 100 face.
        notional: face value.
        maturity_years: remaining maturity.
        repo_rate: repo financing rate.
        funding_spread: unsecured funding spread over repo.
        haircut: repo haircut (unfunded portion).
    """
    T = maturity_years
    funded = notional * bond_dirty_price / 100

    # FVA: cost of unsecured funding on haircut portion
    unfunded_portion = funded * haircut
    fva = unfunded_portion * funding_spread * T

    # KVA: capital cost
    sf = 0.005
    ead = 1.4 * (funded + notional * sf)
    capital = ead * 0.08
    kva = capital * 0.10 * T

    # MVA: initial margin cost
    im = notional * 0.01
    mva = im * funding_spread * T

    total = fva + kva + mva

    return ASWXVAResult(fva=fva, kva=kva, mva=mva, total_xva=total)


# ---------------------------------------------------------------------------
# Structured Credit XVA
# ---------------------------------------------------------------------------

@dataclass
class StructuredCreditXVAResult:
    """XVA for structured credit (risk participation, guaranteed notes, etc.)."""
    cva: float
    fva: float
    kva: float
    wrong_way_adj: float     # wrong-way risk adjustment
    total_xva: float

    def to_dict(self) -> dict:
        return {"cva": self.cva, "fva": self.fva, "kva": self.kva,
                "wwr": self.wrong_way_adj, "total": self.total_xva}


def structured_credit_xva(
    pv: float,
    notional: float,
    maturity_years: float,
    counterparty_spread: float = 0.02,
    obligor_spread: float = 0.03,
    recovery: float = 0.40,
    funding_spread: float = 0.003,
    correlation: float = 0.30,
) -> StructuredCreditXVAResult:
    """XVA for structured credit instruments.

    Structured credit has wrong-way risk: counterparty default is
    correlated with obligor default (both worsen in credit stress).

    Args:
        pv: current PV.
        notional: position notional.
        maturity_years: remaining maturity.
        counterparty_spread: counterparty CDS spread.
        obligor_spread: underlying credit spread.
        recovery: counterparty recovery.
        funding_spread: funding cost.
        correlation: counterparty-obligor default correlation.
    """
    T = maturity_years
    lgd = 1 - recovery

    # Base CVA
    epe = max(pv, 0) + notional * 0.05  # 5% exposure add-on for credit
    cva = lgd * epe * counterparty_spread * T

    # Wrong-way risk: exposure increases when counterparty is stressed
    # WWR adjustment ≈ CVA × (1 + α × ρ) where α calibrated to 2x at ρ=0.5
    alpha = 4.0
    wwr_multiplier = 1 + alpha * max(correlation, 0)
    wwr_adj = cva * (wwr_multiplier - 1)
    cva_with_wwr = cva * wwr_multiplier

    # FVA
    fva = funding_spread * epe * T

    # KVA
    sf = 0.005
    ead = 1.4 * (max(pv, 0) + notional * sf)
    capital = ead * 0.08
    kva = capital * 0.10 * T

    total = cva_with_wwr + fva + kva

    return StructuredCreditXVAResult(
        cva=cva_with_wwr, fva=fva, kva=kva,
        wrong_way_adj=wwr_adj, total_xva=total,
    )
