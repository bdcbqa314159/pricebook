"""CMBS analytics: loan-level LTV/DSCR, balloon risk, defeasance.

Commercial mortgage-backed securities with property-type concentration,
tranche credit enhancement, and loan-level analytics.

* :class:`CMBSLoan` — single commercial mortgage.
* :class:`CMBSPool` — pool of commercial mortgages.
* :class:`CMBSResult` — pricing/analytics result.
* :func:`price_cmbs` — price CMBS tranche.
* :func:`cmbs_stress` — stress LTV/DSCR under property value shocks.

References:
    Fabozzi & Jacob, *The Handbook of Commercial Mortgage-Backed
    Securities*, 2nd ed., 2001.
    Tuckman & Serrat, *Fixed Income Securities*, 3rd ed., Ch. 22.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

import numpy as np

from pricebook.core.discount_curve import DiscountCurve


class PropertyType(Enum):
    """Commercial property types."""
    OFFICE = "office"
    RETAIL = "retail"
    MULTIFAMILY = "multifamily"
    INDUSTRIAL = "industrial"
    HOTEL = "hotel"
    MIXED_USE = "mixed_use"


@dataclass
class CMBSLoan:
    """Single commercial mortgage loan."""
    balance: float
    property_value: float
    noi: float              # net operating income (annual)
    coupon: float           # annual mortgage rate
    maturity_months: int    # remaining months to balloon
    property_type: PropertyType = PropertyType.OFFICE
    amortising: bool = False  # interest-only vs amortising
    amort_months: int = 360   # amort schedule if amortising

    @property
    def ltv(self) -> float:
        """Loan-to-value ratio."""
        return self.balance / self.property_value if self.property_value > 0 else 0

    @property
    def dscr(self) -> float:
        """Debt service coverage ratio: NOI / annual debt service."""
        annual_ds = self.balance * self.coupon
        if self.amortising and self.amort_months > 0:
            monthly_rate = self.coupon / 12
            annual_ds = self.balance * monthly_rate / (1 - (1 + monthly_rate) ** (-self.amort_months)) * 12
        return self.noi / annual_ds if annual_ds > 0 else 0

    @property
    def debt_yield(self) -> float:
        """Debt yield: NOI / loan balance."""
        return self.noi / self.balance if self.balance > 0 else 0

    def to_dict(self) -> dict:
        return {
            "balance": self.balance,
            "property_value": self.property_value,
            "ltv": self.ltv,
            "dscr": self.dscr,
            "debt_yield": self.debt_yield,
            "coupon": self.coupon,
            "property_type": self.property_type.value,
        }


@dataclass
class CMBSPool:
    """Pool of commercial mortgages."""
    loans: list[CMBSLoan]

    @property
    def total_balance(self) -> float:
        return sum(l.balance for l in self.loans)

    @property
    def wa_ltv(self) -> float:
        total = self.total_balance
        if total <= 0:
            return 0
        return sum(l.balance * l.ltv for l in self.loans) / total

    @property
    def wa_dscr(self) -> float:
        total = self.total_balance
        if total <= 0:
            return 0
        return sum(l.balance * l.dscr for l in self.loans) / total

    @property
    def wa_coupon(self) -> float:
        total = self.total_balance
        if total <= 0:
            return 0
        return sum(l.balance * l.coupon for l in self.loans) / total

    def concentration(self) -> dict[str, float]:
        """Property type concentration (% of balance)."""
        total = self.total_balance
        if total <= 0:
            return {}
        conc: dict[str, float] = {}
        for loan in self.loans:
            pt = loan.property_type.value
            conc[pt] = conc.get(pt, 0) + loan.balance / total * 100
        return conc

    def to_dict(self) -> dict:
        return {
            "n_loans": len(self.loans),
            "total_balance": self.total_balance,
            "wa_ltv": self.wa_ltv,
            "wa_dscr": self.wa_dscr,
            "wa_coupon": self.wa_coupon,
            "concentration": self.concentration(),
        }


@dataclass
class CMBSResult:
    """CMBS tranche pricing result."""
    price: float
    wal: float
    credit_enhancement_pct: float
    wa_ltv: float
    wa_dscr: float
    balloon_risk_pct: float     # % of loans with balloon payment
    expected_loss_pct: float

    def to_dict(self) -> dict:
        return vars(self)


def price_cmbs(
    pool: CMBSPool,
    tranches: list[dict],
    discount_curve: DiscountCurve,
    spread: float = 0.0,
    default_rate: float = 0.02,
    loss_severity: float = 0.35,
) -> list[CMBSResult]:
    """Price CMBS tranches.

    Commercial mortgages are typically interest-only with balloon
    payment at maturity. Default risk concentrates at the balloon
    date when refinancing may fail.

    Args:
        pool: CMBS loan pool.
        tranches: list of {"name": str, "notional": float, "coupon": float, "seniority": int}.
        discount_curve: risk-free curve.
        spread: static spread.
        default_rate: annual default rate.
        loss_severity: loss-given-default (1 - recovery).
    """
    ref = discount_curve.reference_date
    sorted_tranches = sorted(tranches, key=lambda t: t.get("seniority", 0))
    total_notional = sum(t["notional"] for t in sorted_tranches)

    results = []
    for tranche in sorted_tranches:
        tr_notional = tranche["notional"]
        tr_coupon = tranche.get("coupon", 0.04)
        tr_seniority = tranche.get("seniority", 0)

        # Credit enhancement: subordination below
        sub_below = sum(
            t["notional"] for t in sorted_tranches
            if t.get("seniority", 0) > tr_seniority
        )
        ce_pct = sub_below / total_notional * 100 if total_notional > 0 else 0

        # Cashflows: use pool WAL for simplicity
        max_months = max(l.maturity_months for l in pool.loans)
        tr_balance = tr_notional
        pv = 0.0
        wal_num = 0.0
        total_prin = 0.0
        pool_balance = pool.total_balance

        for m in range(1, max_months + 1):
            if tr_balance < 0.01 or pool_balance < 0.01:
                break

            t_years = m / 12.0
            try:
                cf_date = ref + timedelta(days=round(t_years * 365.25))
                df = discount_curve.df(cf_date) * math.exp(-spread * t_years)
            except Exception:
                df = math.exp(-(0.04 + spread) * t_years)

            # Pool interest
            pool_interest = pool_balance * pool.wa_coupon / 12.0

            # Defaults
            monthly_default = pool_balance * default_rate / 12.0
            monthly_loss = monthly_default * loss_severity

            # Balloon payments from maturing loans
            balloon = 0.0
            for loan in pool.loans:
                if loan.maturity_months == m:
                    balloon += loan.balance

            # Tranche interest
            tr_interest = tr_balance * tr_coupon / 12.0

            # Principal: balloon + scheduled amort
            avail_principal = balloon
            prin_to_tranche = min(avail_principal, tr_balance)

            pv += (tr_interest + prin_to_tranche) * df
            wal_num += prin_to_tranche * t_years
            total_prin += prin_to_tranche
            tr_balance -= prin_to_tranche

            pool_balance -= (balloon + monthly_default)
            pool_balance = max(pool_balance, 0)

        price = pv / tr_notional * 100 if tr_notional > 0 else 0
        wal = wal_num / total_prin if total_prin > 0 else max_months / 12.0

        # Balloon risk: % of loans that are interest-only
        n_io = sum(1 for l in pool.loans if not l.amortising)
        balloon_risk = n_io / len(pool.loans) * 100 if pool.loans else 0

        # Expected loss for this tranche
        total_pool_loss = pool.total_balance * default_rate * loss_severity * (max_months / 12.0)
        tranche_loss = max(total_pool_loss - sub_below, 0)
        el_pct = min(tranche_loss / tr_notional * 100, 100) if tr_notional > 0 else 0

        results.append(CMBSResult(
            price=price,
            wal=wal,
            credit_enhancement_pct=ce_pct,
            wa_ltv=pool.wa_ltv,
            wa_dscr=pool.wa_dscr,
            balloon_risk_pct=balloon_risk,
            expected_loss_pct=el_pct,
        ))

    return results


def cmbs_stress(
    pool: CMBSPool,
    property_shock: float = -0.20,
    noi_shock: float = -0.10,
) -> dict:
    """Stress CMBS pool under property value and NOI shocks.

    Args:
        property_shock: % change in property values (-0.20 = -20%).
        noi_shock: % change in NOI.

    Returns:
        Stressed pool metrics.
    """
    stressed_loans = []
    for loan in pool.loans:
        stressed = CMBSLoan(
            balance=loan.balance,
            property_value=loan.property_value * (1 + property_shock),
            noi=loan.noi * (1 + noi_shock),
            coupon=loan.coupon,
            maturity_months=loan.maturity_months,
            property_type=loan.property_type,
            amortising=loan.amortising,
            amort_months=loan.amort_months,
        )
        stressed_loans.append(stressed)

    stressed_pool = CMBSPool(stressed_loans)

    # Count breached loans
    ltv_breach = sum(1 for l in stressed_loans if l.ltv > 0.80)
    dscr_breach = sum(1 for l in stressed_loans if l.dscr < 1.20)

    return {
        "base_wa_ltv": pool.wa_ltv,
        "stressed_wa_ltv": stressed_pool.wa_ltv,
        "base_wa_dscr": pool.wa_dscr,
        "stressed_wa_dscr": stressed_pool.wa_dscr,
        "ltv_breach_count": ltv_breach,
        "dscr_breach_count": dscr_breach,
        "n_loans": len(pool.loans),
        "property_shock": property_shock,
        "noi_shock": noi_shock,
    }


# ═══════════════════════════════════════════════════════════════
# Defeasance & Yield Maintenance
# ═══════════════════════════════════════════════════════════════

def defeasance_cost(
    loan_balance: float,
    loan_coupon: float,
    remaining_months: int,
    treasury_rate: float,
) -> float:
    """Cost of defeasance (substituting Treasury collateral).

    The borrower buys Treasury securities sufficient to make
    all remaining debt service payments.

    cost = PV(remaining payments, at treasury rate) − loan_balance.

    Args:
        loan_balance: current loan balance.
        loan_coupon: annual coupon rate.
        remaining_months: months to maturity.
        treasury_rate: risk-free rate for pricing Treasuries.
    """
    monthly_rate = loan_coupon / 12.0
    monthly_treasury = treasury_rate / 12.0

    # Monthly payment
    if monthly_rate > 0 and remaining_months > 0:
        payment = loan_balance * monthly_rate / (1 - (1 + monthly_rate) ** (-remaining_months))
    else:
        payment = loan_balance / max(remaining_months, 1)

    # PV of payments at treasury rate
    pv = 0.0
    for m in range(1, remaining_months + 1):
        df = (1 + monthly_treasury) ** (-m)
        pv += payment * df

    return pv - loan_balance


def yield_maintenance(
    loan_balance: float,
    loan_coupon: float,
    remaining_months: int,
    treasury_rate: float,
) -> float:
    """Yield maintenance prepayment penalty.

    penalty = PV(coupon differential) over remaining term.
    = Σ (loan_coupon − treasury_rate) × balance / 12 × df

    Args:
        loan_balance: current balance.
        loan_coupon: annual coupon.
        remaining_months: months remaining.
        treasury_rate: benchmark treasury rate.
    """
    if loan_coupon <= treasury_rate:
        return 0.0  # no penalty if rates have risen

    monthly_diff = (loan_coupon - treasury_rate) / 12.0
    monthly_treasury = treasury_rate / 12.0

    penalty = 0.0
    for m in range(1, remaining_months + 1):
        df = (1 + monthly_treasury) ** (-m)
        penalty += loan_balance * monthly_diff * df

    return max(penalty, 0)
