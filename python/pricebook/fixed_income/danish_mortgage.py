"""Danish mortgage bonds (realkreditobligationer).

One of the world's largest covered bond markets (~€400bn outstanding).
Danish mortgage bonds are callable at par by the borrower, creating
negative convexity and prepayment risk.

    from pricebook.fixed_income.danish_mortgage import (
        DanishMortgageBond, prepayment_model, MortgageBondResult,
    )

Conventions:
- Day count: ACT/ACT ICMA, annual coupon, T+2
- Borrower can prepay at par (call at 100)
- Prepayment driven by refinancing incentive (current rate vs coupon)
- Two structures: bullet (non-amortising) and pass-through (amortising)
- CPR: Conditional Prepayment Rate (annualised % of remaining balance)

References:
    Nykredit (2024). Danish Mortgage Bond Market Guide.
    Danske Bank (2024). Covered Bond Handbook.
    Association of Danish Mortgage Banks (2024). The Danish Mortgage Model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.schedule import Frequency, generate_schedule


# ═══════════════════════════════════════════════════════════════
# Prepayment model
# ═══════════════════════════════════════════════════════════════

def prepayment_model(
    coupon: float,
    current_rate: float,
    seasoning_years: float = 1.0,
    base_cpr: float = 0.03,
    refi_sensitivity: float = 5.0,
) -> float:
    """CPR (Conditional Prepayment Rate) as a function of refinancing incentive.

    CPR = base_cpr + refi_sensitivity × max(coupon - current_rate, 0)

    When rates fall below the bond coupon, borrowers refinance at lower rates.
    The CPR increases with the refinancing incentive (coupon - market rate).

    Args:
        coupon: bond coupon rate.
        current_rate: current market mortgage rate.
        seasoning_years: time since issue (ramp-up effect).
        base_cpr: baseline CPR (turnover, moves, etc.).
        refi_sensitivity: CPR increase per 1% refinancing incentive.

    Returns:
        Annualised CPR (e.g. 0.15 = 15% of remaining balance per year).
    """
    # Refinancing incentive
    refi_incentive = max(coupon - current_rate, 0.0)

    # Seasoning ramp: CPR ramps up over first 2 years
    seasoning_mult = min(seasoning_years / 2.0, 1.0)

    cpr = (base_cpr + refi_sensitivity * refi_incentive) * seasoning_mult

    return max(0.0, min(cpr, 0.80))  # cap at 80% CPR


def psa_curve(month: int, psa_speed: float = 100.0) -> float:
    """PSA (Public Securities Association) prepayment curve.

    Standard PSA: CPR ramps from 0% to 6% over months 1-30, then flat at 6%.
    PSA speed: 100% = standard, 200% = double speed, etc.

    Args:
        month: months since origination (1-based).
        psa_speed: PSA speed as percentage (100 = standard).

    Returns:
        Monthly CPR (not annualised).
    """
    if month <= 30:
        annual_cpr = 0.06 * month / 30 * psa_speed / 100
    else:
        annual_cpr = 0.06 * psa_speed / 100

    # Convert annual to monthly: 1 - (1-CPR)^(1/12)
    monthly_cpr = 1 - (1 - annual_cpr) ** (1.0 / 12)
    return max(0.0, monthly_cpr)


# ═══════════════════════════════════════════════════════════════
# Mortgage bond
# ═══════════════════════════════════════════════════════════════

@dataclass
class MortgageBondResult:
    """Danish mortgage bond pricing result."""
    dirty_price: float
    oas: float                  # option-adjusted spread (bp)
    effective_duration: float   # duration adjusted for prepayment
    wal: float                  # weighted average life (years)
    expected_cpr: float         # expected annualised CPR
    callable_value: float       # value of prepayment option to borrower

    def to_dict(self) -> dict:
        return vars(self)


class DanishMortgageBond:
    """Danish mortgage bond (realkreditobligation).

    Callable at par by the borrower. Two structures:
    - "bullet": non-amortising (principal at maturity)
    - "pass_through": amortising (principal repaid over life)

    The call option (prepayment at par) creates negative convexity:
    when rates fall, the bond price is capped near par.

    Args:
        issue_date: bond issue date.
        maturity: final maturity date.
        coupon: annual coupon rate.
        structure: "bullet" or "pass_through".
        face: face value (per 100).
        call_price: prepayment price (default 100 = par).
    """

    def __init__(self, issue_date: date, maturity: date, coupon: float,
                 structure: str = "bullet", face: float = 100,
                 call_price: float = 100):
        self.issue_date = issue_date
        self.maturity = maturity
        self.coupon = coupon
        self.structure = structure
        self.face = face
        self.call_price = call_price

    def price(
        self,
        reference_date: date,
        discount_curve: DiscountCurve,
        current_mortgage_rate: float | None = None,
        cpr: float | None = None,
        _compute_duration: bool = True,
    ) -> MortgageBondResult:
        """Price the mortgage bond with prepayment.

        Args:
            discount_curve: risk-free discount curve.
            current_mortgage_rate: current market mortgage rate (for CPR model).
            cpr: override CPR directly. If None, uses prepayment_model().
        """
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.ANNUAL)
        dc = DayCountConvention.ACT_ACT_ICMA

        # Determine CPR
        if cpr is not None:
            expected_cpr = cpr
        elif current_mortgage_rate is not None:
            seasoning = year_fraction(self.issue_date, reference_date,
                                       DayCountConvention.ACT_365_FIXED)
            expected_cpr = prepayment_model(self.coupon, current_mortgage_rate, seasoning)
        else:
            expected_cpr = 0.03  # default 3% baseline

        # Monthly survival factor from CPR
        monthly_survival = (1 - expected_cpr) ** (1.0 / 12)

        # Price with prepayment
        remaining_face = self.face
        pv = 0.0
        cashflows = []  # for WAL computation
        total_principal = 0.0

        for i in range(1, len(schedule)):
            if schedule[i] <= reference_date:
                # Past cashflow — adjust remaining face for pass-through
                if self.structure == "pass_through":
                    months = max(1, int(year_fraction(schedule[i-1], schedule[i],
                                 DayCountConvention.ACT_365_FIXED) * 12))
                    prepaid = remaining_face * (1 - monthly_survival ** months)
                    scheduled = remaining_face / max(len(schedule) - 1, 1)
                    remaining_face -= (prepaid + scheduled)
                    remaining_face = max(remaining_face, 0)
                continue

            tau = year_fraction(schedule[i-1], schedule[i], dc,
                                ref_start=schedule[i-1], ref_end=schedule[i], frequency=1)
            df = discount_curve.df(schedule[i])
            t = year_fraction(reference_date, schedule[i], DayCountConvention.ACT_365_FIXED)

            # Months in this period
            months = max(1, int(tau * 12))

            # Prepayment in this period
            prepaid = remaining_face * (1 - monthly_survival ** months)

            if self.structure == "pass_through":
                # Scheduled amortisation
                n_remaining = len(schedule) - i
                scheduled = remaining_face / max(n_remaining, 1)
            else:
                # Bullet: no scheduled amortisation (principal at maturity)
                scheduled = 0.0

            # Total principal return this period
            principal = prepaid + scheduled

            # Coupon on remaining balance
            coupon_cf = remaining_face * self.coupon * tau

            # Cashflow PV
            pv += (coupon_cf + principal) * df

            # Track for WAL
            cashflows.append((t, principal))
            total_principal += principal

            remaining_face -= principal
            remaining_face = max(remaining_face, 0)

        # Remaining principal at maturity (for bullet)
        if remaining_face > 0.01:
            df_mat = discount_curve.df(self.maturity)
            t_mat = year_fraction(reference_date, self.maturity,
                                   DayCountConvention.ACT_365_FIXED)
            pv += remaining_face * df_mat
            cashflows.append((t_mat, remaining_face))
            total_principal += remaining_face

        # Weighted Average Life
        if total_principal > 0:
            wal = sum(t * p for t, p in cashflows) / total_principal
        else:
            wal = 0.0

        # Non-callable price (for OAS and callable value)
        noncallable_pv = self._price_noncallable(reference_date, discount_curve)

        # OAS: spread that makes callable price = non-callable price
        # Simplified: OAS ≈ (noncallable - callable) / duration / 100
        T = year_fraction(reference_date, self.maturity, DayCountConvention.ACT_365_FIXED)
        oas_bp = (noncallable_pv - pv) / max(T, 0.1) * 100 if T > 0 else 0

        # Effective duration (bump rates ±10bp)
        if _compute_duration:
            eff_dur = self._effective_duration(reference_date, discount_curve,
                                                None, expected_cpr)
        else:
            eff_dur = 0.0

        callable_value = max(noncallable_pv - pv, 0)

        return MortgageBondResult(pv, oas_bp, eff_dur, wal, expected_cpr, callable_value)

    def _price_noncallable(self, ref: date, curve: DiscountCurve) -> float:
        """Price without prepayment (zero CPR)."""
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.ANNUAL)
        dc = DayCountConvention.ACT_ACT_ICMA
        pv = 0.0
        for i in range(1, len(schedule)):
            if schedule[i] <= ref:
                continue
            tau = year_fraction(schedule[i-1], schedule[i], dc,
                                ref_start=schedule[i-1], ref_end=schedule[i], frequency=1)
            pv += self.face * self.coupon * tau * curve.df(schedule[i])
        pv += self.face * curve.df(self.maturity)
        return pv

    def _effective_duration(self, ref, curve, mortgage_rate, cpr, bump_bp=10):
        """Effective duration via parallel rate bump."""
        from pricebook.core.interpolation import InterpolationMethod
        base = self.price(ref, curve, mortgage_rate, cpr, _compute_duration=False).dirty_price

        # Bump curve down
        pillar_dates = curve.pillar_dates
        dfs_down = [curve.df(d) * math.exp(bump_bp / 10_000 *
                    year_fraction(ref, d, DayCountConvention.ACT_365_FIXED))
                    for d in pillar_dates]
        curve_down = DiscountCurve(ref, pillar_dates, dfs_down,
                                    interpolation=InterpolationMethod.LOG_LINEAR)
        px_down = self.price(ref, curve_down, mortgage_rate, cpr, _compute_duration=False).dirty_price

        # Bump curve up
        dfs_up = [curve.df(d) * math.exp(-bump_bp / 10_000 *
                  year_fraction(ref, d, DayCountConvention.ACT_365_FIXED))
                  for d in pillar_dates]
        curve_up = DiscountCurve(ref, pillar_dates, dfs_up,
                                  interpolation=InterpolationMethod.LOG_LINEAR)
        px_up = self.price(ref, curve_up, mortgage_rate, cpr, _compute_duration=False).dirty_price

        if base > 0:
            return (px_down - px_up) / (2 * bump_bp / 10_000 * base)
        return 0.0

    def to_dict(self) -> dict:
        return {"type": "danish_mortgage", "maturity": self.maturity.isoformat(),
                "coupon": self.coupon, "structure": self.structure}


# ═══════════════════════════════════════════════════════════════
# Synthetic data
# ═══════════════════════════════════════════════════════════════

def synthetic_mortgage_quotes(reference_date: date) -> list[dict]:
    """Synthetic Danish mortgage bond quotes (realistic 2024)."""
    from dateutil.relativedelta import relativedelta
    return [
        {"tenor": "2Y", "maturity": reference_date + relativedelta(years=2),
         "coupon": 0.04, "price": 100.5, "structure": "bullet"},
        {"tenor": "5Y", "maturity": reference_date + relativedelta(years=5),
         "coupon": 0.035, "price": 99.0, "structure": "bullet"},
        {"tenor": "10Y", "maturity": reference_date + relativedelta(years=10),
         "coupon": 0.03, "price": 96.0, "structure": "pass_through"},
        {"tenor": "20Y", "maturity": reference_date + relativedelta(years=20),
         "coupon": 0.025, "price": 88.0, "structure": "pass_through"},
        {"tenor": "30Y", "maturity": reference_date + relativedelta(years=30),
         "coupon": 0.02, "price": 78.0, "structure": "pass_through"},
    ]
