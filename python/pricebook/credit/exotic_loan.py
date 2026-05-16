"""
Exotic loan features: prepayment, covenant triggers, TRS on loans.

Prepayment: CPR/PSA models adjust expected cashflows.
Covenant: financial triggers that accelerate the loan.
TRS: total return swap referencing a loan.

    from pricebook.exotic_loan import prepay_adjusted_cashflows, CovenantLoan
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.loan import TermLoan


def cpr_to_smm(cpr: float) -> float:
    """Convert Constant Prepayment Rate (annual) to Single Monthly Mortality.

    SMM = 1 - (1 - CPR)^(1/12)
    """
    return 1 - (1 - cpr) ** (1.0 / 12)


def psa_cpr(month: int, psa_speed: float = 1.0) -> float:
    """PSA prepayment model: CPR ramps up linearly for 30 months.

    At 100% PSA: CPR = 0.2% * month for month 1-30, then 6% flat.
    psa_speed: multiplier (e.g. 1.5 = 150% PSA).
    """
    base_cpr = min(month, 30) * 0.002
    return base_cpr * psa_speed


def prepay_adjusted_loan(
    loan: TermLoan,
    cpr: float,
    projection_curve: DiscountCurve,
) -> list[tuple[float, float, float]]:
    """Adjust loan cashflows for constant prepayment.

    Returns: list of (date, interest, principal) with prepayment.
    """
    smm = cpr_to_smm(cpr)
    base_flows = loan.cashflows(projection_curve)
    outstanding = loan.notional
    adjusted = []

    for d, interest, sched_principal in base_flows:
        # Interest on remaining balance
        if outstanding <= 0:
            adjusted.append((d, 0.0, 0.0))
            continue

        adj_interest = interest * (outstanding / loan.notional)

        # Scheduled principal
        actual_sched = min(sched_principal * (outstanding / loan.notional), outstanding)

        # Prepayment on remaining after scheduled
        remaining_after_sched = outstanding - actual_sched
        prepay = remaining_after_sched * smm

        total_principal = actual_sched + prepay
        total_principal = min(total_principal, outstanding)

        adjusted.append((d, adj_interest, total_principal))
        outstanding -= total_principal

    return adjusted


def prepay_adjusted_wal(
    loan: TermLoan,
    cpr: float,
    projection_curve: DiscountCurve,
) -> float:
    """Weighted average life with prepayment."""
    flows = prepay_adjusted_loan(loan, cpr, projection_curve)
    num = 0.0
    den = 0.0
    for d, _, principal in flows:
        t = year_fraction(loan.start, d, loan.day_count)
        num += t * principal
        den += principal
    return num / den if den > 0 else 0.0


class CovenantLoan:
    """Loan with financial covenant trigger.

    If a covenant is breached, the loan accelerates (all principal due).
    Modelled as a probability of breach per period.

    Args:
        base_loan: the underlying TermLoan.
        breach_prob_per_period: probability of covenant breach each period.
    """

    def __init__(self, base_loan: TermLoan, breach_prob_per_period: float = 0.01):
        self.loan = base_loan
        self.breach_prob = breach_prob_per_period

    def expected_maturity(self, projection_curve: DiscountCurve) -> float:
        """Expected maturity accounting for covenant triggers."""
        flows = self.loan.cashflows(projection_curve)
        outstanding = self.loan.notional
        expected_mat = 0.0
        survival = 1.0

        for i, (d, _, _) in enumerate(flows):
            t = year_fraction(self.loan.start, d, self.loan.day_count)
            # Probability of breach in this period
            breach = survival * self.breach_prob
            # If breach: maturity = t
            expected_mat += breach * t
            survival *= (1 - self.breach_prob)

        # If never breached: maturity = final date
        t_final = year_fraction(
            self.loan.start, flows[-1][0], self.loan.day_count,
        )
        expected_mat += survival * t_final

        return expected_mat

    def pv(
        self,
        discount_curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        """PV with covenant-adjusted cashflows.

        At each period: survival_prob * normal_cashflow + breach_prob * acceleration.
        """
        proj = projection_curve if projection_curve is not None else discount_curve
        flows = self.loan.cashflows(proj)
        outstanding = self.loan.notional
        survival = 1.0
        pv = 0.0

        for d, interest, principal in flows:
            df = discount_curve.df(d)

            # Normal cashflow (conditional on survival)
            pv += survival * (1 - self.breach_prob) * df * (interest + principal)

            # Acceleration (if breached): repay all outstanding
            pv += survival * self.breach_prob * df * outstanding

            outstanding -= principal
            outstanding = max(outstanding, 0.0)
            survival *= (1 - self.breach_prob)

        return pv
