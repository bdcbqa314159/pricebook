"""
CDS Index and Credit-Linked Note (CLN).

CDS Index: portfolio of single-name CDS with equal weights.
    Flat spread: single spread that reprices the index.
    Intrinsic spread: weighted average of constituent spreads.

Vanilla CLN: bond + embedded short CDS protection.
    Coupon = risk-free + credit spread. Principal at risk on default.

    index = CDSIndex(constituents=[cds1, cds2, ...])
    pv = index.pv(discount_curve, survival_curves)

    cln = VanillaCLN(start, end, coupon_rate=0.06, reference_entity="ACME")
    price = cln.dirty_price(discount_curve, survival_curve)
"""

from __future__ import annotations

import math
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.cds import CDS
from pricebook.schedule import Frequency, generate_schedule
from pricebook.solvers import brentq


class CDSIndex:
    """CDS index: equally-weighted portfolio of single-name CDS.

    Args:
        constituents: list of CDS objects (all same maturity assumed).
        notional: index notional (divided equally among constituents).
    """

    def __init__(
        self,
        constituents: list[CDS],
        notional: float = 1_000_000.0,
    ):
        if not constituents:
            raise ValueError("need at least 1 constituent")
        self.constituents = constituents
        self.notional = notional
        self.n = len(constituents)

    def pv(
        self,
        discount_curve: DiscountCurve,
        survival_curves: list[SurvivalCurve],
    ) -> float:
        """PV = sum of constituent PVs (equally weighted).

        Args:
            survival_curves: one per constituent, same order.
        """
        if len(survival_curves) != self.n:
            raise ValueError(
                f"Expected {self.n} survival curves, got {len(survival_curves)}"
            )
        total = 0.0
        weight = self.notional / (self.n * self.constituents[0].notional)
        for cds, sc in zip(self.constituents, survival_curves):
            total += weight * cds.pv(discount_curve, sc)
        return total

    def flat_spread(
        self,
        discount_curve: DiscountCurve,
        survival_curves: list[SurvivalCurve],
    ) -> float:
        """Flat spread: single spread that reprices the index PV.

        Uses the first constituent as a template for the premium leg structure.
        """
        index_pv = self.pv(discount_curve, survival_curves)

        # Average survival curve for the index (simple average of survivals)
        ref = survival_curves[0]

        template = self.constituents[0]

        def objective(s: float) -> float:
            index_cds = CDS(
                template.start, template.end,
                spread=s, notional=self.notional,
                recovery=template.recovery,
            )
            # Use average survival for flat spread
            avg_pv = 0.0
            for sc in survival_curves:
                avg_pv += index_cds.pv(discount_curve, sc)
            avg_pv /= self.n
            return avg_pv - index_pv / self.n

        return brentq(objective, 0.0001, 0.10)

    def intrinsic_spread(
        self,
        discount_curve: DiscountCurve,
        survival_curves: list[SurvivalCurve],
    ) -> float:
        """Intrinsic spread: weighted average of constituent par spreads."""
        total = 0.0
        for cds, sc in zip(self.constituents, survival_curves):
            total += cds.par_spread(discount_curve, sc)
        return total / self.n


class VanillaCLN:
    """Credit-Linked Note: funded credit exposure.

    The investor buys the note (funds the principal). In return:
    - Receives coupon (= risk-free + credit spread) each period
    - At maturity: receives par if no default, recovery if default

    PV = sum(coupon * df * survival) + par * df_T * surv_T
         + recovery * sum(df * default_prob_per_period)

    Args:
        start: issue date.
        end: maturity.
        coupon_rate: annual coupon rate (includes credit spread).
        notional: face value.
        recovery: recovery rate on default.
        frequency: coupon frequency.
    """

    def __init__(
        self,
        start: date,
        end: date,
        coupon_rate: float,
        notional: float = 100.0,
        recovery: float = 0.4,
        frequency: Frequency = Frequency.SEMI_ANNUAL,
        day_count: DayCountConvention = DayCountConvention.THIRTY_360,
    ):
        self.start = start
        self.end = end
        self.coupon_rate = coupon_rate
        self.notional = notional
        self.recovery = recovery
        self.day_count = day_count
        self.schedule = generate_schedule(start, end, frequency)

    def dirty_price(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """Full price per face value, accounting for default risk."""
        pv = 0.0
        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, self.day_count)
            df = discount_curve.df(t_end)
            surv = survival_curve.survival(t_end)
            surv_prev = survival_curve.survival(t_start)

            # Coupon conditional on survival
            pv += self.notional * self.coupon_rate * yf * df * surv

            # Recovery on default in this period
            default_prob = surv_prev - surv
            pv += self.recovery * self.notional * default_prob * df

        # Principal at maturity conditional on survival
        pv += self.notional * discount_curve.df(self.end) * survival_curve.survival(self.end)

        return pv

    def risk_free_equivalent_price(self, discount_curve: DiscountCurve) -> float:
        """Price if there were no credit risk (for comparison)."""
        pv = 0.0
        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, self.day_count)
            pv += self.notional * self.coupon_rate * yf * discount_curve.df(t_end)
        pv += self.notional * discount_curve.df(self.end)
        return pv

    def credit_spread(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """Implied credit spread: CLN yield - risk-free yield.

        Approximate: (risk_free_price - risky_price) / (notional * annuity).
        """
        risky = self.dirty_price(discount_curve, survival_curve)
        riskfree = self.risk_free_equivalent_price(discount_curve)
        annuity = 0.0
        for i in range(1, len(self.schedule)):
            yf = year_fraction(self.schedule[i-1], self.schedule[i], self.day_count)
            annuity += yf * discount_curve.df(self.schedule[i])
        if annuity == 0:
            return 0.0
        return (riskfree - risky) / (self.notional * annuity)
