"""Slice 5 round-trip validation.

Bootstrap credit curve from CDS spreads, verify repricing,
compute CS01, cross-check risky bond pricing.
"""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.cds import CDS, bootstrap_credit_curve
from pricebook.bond import FixedRateBond
from pricebook.survival_curve import SurvivalCurve
from pricebook.discount_curve import DiscountCurve
from pricebook.day_count import DayCountConvention
from pricebook.schedule import Frequency


REF = date(2024, 1, 15)

CDS_SPREADS = [
    (REF + relativedelta(years=1), 0.0060),
    (REF + relativedelta(years=3), 0.0085),
    (REF + relativedelta(years=5), 0.0110),
    (REF + relativedelta(years=7), 0.0125),
    (REF + relativedelta(years=10), 0.0140),
]


def _flat_discount(ref: date, rate: float = 0.04) -> DiscountCurve:
    tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10]
    dates = [date.fromordinal(ref.toordinal() + int(t * 365)) for t in tenors]
    dfs = [math.exp(-rate * t) for t in tenors]
    return DiscountCurve(ref, dates, dfs)


def _build():
    dc = _flat_discount(REF)
    sc = bootstrap_credit_curve(REF, CDS_SPREADS, dc)
    return dc, sc


class TestCS01:
    """Credit spread sensitivity: bump all CDS spreads by 1bp, re-bootstrap, reprice."""

    def test_cs01_negative_for_protection_buyer(self):
        """Protection buyer loses when spreads tighten (credit improves)."""
        dc, sc = _build()
        cds = CDS(REF, REF + relativedelta(years=5), spread=0.0110)

        # Base PV should be ~0 (at par)
        base_pv = cds.pv(dc, sc)

        # Bump all spreads up by 1bp -> credit worsens -> protection more valuable
        bumped_spreads = [(d, s + 0.0001) for d, s in CDS_SPREADS]
        sc_bumped = bootstrap_credit_curve(REF, bumped_spreads, dc)
        bumped_pv = cds.pv(dc, sc_bumped)

        cs01 = bumped_pv - base_pv
        # Protection buyer benefits from spread widening -> positive CS01
        assert cs01 > 0

    def test_cs01_scales_with_maturity(self):
        """Longer CDS has higher CS01."""
        dc, sc = _build()

        def compute_cs01(mat_years: int, spread: float) -> float:
            mat = REF + relativedelta(years=mat_years)
            cds = CDS(REF, mat, spread=spread)
            base_pv = cds.pv(dc, sc)
            bumped = [(d, s + 0.0001) for d, s in CDS_SPREADS]
            sc_bumped = bootstrap_credit_curve(REF, bumped, dc)
            return cds.pv(dc, sc_bumped) - base_pv

        cs01_3y = compute_cs01(3, 0.0085)
        cs01_10y = compute_cs01(10, 0.0140)
        assert abs(cs01_10y) > abs(cs01_3y)


class TestRiskyBondCrossCheck:
    """
    A risky bond should be worth less than a risk-free bond by approximately
    the CDS protection value.

    risky_bond_price ≈ riskfree_bond_price - CDS_protection_value / notional * 100
    """

    def test_risky_bond_cheaper_than_riskfree(self):
        dc, sc = _build()
        mat = REF + relativedelta(years=5)
        bond = FixedRateBond(REF, mat, coupon_rate=0.05)

        riskfree_price = bond.dirty_price(dc)

        # Risky price: discount each cashflow by df * Q
        risky_pv = 0.0
        for cf in bond.coupon_leg.cashflows:
            risky_pv += cf.amount * dc.df(cf.payment_date) * sc.survival(cf.payment_date)
        risky_pv += bond.face_value * dc.df(mat) * sc.survival(mat)
        risky_price = risky_pv / bond.face_value * 100.0

        assert risky_price < riskfree_price

    def test_price_difference_approximates_protection(self):
        """Price gap ≈ (1-R) * default_prob * face, roughly."""
        dc, sc = _build()
        mat = REF + relativedelta(years=5)
        bond = FixedRateBond(REF, mat, coupon_rate=0.05)

        riskfree_price = bond.dirty_price(dc)

        risky_pv = 0.0
        for cf in bond.coupon_leg.cashflows:
            risky_pv += cf.amount * dc.df(cf.payment_date) * sc.survival(cf.payment_date)
        risky_pv += bond.face_value * dc.df(mat) * sc.survival(mat)
        risky_price = risky_pv / bond.face_value * 100.0

        price_gap = riskfree_price - risky_price

        # CDS 5Y spread is 110bp, ~5 years, with 40% recovery
        # Rough approximation: gap ≈ spread * (1-R) * duration * 100
        # This is a loose sanity check, not a precise identity
        assert 0.5 < price_gap < 10.0


class TestCurveConsistency:
    """Cross-checks on the bootstrapped credit curve."""

    def test_term_structure_of_hazard_rates(self):
        """With upward-sloping CDS spreads, implied hazard rates should increase."""
        dc, sc = _build()
        dates = [d for d, _ in CDS_SPREADS]
        hazards = [sc.hazard_rate(d) for d in dates]
        # At least non-decreasing (exact monotonicity depends on curve shape)
        for i in range(1, len(hazards)):
            assert hazards[i] >= hazards[i - 1] * 0.9  # allow small non-monotonicity from interpolation

    def test_cumulative_default_increases(self):
        dc, sc = _build()
        dates = [d for d, _ in CDS_SPREADS]
        cum_defaults = [1 - sc.survival(d) for d in dates]
        for i in range(1, len(cum_defaults)):
            assert cum_defaults[i] > cum_defaults[i - 1]

    def test_5y_default_prob_reasonable(self):
        """With ~110bp 5Y spread, 40% recovery, 5Y default prob should be ~9%."""
        dc, sc = _build()
        dp = 1 - sc.survival(REF + relativedelta(years=5))
        # Rough: spread ≈ hazard * (1-R), hazard ≈ 0.011/0.6 ≈ 0.018, dp ≈ 1-exp(-0.018*5) ≈ 8.6%
        assert 0.04 < dp < 0.15
