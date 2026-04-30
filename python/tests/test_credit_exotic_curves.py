"""Tests for credit_exotic curve integration: verify curves are actually used."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.credit_exotic import (
    capped_coupon_bond, digital_cds,
    credit_range_accrual, credit_linked_loan,
)
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 1, 15)


class TestCappedCouponBondCurves:

    def test_flat_curve_matches_flat_scalar(self):
        """Flat curves should give same result as flat scalars."""
        dc = make_flat_curve(REF, 0.05)
        sc = make_flat_survival(REF, 0.02)

        with_curves = capped_coupon_bond(
            100, 0.04, 0.02, 0.08, 5,
            discount_curve=dc, survival_curve=sc,
            flat_rate=0.05, flat_hazard=0.02,
        )
        without_curves = capped_coupon_bond(
            100, 0.04, 0.02, 0.08, 5,
            flat_rate=0.05, flat_hazard=0.02,
        )
        assert with_curves.dirty_price == pytest.approx(without_curves.dirty_price, rel=0.01)

    def test_curves_actually_used(self):
        """Different curve rates should give different results."""
        dc_low = make_flat_curve(REF, 0.02)
        dc_high = make_flat_curve(REF, 0.08)
        sc = make_flat_survival(REF, 0.02)

        low = capped_coupon_bond(
            100, 0.04, 0.02, 0.08, 5,
            discount_curve=dc_low, survival_curve=sc,
            flat_rate=0.05, flat_hazard=0.02,
        )
        high = capped_coupon_bond(
            100, 0.04, 0.02, 0.08, 5,
            discount_curve=dc_high, survival_curve=sc,
            flat_rate=0.05, flat_hazard=0.02,
        )
        # Lower discount rate → higher price
        assert low.dirty_price > high.dirty_price


class TestDigitalCDSCurves:

    def test_flat_matches(self):
        dc = make_flat_curve(REF, 0.05)
        sc = make_flat_survival(REF, 0.02)

        with_curves = digital_cds(
            1_000_000, 500_000, 0.01, 5,
            discount_curve=dc, survival_curve=sc,
            flat_rate=0.05, flat_hazard=0.02,
        )
        without_curves = digital_cds(
            1_000_000, 500_000, 0.01, 5,
            flat_rate=0.05, flat_hazard=0.02,
        )
        assert with_curves.pv == pytest.approx(without_curves.pv, rel=0.05)

    def test_curves_differ(self):
        dc = make_flat_curve(REF, 0.05)
        sc_low = make_flat_survival(REF, 0.01)
        sc_high = make_flat_survival(REF, 0.05)

        low = digital_cds(
            1_000_000, 500_000, 0.01, 5,
            discount_curve=dc, survival_curve=sc_low,
        )
        high = digital_cds(
            1_000_000, 500_000, 0.01, 5,
            discount_curve=dc, survival_curve=sc_high,
        )
        # Higher hazard → more defaults → higher protection PV
        assert high.pv > low.pv


class TestRangeAccrualCurves:

    def test_flat_matches(self):
        dc = make_flat_curve(REF, 0.05)
        sc = make_flat_survival(REF, 0.02)

        with_c = credit_range_accrual(
            100, 0.06, 0.01, 0.05, 3.0, 0.03, 0.01,
            discount_curve=dc, survival_curve=sc,
            flat_rate=0.05, flat_hazard=0.02,
        )
        without_c = credit_range_accrual(
            100, 0.06, 0.01, 0.05, 3.0, 0.03, 0.01,
            flat_rate=0.05, flat_hazard=0.02,
        )
        assert with_c.pv == pytest.approx(without_c.pv, rel=0.01)


class TestCreditLinkedLoanCurves:

    def test_flat_matches(self):
        dc = make_flat_curve(REF, 0.05)
        sc = make_flat_survival(REF, 0.02)

        with_c = credit_linked_loan(
            1_000_000, 0.04, 0.02, 5,
            discount_curve=dc, survival_curve=sc,
            flat_rate=0.05, flat_hazard=0.02,
        )
        without_c = credit_linked_loan(
            1_000_000, 0.04, 0.02, 5,
            flat_rate=0.05, flat_hazard=0.02,
        )
        assert with_c.pv == pytest.approx(without_c.pv, rel=0.01)

    def test_curves_differ(self):
        dc = make_flat_curve(REF, 0.05)
        sc_low = make_flat_survival(REF, 0.01)
        sc_high = make_flat_survival(REF, 0.05)

        low = credit_linked_loan(
            1_000_000, 0.04, 0.02, 5,
            discount_curve=dc, survival_curve=sc_low,
        )
        high = credit_linked_loan(
            1_000_000, 0.04, 0.02, 5,
            discount_curve=dc, survival_curve=sc_high,
        )
        # Higher hazard → more defaults → lower PV (more expected loss)
        assert low.pv > high.pv
