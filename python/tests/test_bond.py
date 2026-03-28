"""Tests for fixed-rate bond."""

import math
import pytest
from datetime import date

from pricebook.bond import FixedRateBond
from pricebook.schedule import Frequency
from pricebook.day_count import DayCountConvention
from pricebook.discount_curve import DiscountCurve


def _flat_curve(ref: date, rate: float = 0.05) -> DiscountCurve:
    tenors_years = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    dates = [date.fromordinal(ref.toordinal() + int(t * 365)) for t in tenors_years]
    dfs = [math.exp(-rate * t) for t in tenors_years]
    return DiscountCurve(ref, dates, dfs)


class TestDirtyPrice:

    def test_par_bond_prices_near_100(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        dp = bond.dirty_price(curve)
        assert 95.0 < dp < 105.0

    def test_zero_coupon_dirty_price(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.0)
        dp = bond.dirty_price(curve)
        expected = curve.df(date(2029, 1, 15)) * 100.0
        assert dp == pytest.approx(expected, rel=1e-6)

    def test_higher_coupon_higher_price(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        bond_low = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.03)
        bond_high = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.07)
        assert bond_high.dirty_price(curve) > bond_low.dirty_price(curve)

    def test_higher_yield_lower_price(self):
        ref = date(2024, 1, 15)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        dp_low = bond.dirty_price(_flat_curve(ref, rate=0.03))
        dp_high = bond.dirty_price(_flat_curve(ref, rate=0.08))
        assert dp_low > dp_high


class TestAccruedInterest:

    def test_accrued_at_coupon_date_is_zero(self):
        bond = FixedRateBond(date(2024, 1, 15), date(2026, 1, 15), coupon_rate=0.06)
        ai = bond.accrued_interest(date(2024, 1, 15))
        assert ai == pytest.approx(0.0)

    def test_accrued_mid_period(self):
        bond = FixedRateBond(
            date(2024, 1, 15), date(2025, 1, 15),
            coupon_rate=0.06, frequency=Frequency.SEMI_ANNUAL,
            day_count=DayCountConvention.THIRTY_360,
        )
        ai = bond.accrued_interest(date(2024, 4, 15))
        assert ai == pytest.approx(1.5, rel=1e-2)

    def test_accrued_after_maturity_is_zero(self):
        bond = FixedRateBond(date(2024, 1, 15), date(2025, 1, 15), coupon_rate=0.06)
        ai = bond.accrued_interest(date(2025, 6, 15))
        assert ai == pytest.approx(0.0)


class TestCleanPrice:

    def test_clean_equals_dirty_at_coupon_date(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        assert bond.clean_price(curve) == pytest.approx(bond.dirty_price(curve), rel=1e-6)

    def test_clean_less_than_dirty_mid_period(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        settlement = date(2024, 4, 15)
        clean = bond.clean_price(curve, settlement)
        dirty = bond.dirty_price(curve)
        assert clean < dirty

    def test_clean_price_default_settlement(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        assert bond.clean_price(curve) == bond.clean_price(curve, ref)


class TestValidation:

    def test_negative_face_value_raises(self):
        with pytest.raises(ValueError):
            FixedRateBond(date(2024, 1, 1), date(2025, 1, 1), coupon_rate=0.05, face_value=-100.0)

    def test_zero_face_value_raises(self):
        with pytest.raises(ValueError):
            FixedRateBond(date(2024, 1, 1), date(2025, 1, 1), coupon_rate=0.05, face_value=0.0)

    def test_issue_after_maturity_raises(self):
        with pytest.raises(ValueError):
            FixedRateBond(date(2026, 1, 1), date(2025, 1, 1), coupon_rate=0.05)
