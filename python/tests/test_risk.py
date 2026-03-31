"""Tests for curve-based risk."""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.risk import dv01_curve, key_rate_durations
from pricebook.bond import FixedRateBond
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.bootstrap import bootstrap
from pricebook.schedule import Frequency
from pricebook.day_count import DayCountConvention
from pricebook.discount_curve import DiscountCurve


REF = date(2024, 1, 15)

DEPOSITS = [
    (REF + relativedelta(days=1), 0.0530),
    (REF + relativedelta(weeks=1), 0.0528),
    (REF + relativedelta(months=1), 0.0525),
    (REF + relativedelta(months=2), 0.0520),
    (REF + relativedelta(months=3), 0.0515),
    (REF + relativedelta(months=6), 0.0500),
]

SWAPS = [
    (REF + relativedelta(years=1), 0.0480),
    (REF + relativedelta(years=2), 0.0460),
    (REF + relativedelta(years=3), 0.0450),
    (REF + relativedelta(years=5), 0.0440),
]


def _build_curve():
    return bootstrap(REF, DEPOSITS, SWAPS)


class TestBumpCurve:

    def test_bumped_curve_has_higher_zero_rate(self):
        curve = _build_curve()
        bumped = curve.bumped_at(5, 0.0001)
        pillar_dates = curve.pillar_dates
        assert bumped.zero_rate(pillar_dates[5]) > curve.zero_rate(pillar_dates[5])

    def test_unbumped_pillars_unchanged(self):
        curve = _build_curve()
        pillar_dates = curve.pillar_dates
        bumped = curve.bumped_at(5, 0.0001)
        for i in [0, 1, 2, 3, 4]:
            assert bumped.df(pillar_dates[i]) == pytest.approx(
                curve.df(pillar_dates[i]), rel=1e-6,
            )

    def test_parallel_bump_shifts_all(self):
        curve = _build_curve()
        bumped = curve.bumped(0.0001)
        for d in curve.pillar_dates:
            assert bumped.zero_rate(d) > curve.zero_rate(d)


class TestDV01Curve:

    def test_bond_dv01_negative(self):
        """Bond price falls when rates rise -> DV01 is negative."""
        curve = _build_curve()
        bond = FixedRateBond(REF, REF + relativedelta(years=5), coupon_rate=0.05)
        dv01 = dv01_curve(lambda c: bond.dirty_price(c), curve)
        assert dv01 < 0

    def test_longer_bond_larger_dv01(self):
        curve = _build_curve()
        bond_2y = FixedRateBond(REF, REF + relativedelta(years=2), coupon_rate=0.05)
        bond_5y = FixedRateBond(REF, REF + relativedelta(years=5), coupon_rate=0.05)
        dv01_2y = dv01_curve(lambda c: bond_2y.dirty_price(c), curve)
        dv01_5y = dv01_curve(lambda c: bond_5y.dirty_price(c), curve)
        assert abs(dv01_5y) > abs(dv01_2y)

    def test_swap_dv01(self):
        """Payer swap PV increases when rates rise -> positive DV01."""
        curve = _build_curve()
        swap = InterestRateSwap(REF, REF + relativedelta(years=5), fixed_rate=0.045)
        dv01 = dv01_curve(lambda c: swap.pv(c), curve)
        assert dv01 > 0


class TestKeyRateDurations:

    def test_returns_one_entry_per_pillar(self):
        curve = _build_curve()
        bond = FixedRateBond(REF, REF + relativedelta(years=5), coupon_rate=0.05)
        krd = key_rate_durations(lambda c: bond.dirty_price(c), curve)
        assert len(krd) == len(curve.pillar_dates)

    def test_key_rates_sum_approximates_parallel(self):
        """Sum of key rate sensitivities ≈ parallel DV01."""
        curve = _build_curve()
        bond = FixedRateBond(REF, REF + relativedelta(years=5), coupon_rate=0.05)
        pricer = lambda c: bond.dirty_price(c)

        krd = key_rate_durations(pricer, curve)
        krd_sum = sum(delta for _, delta in krd)
        par_dv01 = dv01_curve(pricer, curve)

        assert krd_sum == pytest.approx(par_dv01, rel=0.05)

    def test_sensitivity_concentrated_near_maturity(self):
        """A 5Y bond should be most sensitive to the 5Y pillar."""
        curve = _build_curve()
        bond = FixedRateBond(REF, REF + relativedelta(years=5), coupon_rate=0.05)
        krd = key_rate_durations(lambda c: bond.dirty_price(c), curve)

        # Find the maximum sensitivity
        max_abs = max(abs(delta) for _, delta in krd)
        # The last few pillars (near 5Y) should have the largest sensitivity
        last_delta = abs(krd[-1][1])
        assert last_delta == pytest.approx(max_abs, rel=0.3)
