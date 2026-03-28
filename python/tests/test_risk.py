"""Tests for curve-based risk."""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.risk import dv01_curve, key_rate_durations, parallel_bump_curve, bump_curve
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

PILLAR_DATES = [d for d, _ in DEPOSITS] + [d for d, _ in SWAPS]


def _build_curve():
    return bootstrap(REF, DEPOSITS, SWAPS)


def _pillar_dfs(curve):
    return [curve.df(d) for d in PILLAR_DATES]


class TestBumpCurve:

    def test_bumped_curve_has_higher_zero_rate(self):
        curve = _build_curve()
        dfs = _pillar_dfs(curve)
        bumped = bump_curve(curve, PILLAR_DATES, dfs, pillar_index=5, bump_bps=1.0)
        # Bumped pillar should have higher zero rate
        assert bumped.zero_rate(PILLAR_DATES[5]) > curve.zero_rate(PILLAR_DATES[5])

    def test_unbumped_pillars_unchanged(self):
        curve = _build_curve()
        dfs = _pillar_dfs(curve)
        bumped = bump_curve(curve, PILLAR_DATES, dfs, pillar_index=5, bump_bps=1.0)
        # Other pillars should have same df
        for i in [0, 1, 2, 3, 4]:
            assert bumped.df(PILLAR_DATES[i]) == pytest.approx(curve.df(PILLAR_DATES[i]), rel=1e-10)

    def test_parallel_bump_shifts_all(self):
        curve = _build_curve()
        dfs = _pillar_dfs(curve)
        bumped = parallel_bump_curve(curve, PILLAR_DATES, dfs, bump_bps=1.0)
        for d in PILLAR_DATES:
            assert bumped.zero_rate(d) > curve.zero_rate(d)


class TestDV01Curve:

    def test_bond_dv01_negative(self):
        """Bond price falls when rates rise -> DV01 is negative."""
        curve = _build_curve()
        dfs = _pillar_dfs(curve)
        bond = FixedRateBond(REF, REF + relativedelta(years=5), coupon_rate=0.05)
        dv01 = dv01_curve(lambda c: bond.dirty_price(c), curve, PILLAR_DATES, dfs)
        assert dv01 < 0

    def test_longer_bond_larger_dv01(self):
        curve = _build_curve()
        dfs = _pillar_dfs(curve)
        bond_2y = FixedRateBond(REF, REF + relativedelta(years=2), coupon_rate=0.05)
        bond_5y = FixedRateBond(REF, REF + relativedelta(years=5), coupon_rate=0.05)
        dv01_2y = dv01_curve(lambda c: bond_2y.dirty_price(c), curve, PILLAR_DATES, dfs)
        dv01_5y = dv01_curve(lambda c: bond_5y.dirty_price(c), curve, PILLAR_DATES, dfs)
        assert abs(dv01_5y) > abs(dv01_2y)

    def test_swap_dv01(self):
        """Payer swap PV increases when rates rise -> positive DV01."""
        curve = _build_curve()
        dfs = _pillar_dfs(curve)
        swap = InterestRateSwap(REF, REF + relativedelta(years=5), fixed_rate=0.045)
        dv01 = dv01_curve(lambda c: swap.pv(c), curve, PILLAR_DATES, dfs)
        assert dv01 > 0


class TestKeyRateDurations:

    def test_returns_one_entry_per_pillar(self):
        curve = _build_curve()
        dfs = _pillar_dfs(curve)
        bond = FixedRateBond(REF, REF + relativedelta(years=5), coupon_rate=0.05)
        krd = key_rate_durations(lambda c: bond.dirty_price(c), curve, PILLAR_DATES, dfs)
        assert len(krd) == len(PILLAR_DATES)

    def test_key_rates_sum_approximates_parallel(self):
        """Sum of key rate sensitivities ≈ parallel DV01."""
        curve = _build_curve()
        dfs = _pillar_dfs(curve)
        bond = FixedRateBond(REF, REF + relativedelta(years=5), coupon_rate=0.05)
        pricer = lambda c: bond.dirty_price(c)

        krd = key_rate_durations(pricer, curve, PILLAR_DATES, dfs)
        krd_sum = sum(delta for _, delta in krd)
        par_dv01 = dv01_curve(pricer, curve, PILLAR_DATES, dfs)

        assert krd_sum == pytest.approx(par_dv01, rel=0.05)

    def test_sensitivity_concentrated_near_maturity(self):
        """A 5Y bond should be most sensitive to the 5Y pillar."""
        curve = _build_curve()
        dfs = _pillar_dfs(curve)
        bond = FixedRateBond(REF, REF + relativedelta(years=5), coupon_rate=0.05)
        krd = key_rate_durations(lambda c: bond.dirty_price(c), curve, PILLAR_DATES, dfs)

        # Find the 5Y pillar (last swap)
        five_year_pillar = REF + relativedelta(years=5)
        five_year_delta = None
        max_abs_delta = 0.0
        for d, delta in krd:
            if d == five_year_pillar:
                five_year_delta = delta
            if abs(delta) > max_abs_delta:
                max_abs_delta = abs(delta)

        assert five_year_delta is not None
        assert abs(five_year_delta) == pytest.approx(max_abs_delta, rel=0.1)
