"""Tests for discount curve."""

import math
import pytest
from datetime import date

from pricebook.day_count import DayCountConvention
from pricebook.interpolation import InterpolationMethod
from pricebook.discount_curve import DiscountCurve
from pricebook.deposit import Deposit


@pytest.fixture
def simple_curve():
    """A curve from 3 deposit-like pillars."""
    ref = date(2024, 1, 15)
    dates = [date(2024, 4, 15), date(2024, 7, 15), date(2025, 1, 15)]
    # Rates: 5%, 5.1%, 5.2% -> compute dfs from deposit formula
    rates = [0.05, 0.051, 0.052]
    dfs = []
    for d, r in zip(dates, rates):
        dep = Deposit(ref, d, r, day_count=DayCountConvention.ACT_365_FIXED)
        dfs.append(dep.discount_factor)
    return DiscountCurve(ref, dates, dfs)


class TestDiscountFactor:
    """Discount factor queries."""

    def test_df_at_reference_date(self, simple_curve):
        assert simple_curve.df(date(2024, 1, 15)) == pytest.approx(1.0)

    def test_df_at_pillar(self, simple_curve):
        # Should recover the input df at 3M
        dep = Deposit(
            date(2024, 1, 15), date(2024, 4, 15), 0.05,
            day_count=DayCountConvention.ACT_365_FIXED,
        )
        assert simple_curve.df(date(2024, 4, 15)) == pytest.approx(dep.discount_factor)

    def test_df_between_pillars(self, simple_curve):
        # Between 3M and 6M, df should be between the two pillar values
        df_3m = simple_curve.df(date(2024, 4, 15))
        df_6m = simple_curve.df(date(2024, 7, 15))
        df_mid = simple_curve.df(date(2024, 6, 1))
        assert df_6m < df_mid < df_3m

    def test_df_before_reference_is_one(self, simple_curve):
        assert simple_curve.df(date(2024, 1, 1)) == pytest.approx(1.0)

    def test_df_decreasing(self, simple_curve):
        d1 = simple_curve.df(date(2024, 4, 15))
        d2 = simple_curve.df(date(2024, 7, 15))
        d3 = simple_curve.df(date(2025, 1, 15))
        assert d1 > d2 > d3


class TestZeroRate:
    """Continuously compounded zero rates."""

    def test_zero_rate_at_reference(self, simple_curve):
        assert simple_curve.zero_rate(date(2024, 1, 15)) == pytest.approx(0.0)

    def test_zero_rate_positive(self, simple_curve):
        r = simple_curve.zero_rate(date(2024, 7, 15))
        assert r > 0

    def test_zero_rate_consistent_with_df(self, simple_curve):
        # df = exp(-r * t) => r = -ln(df) / t
        d = date(2024, 7, 15)
        df = simple_curve.df(d)
        from pricebook.day_count import year_fraction
        t = year_fraction(date(2024, 1, 15), d, DayCountConvention.ACT_365_FIXED)
        expected_r = -math.log(df) / t
        assert simple_curve.zero_rate(d) == pytest.approx(expected_r)


class TestForwardRate:
    """Simply compounded forward rates."""

    def test_forward_rate_positive(self, simple_curve):
        f = simple_curve.forward_rate(date(2024, 4, 15), date(2024, 7, 15))
        assert f > 0

    def test_forward_rate_formula(self, simple_curve):
        # F = (df1/df2 - 1) / tau
        d1, d2 = date(2024, 4, 15), date(2024, 7, 15)
        df1 = simple_curve.df(d1)
        df2 = simple_curve.df(d2)
        from pricebook.day_count import year_fraction
        tau = year_fraction(d1, d2, DayCountConvention.ACT_365_FIXED)
        expected = (df1 / df2 - 1.0) / tau
        assert simple_curve.forward_rate(d1, d2) == pytest.approx(expected)

    def test_forward_rate_d1_after_d2_raises(self, simple_curve):
        with pytest.raises(ValueError):
            simple_curve.forward_rate(date(2024, 7, 15), date(2024, 4, 15))


class TestInterpolationMethods:
    """Curve should work with all interpolation methods."""

    def test_all_methods_recover_pillars(self):
        ref = date(2024, 1, 1)
        dates = [date(2024, 7, 1), date(2025, 1, 1)]
        dfs = [0.975, 0.95]
        for method in InterpolationMethod:
            curve = DiscountCurve(ref, dates, dfs, interpolation=method)
            assert curve.df(dates[0]) == pytest.approx(dfs[0], abs=1e-10)
            assert curve.df(dates[1]) == pytest.approx(dfs[1], abs=1e-10)


class TestRoundTrip:
    """Bootstrap round-trip: deposits -> curve -> reprice deposits."""

    def test_deposit_round_trip(self):
        ref = date(2024, 1, 15)
        tenors = [date(2024, 2, 15), date(2024, 4, 15), date(2024, 7, 15), date(2025, 1, 15)]
        rates = [0.048, 0.050, 0.051, 0.052]

        # Build deposits and extract dfs
        deposits = []
        dfs = []
        for d, r in zip(tenors, rates):
            dep = Deposit(ref, d, r, day_count=DayCountConvention.ACT_360)
            deposits.append(dep)
            dfs.append(dep.discount_factor)

        # Build curve
        curve = DiscountCurve(ref, tenors, dfs, day_count=DayCountConvention.ACT_360)

        # Reprice each deposit: PV should be zero
        for dep in deposits:
            df = curve.df(dep.end)
            assert dep.pv(df) == pytest.approx(0.0, abs=1e-12)
