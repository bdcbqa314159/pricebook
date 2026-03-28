"""Slice 3 round-trip validation.

Bootstrap a curve, price a bond, compute yield, verify that
analytical risk measures agree with bump-and-reprice.
"""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.bootstrap import bootstrap
from pricebook.bond import FixedRateBond
from pricebook.risk import dv01_curve, key_rate_durations
from pricebook.schedule import Frequency
from pricebook.day_count import DayCountConvention


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


class TestBondPricingFromBootstrappedCurve:
    """Price a bond off a bootstrapped curve and verify consistency."""

    def test_ytm_from_curve_price_is_consistent(self):
        """Compute dirty price from curve, extract YTM, reprice from YTM — must match."""
        curve = bootstrap(REF, DEPOSITS, SWAPS)
        bond = FixedRateBond(REF, REF + relativedelta(years=5), coupon_rate=0.045)

        dirty = bond.dirty_price(curve)
        ytm = bond.yield_to_maturity(dirty)
        dirty_from_ytm = bond._price_from_ytm(ytm)

        assert dirty_from_ytm == pytest.approx(dirty, abs=0.01)

    def test_zero_coupon_yield_matches_zero_rate(self):
        """For a zero-coupon bond, YTM should be close to the curve's zero rate."""
        curve = bootstrap(REF, DEPOSITS, SWAPS)
        mat = REF + relativedelta(years=3)
        bond = FixedRateBond(REF, mat, coupon_rate=0.0)

        dirty = bond.dirty_price(curve)
        ytm = bond.yield_to_maturity(dirty)
        zero = curve.zero_rate(mat)

        # Not exactly equal (compounding convention differs) but close
        assert abs(ytm - zero) < 0.005


class TestAnalyticalVsBumpRisk:
    """Verify that analytical duration ≈ bump-and-reprice DV01."""

    def test_yield_dv01_matches_curve_dv01(self):
        """Analytical yield DV01 should be close to curve parallel DV01."""
        curve = bootstrap(REF, DEPOSITS, SWAPS)
        dfs = [curve.df(d) for d in PILLAR_DATES]
        bond = FixedRateBond(REF, REF + relativedelta(years=5), coupon_rate=0.045)

        dirty = bond.dirty_price(curve)
        ytm = bond.yield_to_maturity(dirty)

        yield_dv01 = bond.dv01_yield(ytm)
        curve_dv01 = abs(dv01_curve(lambda c: bond.dirty_price(c), curve, PILLAR_DATES, dfs))

        # These won't be identical (yield shift vs parallel zero rate shift)
        # but should be in the same order of magnitude
        assert yield_dv01 == pytest.approx(curve_dv01, rel=0.15)

    def test_convexity_improves_large_shift(self):
        """For a 50bp parallel shift, duration + convexity should beat duration alone."""
        curve = bootstrap(REF, DEPOSITS, SWAPS)
        bond = FixedRateBond(REF, REF + relativedelta(years=5), coupon_rate=0.045)

        dirty = bond.dirty_price(curve)
        ytm = bond.yield_to_maturity(dirty)
        mod_dur = bond.modified_duration(ytm)
        conv = bond.convexity(ytm)

        dy = 0.005  # 50bp
        actual = bond._price_from_ytm(ytm + dy) - dirty
        dur_only = -mod_dur * dirty * dy
        dur_conv = dur_only + 0.5 * conv * dirty * dy ** 2

        assert abs(actual - dur_conv) < abs(actual - dur_only)


class TestKeyRateRoundTrip:
    """Key rate durations should decompose the parallel DV01."""

    def test_key_rates_sum_to_parallel(self):
        curve = bootstrap(REF, DEPOSITS, SWAPS)
        dfs = [curve.df(d) for d in PILLAR_DATES]
        bond = FixedRateBond(REF, REF + relativedelta(years=5), coupon_rate=0.045)
        pricer = lambda c: bond.dirty_price(c)

        krd = key_rate_durations(pricer, curve, PILLAR_DATES, dfs)
        par_dv01 = dv01_curve(pricer, curve, PILLAR_DATES, dfs)

        krd_sum = sum(d for _, d in krd)
        assert krd_sum == pytest.approx(par_dv01, rel=0.05)

    def test_short_bond_insensitive_to_long_pillars(self):
        """A 1Y bond should have negligible sensitivity to the 5Y pillar."""
        curve = bootstrap(REF, DEPOSITS, SWAPS)
        dfs = [curve.df(d) for d in PILLAR_DATES]
        bond = FixedRateBond(REF, REF + relativedelta(years=1), coupon_rate=0.05)

        krd = key_rate_durations(lambda c: bond.dirty_price(c), curve, PILLAR_DATES, dfs)

        five_year = REF + relativedelta(years=5)
        for d, delta in krd:
            if d == five_year:
                assert abs(delta) < 0.001
