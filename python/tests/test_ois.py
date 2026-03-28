"""Tests for OIS swap and OIS bootstrap."""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.ois import OISSwap, bootstrap_ois
from pricebook.schedule import Frequency
from pricebook.day_count import DayCountConvention
from pricebook.discount_curve import DiscountCurve


def _flat_curve(ref: date, rate: float = 0.05) -> DiscountCurve:
    tenors_years = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    dates = [date.fromordinal(ref.toordinal() + int(t * 365)) for t in tenors_years]
    dfs = [math.exp(-rate * t) for t in tenors_years]
    return DiscountCurve(ref, dates, dfs)


REF = date(2024, 1, 15)

# USD SOFR OIS par rates
OIS_RATES = [
    (REF + relativedelta(months=1), 0.0530),
    (REF + relativedelta(months=3), 0.0525),
    (REF + relativedelta(months=6), 0.0510),
    (REF + relativedelta(years=1), 0.0490),
    (REF + relativedelta(years=2), 0.0470),
    (REF + relativedelta(years=3), 0.0455),
    (REF + relativedelta(years=5), 0.0440),
]


class TestOISSwapPricing:

    def test_par_rate_positive(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        ois = OISSwap(ref, date(2026, 1, 15), fixed_rate=0.0)
        assert ois.par_rate(curve) > 0

    def test_pv_zero_at_par(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        ois = OISSwap(ref, date(2026, 1, 15), fixed_rate=0.0)
        par = ois.par_rate(curve)
        ois_at_par = OISSwap(ref, date(2026, 1, 15), fixed_rate=par)
        assert ois_at_par.pv(curve) == pytest.approx(0.0, abs=1.0)

    def test_pv_positive_when_fixed_below_market(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        ois = OISSwap(ref, date(2026, 1, 15), fixed_rate=0.01)
        assert ois.pv(curve) > 0

    def test_pv_float_telescopes(self):
        """PV_float = notional * (df(start) - df(end))."""
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        notional = 1_000_000.0
        end = date(2027, 1, 15)
        ois = OISSwap(ref, end, fixed_rate=0.05, notional=notional)
        expected = notional * (curve.df(ref) - curve.df(end))
        assert ois.pv_float(curve) == pytest.approx(expected, rel=1e-10)


class TestOISBootstrap:

    def test_round_trip_reprices(self):
        """Bootstrapped OIS curve reprices all input OIS swaps at par."""
        curve = bootstrap_ois(REF, OIS_RATES)

        for mat, par_rate in OIS_RATES:
            ois = OISSwap(REF, mat, fixed_rate=par_rate)
            pv = ois.pv(curve)
            assert abs(pv) < 50.0, \
                f"OIS {mat} not at par: PV={pv:.2f}"

    def test_par_rates_recovered(self):
        curve = bootstrap_ois(REF, OIS_RATES)

        for mat, par_rate in OIS_RATES:
            ois = OISSwap(REF, mat, fixed_rate=0.0)
            recovered = ois.par_rate(curve)
            assert recovered == pytest.approx(par_rate, abs=5e-4), \
                f"OIS {mat}: input={par_rate:.4f}, recovered={recovered:.4f}"

    def test_discount_factors_decreasing(self):
        curve = bootstrap_ois(REF, OIS_RATES)
        dfs = [curve.df(d) for d, _ in OIS_RATES]
        for i in range(1, len(dfs)):
            assert dfs[i] <= dfs[i - 1]

    def test_discount_factors_positive(self):
        curve = bootstrap_ois(REF, OIS_RATES)
        for d, _ in OIS_RATES:
            assert curve.df(d) > 0

    def test_zero_rates_positive(self):
        curve = bootstrap_ois(REF, OIS_RATES)
        for d, _ in OIS_RATES:
            assert curve.zero_rate(d) > 0

    def test_unsorted_raises(self):
        bad = [(REF + relativedelta(years=2), 0.05), (REF + relativedelta(years=1), 0.04)]
        with pytest.raises(ValueError):
            bootstrap_ois(REF, bad)
