"""Tests for global multi-curve solver."""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.global_solver import global_bootstrap, coupled_bootstrap
from pricebook.bootstrap import bootstrap
from pricebook.swap import InterestRateSwap


REF = date(2024, 1, 15)


def _deposits():
    return [
        (REF + relativedelta(months=1), 0.053),
        (REF + relativedelta(months=3), 0.052),
        (REF + relativedelta(months=6), 0.051),
    ]


def _swaps():
    return [
        (REF + relativedelta(years=1), 0.050),
        (REF + relativedelta(years=2), 0.048),
        (REF + relativedelta(years=5), 0.046),
        (REF + relativedelta(years=10), 0.044),
    ]


class TestGlobalBootstrap:
    def test_builds_curve(self):
        curve = global_bootstrap(REF, _deposits(), _swaps())
        d5y = REF + relativedelta(years=5)
        assert 0.7 < curve.df(d5y) < 0.85

    def test_matches_sequential(self):
        """Global Newton should match sequential bootstrap."""
        seq_curve = bootstrap(REF, _deposits(), _swaps())
        glob_curve = global_bootstrap(REF, _deposits(), _swaps())

        for t in [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
            d = date.fromordinal(REF.toordinal() + int(t * 365))
            assert glob_curve.df(d) == pytest.approx(seq_curve.df(d), rel=0.02)

    def test_reprices_deposits(self):
        curve = global_bootstrap(REF, _deposits(), [])
        for mat, rate in _deposits():
            tau = (mat - REF).days / 360.0
            expected_df = 1.0 / (1.0 + rate * tau)
            assert curve.df(mat) == pytest.approx(expected_df, rel=0.01)

    def test_reprices_swaps(self):
        """Par rate from built curve should match input swap rate."""
        curve = global_bootstrap(REF, _deposits(), _swaps())
        for mat, rate in _swaps():
            swap = InterestRateSwap(REF, mat, rate)
            par = swap.par_rate(curve)
            assert par == pytest.approx(rate, abs=0.003)


class TestCoupledBootstrap:
    def test_builds_two_curves(self):
        ois, proj = coupled_bootstrap(
            REF,
            ois_deposits=_deposits(),
            ois_swaps=[(REF + relativedelta(years=2), 0.048),
                       (REF + relativedelta(years=5), 0.046)],
            projection_swaps=[(REF + relativedelta(years=2), 0.049),
                              (REF + relativedelta(years=5), 0.047)],
        )
        d5y = REF + relativedelta(years=5)
        assert 0.7 < ois.df(d5y) < 0.85
        assert 0.7 < proj.df(d5y) < 0.85

    def test_ois_different_from_projection(self):
        """With different swap rates, curves should differ."""
        ois, proj = coupled_bootstrap(
            REF,
            ois_deposits=_deposits(),
            ois_swaps=[(REF + relativedelta(years=5), 0.046)],
            projection_swaps=[(REF + relativedelta(years=5), 0.050)],
        )
        d5y = REF + relativedelta(years=5)
        assert ois.df(d5y) != pytest.approx(proj.df(d5y), abs=0.001)

    def test_same_rates_similar_curves(self):
        """With identical rates, OIS and projection should be close."""
        rate = 0.048
        ois, proj = coupled_bootstrap(
            REF,
            ois_deposits=_deposits(),
            ois_swaps=[(REF + relativedelta(years=5), rate)],
            projection_swaps=[(REF + relativedelta(years=5), rate)],
        )
        d5y = REF + relativedelta(years=5)
        assert ois.df(d5y) == pytest.approx(proj.df(d5y), rel=0.05)
