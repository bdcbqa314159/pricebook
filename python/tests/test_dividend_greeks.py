"""Tests for enhanced dividend Greeks."""

import pytest
import math
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.equity.dividend_greeks import (
    compute_dividend_greeks, theta_decomposition, dividend_scenario_ladder,
)
from pricebook.equity.dividend_model import Dividend
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.interpolation import InterpolationMethod
from pricebook.models.black76 import OptionType


REF = date(2024, 11, 4)


def _make_curve():
    dates = [REF + relativedelta(years=y) for y in [1, 2, 5]]
    dfs = [math.exp(-0.05 * y) for y in [1, 2, 5]]
    return DiscountCurve(REF, dates, dfs, interpolation=InterpolationMethod.LOG_LINEAR)


def _make_divs():
    return [
        Dividend(REF + relativedelta(months=3), 1.5),
        Dividend(REF + relativedelta(months=6), 1.5),
        Dividend(REF + relativedelta(months=9), 1.5),
    ]


class TestDividendGreeks:
    def test_call_div_delta_negative(self):
        """Higher dividends reduce call value."""
        curve = _make_curve()
        divs = _make_divs()
        g = compute_dividend_greeks(100, 100, 0.20, 0.05,
                                     REF + relativedelta(years=1), divs, curve,
                                     OptionType.CALL)
        assert g.div_delta < 0  # more divs → lower call

    def test_put_div_delta_positive(self):
        """Higher dividends increase put value."""
        curve = _make_curve()
        divs = _make_divs()
        g = compute_dividend_greeks(100, 100, 0.20, 0.05,
                                     REF + relativedelta(years=1), divs, curve,
                                     OptionType.PUT)
        assert g.div_delta > 0  # more divs → higher put

    def test_cross_gamma_exists(self):
        """Cross-gamma should be finite."""
        curve = _make_curve()
        divs = _make_divs()
        g = compute_dividend_greeks(100, 100, 0.20, 0.05,
                                     REF + relativedelta(years=1), divs, curve)
        assert math.isfinite(g.cross_gamma_spot_div)

    def test_spot_delta_positive_for_call(self):
        curve = _make_curve()
        divs = _make_divs()
        g = compute_dividend_greeks(100, 100, 0.20, 0.05,
                                     REF + relativedelta(years=1), divs, curve,
                                     OptionType.CALL)
        assert g.spot_delta > 0

    def test_to_dict(self):
        curve = _make_curve()
        divs = _make_divs()
        g = compute_dividend_greeks(100, 100, 0.20, 0.05,
                                     REF + relativedelta(years=1), divs, curve)
        d = g.to_dict()
        assert "cross_gamma_spot_div" in d
        assert "div_delta" in d


class TestThetaDecomposition:
    def test_components_finite(self):
        curve = _make_curve()
        divs = _make_divs()
        td = theta_decomposition(100, 100, 0.20, 0.05,
                                  REF + relativedelta(years=1), divs, curve)
        assert math.isfinite(td["total_theta"])
        assert math.isfinite(td["carry_theta"])
        assert math.isfinite(td["div_theta"])

    def test_carry_theta_negative(self):
        """Carry theta (cost of financing) should be negative for positive rate."""
        curve = _make_curve()
        divs = _make_divs()
        td = theta_decomposition(100, 100, 0.20, 0.05,
                                  REF + relativedelta(years=1), divs, curve)
        assert td["carry_theta"] < 0


class TestScenarioLadder:
    def test_default_shifts(self):
        curve = _make_curve()
        divs = _make_divs()
        ladder = dividend_scenario_ladder(100, 100, 0.20, 0.05,
                                           REF + relativedelta(years=1), divs, curve)
        assert len(ladder) == 7  # default 7 shifts

    def test_base_zero_change(self):
        curve = _make_curve()
        divs = _make_divs()
        ladder = dividend_scenario_ladder(100, 100, 0.20, 0.05,
                                           REF + relativedelta(years=1), divs, curve)
        base_entry = [l for l in ladder if l["div_shift_pct"] == 0.0]
        assert len(base_entry) == 1
        assert abs(base_entry[0]["change"]) < 1e-10

    def test_call_monotone(self):
        """Call price should decrease as dividends increase."""
        curve = _make_curve()
        divs = _make_divs()
        ladder = dividend_scenario_ladder(100, 100, 0.20, 0.05,
                                           REF + relativedelta(years=1), divs, curve,
                                           OptionType.CALL)
        prices = [l["price"] for l in ladder]
        # Should be roughly decreasing (higher divs → lower call)
        assert prices[0] > prices[-1]

    def test_custom_shifts(self):
        curve = _make_curve()
        divs = _make_divs()
        ladder = dividend_scenario_ladder(100, 100, 0.20, 0.05,
                                           REF + relativedelta(years=1), divs, curve,
                                           div_shifts=[-0.5, 0, 0.5])
        assert len(ladder) == 3
