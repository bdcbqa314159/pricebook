"""Tests for commodity instruments."""

import pytest
import math
from datetime import date

from pricebook.commodity import (
    CommodityForwardCurve,
    CommoditySwap,
    commodity_option_price,
)
from pricebook.schedule import Frequency
from pricebook.black76 import OptionType, black76_price
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


def _fwd_curve(spot=80.0):
    """Simple commodity forward curve (contango)."""
    dates = [
        date(2024, 2, 15), date(2024, 4, 15), date(2024, 7, 15),
        date(2024, 10, 15), date(2025, 1, 15), date(2025, 7, 15),
    ]
    # Slight contango: forward > spot
    forwards = [spot, spot * 1.01, spot * 1.02, spot * 1.025, spot * 1.03, spot * 1.04]
    return CommodityForwardCurve(REF, dates, forwards)


class TestForwardCurve:
    def test_at_pillar(self):
        c = _fwd_curve()
        assert c.forward(date(2024, 7, 15)) == pytest.approx(80 * 1.02)

    def test_interpolates(self):
        c = _fwd_curve()
        f = c.forward(date(2024, 5, 15))
        # Between Apr (81.6) and Jul (81.6)
        assert 80 < f < 83

    def test_contango(self):
        """Forward curve in contango: later dates have higher prices."""
        c = _fwd_curve()
        f_near = c.forward(date(2024, 4, 15))
        f_far = c.forward(date(2025, 1, 15))
        assert f_far > f_near

    def test_single_point(self):
        c = CommodityForwardCurve(REF, [date(2025, 1, 15)], [85.0])
        assert c.forward(date(2025, 1, 15)) == 85.0
        assert c.forward(date(2025, 7, 15)) == 85.0  # flat extrap

    def test_mismatched_raises(self):
        with pytest.raises(ValueError, match="same length"):
            CommodityForwardCurve(REF, [date(2025, 1, 15)], [80, 85])

    def test_convenience_yield(self):
        c = _fwd_curve()
        disc = make_flat_curve(REF, 0.05)
        cy = c.convenience_yield(date(2025, 1, 15), disc)
        # With contango and positive rates, convenience yield should be < rate
        assert cy < 0.05


class TestCommoditySwap:
    def test_pv_zero_at_par(self):
        c = _fwd_curve()
        disc = make_flat_curve(REF, 0.05)
        swap = CommoditySwap(REF, date(2025, 1, 15), fixed_price=80.0)
        par = swap.par_price(c, disc)

        swap_at_par = CommoditySwap(REF, date(2025, 1, 15), fixed_price=par)
        assert swap_at_par.pv(c, disc) == pytest.approx(0.0, abs=0.01)

    def test_pv_positive_below_par(self):
        """Fixed price below market → receiver floating gains."""
        c = _fwd_curve()
        disc = make_flat_curve(REF, 0.05)
        swap = CommoditySwap(REF, date(2025, 1, 15), fixed_price=70.0)
        assert swap.pv(c, disc) > 0

    def test_pv_negative_above_par(self):
        c = _fwd_curve()
        disc = make_flat_curve(REF, 0.05)
        swap = CommoditySwap(REF, date(2025, 1, 15), fixed_price=90.0)
        assert swap.pv(c, disc) < 0

    def test_par_price_reasonable(self):
        c = _fwd_curve()
        disc = make_flat_curve(REF, 0.05)
        swap = CommoditySwap(REF, date(2025, 1, 15), fixed_price=80.0)
        par = swap.par_price(c, disc)
        # Par should be near the average forward
        assert 79 < par < 84

    def test_quantity_scales_pv(self):
        c = _fwd_curve()
        disc = make_flat_curve(REF, 0.05)
        swap1 = CommoditySwap(REF, date(2025, 1, 15), fixed_price=80.0, quantity=1.0)
        swap10 = CommoditySwap(REF, date(2025, 1, 15), fixed_price=80.0, quantity=10.0)
        assert swap10.pv(c, disc) == pytest.approx(10 * swap1.pv(c, disc))


class TestCommodityOption:
    def test_call_positive(self):
        p = commodity_option_price(80.0, 80.0, 0.30, 1.0, 0.95)
        assert p > 0

    def test_put_call_parity(self):
        F, K, vol, T, df = 80.0, 85.0, 0.30, 1.0, math.exp(-0.05)
        c = commodity_option_price(F, K, vol, T, df, OptionType.CALL)
        p = commodity_option_price(F, K, vol, T, df, OptionType.PUT)
        assert c - p == pytest.approx(df * (F - K), abs=1e-10)

    def test_matches_black76(self):
        """commodity_option_price is just Black-76."""
        F, K, vol, T, df = 80.0, 85.0, 0.30, 1.0, 0.95
        assert commodity_option_price(F, K, vol, T, df) == \
            pytest.approx(black76_price(F, K, vol, T, df))

    def test_higher_vol_higher_price(self):
        p1 = commodity_option_price(80.0, 80.0, 0.20, 1.0, 0.95)
        p2 = commodity_option_price(80.0, 80.0, 0.40, 1.0, 0.95)
        assert p2 > p1
