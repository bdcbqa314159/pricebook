"""Tests for FX swap."""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.fx_swap import FXSwap
from pricebook.fx_forward import FXForward
from pricebook.currency import Currency, CurrencyPair
from pricebook.discount_curve import DiscountCurve


REF = date(2024, 1, 15)
EURUSD = CurrencyPair(Currency.EUR, Currency.USD)
SPOT = 1.10


def _flat_curve(ref: date, rate: float) -> DiscountCurve:
    tenors = [0.25, 0.5, 1.0, 2.0, 5.0]
    dates = [date.fromordinal(ref.toordinal() + int(t * 365)) for t in tenors]
    dfs = [math.exp(-rate * t) for t in tenors]
    return DiscountCurve(ref, dates, dfs)


class TestPV:

    def test_pv_zero_at_fair_rates(self):
        """FX swap at fair forward rates has PV = 0."""
        eur = _flat_curve(REF, rate=0.04)
        usd = _flat_curve(REF, rate=0.05)
        near = REF + relativedelta(months=1)
        far = REF + relativedelta(months=6)
        near_rate = FXForward.forward_rate(SPOT, near, eur, usd)
        far_rate = FXForward.forward_rate(SPOT, far, eur, usd)
        swap = FXSwap(EURUSD, near, far, near_rate, far_rate)
        assert swap.pv(SPOT, eur, usd) == pytest.approx(0.0, abs=1.0)

    def test_pv_positive_when_far_rate_above_fair(self):
        """Selling base at a rate above fair -> positive PV."""
        eur = _flat_curve(REF, rate=0.04)
        usd = _flat_curve(REF, rate=0.05)
        near = REF + relativedelta(months=1)
        far = REF + relativedelta(months=6)
        near_rate = FXForward.forward_rate(SPOT, near, eur, usd)
        far_rate = FXForward.forward_rate(SPOT, far, eur, usd) + 0.01
        swap = FXSwap(EURUSD, near, far, near_rate, far_rate)
        assert swap.pv(SPOT, eur, usd) > 0

    def test_pv_scales_with_notional(self):
        eur = _flat_curve(REF, rate=0.04)
        usd = _flat_curve(REF, rate=0.05)
        near = REF + relativedelta(months=1)
        far = REF + relativedelta(months=6)
        swap1 = FXSwap(EURUSD, near, far, 1.10, 1.11, notional=1_000_000.0)
        swap2 = FXSwap(EURUSD, near, far, 1.10, 1.11, notional=2_000_000.0)
        assert swap2.pv(SPOT, eur, usd) == pytest.approx(
            2 * swap1.pv(SPOT, eur, usd), rel=1e-10)


class TestSwapPoints:

    def test_swap_points(self):
        swap = FXSwap(EURUSD, REF + relativedelta(months=1),
                      REF + relativedelta(months=6), 1.1005, 1.1030)
        assert swap.swap_points == pytest.approx(0.0025, rel=1e-10)

    def test_fair_swap_points_consistent(self):
        """Fair swap points = F_far - F_near."""
        eur = _flat_curve(REF, rate=0.04)
        usd = _flat_curve(REF, rate=0.05)
        near = REF + relativedelta(months=1)
        far = REF + relativedelta(months=6)
        pts = FXSwap.fair_swap_points(SPOT, near, far, eur, usd)
        fwd_near = FXForward.forward_rate(SPOT, near, eur, usd)
        fwd_far = FXForward.forward_rate(SPOT, far, eur, usd)
        assert pts == pytest.approx(fwd_far - fwd_near, rel=1e-10)

    def test_fair_swap_points_zero_equal_rates(self):
        curve = _flat_curve(REF, rate=0.04)
        near = REF + relativedelta(months=1)
        far = REF + relativedelta(months=6)
        pts = FXSwap.fair_swap_points(SPOT, near, far, curve, curve)
        assert pts == pytest.approx(0.0, abs=1e-6)


class TestValidation:

    def test_near_after_far_raises(self):
        with pytest.raises(ValueError):
            FXSwap(EURUSD, REF + relativedelta(months=6),
                   REF + relativedelta(months=1), 1.10, 1.11)

    def test_negative_notional_raises(self):
        with pytest.raises(ValueError):
            FXSwap(EURUSD, REF + relativedelta(months=1),
                   REF + relativedelta(months=6), 1.10, 1.11, notional=-1.0)

    def test_negative_rate_raises(self):
        with pytest.raises(ValueError):
            FXSwap(EURUSD, REF + relativedelta(months=1),
                   REF + relativedelta(months=6), -1.10, 1.11)
