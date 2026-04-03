"""Tests for IR futures: SOFR futures and convexity adjustment."""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.ir_futures import (
    IRFuture,
    FuturesType,
    hw_convexity_adjustment,
    futures_strip_rates,
)
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


class TestIRFuture:
    def test_price_from_flat_curve(self):
        """On a flat 5% curve, futures price ≈ 95."""
        curve = make_flat_curve(REF, 0.05)
        fut = IRFuture(date(2024, 4, 15), date(2024, 7, 15))
        price = fut.price(curve)
        assert price == pytest.approx(95.0, abs=0.5)

    def test_implied_forward(self):
        curve = make_flat_curve(REF, 0.05)
        fut = IRFuture(date(2024, 4, 15), date(2024, 7, 15))
        fwd = fut.implied_forward(curve)
        assert fwd == pytest.approx(0.05, rel=0.02)

    def test_futures_rate_no_convexity(self):
        curve = make_flat_curve(REF, 0.05)
        fut = IRFuture(date(2024, 4, 15), date(2024, 7, 15))
        rate = fut.futures_rate(curve, convexity=0.0)
        fwd = fut.implied_forward(curve)
        assert rate == pytest.approx(fwd)

    def test_futures_rate_with_convexity(self):
        curve = make_flat_curve(REF, 0.05)
        fut = IRFuture(date(2024, 4, 15), date(2024, 7, 15))
        ca = 0.0005  # 5bp convexity
        rate = fut.futures_rate(curve, convexity=ca)
        fwd = fut.implied_forward(curve)
        assert rate == pytest.approx(fwd + ca)

    def test_price_inverse_of_rate(self):
        curve = make_flat_curve(REF, 0.04)
        fut = IRFuture(date(2024, 4, 15), date(2024, 7, 15))
        rate = fut.futures_rate(curve)
        price = fut.price(curve)
        assert price == pytest.approx(100.0 - rate * 100.0)

    def test_pv_profit(self):
        """Bought at 95, now worth 95.5 → positive PV."""
        curve = make_flat_curve(REF, 0.045)  # rates dropped
        fut = IRFuture(date(2024, 4, 15), date(2024, 7, 15))
        price_now = fut.price(curve)
        # Bought when rates were higher
        pv = fut.pv(curve, trade_price=price_now - 0.5)
        assert pv > 0

    def test_pv_zero_at_trade(self):
        curve = make_flat_curve(REF, 0.05)
        fut = IRFuture(date(2024, 4, 15), date(2024, 7, 15))
        trade_price = fut.price(curve)
        assert fut.pv(curve, trade_price) == pytest.approx(0.0)

    def test_dv01(self):
        curve = make_flat_curve(REF, 0.05)
        fut = IRFuture(date(2024, 4, 15), date(2024, 7, 15))
        dv01 = fut.dv01(curve)
        assert dv01 > 0  # tick value is positive

    def test_1m_vs_3m(self):
        curve = make_flat_curve(REF, 0.05)
        fut_1m = IRFuture(date(2024, 4, 15), date(2024, 5, 15), FuturesType.SOFR_1M)
        fut_3m = IRFuture(date(2024, 4, 15), date(2024, 7, 15), FuturesType.SOFR_3M)
        # Both should give similar rates on flat curve
        assert fut_1m.implied_forward(curve) == pytest.approx(
            fut_3m.implied_forward(curve), rel=0.05
        )

    def test_accrual_fraction(self):
        fut = IRFuture(date(2024, 4, 15), date(2024, 7, 15))
        tau = fut.accrual_fraction
        assert 0.2 < tau < 0.3  # roughly 3 months


class TestConvexityAdjustment:
    def test_zero_vol_zero_convexity(self):
        ca = hw_convexity_adjustment(a=0.05, sigma=0.0, t=0.0, T1=1.0, T2=1.25)
        assert ca == pytest.approx(0.0)

    def test_positive_convexity(self):
        """Futures rate > forward rate."""
        ca = hw_convexity_adjustment(a=0.05, sigma=0.01, t=0.0, T1=1.0, T2=1.25)
        assert ca > 0

    def test_convexity_increases_with_vol(self):
        ca_low = hw_convexity_adjustment(a=0.05, sigma=0.005, t=0.0, T1=2.0, T2=2.25)
        ca_high = hw_convexity_adjustment(a=0.05, sigma=0.015, t=0.0, T1=2.0, T2=2.25)
        assert ca_high > ca_low

    def test_convexity_increases_with_maturity(self):
        """Longer-dated futures have larger convexity bias."""
        ca_near = hw_convexity_adjustment(a=0.05, sigma=0.01, t=0.0, T1=0.5, T2=0.75)
        ca_far = hw_convexity_adjustment(a=0.05, sigma=0.01, t=0.0, T1=5.0, T2=5.25)
        assert ca_far > ca_near

    def test_no_mean_reversion(self):
        """With a=0, use simplified formula."""
        ca = hw_convexity_adjustment(a=0.0, sigma=0.01, t=0.0, T1=1.0, T2=1.25)
        expected = 0.5 * 0.01**2 * 0.25 * 1.0
        assert ca == pytest.approx(expected)

    def test_small_a_converges(self):
        """As a→0, HW CA converges smoothly."""
        ca_a1 = hw_convexity_adjustment(a=0.001, sigma=0.01, t=0.0, T1=1.0, T2=1.25)
        ca_a2 = hw_convexity_adjustment(a=0.0001, sigma=0.01, t=0.0, T1=1.0, T2=1.25)
        # Should converge as a shrinks
        assert abs(ca_a1 - ca_a2) < abs(ca_a1) * 0.1

    def test_convexity_order_of_magnitude(self):
        """Typical convexity: a few bp for near-term, 10s of bp for far-term."""
        ca_2y = hw_convexity_adjustment(a=0.03, sigma=0.01, t=0.0, T1=2.0, T2=2.25)
        assert 0.0 < ca_2y < 0.01  # < 100bp

    def test_convexity_quadratic_in_sigma(self):
        """CA ∝ σ², so doubling vol quadruples convexity."""
        ca1 = hw_convexity_adjustment(a=0.05, sigma=0.01, t=0.0, T1=1.0, T2=1.25)
        ca2 = hw_convexity_adjustment(a=0.05, sigma=0.02, t=0.0, T1=1.0, T2=1.25)
        assert ca2 == pytest.approx(4.0 * ca1, rel=0.01)


class TestFuturesStrip:
    def test_strip_length(self):
        curve = make_flat_curve(REF, 0.05)
        futures = []
        start = date(2024, 3, 15)
        for i in range(4):
            s = start + relativedelta(months=3 * i)
            e = s + relativedelta(months=3)
            futures.append(IRFuture(s, e))

        results = futures_strip_rates(futures, curve)
        assert len(results) == 4

    def test_strip_forward_rates(self):
        curve = make_flat_curve(REF, 0.05)
        futures = []
        start = date(2024, 3, 15)
        for i in range(4):
            s = start + relativedelta(months=3 * i)
            e = s + relativedelta(months=3)
            futures.append(IRFuture(s, e))

        results = futures_strip_rates(futures, curve)
        for r in results:
            assert r["forward"] == pytest.approx(0.05, rel=0.03)

    def test_strip_with_convexity(self):
        curve = make_flat_curve(REF, 0.05)
        futures = []
        start = date(2024, 3, 15)
        for i in range(8):
            s = start + relativedelta(months=3 * i)
            e = s + relativedelta(months=3)
            futures.append(IRFuture(s, e))

        results = futures_strip_rates(futures, curve, a=0.05, sigma=0.01)
        # All convexity adjustments should be positive and increasing
        convexities = [r["convexity"] for r in results]
        assert all(c > 0 for c in convexities)
        assert convexities == sorted(convexities)  # increasing with maturity

    def test_strip_no_vol_no_convexity(self):
        curve = make_flat_curve(REF, 0.05)
        futures = [IRFuture(date(2024, 6, 15), date(2024, 9, 15))]
        results = futures_strip_rates(futures, curve, a=0.05, sigma=0.0)
        assert results[0]["convexity"] == pytest.approx(0.0)
        assert results[0]["futures_rate"] == pytest.approx(results[0]["forward"])

    def test_strip_prices(self):
        curve = make_flat_curve(REF, 0.04)
        futures = [IRFuture(date(2024, 6, 15), date(2024, 9, 15))]
        results = futures_strip_rates(futures, curve)
        assert results[0]["price"] == pytest.approx(100.0 - results[0]["futures_rate"] * 100.0)
