"""Tests for advanced inflation models."""

import math

import numpy as np
import pytest

from pricebook.inflation_advanced import (
    InflationSwaptionResult,
    LPIResult,
    RealRateSwaptionResult,
    YoYConvexityResult,
    inflation_swaption_price,
    lpi_swap_price,
    real_rate_swaption_price,
    yoy_convexity_adjustment,
)


# ---- YoY convexity ----

class TestYoYConvexity:
    def test_positive_adjustment_positive_corr(self):
        result = yoy_convexity_adjustment(0.025, 0.01, 0.005, 0.008, 0.3, 1.0, 5)
        assert isinstance(result, YoYConvexityResult)
        assert result.convexity_adjustment > 0
        assert result.yoy_rate > result.zc_rate

    def test_negative_corr_negative_adjustment(self):
        result = yoy_convexity_adjustment(0.025, 0.01, 0.005, 0.008, -0.3, 1.0, 5)
        assert result.convexity_adjustment < 0
        assert result.yoy_rate < result.zc_rate

    def test_zero_corr_no_adjustment(self):
        result = yoy_convexity_adjustment(0.025, 0.01, 0.005, 0.008, 0.0, 1.0, 5)
        assert result.convexity_adjustment == 0.0

    def test_adjustment_grows_with_year(self):
        early = yoy_convexity_adjustment(0.025, 0.01, 0.005, 0.008, 0.3, 1.0, 2)
        late = yoy_convexity_adjustment(0.025, 0.01, 0.005, 0.008, 0.3, 1.0, 10)
        assert abs(late.convexity_adjustment) > abs(early.convexity_adjustment)

    def test_higher_vol_larger_adjustment(self):
        low = yoy_convexity_adjustment(0.025, 0.005, 0.003, 0.004, 0.3, 1.0, 5)
        high = yoy_convexity_adjustment(0.025, 0.015, 0.010, 0.012, 0.3, 1.0, 5)
        assert abs(high.convexity_adjustment) > abs(low.convexity_adjustment)


# ---- LPI ----

class TestLPISwap:
    def test_basic_pricing(self):
        result = lpi_swap_price(1e6, 0.03, cap=0.05, floor=0.00, maturity=10, inflation_vol=0.02)
        assert isinstance(result, LPIResult)
        assert result.n_periods == 10
        assert result.method == "capfloor_decomposition"

    def test_wide_collar_near_plain_yoy(self):
        """Very wide cap/floor → LPI ≈ plain YoY swap."""
        plain = lpi_swap_price(1e6, 0.03, cap=1.0, floor=-1.0, maturity=5, inflation_vol=0.02)
        lpi = lpi_swap_price(1e6, 0.03, cap=0.05, floor=0.00, maturity=5, inflation_vol=0.02)
        # LPI with cap/floor should be cheaper than uncapped
        assert lpi.price <= plain.price + 1e-6

    def test_tighter_collar_lower_price(self):
        """Tighter collar → lower price (less upside)."""
        wide = lpi_swap_price(1e6, 0.03, cap=0.10, floor=-0.05, maturity=10, inflation_vol=0.02)
        tight = lpi_swap_price(1e6, 0.03, cap=0.04, floor=0.01, maturity=10, inflation_vol=0.02)
        assert tight.price < wide.price

    def test_zero_vol(self):
        """Zero vol → deterministic: LPI = min(cap, max(floor, expected))."""
        result = lpi_swap_price(1e6, 0.03, cap=0.05, floor=0.00, maturity=5, inflation_vol=0.0)
        assert result.price > 0

    def test_higher_notional_proportional(self):
        small = lpi_swap_price(1e6, 0.03, cap=0.05, floor=0.0, maturity=5, inflation_vol=0.02)
        big = lpi_swap_price(2e6, 0.03, cap=0.05, floor=0.0, maturity=5, inflation_vol=0.02)
        assert big.price == pytest.approx(2 * small.price, rel=0.01)


# ---- Inflation swaption ----

class TestInflationSwaption:
    def test_payer_positive(self):
        result = inflation_swaption_price(1e6, 0.025, 0.025, 1.0, 5.0, 0.005)
        assert isinstance(result, InflationSwaptionResult)
        assert result.price > 0
        assert result.is_payer is True

    def test_receiver_positive(self):
        result = inflation_swaption_price(1e6, 0.025, 0.025, 1.0, 5.0, 0.005, is_payer=False)
        assert result.price > 0

    def test_deep_otm_small(self):
        """Deep OTM payer (strike >> forward): near zero."""
        result = inflation_swaption_price(1e6, 0.02, 0.10, 1.0, 5.0, 0.005)
        assert result.price < 100  # very small for 1M notional

    def test_higher_vol_higher_price(self):
        low = inflation_swaption_price(1e6, 0.025, 0.025, 1.0, 5.0, 0.003)
        high = inflation_swaption_price(1e6, 0.025, 0.025, 1.0, 5.0, 0.010)
        assert high.price > low.price

    def test_longer_expiry_higher_price(self):
        short = inflation_swaption_price(1e6, 0.025, 0.025, 0.5, 5.0, 0.005)
        long = inflation_swaption_price(1e6, 0.025, 0.025, 3.0, 5.0, 0.005)
        assert long.price > short.price

    def test_put_call_parity_approx(self):
        """Payer - Receiver ≈ forward value (approximately)."""
        fwd = 0.03
        K = 0.025
        payer = inflation_swaption_price(1e6, fwd, K, 1.0, 5.0, 0.005)
        receiver = inflation_swaption_price(1e6, fwd, K, 1.0, 5.0, 0.005, is_payer=False)
        # payer - receiver = annuity × (F - K)
        diff = payer.price - receiver.price
        assert diff > 0  # F > K → payer > receiver


# ---- Real rate swaption ----

class TestRealRateSwaption:
    def test_positive_price(self):
        result = real_rate_swaption_price(1e6, 0.01, 0.01, 1.0, 5.0, 0.005)
        assert isinstance(result, RealRateSwaptionResult)
        assert result.price > 0

    def test_negative_real_rate(self):
        """Should handle negative real rates via shift."""
        result = real_rate_swaption_price(1e6, -0.005, -0.005, 1.0, 5.0, 0.005)
        assert result.price > 0

    def test_higher_vol_higher_price(self):
        low = real_rate_swaption_price(1e6, 0.01, 0.01, 1.0, 5.0, 0.003)
        high = real_rate_swaption_price(1e6, 0.01, 0.01, 1.0, 5.0, 0.010)
        assert high.price > low.price

    def test_payer_vs_receiver(self):
        payer = real_rate_swaption_price(1e6, 0.015, 0.01, 1.0, 5.0, 0.005, is_payer=True)
        receiver = real_rate_swaption_price(1e6, 0.015, 0.01, 1.0, 5.0, 0.005, is_payer=False)
        # Forward > strike → payer worth more
        assert payer.price > receiver.price
