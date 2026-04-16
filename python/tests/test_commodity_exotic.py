"""Tests for commodity exotic options."""

import math

import numpy as np
import pytest

from pricebook.commodity_exotic import (
    CommodityAsianResult,
    CommodityBarrierResult,
    CommodityLookbackResult,
    QuantoCommodityResult,
    commodity_asian_monthly,
    commodity_barrier_smile,
    commodity_lookback,
    quanto_commodity_option,
)


# ---- Barrier ----

class TestCommodityBarrier:
    def test_basic(self):
        result = commodity_barrier_smile(
            spot=80, strike=80, barrier=90,
            rate=0.03, convenience_yield=0.05,
            vol_atm=0.30, vol_25d_call=0.31, vol_25d_put=0.32, T=1.0,
            is_up=True, is_knock_in=False, is_call=True,
        )
        assert isinstance(result, CommodityBarrierResult)
        assert result.price >= 0

    def test_ko_less_than_vanilla(self):
        from pricebook.black76 import black76_price, OptionType
        F = 80 * math.exp((0.03 - 0.05) * 1.0)  # backwardation
        vanilla = black76_price(F, 80, 0.30, 1.0, math.exp(-0.03), OptionType.CALL)
        ko = commodity_barrier_smile(
            80, 80, 100, 0.03, 0.05, 0.30, 0.31, 0.32, 1.0,
            is_up=True, is_knock_in=False,
        )
        assert ko.price <= vanilla + 1e-2

    def test_knock_in_knock_out_parity(self):
        """KI + KO = vanilla (approximate)."""
        ko = commodity_barrier_smile(80, 80, 95, 0.03, 0.05, 0.30, 0.30, 0.30, 1.0,
                                       is_up=True, is_knock_in=False)
        ki = commodity_barrier_smile(80, 80, 95, 0.03, 0.05, 0.30, 0.30, 0.30, 1.0,
                                       is_up=True, is_knock_in=True)
        from pricebook.black76 import black76_price, OptionType
        F = 80 * math.exp((0.03 - 0.05) * 1.0)
        vanilla = black76_price(F, 80, 0.30, 1.0, math.exp(-0.03), OptionType.CALL)
        # parity approximate due to VV smile adjustments
        assert ko.price + ki.price == pytest.approx(vanilla, rel=0.15)


# ---- Lookback ----

class TestCommodityLookback:
    def test_floating_basic(self):
        result = commodity_lookback(80, 0.03, 0.05, 0.30, 1.0, is_floating=True)
        assert isinstance(result, CommodityLookbackResult)
        assert result.is_floating
        assert result.price > 0

    def test_fixed_basic(self):
        result = commodity_lookback(80, 0.03, 0.05, 0.30, 1.0,
                                     is_floating=False, strike=80,
                                     n_observations=100, n_paths=2000, seed=42)
        assert not result.is_floating
        assert result.price > 0

    def test_fixed_exceeds_vanilla(self):
        """Fixed-strike lookback call ≥ vanilla (max(S) ≥ S_T)."""
        from pricebook.black76 import black76_price, OptionType
        F = 80 * math.exp((0.03 - 0.05) * 1.0)
        vanilla = black76_price(F, 80, 0.30, 1.0, math.exp(-0.03), OptionType.CALL)
        lb = commodity_lookback(80, 0.03, 0.05, 0.30, 1.0,
                                 is_floating=False, strike=80,
                                 n_observations=252, n_paths=20_000, seed=42)
        assert lb.price >= vanilla

    def test_higher_vol_higher_price(self):
        low = commodity_lookback(80, 0.03, 0.05, 0.10, 1.0, is_floating=True)
        high = commodity_lookback(80, 0.03, 0.05, 0.40, 1.0, is_floating=True)
        assert high.price > low.price


# ---- Asian ----

class TestCommodityAsian:
    def test_geometric_closed_form(self):
        result = commodity_asian_monthly(
            80, 80, 0.03, 0.05, 0.30, 1.0, n_fixings=12,
            is_arithmetic=False,
        )
        assert isinstance(result, CommodityAsianResult)
        assert not result.is_arithmetic
        assert result.price > 0

    def test_arithmetic_mc(self):
        result = commodity_asian_monthly(
            80, 80, 0.03, 0.05, 0.30, 1.0,
            n_fixings=12, is_arithmetic=True,
            n_paths=5000, seed=42,
        )
        assert result.is_arithmetic
        assert result.price > 0

    def test_arith_geq_geo(self):
        """Arithmetic ≥ Geometric (AM ≥ GM)."""
        arith = commodity_asian_monthly(80, 80, 0.03, 0.05, 0.30, 1.0,
                                          is_arithmetic=True, n_paths=10_000, seed=42)
        geo = commodity_asian_monthly(80, 80, 0.03, 0.05, 0.30, 1.0,
                                        is_arithmetic=False)
        assert arith.price >= geo.price * 0.95  # allow MC noise

    def test_asian_less_than_vanilla(self):
        """Asian < vanilla (averaging reduces effective vol)."""
        from pricebook.black76 import black76_price, OptionType
        F = 80 * math.exp((0.03 - 0.05) * 1.0)
        vanilla = black76_price(F, 80, 0.30, 1.0, math.exp(-0.03), OptionType.CALL)
        asian = commodity_asian_monthly(80, 80, 0.03, 0.05, 0.30, 1.0, is_arithmetic=False)
        assert asian.price < vanilla

    def test_control_variate_reduces_noise(self):
        """With control variate, adjustment should be small for decent MC size."""
        result = commodity_asian_monthly(
            80, 80, 0.03, 0.05, 0.30, 1.0,
            n_fixings=12, is_arithmetic=True,
            n_paths=10_000, seed=42,
        )
        # CV adjustment should be small relative to price
        assert abs(result.control_variate_adjustment) < result.price


# ---- Quanto commodity ----

class TestQuantoCommodity:
    def test_basic(self):
        result = quanto_commodity_option(
            spot=80, strike=80, fx_spot=0.9,
            rate_quanto=0.02, rate_native=0.03,
            convenience_yield=0.04,
            vol_commodity=0.30, vol_fx=0.10,
            correlation=0.2, T=1.0,
        )
        assert isinstance(result, QuantoCommodityResult)
        assert result.price > 0

    def test_zero_correlation_no_adjustment(self):
        result = quanto_commodity_option(
            80, 80, 0.9, 0.02, 0.03, 0.04, 0.30, 0.10,
            correlation=0.0, T=1.0,
        )
        assert result.quanto_forward == pytest.approx(result.native_forward)

    def test_positive_correlation_lower_quanto_forward(self):
        """ρ > 0 → quanto adjustment reduces forward (call becomes cheaper)."""
        pos = quanto_commodity_option(80, 80, 0.9, 0.02, 0.03, 0.04,
                                        0.30, 0.10, 0.5, 1.0)
        neg = quanto_commodity_option(80, 80, 0.9, 0.02, 0.03, 0.04,
                                        0.30, 0.10, -0.5, 1.0)
        assert pos.quanto_forward < neg.quanto_forward

    def test_higher_correlation_cheaper_call(self):
        pos = quanto_commodity_option(80, 80, 0.9, 0.02, 0.03, 0.04,
                                        0.30, 0.10, 0.5, 1.0)
        zero = quanto_commodity_option(80, 80, 0.9, 0.02, 0.03, 0.04,
                                         0.30, 0.10, 0.0, 1.0)
        assert pos.price < zero.price
