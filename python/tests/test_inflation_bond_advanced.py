"""Tests for inflation bonds advanced."""
import math
import numpy as np
import pytest
from pricebook.inflation_bond_advanced import (
    real_yield_curve_bootstrap, breakeven_trade, seasonality_adjusted_breakeven,
    linker_asw, deflation_floor_value,
)

class TestRealYieldCurveBootstrap:
    def test_basic(self):
        # 2Y and 10Y TIPS
        r = real_yield_curve_bootstrap(
            linker_prices=[102, 98],
            notionals=[100, 100],
            coupon_rates=[0.01, 0.02],
            maturities=[2, 10],
            cpi_base=250, cpi_current=260,
        )
        assert len(r.real_yields) == 2
        assert all(y > -0.05 and y < 0.20 for y in r.real_yields)

    def test_higher_price_lower_yield(self):
        r = real_yield_curve_bootstrap(
            [110, 90], [100, 100], [0.02, 0.02], [5, 5],
            250, 260,
        )
        assert r.real_yields[0] < r.real_yields[1]

    def test_at_par(self):
        """Price near par × index_ratio → yield near coupon."""
        index_ratio = 260 / 250
        price = 100 * index_ratio  # at par (indexed)
        r = real_yield_curve_bootstrap([price], [100], [0.02], [5], 250, 260)
        assert r.real_yields[0] == pytest.approx(0.02, abs=0.01)

class TestBreakevenTrade:
    def test_basic(self):
        r = breakeven_trade(4.50, 2.00)
        assert r.breakeven_pct == pytest.approx(2.50)

    def test_risk_premium(self):
        r = breakeven_trade(4.50, 2.00, expected_inflation_pct=2.30)
        assert r.risk_premium_est == pytest.approx(0.20)

    def test_no_expected_zero_premium(self):
        r = breakeven_trade(4.50, 2.00)
        assert r.risk_premium_est == 0.0

class TestSeasonalityAdjusted:
    def test_positive_seasonal(self):
        r = seasonality_adjusted_breakeven(2.50, seasonal_factor=0.30)
        assert r.adjusted_breakeven_pct == pytest.approx(2.20)

    def test_negative_seasonal(self):
        r = seasonality_adjusted_breakeven(2.50, -0.20)
        assert r.adjusted_breakeven_pct == pytest.approx(2.70)

    def test_zero_seasonal(self):
        r = seasonality_adjusted_breakeven(2.50, 0.0)
        assert r.adjusted_breakeven_pct == 2.50

class TestLinkerASW:
    def test_positive_spread(self):
        r = linker_asw(1.50, 1.20)
        assert r.real_asw_bps == pytest.approx(30)

    def test_negative_spread(self):
        r = linker_asw(1.50, 1.80)
        assert r.real_asw_bps < 0

    def test_zero(self):
        r = linker_asw(1.50, 1.50)
        assert r.real_asw_bps == 0.0

class TestDeflationFloor:
    def test_low_breakeven_high_floor(self):
        """Low breakeven → deflation more likely → higher floor."""
        low = deflation_floor_value(0.005, 0.015, 10.0)   # 0.5%, 1.5% vol
        high = deflation_floor_value(0.03, 0.015, 10.0)    # 3.0%, 1.5% vol
        assert low.floor_value > high.floor_value

    def test_higher_vol_higher_floor(self):
        low_vol = deflation_floor_value(0.02, 0.005, 10.0)  # 2%, 0.5% vol
        high_vol = deflation_floor_value(0.02, 0.03, 10.0)  # 2%, 3.0% vol
        assert high_vol.floor_value >= low_vol.floor_value

    def test_zero_vol_deterministic(self):
        r = deflation_floor_value(0.02, 0.0, 10.0)  # 2% breakeven, 0 vol
        # Positive breakeven → no deflation
        assert r.floor_value == 0.0
        assert r.probability_deflation == 0.0

    def test_negative_breakeven(self):
        """Negative breakeven → deflation expected → floor very valuable."""
        r = deflation_floor_value(-0.01, 0.01, 5.0)  # -1%, 1% vol
        assert r.floor_value > 0
        assert r.probability_deflation > 0.5

    def test_probability_bounded(self):
        r = deflation_floor_value(0.02, 0.015, 10.0)  # 2%, 1.5% vol
        assert 0 <= r.probability_deflation <= 1
