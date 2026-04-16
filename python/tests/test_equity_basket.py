"""Tests for equity basket and correlation trading."""

import math

import numpy as np
import pytest

from pricebook.equity_basket import (
    CorrelationSwapResult,
    DispersionTradeResult,
    EquityBasketResult,
    MargrabeEquityResult,
    MaxMinResult,
    correlation_swap_price,
    dispersion_trade_value,
    equity_basket_mc,
    implied_correlation_from_dispersion,
    johnson_max_call,
    johnson_min_call,
    margrabe_equity,
)


# ---- Margrabe ----

class TestMargrabeEquity:
    def test_basic(self):
        result = margrabe_equity(100, 100, 0.03, 0.02, 0.02, 0.20, 0.25, 0.3, 1.0)
        assert isinstance(result, MargrabeEquityResult)
        assert result.price > 0

    def test_zero_correlation_higher_price(self):
        neg = margrabe_equity(100, 100, 0.03, 0.02, 0.02, 0.20, 0.25, -0.5, 1.0)
        pos = margrabe_equity(100, 100, 0.03, 0.02, 0.02, 0.20, 0.25, 0.5, 1.0)
        # Negative correlation → higher combined vol → higher price
        assert neg.price > pos.price

    def test_vol_combined(self):
        result = margrabe_equity(100, 100, 0.03, 0.02, 0.02, 0.10, 0.10, 0.0, 1.0)
        # σ² = 0.01 + 0.01 - 0 = 0.02; σ = 0.1414
        assert result.vol_combined == pytest.approx(math.sqrt(0.02))

    def test_identical_perfect_corr_zero(self):
        result = margrabe_equity(100, 100, 0.03, 0.02, 0.02, 0.20, 0.20, 1.0, 1.0)
        # Perfect corr + same vol + same spot + same div → zero (up to floor)
        assert result.price < 0.01


# ---- Johnson max/min ----

class TestJohnsonMaxMin:
    def test_max_call(self):
        result = johnson_max_call(100, 100, 100, 0.03, 0.02, 0.02, 0.20, 0.20, 0.3, 1.0)
        assert isinstance(result, MaxMinResult)
        assert result.option_style == "max_call"
        assert result.price > 0

    def test_min_call(self):
        result = johnson_min_call(100, 100, 100, 0.03, 0.02, 0.02, 0.20, 0.20, 0.3, 1.0)
        assert result.option_style == "min_call"
        assert result.price > 0

    def test_max_greater_than_min(self):
        """Max call ≥ min call."""
        max_c = johnson_max_call(100, 100, 100, 0.03, 0.02, 0.02, 0.20, 0.20, 0.3, 1.0)
        min_c = johnson_min_call(100, 100, 100, 0.03, 0.02, 0.02, 0.20, 0.20, 0.3, 1.0)
        assert max_c.price >= min_c.price

    def test_max_greater_than_single(self):
        """Max call ≥ single-asset call on either leg."""
        from pricebook.black76 import black76_price, OptionType
        F = 100 * math.exp((0.03 - 0.02) * 1.0)
        single = black76_price(F, 100, 0.20, 1.0, math.exp(-0.03), OptionType.CALL)
        max_c = johnson_max_call(100, 100, 100, 0.03, 0.02, 0.02, 0.20, 0.20, 0.0, 1.0)
        assert max_c.price >= single * 0.95


# ---- Basket MC ----

class TestEquityBasketMC:
    def _corr(self, rho=0.3):
        return np.array([[1.0, rho], [rho, 1.0]])

    def test_basic_average(self):
        corr = self._corr()
        result = equity_basket_mc(
            [100, 100], [0.5, 0.5], 100,
            rate=0.03, dividend_yields=[0.02, 0.02],
            vols=[0.20, 0.25], correlations=corr, T=1.0,
            basket_type="average", n_paths=5000, seed=42,
        )
        assert isinstance(result, EquityBasketResult)
        assert result.price > 0

    def test_min_cheaper_than_average_call(self):
        corr = self._corr()
        avg = equity_basket_mc([100, 100], [0.5, 0.5], 100,
                                0.03, [0.02, 0.02], [0.20, 0.25], corr, 1.0,
                                basket_type="average", n_paths=5000, seed=42)
        mn = equity_basket_mc([100, 100], [0.5, 0.5], 100,
                               0.03, [0.02, 0.02], [0.20, 0.25], corr, 1.0,
                               basket_type="min", n_paths=5000, seed=42)
        assert mn.price <= avg.price + 1.0

    def test_n_assets(self):
        corr = np.eye(3)
        result = equity_basket_mc([100, 110, 90], [1/3]*3, 100,
                                   0.03, [0.02]*3, [0.20]*3, corr, 1.0,
                                   n_paths=2000, seed=42)
        assert result.n_assets == 3


# ---- Correlation swap ----

class TestCorrelationSwap:
    def test_basic(self):
        result = correlation_swap_price(0.6, 0.5)
        assert isinstance(result, CorrelationSwapResult)
        assert result.price == pytest.approx(0.1)

    def test_zero_spread(self):
        result = correlation_swap_price(0.5, 0.5)
        assert result.price == 0.0

    def test_negative_realised(self):
        result = correlation_swap_price(0.3, 0.6)
        assert result.price < 0


class TestImpliedCorrelationFromDispersion:
    def test_basic(self):
        """Compute implied correlation from index and constituent variances."""
        variances = [0.04, 0.04]    # σ = 0.20 each
        weights = [0.5, 0.5]
        # σ²_index = 0.25×0.04 + 0.25×0.04 + ρ×0.5×0.5×0.2×0.2×2 = 0.02 + 0.02ρ
        # For ρ=0.5: σ²_index = 0.03
        rho = implied_correlation_from_dispersion(0.03, variances, weights)
        assert rho == pytest.approx(0.5, abs=0.05)

    def test_diagonal_correlation(self):
        """σ²_index = weighted constituent variance → ρ=1."""
        variances = [0.04, 0.04]
        weights = [0.5, 0.5]
        # σ_index = weighted average → σ²_index = 0.04
        rho = implied_correlation_from_dispersion(0.04, variances, weights)
        assert rho == pytest.approx(1.0, abs=0.1)


# ---- Dispersion ----

class TestDispersionTrade:
    def test_basic(self):
        result = dispersion_trade_value(0.03, [0.04, 0.04], [0.5, 0.5])
        assert isinstance(result, DispersionTradeResult)

    def test_dispersion_gap(self):
        """Dispersion gap = weighted components - index."""
        result = dispersion_trade_value(0.03, [0.04, 0.04], [0.5, 0.5])
        # weighted_var = 0.25×0.04 + 0.25×0.04 = 0.02
        # gap = 0.02 - 0.03 = -0.01
        assert result.weighted_component_variance == pytest.approx(0.02)
        assert result.dispersion_gap == pytest.approx(-0.01)

    def test_low_correlation_positive_pnl(self):
        """Low realised correlation → dispersion pays off."""
        # Index var low because realised corr low
        low_corr_index = 0.022  # near diagonal term
        result = dispersion_trade_value(low_corr_index, [0.04, 0.04], [0.5, 0.5])
        assert result.implied_correlation < 0.5
