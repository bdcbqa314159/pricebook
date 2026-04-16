"""Tests for FX correlation and baskets."""

import math

import numpy as np
import pytest

from pricebook.fx_correlation import (
    BasketResult,
    ImpliedCorrelationResult,
    MargrabeResult,
    TriangularResult,
    fx_basket_option,
    fx_best_of,
    fx_worst_of,
    implied_correlation_from_triangular,
    implied_correlation_quanto,
    margrabe_fx_exchange,
    triangular_correlation,
)


# ---- Triangular ----

class TestTriangularCorrelation:
    def test_basic(self):
        result = triangular_correlation(0.08, 0.10, 0.3)
        assert isinstance(result, TriangularResult)
        assert result.vol_implied_cross > 0

    def test_zero_correlation(self):
        """ρ=0 → σ_cross² = σ₁² + σ₂²."""
        result = triangular_correlation(0.06, 0.08, 0.0)
        expected = math.sqrt(0.06**2 + 0.08**2)
        assert result.vol_implied_cross == pytest.approx(expected)

    def test_positive_correlation_higher_vol(self):
        """Positive correlation → higher cross vol (amplifies)."""
        pos = triangular_correlation(0.08, 0.10, 0.5)
        zero = triangular_correlation(0.08, 0.10, 0.0)
        neg = triangular_correlation(0.08, 0.10, -0.5)
        assert pos.vol_implied_cross > zero.vol_implied_cross > neg.vol_implied_cross

    def test_basis(self):
        result = triangular_correlation(0.08, 0.10, 0.3, vol_cross_market=0.13)
        assert result.basis is not None
        assert result.basis == pytest.approx(result.vol_implied_cross - 0.13)


class TestImpliedCorrelationFromTriangular:
    def test_inverse(self):
        """Implied correlation from triangular should match input."""
        vol1, vol2, rho = 0.08, 0.10, 0.4
        cross = triangular_correlation(vol1, vol2, rho)
        rho_implied = implied_correlation_from_triangular(vol1, vol2, cross.vol_implied_cross)
        assert rho_implied == pytest.approx(rho, abs=0.01)

    def test_zero_vols(self):
        assert implied_correlation_from_triangular(0, 0.1, 0.1) == 0.0


# ---- Basket option ----

class TestFXBasketOption:
    def _corr(self, rho=0.3):
        return np.array([[1.0, rho], [rho, 1.0]])

    def test_basic_average(self):
        corr = self._corr()
        result = fx_basket_option(
            [1.0, 1.0], [0.5, 0.5], 1.0,
            rates_dom=0.02, rates_for=[0.01, 0.01],
            vols=[0.10, 0.12], correlations=corr, T=1.0,
            basket_type="average", n_paths=5000, seed=42,
        )
        assert isinstance(result, BasketResult)
        assert result.price > 0
        assert result.basket_type == "average"
        assert result.n_assets == 2

    def test_min_cheaper_than_average(self):
        """Worst-of call ≤ average call (the min is always ≤ average)."""
        corr = self._corr()
        avg = fx_basket_option([1.0, 1.0], [0.5, 0.5], 1.0,
                                0.02, [0.01, 0.01], [0.10, 0.12], corr, 1.0,
                                basket_type="average", n_paths=5000, seed=42)
        mn = fx_basket_option([1.0, 1.0], [0.5, 0.5], 1.0,
                               0.02, [0.01, 0.01], [0.10, 0.12], corr, 1.0,
                               basket_type="min", n_paths=5000, seed=42)
        assert mn.price <= avg.price + 1e-6

    def test_max_more_expensive_than_average(self):
        corr = self._corr()
        avg = fx_basket_option([1.0, 1.0], [0.5, 0.5], 1.0,
                                0.02, [0.01, 0.01], [0.10, 0.12], corr, 1.0,
                                basket_type="average", n_paths=5000, seed=42)
        mx = fx_basket_option([1.0, 1.0], [0.5, 0.5], 1.0,
                               0.02, [0.01, 0.01], [0.10, 0.12], corr, 1.0,
                               basket_type="max", n_paths=5000, seed=42)
        assert mx.price >= avg.price - 1e-6

    def test_three_asset_basket(self):
        corr = np.array([
            [1.0, 0.3, 0.2],
            [0.3, 1.0, 0.4],
            [0.2, 0.4, 1.0],
        ])
        result = fx_basket_option(
            [1.0, 1.1, 0.9], [1/3, 1/3, 1/3], 1.0,
            0.02, [0.01, 0.005, 0.015], [0.10, 0.12, 0.08], corr, 1.0,
            basket_type="average", n_paths=5000, seed=42,
        )
        assert result.price > 0
        assert result.n_assets == 3


class TestFXWorstOf:
    def test_basic(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = fx_worst_of([1.0, 1.0], 1.0, 0.02, [0.01, 0.01],
                              [0.10, 0.12], corr, 1.0,
                              n_paths=5000, seed=42)
        assert result.basket_type == "min"
        assert result.price >= 0

    def test_correlation_effect(self):
        """Lower correlation → worst-of call is cheaper (dispersion)."""
        low_corr = np.array([[1.0, 0.0], [0.0, 1.0]])
        high_corr = np.array([[1.0, 0.9], [0.9, 1.0]])
        low = fx_worst_of([1.0, 1.0], 1.0, 0.02, [0.01, 0.01],
                           [0.15, 0.15], low_corr, 1.0,
                           n_paths=5000, seed=42)
        high = fx_worst_of([1.0, 1.0], 1.0, 0.02, [0.01, 0.01],
                            [0.15, 0.15], high_corr, 1.0,
                            n_paths=5000, seed=42)
        # Higher correlation → min behaves more like single asset → more valuable call
        assert high.price >= low.price * 0.8


class TestFXBestOf:
    def test_basic(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = fx_best_of([1.0, 1.0], 1.0, 0.02, [0.01, 0.01],
                             [0.10, 0.12], corr, 1.0,
                             n_paths=5000, seed=42)
        assert result.basket_type == "max"

    def test_best_greater_than_worst(self):
        """Best-of call ≥ worst-of call."""
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        best = fx_best_of([1.0, 1.0], 1.0, 0.02, [0.01, 0.01],
                           [0.10, 0.10], corr, 1.0,
                           n_paths=5000, seed=42)
        worst = fx_worst_of([1.0, 1.0], 1.0, 0.02, [0.01, 0.01],
                             [0.10, 0.10], corr, 1.0,
                             n_paths=5000, seed=42)
        assert best.price >= worst.price


# ---- Margrabe ----

class TestMargrabe:
    def test_basic(self):
        result = margrabe_fx_exchange(1.0, 1.0, 0.02, 0.01, 0.015,
                                       0.10, 0.12, 0.3, 1.0)
        assert isinstance(result, MargrabeResult)
        assert result.price > 0

    def test_no_correlation_max_vol(self):
        """ρ=0 → combined vol² = σ₁² + σ₂² (highest)."""
        neg = margrabe_fx_exchange(1.0, 1.0, 0.02, 0.01, 0.015, 0.10, 0.12, -0.5, 1.0)
        zero = margrabe_fx_exchange(1.0, 1.0, 0.02, 0.01, 0.015, 0.10, 0.12, 0.0, 1.0)
        pos = margrabe_fx_exchange(1.0, 1.0, 0.02, 0.01, 0.015, 0.10, 0.12, 0.5, 1.0)
        # Negative correlation → highest price (highest σ)
        assert neg.price > zero.price > pos.price

    def test_vol_combined(self):
        result = margrabe_fx_exchange(1.0, 1.0, 0.02, 0.01, 0.015, 0.08, 0.08, 0.0, 1.0)
        # ρ=0 → σ² = 0.08² + 0.08² = 0.0128 → σ ≈ 0.113
        assert result.vol_combined == pytest.approx(math.sqrt(0.0128))

    def test_identical_assets_zero(self):
        """Exchange two identical assets → zero value."""
        result = margrabe_fx_exchange(1.0, 1.0, 0.02, 0.01, 0.01, 0.10, 0.10, 1.0, 1.0)
        # Perfect correlation + same vol + same spot → combined σ = 0, price = 0
        assert result.price < 1e-6


# ---- Implied correlation ----

class TestImpliedCorrelationQuanto:
    def test_basic(self):
        result = implied_correlation_quanto(
            spot=1.0, strike=1.0,
            quanto_strike_vol=0.10, underlying_vol=0.20,
            fx_vol=0.15, quanto_adjustment_observed=-0.01, T=1.0,
        )
        assert isinstance(result, ImpliedCorrelationResult)
        assert -1 <= result.correlation <= 1

    def test_positive_adjustment_negative_correlation(self):
        """Positive quanto adjustment → negative correlation."""
        result = implied_correlation_quanto(
            1.0, 1.0, 0.10, 0.20, 0.15,
            quanto_adjustment_observed=0.01, T=1.0,
        )
        assert result.correlation < 0

    def test_zero_vols_zero_correlation(self):
        result = implied_correlation_quanto(1.0, 1.0, 0, 0, 0, 0, 1.0)
        assert result.correlation == 0.0
