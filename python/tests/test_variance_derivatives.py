"""Tests for variance and volatility derivatives."""

import math

import numpy as np
import pytest

from pricebook.variance_derivatives import (
    VRPResult,
    VarianceFuturesResult,
    VarianceSwapResult,
    VolatilitySwapResult,
    brockhaus_long_approx,
    variance_future_price,
    variance_risk_premium,
    variance_swap_replication,
    volatility_swap_heston,
)


# ---- Variance swap replication ----

class TestVarianceSwapReplication:
    def test_basic(self):
        strikes = [80, 90, 100, 110, 120]
        vols = [0.22, 0.21, 0.20, 0.21, 0.22]
        result = variance_swap_replication(
            spot=100, forward=102, rate=0.02, dividend_yield=0.0, T=1.0,
            strikes=strikes, call_vols=vols,
        )
        assert isinstance(result, VarianceSwapResult)
        assert result.fair_variance > 0
        assert result.fair_vol_strike > 0

    def test_fair_vol_near_atm_vol(self):
        """Flat smile with σ = 0.20 → fair vol ≈ 0.20."""
        strikes = np.linspace(50, 200, 30).tolist()
        vols = [0.20] * len(strikes)
        result = variance_swap_replication(
            spot=100, forward=100, rate=0.0, dividend_yield=0.0, T=1.0,
            strikes=strikes, call_vols=vols,
        )
        assert result.fair_vol_strike == pytest.approx(0.20, abs=0.03)

    def test_higher_smile_higher_strike(self):
        """Smile with higher wings → higher variance strike."""
        strikes = np.linspace(50, 200, 30).tolist()
        flat = [0.20] * len(strikes)
        # Smile: U-shape
        smile = [0.20 + 0.001 * (k - 100)**2 / 100 for k in strikes]

        result_flat = variance_swap_replication(
            100, 100, 0.0, 0.0, 1.0, strikes, flat,
        )
        result_smile = variance_swap_replication(
            100, 100, 0.0, 0.0, 1.0, strikes, smile,
        )
        assert result_smile.fair_variance > result_flat.fair_variance

    def test_n_strikes_used(self):
        strikes = [80, 90, 100, 110, 120]
        vols = [0.20] * 5
        result = variance_swap_replication(100, 100, 0.02, 0.0, 1.0, strikes, vols)
        assert result.n_strikes_used == 5


# ---- Volatility swap ----

class TestVolatilitySwapHeston:
    def test_basic(self):
        result = volatility_swap_heston(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, T=1.0)
        assert isinstance(result, VolatilitySwapResult)
        assert result.fair_vol > 0

    def test_vol_less_than_sqrt_variance(self):
        """Jensen: E[√var] < √E[var]."""
        result = volatility_swap_heston(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, T=1.0)
        assert result.fair_vol <= result.variance_strike
        assert result.convexity_adjustment >= 0

    def test_zero_vol_of_vol_no_convexity(self):
        """ξ=0 → variance is deterministic → fair vol = √variance."""
        result = volatility_swap_heston(0.04, 2.0, 0.04, 0.0, 1.0)
        assert result.convexity_adjustment == pytest.approx(0.0, abs=1e-6)

    def test_constant_variance_matches_vol(self):
        """v0 = θ → E[var] = v0 × T."""
        result = volatility_swap_heston(0.04, 2.0, 0.04, 0.0, 1.0)
        assert result.variance_strike == pytest.approx(0.20, abs=0.01)


class TestBrockhausLong:
    def test_zero_skew_matches_atm(self):
        result = brockhaus_long_approx(atm_vol=0.20, skew=0.0, T=1.0)
        assert result.fair_vol == pytest.approx(0.20)

    def test_positive_skew_reduces_vol(self):
        no_skew = brockhaus_long_approx(0.20, 0.0, 1.0)
        with_skew = brockhaus_long_approx(0.20, 0.5, 1.0)
        assert with_skew.fair_vol < no_skew.fair_vol

    def test_zero_atm_vol(self):
        result = brockhaus_long_approx(0.0, 0.1, 1.0)
        assert result.fair_vol == 0.0


# ---- Variance futures ----

class TestVarianceFutures:
    def test_basic(self):
        strikes = np.linspace(50, 200, 20).tolist()
        vols = [0.20] * 20
        result = variance_future_price(100, 0.02, 1.0, strikes, vols)
        assert isinstance(result, VarianceFuturesResult)
        assert result.volatility_index > 0

    def test_vix_like_scale(self):
        """For σ=0.20, VIX-like index ≈ 20 (in percent)."""
        strikes = np.linspace(50, 200, 30).tolist()
        vols = [0.20] * 30
        result = variance_future_price(100, 0.0, 1.0, strikes, vols)
        # Volatility index should be near 0.20
        assert result.volatility_index == pytest.approx(0.20, abs=0.05)


# ---- Variance risk premium ----

class TestVRP:
    def test_positive_vrp_typical(self):
        """Typical: implied > realised → positive VRP."""
        result = variance_risk_premium(0.04, 0.03)
        assert isinstance(result, VRPResult)
        assert result.vrp > 0
        assert result.vrp_as_ratio > 0

    def test_negative_vrp(self):
        """Realised > implied can happen in crises."""
        result = variance_risk_premium(0.03, 0.05)
        assert result.vrp < 0

    def test_zero_vrp(self):
        result = variance_risk_premium(0.04, 0.04)
        assert result.vrp == 0.0

    def test_vrp_vol_matches(self):
        """VRP in vol terms = sqrt(implied) - sqrt(realised)."""
        result = variance_risk_premium(0.04, 0.03)
        expected = math.sqrt(0.04) - math.sqrt(0.03)
        assert result.vrp_in_vol_terms == pytest.approx(expected)

    def test_zero_realised_ratio_zero(self):
        result = variance_risk_premium(0.04, 0.0)
        assert result.vrp_as_ratio == 0.0
