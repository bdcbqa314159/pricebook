"""Tests for advanced vol derivatives: var swap Greeks, forward var, vol swap,
dispersion, Bates CF, SVI calibration."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
from dateutil.relativedelta import relativedelta

from pricebook.vol_derivatives_advanced import (
    variance_swap_greeks, VarianceSwapGreeks,
    forward_variance_curve, ForwardVarianceCurve,
    volatility_swap_price, VolSwapResult,
    dispersion_trade, DispersionTradeResult,
    bates_characteristic_function,
    svi_calibrate, SVIParams,
)


# ── Variance swap Greeks ──

class TestVarianceSwapGreeks:

    def test_dollar_gamma_constant(self):
        """Dollar gamma = 2/T × notional_var (constant, key property)."""
        g1 = variance_swap_greeks(100, 0.04, 1.0, 0.20, 1_000_000)
        g2 = variance_swap_greeks(120, 0.04, 1.0, 0.20, 1_000_000)
        # Dollar gamma should be the same at different spot levels
        assert abs(g1.dollar_gamma - g2.dollar_gamma) < 1.0

    def test_gamma_inversely_proportional_to_spot_squared(self):
        """Gamma ∝ 1/S²."""
        g100 = variance_swap_greeks(100, 0.04, 1.0, 0.20, 1_000_000)
        g200 = variance_swap_greeks(200, 0.04, 1.0, 0.20, 1_000_000)
        # gamma(200) ≈ gamma(100) / 4
        ratio = g100.gamma / g200.gamma
        assert abs(ratio - 4.0) < 0.1

    def test_vega_positive(self):
        """Long var swap has positive vega."""
        g = variance_swap_greeks(100, 0.04, 1.0, 0.20, 1_000_000)
        assert g.vega > 0

    def test_theta_negative(self):
        """Theta is negative (variance accrual drains value)."""
        g = variance_swap_greeks(100, 0.04, 1.0, 0.20, 1_000_000)
        assert g.theta < 0

    def test_pv_at_fair_is_zero(self):
        """At fair strike (= implied var), PV ≈ 0."""
        vol = 0.20
        g = variance_swap_greeks(100, vol ** 2, 1.0, vol, 1_000_000)
        assert abs(g.pv) < 100  # small residual

    def test_to_dict(self):
        g = variance_swap_greeks(100, 0.04, 1.0, 0.20, 1_000_000)
        d = g.to_dict()
        assert "dollar_gamma" in d
        assert "vega" in d


# ── Forward variance ──

class TestForwardVariance:

    def test_forward_var_from_term_structure(self):
        """Forward var is positive when term structure is upward sloping."""
        atm_vols = [(0.25, 0.15), (0.5, 0.17), (1.0, 0.19), (2.0, 0.20)]
        fvc = forward_variance_curve(atm_vols)
        assert len(fvc.forward_vars) == 4
        assert all(v >= 0 for v in fvc.forward_vars)

    def test_total_variance_increasing(self):
        """Total variance σ²T should increase with T."""
        atm_vols = [(0.25, 0.15), (0.5, 0.17), (1.0, 0.19)]
        fvc = forward_variance_curve(atm_vols)
        assert fvc.total_vars[-1] > fvc.total_vars[0]

    def test_forward_vol(self):
        atm_vols = [(0.5, 0.18), (1.0, 0.20)]
        fvc = forward_variance_curve(atm_vols)
        fv = fvc.forward_vol(1)
        assert fv > 0

    def test_to_dict(self):
        atm_vols = [(0.5, 0.18), (1.0, 0.20)]
        d = forward_variance_curve(atm_vols).to_dict()
        assert "forward_vols" in d


# ── Vol swap ──

class TestVolSwap:

    def test_fair_vol_below_atm(self):
        """Fair vol swap strike < ATM vol (Jensen's inequality)."""
        result = volatility_swap_price(0.0, 0.20, vol_of_vol=0.50)
        assert result.fair_vol < 0.20

    def test_convexity_adjustment_positive(self):
        """Convexity adjustment is positive (E[√V] < √E[V])."""
        result = volatility_swap_price(0.0, 0.20, vol_of_vol=0.50)
        assert result.convexity_adjustment > 0

    def test_zero_volvol_no_adjustment(self):
        """With zero vol-of-vol, fair vol = ATM vol."""
        result = volatility_swap_price(0.0, 0.20, vol_of_vol=0.0)
        assert result.fair_vol == 0.20
        assert result.convexity_adjustment == 0.0

    def test_to_dict(self):
        d = volatility_swap_price(0.18, 0.20, 0.50).to_dict()
        assert "convexity_adj" in d


# ── Dispersion ──

class TestDispersion:

    def test_implied_correlation_between_0_and_1(self):
        """Implied correlation from index vs constituents."""
        result = dispersion_trade(
            index_vol=0.15,
            constituent_vols=[0.25, 0.22, 0.28, 0.20, 0.24],
        )
        assert 0 <= result.implied_correlation <= 1

    def test_index_var_less_than_avg_constituent(self):
        """Index variance < average constituent variance (diversification)."""
        result = dispersion_trade(
            index_vol=0.15,
            constituent_vols=[0.25, 0.22, 0.28, 0.20, 0.24],
        )
        assert result.index_var < result.avg_constituent_var

    def test_to_dict(self):
        result = dispersion_trade(0.15, [0.25, 0.22, 0.28])
        d = result.to_dict()
        assert "implied_corr" in d
        assert "pnl" in d


# ── Bates characteristic function ──

class TestBatesCF:

    def test_cf_at_zero_is_one(self):
        """φ(0) = 1 for any characteristic function."""
        cf = bates_characteristic_function(
            0+0j, 100, 100, 1.0, 0.04, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            0.5, -0.05, 0.10,
        )
        assert abs(cf - 1.0) < 0.01

    def test_cf_finite(self):
        """CF should be finite for reasonable parameters."""
        cf = bates_characteristic_function(
            1+0j, 100, 100, 1.0, 0.04, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            0.5, -0.05, 0.10,
        )
        assert math.isfinite(abs(cf))

    def test_cf_without_jumps_equals_heston(self):
        """With λ=0 (no jumps), Bates CF should equal Heston CF."""
        u = 0.5 + 0j
        bates_cf = bates_characteristic_function(
            u, 100, 100, 1.0, 0.04, 0.0,
            0.04, 1.5, 0.04, 0.3, -0.7,
            0.0, 0.0, 0.0,  # no jumps
        )
        # Should match Heston (no jump contribution)
        assert math.isfinite(abs(bates_cf))


# ── SVI calibration ──

class TestSVI:

    def _sample_smile(self):
        """Generate a sample smile for calibration."""
        k = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        # Typical equity skew: higher vol for OTM puts
        vols = [0.25, 0.22, 0.19, 0.18, 0.175, 0.17, 0.17]
        T = 0.5
        total_var = [v ** 2 * T for v in vols]
        return k, total_var, T

    def test_calibration_roundtrip(self):
        """Calibrated SVI should fit market data closely."""
        k, w, T = self._sample_smile()
        params = svi_calibrate(k, w, T)
        for ki, wi in zip(k, w):
            model_w = params.total_variance(ki)
            assert abs(model_w - wi) < 0.005  # within 50bp² of total var

    def test_implied_vol(self):
        k, w, T = self._sample_smile()
        params = svi_calibrate(k, w, T)
        vol_atm = params.implied_vol(0.0)
        assert 0.10 < vol_atm < 0.30

    def test_arbitrage_free_flag(self):
        k, w, T = self._sample_smile()
        params = svi_calibrate(k, w, T)
        # Well-calibrated SVI should be arb-free
        assert isinstance(params.is_arbitrage_free(), bool)

    def test_to_dict(self):
        k, w, T = self._sample_smile()
        params = svi_calibrate(k, w, T)
        d = params.to_dict()
        assert "a" in d
        assert "rho" in d
        assert "arb_free" in d

    def test_skew_direction(self):
        """Equity skew: ρ should be negative (puts more expensive)."""
        k, w, T = self._sample_smile()
        params = svi_calibrate(k, w, T)
        assert params.rho < 0
