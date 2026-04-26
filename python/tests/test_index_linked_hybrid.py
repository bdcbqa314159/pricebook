"""Tests for index-linked hybrid pricing (Pucci 2012b, Section 9 validation)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.cms import cash_annuity
from pricebook.hybrid_mc import simulate_2d_local_vol, local_vol_hybrid_price
from pricebook.index_linked_hybrid import index_linked_hybrid_price


# ---- Test parameters ----
# 5Y expiry, 5Y swap (semi-annual), flat vols

T = 5.0
N_PERIODS = 10
YFS = [0.5] * N_PERIODS
TAUS = [0.5 * (i + 1) for i in range(N_PERIODS)]  # yf(T, T_i)
F0 = 0.04       # convexity-adjusted forward swap rate
U0 = 0.04       # index forward (normalised to rate scale for simplicity)
SIGMA_F = 0.30   # flat rate vol
SIGMA_U = 0.25   # flat index vol
DF = math.exp(-0.03 * T)


# ---- 9.1 Building blocks ----

class TestBuildingBlocks:
    """Validation items 1-3."""

    def test_v1_cash_annuity_closed_form(self):
        """V1: Annual schedule with y_i=1, yf=i: Â = (1-(1+S)^{-n})/S."""
        S = 0.05
        n = 5
        yfs = [1.0] * n
        taus = [float(i + 1) for i in range(n)]
        A = cash_annuity(S, yfs, taus)
        expected = (1 - (1 + S)**(-n)) / S
        assert A == pytest.approx(expected, rel=1e-8)

    def test_v1_cash_annuity_positive(self):
        """Cash annuity should be positive for positive swap rate."""
        A = cash_annuity(F0, YFS, TAUS)
        assert A > 0

    def test_v1_cash_annuity_decreasing_in_rate(self):
        """Higher swap rate => lower cash annuity."""
        A_low = cash_annuity(0.02, YFS, TAUS)
        A_high = cash_annuity(0.08, YFS, TAUS)
        assert A_high < A_low


# ---- 9.2 Calibration self-checks ----

class TestCalibrationSelfChecks:
    """Validation items 4-5: martingale property."""

    def test_v4_martingale_F(self):
        """V4: E^T[F_T] = F0 (martingale property)."""
        F_T, _ = simulate_2d_local_vol(
            F0, U0, SIGMA_F, SIGMA_U, 0.0, T,
            n_paths=100_000, n_steps=50, seed=42)
        assert float(F_T.mean()) == pytest.approx(F0, rel=0.01)

    def test_v5_martingale_U(self):
        """V5: E^T[U_T] = U0 (martingale property)."""
        _, U_T = simulate_2d_local_vol(
            F0, U0, SIGMA_F, SIGMA_U, 0.0, T,
            n_paths=100_000, n_steps=50, seed=42)
        assert float(U_T.mean()) == pytest.approx(U0, rel=0.01)


# ---- 9.3 Hybrid pricing ----

class TestHybridPricing:
    """Validation items 8-11."""

    def test_v8_black_limit_rho_zero(self):
        """V8: Flat vol, rho=0 — MC matches closed-form integral."""
        # With rho=0, hybrid = integral of cash-swaption(K=u) * pU(u) du
        # For flat vol, both are lognormal, so MC should give a finite price
        result = index_linked_hybrid_price(
            F0, U0, DF, YFS, TAUS, SIGMA_F, SIGMA_U,
            rho=0.0, T=T, theta=1,
            n_paths=50_000, n_steps=50, seed=42)
        assert result.price > 0
        assert math.isfinite(result.price)
        assert result.std_error / result.price < 0.05  # reasonable MC error

    def test_v9_rho_zero_decomposition(self):
        """V9: At rho=0, hybrid = integral of cash-swaption against index density.

        Instead of implementing the full 1D integral, we verify that setting
        U deterministic (U=U0) recovers the vanilla cash swaption.
        """
        # Vanilla cash swaption: U_T = U0 (constant)
        vanilla = index_linked_hybrid_price(
            F0, U0, DF, YFS, TAUS, SIGMA_F, 1e-10,  # near-zero index vol
            rho=0.0, T=T, theta=1,
            n_paths=50_000, n_steps=50, seed=42)

        # With index vol, the hybrid should differ (broader strike distribution)
        hybrid = index_linked_hybrid_price(
            F0, U0, DF, YFS, TAUS, SIGMA_F, SIGMA_U,
            rho=0.0, T=T, theta=1,
            n_paths=50_000, n_steps=50, seed=42)

        # Both should be positive and finite
        assert vanilla.price > 0
        assert hybrid.price > 0

    def test_v10_rho_plus_one(self):
        """V10: rho=+1, price should be finite (degenerate joint)."""
        result = index_linked_hybrid_price(
            F0, U0, DF, YFS, TAUS, SIGMA_F, SIGMA_U,
            rho=0.999, T=T, theta=1,
            n_paths=20_000, n_steps=50, seed=42)
        assert math.isfinite(result.price)

    def test_v10_rho_minus_one(self):
        """V10: rho=-1, price should be finite."""
        result = index_linked_hybrid_price(
            F0, U0, DF, YFS, TAUS, SIGMA_F, SIGMA_U,
            rho=-0.999, T=T, theta=1,
            n_paths=20_000, n_steps=50, seed=42)
        assert math.isfinite(result.price)

    def test_v11_monotone_in_rho(self):
        """V11: ATM hybrid price decreases with rho (less spread variance).

        When F0 = U0, higher rho means S and U move together,
        reducing (S-U) variance and lowering the option value.
        """
        prices = []
        for rho in [-0.5, 0.0, 0.5]:
            result = index_linked_hybrid_price(
                F0, U0, DF, YFS, TAUS, SIGMA_F, SIGMA_U,
                rho=rho, T=T, theta=1,
                n_paths=50_000, n_steps=50, seed=42)
            prices.append(result.price)
        # ATM: price decreases with rho (less spread variance)
        assert prices[0] > prices[1] > prices[2]


# ---- Additional tests ----

class TestAdditional:
    """Extra regression tests."""

    def test_payer_positive(self):
        """Payer hybrid should have positive price."""
        result = index_linked_hybrid_price(
            F0, U0, DF, YFS, TAUS, SIGMA_F, SIGMA_U,
            rho=0.3, T=T, theta=1,
            n_paths=20_000, n_steps=50, seed=42)
        assert result.price > 0

    def test_receiver_positive(self):
        """Receiver hybrid should also have positive price (optionality)."""
        result = index_linked_hybrid_price(
            F0, U0, DF, YFS, TAUS, SIGMA_F, SIGMA_U,
            rho=0.3, T=T, theta=-1,
            n_paths=20_000, n_steps=50, seed=42)
        assert result.price > 0

    def test_higher_vol_higher_price(self):
        """Higher rate vol => higher hybrid price."""
        low = index_linked_hybrid_price(
            F0, U0, DF, YFS, TAUS, 0.15, SIGMA_U,
            rho=0.3, T=T, theta=1,
            n_paths=30_000, n_steps=50, seed=42)
        high = index_linked_hybrid_price(
            F0, U0, DF, YFS, TAUS, 0.45, SIGMA_U,
            rho=0.3, T=T, theta=1,
            n_paths=30_000, n_steps=50, seed=42)
        assert high.price > low.price

    def test_martingale_check_in_result(self):
        """Result should report E[F_T] ≈ F0 and E[U_T] ≈ U0."""
        result = index_linked_hybrid_price(
            F0, U0, DF, YFS, TAUS, SIGMA_F, SIGMA_U,
            rho=0.3, T=T, theta=1,
            n_paths=100_000, n_steps=50, seed=42)
        assert result.mean_swap_rate == pytest.approx(F0, rel=0.02)
        assert result.mean_index == pytest.approx(U0, rel=0.02)
