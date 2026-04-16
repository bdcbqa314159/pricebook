"""Tests for advanced dividend modelling."""

import math

import numpy as np
import pytest

from pricebook.dividend_advanced import (
    BuhlerResult,
    BuhlerStochasticDividend,
    DividendBasisResult,
    DividendCurve,
    DividendHedgeResult,
    ImpliedDividendResult,
    dividend_basis_trade,
    dividend_curve_bootstrap,
    dividend_hedge_ratio,
    implied_dividend_yield,
)


# ---- Bühler stochastic dividend ----

class TestBuhler:
    def test_basic(self):
        model = BuhlerStochasticDividend(
            r=0.03, q0=0.02, kappa_q=2.0, theta_q=0.02,
            xi_q=0.01, sigma_s=0.20, rho=0.3,
        )
        result = model.simulate(100, 1.0, n_paths=500, n_steps=50, seed=42)
        assert isinstance(result, BuhlerResult)

    def test_paths_positive(self):
        model = BuhlerStochasticDividend(0.03, 0.02, 2.0, 0.02, 0.01, 0.20, 0.3)
        result = model.simulate(100, 1.0, n_paths=500, n_steps=50, seed=42)
        assert np.all(result.spot_paths >= 0)
        assert np.all(result.dividend_yield_paths >= 0)

    def test_implied_forward_near_analytical(self):
        """Model forward should be near spot × exp((r-q̄)T)."""
        model = BuhlerStochasticDividend(0.03, 0.02, 5.0, 0.02, 0.005, 0.20, 0.0)
        F_model = model.implied_forward(100, 1.0, n_paths=10_000, seed=42)
        F_analytical = 100 * math.exp((0.03 - 0.02) * 1.0)
        assert F_model == pytest.approx(F_analytical, rel=0.02)

    def test_zero_vol_of_q_deterministic(self):
        """ξ_q=0 → dividend yield stays at q0 (no drift diffusion)."""
        model = BuhlerStochasticDividend(0.03, 0.02, 0.0, 0.02, 0.0, 0.20, 0.0)
        result = model.simulate(100, 1.0, n_paths=100, n_steps=50, seed=42)
        assert result.dividend_yield_paths[:, -1].std() < 1e-8


# ---- Dividend curve ----

class TestDividendCurve:
    def test_basic(self):
        tenors = [0.5, 1.0, 2.0, 5.0]
        futures = [2.0, 4.0, 8.0, 20.0]
        curve = dividend_curve_bootstrap(100, 0.03, tenors, futures)
        assert isinstance(curve, DividendCurve)
        assert len(curve.tenors) == 4

    def test_yields_positive(self):
        tenors = [1.0, 2.0]
        futures = [2.0, 4.5]
        curve = dividend_curve_bootstrap(100, 0.03, tenors, futures)
        assert np.all(curve.implied_yields > 0)

    def test_monotone_dividends(self):
        """Futures prices should be increasing in T."""
        tenors = [1.0, 2.0, 5.0]
        futures = [2.0, 4.0, 9.5]
        curve = dividend_curve_bootstrap(100, 0.03, tenors, futures)
        assert np.all(np.diff(curve.cumulative_dividends) > 0)


# ---- Implied dividend yield ----

class TestImpliedDividendYield:
    def test_basic(self):
        # Fabricate: S=100, K=100, T=1, r=0.03, q=0.02 → F=101.005
        # C - P = S e^{-qT} - K e^{-rT} = 100×0.9802 - 100×0.9704 = 0.98
        # So q = 0.02 via parity
        # Use approximate call/put prices consistent with q=0.02
        from pricebook.black76 import black76_price, OptionType
        F = 100 * math.exp((0.03 - 0.02) * 1.0)
        df = math.exp(-0.03)
        C = black76_price(F, 100, 0.20, 1.0, df, OptionType.CALL)
        P = black76_price(F, 100, 0.20, 1.0, df, OptionType.PUT)

        result = implied_dividend_yield(100, 100, C, P, 0.03, 1.0)
        assert isinstance(result, ImpliedDividendResult)
        assert result.implied_yield == pytest.approx(0.02, abs=0.005)

    def test_zero_dividend(self):
        """If C-P = S - K×DF, implied q=0."""
        from pricebook.black76 import black76_price, OptionType
        F = 100 * math.exp(0.03 * 1.0)
        df = math.exp(-0.03)
        C = black76_price(F, 100, 0.20, 1.0, df, OptionType.CALL)
        P = black76_price(F, 100, 0.20, 1.0, df, OptionType.PUT)

        result = implied_dividend_yield(100, 100, C, P, 0.03, 1.0)
        assert result.implied_yield == pytest.approx(0.0, abs=0.005)

    def test_forward_consistent(self):
        from pricebook.black76 import black76_price, OptionType
        F_true = 100 * math.exp((0.03 - 0.025) * 1.0)
        df = math.exp(-0.03)
        C = black76_price(F_true, 100, 0.20, 1.0, df, OptionType.CALL)
        P = black76_price(F_true, 100, 0.20, 1.0, df, OptionType.PUT)

        result = implied_dividend_yield(100, 100, C, P, 0.03, 1.0)
        assert result.forward == pytest.approx(F_true, rel=0.01)


# ---- Dividend basis ----

class TestDividendBasis:
    def test_basic(self):
        dividends = [(0.25, 1.0), (0.50, 1.0), (0.75, 1.0), (1.00, 1.0)]
        result = dividend_basis_trade(dividends, 0.03, 1.0, future_price=4.0)
        assert isinstance(result, DividendBasisResult)
        assert result.cash_dividend_pv > 0

    def test_no_basis_consistent(self):
        """Future price = PV(cash) → zero basis."""
        dividends = [(0.5, 2.0), (1.0, 2.0)]
        pv = 2.0 * math.exp(-0.03 * 0.5) + 2.0 * math.exp(-0.03 * 1.0)
        result = dividend_basis_trade(dividends, 0.03, 1.0, future_price=pv)
        assert result.basis == pytest.approx(0.0, abs=1e-6)

    def test_positive_basis(self):
        """Higher futures → positive basis."""
        dividends = [(0.5, 2.0), (1.0, 2.0)]
        pv = 2.0 * math.exp(-0.03 * 0.5) + 2.0 * math.exp(-0.03 * 1.0)
        result = dividend_basis_trade(dividends, 0.03, 1.0, future_price=pv + 0.5)
        assert result.basis > 0


# ---- Dividend hedge ratio ----

class TestDividendHedge:
    def test_basic(self):
        result = dividend_hedge_ratio(100, 0.02, 1.0)
        assert isinstance(result, DividendHedgeResult)

    def test_hedge_ratio_sign(self):
        """Spot has negative dividend exposure; hedge ratio flips sign."""
        result = dividend_hedge_ratio(100, 0.02, 1.0)
        assert result.spot_sensitivity < 0
        assert result.future_hedge_ratio > 0

    def test_residual_zero(self):
        """Perfect hedge → residual ≈ 0."""
        result = dividend_hedge_ratio(100, 0.02, 1.0)
        assert abs(result.residual_risk) < 0.01

    def test_beta_scaling(self):
        """Higher beta → larger hedge ratio magnitude."""
        h1 = dividend_hedge_ratio(100, 0.02, 1.0, beta_div=1.0)
        h2 = dividend_hedge_ratio(100, 0.02, 1.0, beta_div=2.0)
        assert abs(h2.future_hedge_ratio) > abs(h1.future_hedge_ratio)
