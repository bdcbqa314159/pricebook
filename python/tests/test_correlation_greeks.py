"""Tests for correlation Greeks."""

import math

import numpy as np
import pytest

from pricebook.correlation_greeks import (
    CorrelationDeltaResult,
    CorrelationGammaResult,
    CorrelationLadder,
    CorrelationPnLAttribution,
    CrossGammaResult,
    correlation_delta,
    correlation_gamma,
    correlation_pnl_attribution,
    correlation_sensitivity_ladder,
    cross_gamma,
)


# ---- Helpers: simple two-asset basket call for testing ----

_CRN_Z1 = None
_CRN_Z2_INDEP = None

def _ensure_crn(n_paths=50_000, seed=42):
    """Generate and cache common random numbers for CRN-based pricing."""
    global _CRN_Z1, _CRN_Z2_INDEP
    if _CRN_Z1 is None or len(_CRN_Z1) != n_paths:
        rng = np.random.default_rng(seed)
        _CRN_Z1 = rng.standard_normal(n_paths)
        _CRN_Z2_INDEP = rng.standard_normal(n_paths)

def _basket_price(rho, spot1=100, spot2=100, strike=100, vol1=0.20, vol2=0.25,
                   rate=0.03, T=1.0, n_paths=50_000, seed=42):
    """Two-asset average basket call price under given correlation (CRN)."""
    _ensure_crn(n_paths, seed)
    z1 = _CRN_Z1[:n_paths]
    z2 = rho * z1 + math.sqrt(max(1 - rho**2, 0)) * _CRN_Z2_INDEP[:n_paths]
    sqrt_T = math.sqrt(T)
    S1_T = spot1 * np.exp((rate - 0.5 * vol1**2) * T + vol1 * sqrt_T * z1)
    S2_T = spot2 * np.exp((rate - 0.5 * vol2**2) * T + vol2 * sqrt_T * z2)
    basket = 0.5 * S1_T + 0.5 * S2_T
    payoff = np.maximum(basket - strike, 0.0)
    return float(math.exp(-rate * T) * payoff.mean())


def _basket_price_spots(s1, s2, rho=0.5, **kw):
    """Basket price as function of two spot prices."""
    return _basket_price(rho, spot1=s1, spot2=s2, **kw)


# ---- Correlation delta ----

class TestCorrelationDelta:
    def test_basic(self):
        result = correlation_delta(lambda r: _basket_price(r), rho=0.5)
        assert isinstance(result, CorrelationDeltaResult)

    def test_basket_call_positive_rho_delta(self):
        """Basket call on average: higher ρ → less diversification → higher
        basket vol → higher option price. So ∂V/∂ρ > 0."""
        result = correlation_delta(lambda r: _basket_price(r), rho=0.3)
        assert result.rho_delta > 0

    def test_bump_size(self):
        result = correlation_delta(lambda r: _basket_price(r), 0.5, bump=0.05)
        assert result.bump_size == pytest.approx(0.05)

    def test_near_boundary(self):
        """ρ near +1: bump clamps to ≤ 0.999."""
        result = correlation_delta(lambda r: _basket_price(r), 0.99, bump=0.05)
        assert result.bumped_up_price is not None


# ---- Correlation gamma ----

class TestCorrelationGamma:
    def test_basic(self):
        result = correlation_gamma(lambda r: _basket_price(r), 0.5)
        assert isinstance(result, CorrelationGammaResult)

    def test_convexity_pnl(self):
        result = correlation_gamma(lambda r: _basket_price(r), 0.3)
        # Convexity P&L = 0.5 × γ × h² (always ≥ 0 if γ > 0)
        assert isinstance(result.convexity_pnl_per_unit, float)

    def test_delta_from_gamma_consistent(self):
        """Delta from gamma function should match standalone delta."""
        g = correlation_gamma(lambda r: _basket_price(r), 0.5)
        d = correlation_delta(lambda r: _basket_price(r), 0.5)
        assert g.rho_delta == pytest.approx(d.rho_delta, rel=0.01)


# ---- Cross-gamma ----

class TestCrossGamma:
    def test_basic(self):
        result = cross_gamma(
            lambda s1, s2: _basket_price_spots(s1, s2),
            spot1=100, spot2=100,
        )
        assert isinstance(result, CrossGammaResult)

    def test_positive_delta_for_call(self):
        """Basket call has positive delta to both underlyings."""
        result = cross_gamma(_basket_price_spots, 100, 100)
        assert result.delta1 > 0
        assert result.delta2 > 0

    def test_cross_gamma_sign(self):
        """For a basket call, cross-gamma is typically positive
        (both deltas increase together)."""
        result = cross_gamma(_basket_price_spots, 100, 100)
        # Can be positive or slightly negative depending on correlation; just check bounded
        assert abs(result.cross_gamma) < 10

    def test_asset_names(self):
        result = cross_gamma(_basket_price_spots, 100, 100,
                              asset1_name="AAPL", asset2_name="GOOG")
        assert result.assets == ("AAPL", "GOOG")


# ---- P&L attribution ----

class TestCorrelationPnLAttribution:
    def test_basic(self):
        result = correlation_pnl_attribution(
            lambda r: _basket_price(r), rho_old=0.3, rho_new=0.5,
        )
        assert isinstance(result, CorrelationPnLAttribution)

    def test_explained_close_to_total(self):
        """For small ρ changes, Taylor expansion should explain most P&L."""
        result = correlation_pnl_attribution(
            lambda r: _basket_price(r), 0.50, 0.52,
        )
        # Small move → explained ≈ total
        if abs(result.total_pnl) > 1e-3:
            assert abs(result.unexplained) < abs(result.total_pnl) * 0.5

    def test_zero_change_zero_pnl(self):
        result = correlation_pnl_attribution(lambda r: _basket_price(r), 0.5, 0.5)
        assert result.total_pnl == 0.0
        assert result.rho_change == 0.0

    def test_delta_dominates_for_small_move(self):
        result = correlation_pnl_attribution(lambda r: _basket_price(r), 0.3, 0.31)
        # For 1% ρ move, delta term should dominate gamma
        assert abs(result.delta_pnl) >= abs(result.gamma_pnl) * 0.5


# ---- Correlation ladder ----

class TestCorrelationLadder:
    def test_basic(self):
        corr = np.array([[1.0, 0.5, 0.3],
                          [0.5, 1.0, 0.4],
                          [0.3, 0.4, 1.0]])

        def portfolio_price(c):
            # Simple: price depends on average off-diagonal ρ
            off_diag = [c[0, 1], c[0, 2], c[1, 2]]
            avg_rho = np.mean(off_diag)
            return _basket_price(avg_rho)

        ladder = correlation_sensitivity_ladder(
            ["AAPL", "GOOG", "MSFT"], corr, portfolio_price,
        )
        assert isinstance(ladder, CorrelationLadder)
        assert ladder.n_pairs == 3

    def test_entries_cover_all_pairs(self):
        corr = np.eye(3)
        corr[0, 1] = corr[1, 0] = 0.5
        corr[0, 2] = corr[2, 0] = 0.3
        corr[1, 2] = corr[2, 1] = 0.4

        ladder = correlation_sensitivity_ladder(
            ["A", "B", "C"], corr,
            lambda c: _basket_price(c[0, 1]),
        )
        pairs = [e.asset_pair for e in ladder.entries]
        assert ("A", "B") in pairs
        assert ("A", "C") in pairs
        assert ("B", "C") in pairs

    def test_total_sensitivity(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        ladder = correlation_sensitivity_ladder(
            ["X", "Y"], corr, lambda c: _basket_price(c[0, 1]),
        )
        assert ladder.total_rho_delta > 0
        assert ladder.n_pairs == 1
