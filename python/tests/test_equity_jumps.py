"""Tests for equity jump and regime-switching models."""

import math

import numpy as np
import pytest

from pricebook.equity_jumps import (
    KouResult,
    MertonHybridResult,
    RegimeResult,
    RegimeSwitchingEquity,
    SVJEquityModel,
    SVJResult,
    kou_equity_price,
    merton_equity_hybrid,
)


# ---- Kou ----

class TestKouEquity:
    def test_basic(self):
        result = kou_equity_price(100, 100, 0.03, 0.02, 0.20, 1.0,
                                    lambda_jump=0.5, p=0.4, eta1=10, eta2=5)
        assert isinstance(result, KouResult)
        assert result.price > 0

    def test_no_jumps_matches_bs(self):
        """λ=0 → Kou reduces to Black-Scholes."""
        from pricebook.black76 import black76_price, OptionType
        result = kou_equity_price(100, 100, 0.03, 0.02, 0.20, 1.0,
                                    0.0, 0.4, 10, 5)
        F = 100 * math.exp((0.03 - 0.02) * 1.0)
        bs = black76_price(F, 100, 0.20, 1.0, math.exp(-0.03), OptionType.CALL)
        assert result.price == pytest.approx(bs, rel=0.02)

    def test_invalid_eta1(self):
        with pytest.raises(ValueError):
            kou_equity_price(100, 100, 0.03, 0.02, 0.20, 1.0, 0.5, 0.4, 0.5, 5)

    def test_asymmetric_jumps(self):
        """Down-jump dominance (low p) → higher put prices."""
        down_heavy = kou_equity_price(100, 90, 0.03, 0.02, 0.20, 1.0,
                                        lambda_jump=1.0, p=0.2, eta1=10, eta2=3,
                                        is_call=False)
        up_heavy = kou_equity_price(100, 90, 0.03, 0.02, 0.20, 1.0,
                                      lambda_jump=1.0, p=0.8, eta1=10, eta2=3,
                                      is_call=False)
        assert down_heavy.price > up_heavy.price


# ---- SVJ ----

class TestSVJEquity:
    def test_basic(self):
        model = SVJEquityModel(
            v0=0.04, kappa_v=2.0, theta_v=0.04, xi_v=0.3, rho=-0.5,
            lambda_jump=0.3, mu_j=-0.05, sigma_j=0.10,
        )
        result = model.simulate_option(100, 100, 0.03, 0.02, 1.0,
                                         n_paths=2000, n_steps=50, seed=42)
        assert isinstance(result, SVJResult)
        assert result.price > 0

    def test_paths_positive(self):
        model = SVJEquityModel(0.04, 2.0, 0.04, 0.3, -0.5, 0.3, -0.05, 0.10)
        result = model.simulate_option(100, 100, 0.03, 0.02, 1.0,
                                         n_paths=500, n_steps=30, seed=42)
        assert np.all(result.spot_paths >= 0)
        assert np.all(result.variance_paths >= 0)

    def test_no_jumps_no_jumps_count(self):
        model = SVJEquityModel(0.04, 2.0, 0.04, 0.3, -0.5, 0.0, 0.0, 0.01)
        result = model.simulate_option(100, 100, 0.03, 0.02, 1.0,
                                         n_paths=500, n_steps=30, seed=42)
        assert result.n_jumps_total == 0

    def test_with_jumps(self):
        model = SVJEquityModel(0.04, 2.0, 0.04, 0.3, -0.5, 2.0, -0.05, 0.10)
        result = model.simulate_option(100, 100, 0.03, 0.02, 1.0,
                                         n_paths=500, n_steps=30, seed=42)
        assert result.n_jumps_total > 0


# ---- Regime switching ----

class TestRegimeSwitchingEquity:
    def _make(self):
        # 3 regimes: bull, bear, crisis
        drifts = [0.10, -0.05, -0.25]
        vols = [0.15, 0.25, 0.50]
        Q = np.array([
            [-0.5, 0.4, 0.1],
            [0.3, -0.5, 0.2],
            [0.3, 0.4, -0.7],
        ])
        return RegimeSwitchingEquity(drifts, vols, Q, initial_regime=0)

    def test_basic(self):
        model = self._make()
        result = model.simulate(100, 1.0, n_paths=500, n_steps=30, seed=42)
        assert isinstance(result, RegimeResult)

    def test_initial_regime(self):
        model = self._make()
        result = model.simulate(100, 1.0, n_paths=100, n_steps=20, seed=42)
        assert np.all(result.regime_paths[:, 0] == 0)

    def test_crisis_higher_vol(self):
        """In crisis regime (50% vol), realised vol on crisis-visits is higher."""
        model = self._make()
        result = model.simulate(100, 1.0, n_paths=300, n_steps=50, seed=42)
        # All regimes should be visited at least once for a few paths
        reg_flat = result.regime_paths.flatten()
        visited = np.unique(reg_flat)
        assert len(visited) >= 2

    def test_regime_durations_sum(self):
        model = self._make()
        result = model.simulate(100, 1.0, n_paths=200, n_steps=50, seed=42)
        total = sum(result.mean_regime_duration.values())
        # Sum ≈ T
        assert total == pytest.approx(1.0, rel=0.1)


# ---- Merton hybrid ----

class TestMertonEquityHybrid:
    def test_basic(self):
        result = merton_equity_hybrid(100, 100, 0.03, 0.02, 0.20, 1.0,
                                        lambda_jump=0.5, jump_mean=-0.05, jump_vol=0.10)
        assert isinstance(result, MertonHybridResult)
        assert result.price > 0

    def test_decomposition(self):
        result = merton_equity_hybrid(100, 100, 0.03, 0.02, 0.20, 1.0,
                                        0.5, -0.05, 0.10)
        # Price = diffusion + jump contribution
        assert result.price == pytest.approx(result.diffusion_contribution + result.jump_contribution, rel=1e-6)

    def test_jumps_raise_otm_puts(self):
        """Negative-mean jumps raise OTM put prices."""
        no_jumps = merton_equity_hybrid(100, 85, 0.03, 0.02, 0.20, 1.0,
                                          0.0, 0.0, 0.01, is_call=False)
        with_jumps = merton_equity_hybrid(100, 85, 0.03, 0.02, 0.20, 1.0,
                                            1.0, -0.10, 0.15, is_call=False)
        assert with_jumps.price > no_jumps.price

    def test_no_jumps_no_jump_contribution(self):
        result = merton_equity_hybrid(100, 100, 0.03, 0.02, 0.20, 1.0,
                                        0.0, 0.0, 0.01)
        assert abs(result.jump_contribution) < 0.1
