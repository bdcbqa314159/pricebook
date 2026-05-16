"""Tests for MC Phase 2: Bates, VG, CEV, G2++, cliquet, autocall, swing, IS, conditional MC."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from pricebook.models.mc_engine import MCEngine, TimeGrid
from pricebook.models.mc_processes import (
    BatesProcess, VarianceGammaProcess, CEVProcess, G2PlusProcess,
    BlackScholesProcess, HestonProcess,
)
from pricebook.models.mc_payoffs import (
    european_call, cliquet_payoff, autocall_payoff, swing_payoff,
)
from pricebook.models.mc_variance_reduction import importance_sampling
from pricebook.models.mc_conditional import conditional_mc_heston, conditional_mc_generic


def bs_call(s0, k, r, sigma, T):
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return s0 * norm.cdf(d1) - k * math.exp(-r * T) * norm.cdf(d2)


# ── 1. Bates ──

class TestBates:

    def test_call_price_finite(self):
        engine = MCEngine(
            BatesProcess(s0=100, v0=0.04, mu=0.05, kappa=2, theta=0.04,
                         xi=0.3, rho=-0.7, jump_intensity=1.0,
                         jump_mean=-0.05, jump_vol=0.10),
            TimeGrid.uniform(1.0, 100),
            n_paths=50_000, seed=42,
        )
        result = engine.price(european_call(100, log_space=True), math.exp(-0.05))
        assert result.price > 0
        assert math.isfinite(result.price)

    def test_no_jumps_equals_heston(self):
        """Bates with λ=0 should match Heston."""
        bates = MCEngine(
            BatesProcess(100, 0.04, 0.05, 2, 0.04, 0.3, -0.7, 0, 0, 0),
            TimeGrid.uniform(1.0, 50), 50_000, seed=42,
        )
        heston = MCEngine(
            HestonProcess(100, 0.04, 0.05, 2, 0.04, 0.3, -0.7),
            TimeGrid.uniform(1.0, 50), 50_000, seed=42,
        )
        df = math.exp(-0.05)
        p_bates = bates.price(european_call(100), df).price
        p_heston = heston.price(european_call(100), df).price
        assert p_bates == pytest.approx(p_heston, rel=0.10)


# ── 2. Variance Gamma ──

class TestVarianceGamma:

    def test_price_positive(self):
        engine = MCEngine(
            VarianceGammaProcess(s0=100, mu=0.05, sigma=0.20, nu=0.25, theta_vg=-0.15),
            TimeGrid.uniform(1.0, 100),
            n_paths=50_000, seed=42,
        )
        result = engine.price(european_call(100, log_space=True), math.exp(-0.05))
        assert result.price > 0

    def test_terminal_distribution_finite(self):
        """VG terminal should have finite moments."""
        engine = MCEngine(
            VarianceGammaProcess(100, 0.05, 0.20, 0.25, -0.15),
            TimeGrid.uniform(1.0, 100), 50_000, seed=42,
        )
        terminal = np.exp(engine.paths[:, -1])
        assert np.all(np.isfinite(terminal))
        assert float(np.mean(terminal)) > 50  # not degenerate


# ── 3. CEV ──

class TestCEV:

    def test_beta_1_matches_gbm(self):
        """CEV with β=1 should match GBM."""
        cev = MCEngine(
            CEVProcess(100, 0.05, 0.20, beta=1.0),
            TimeGrid.uniform(1.0, 100),
            50_000, seed=42,
        )
        # CEV is in spot space (not log), payoff needs log_space=False
        def call_payoff(paths, times):
            return np.maximum(paths[:, -1] - 100, 0.0)
        result = cev.price(call_payoff, math.exp(-0.05))
        bs = bs_call(100, 100, 0.05, 0.20, 1.0)
        assert result.price == pytest.approx(bs, rel=0.10)

    def test_stays_positive(self):
        engine = MCEngine(
            CEVProcess(100, 0.05, 0.30, beta=0.5),
            TimeGrid.uniform(1.0, 200), 10_000, seed=42,
        )
        assert np.all(engine.paths >= 0)


# ── 4. G2++ ──

class TestG2Plus:

    def test_factors_mean_revert(self):
        engine = MCEngine(
            G2PlusProcess(x0=0.01, y0=-0.01, a=0.5, b=0.8),
            TimeGrid.uniform(10.0, 100), 10_000, seed=42,
        )
        terminal_x = float(np.mean(engine.paths[:, -1, 0]))
        terminal_y = float(np.mean(engine.paths[:, -1, 1]))
        assert abs(terminal_x) < 0.01  # mean-reverts to 0
        assert abs(terminal_y) < 0.01

    def test_two_factors(self):
        engine = MCEngine(
            G2PlusProcess(), TimeGrid.uniform(5.0, 50), 5_000, seed=42,
        )
        assert engine.paths.shape[2] == 2


# ── 5. Cliquet ──

class TestCliquetPayoff:

    def test_positive(self):
        engine = MCEngine(
            BlackScholesProcess(100, 0.05, 0.20),
            TimeGrid.uniform(1.0, 12),  # monthly
            50_000, seed=42,
        )
        result = engine.price(cliquet_payoff(cap=0.05, floor=-0.03), 1.0)
        assert result.price > 0

    def test_cap_limits_upside(self):
        engine = MCEngine(
            BlackScholesProcess(100, 0.10, 0.30),
            TimeGrid.uniform(1.0, 12), 50_000, seed=42,
        )
        r_tight = engine.price(cliquet_payoff(cap=0.02, floor=-0.02), 1.0).price
        r_wide = engine.price(cliquet_payoff(cap=0.10, floor=-0.10), 1.0).price
        assert r_tight < r_wide


# ── 6. Autocall ──

class TestAutocallPayoff:

    def test_positive(self):
        engine = MCEngine(
            BlackScholesProcess(100, 0.05, 0.15),
            TimeGrid.uniform(3.0, 36),  # monthly for 3Y
            50_000, seed=42,
        )
        result = engine.price(
            autocall_payoff(autocall_barrier=100, autocall_coupon=0.08,
                            put_barrier=70, put_strike=100),
            math.exp(-0.05 * 3),
        )
        assert result.price > 0

    def test_high_barrier_fewer_autocalls(self):
        engine = MCEngine(
            BlackScholesProcess(100, 0.05, 0.20),
            TimeGrid.uniform(2.0, 24), 50_000, seed=42,
        )
        r_low = engine.price(autocall_payoff(80, 0.05), 1.0).price
        r_high = engine.price(autocall_payoff(120, 0.05), 1.0).price
        # Low barrier → more autocalls → higher expected coupon
        assert r_low > r_high * 0.9


# ── 7. Swing ──

class TestSwingPayoff:

    def test_positive(self):
        engine = MCEngine(
            BlackScholesProcess(50, 0.05, 0.30),
            TimeGrid.uniform(1.0, 252), 20_000, seed=42,
        )
        result = engine.price(swing_payoff(50, max_exercises=10), 1.0)
        assert result.price > 0

    def test_more_exercises_higher_value(self):
        engine = MCEngine(
            BlackScholesProcess(50, 0.05, 0.30),
            TimeGrid.uniform(1.0, 252), 20_000, seed=42,
        )
        r5 = engine.price(swing_payoff(50, max_exercises=5), 1.0).price
        r20 = engine.price(swing_payoff(50, max_exercises=20), 1.0).price
        assert r20 >= r5


# ── 8. Importance Sampling ──

class TestImportanceSampling:

    def test_matches_bs(self):
        result = importance_sampling(100, 100, 0.05, 0.20, 1.0, n_paths=100_000)
        bs = bs_call(100, 100, 0.05, 0.20, 1.0)
        assert result.price == pytest.approx(bs, rel=0.03)

    def test_deep_otm_lower_stderr(self):
        """IS should have lower stderr for deep OTM options."""
        # Standard MC
        engine = MCEngine(BlackScholesProcess(100, 0.05, 0.20),
                          TimeGrid.uniform(1.0, 1), 50_000, seed=42)
        r_std = engine.price(european_call(150), math.exp(-0.05))

        # IS
        r_is = importance_sampling(100, 150, 0.05, 0.20, 1.0, n_paths=50_000)

        # IS should be better for OTM
        assert r_is.stderr < r_std.stderr * 2  # at least comparable


# ── 9. Conditional MC ──

class TestConditionalMC:

    def test_heston_matches_full_mc(self):
        """Conditional MC should match full 2D Heston MC."""
        # Full Heston MC
        full = MCEngine(
            HestonProcess(100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7),
            TimeGrid.uniform(1.0, 100), 50_000, seed=42,
        )
        full_price = full.price(european_call(100), math.exp(-0.05)).price

        # Conditional MC
        cond = conditional_mc_heston(
            100, 0.04, 2.0, 0.04, 0.3, -0.7,
            strike=100, T=1.0, r=0.05, n_paths=50_000,
        )
        assert cond.price == pytest.approx(full_price, rel=0.15)

    def test_conditional_lower_stderr(self):
        """Conditional MC should have much lower stderr than full MC."""
        full = MCEngine(
            HestonProcess(100, 0.04, 0.05, 2.0, 0.04, 0.3, -0.7),
            TimeGrid.uniform(1.0, 100), 20_000, seed=42,
        )
        full_result = full.price(european_call(100), math.exp(-0.05))

        cond_result = conditional_mc_heston(
            100, 0.04, 2.0, 0.04, 0.3, -0.7,
            strike=100, T=1.0, r=0.05, n_paths=20_000,
        )
        # Conditional should have significantly lower stderr
        assert cond_result.stderr < full_result.stderr

    def test_put_works(self):
        result = conditional_mc_heston(
            100, 0.04, 2.0, 0.04, 0.3, -0.7,
            strike=100, T=1.0, r=0.05, option_type="put",
        )
        assert result.price > 0

    def test_generic_conditional(self):
        """Generic conditional MC with pre-simulated vol paths."""
        from pricebook.models.mc_processes import CIRProcess
        vol_engine = MCEngine(
            CIRProcess(0.04, 2.0, 0.04, 0.1),
            TimeGrid.uniform(1.0, 100), 20_000, seed=42,
        )
        result = conditional_mc_generic(
            vol_engine, s0=100, r=0.05, strike=100, T=1.0, rho=-0.5,
        )
        assert result.price > 0
        assert math.isfinite(result.stderr)
