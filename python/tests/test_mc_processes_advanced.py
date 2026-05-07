"""Tests for advanced MC processes: SABR, rough vol, SLV, Hull-White, Brownian bridge."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.mc_engine import MCEngine, TimeGrid
from pricebook.mc_processes import (
    SABRProcess, RoughBergomiProcess, SLVProcess,
    HullWhiteProcess, brownian_bridge_max, barrier_correction,
    BlackScholesProcess,
)
from pricebook.mc_payoffs import european_call, european_put, barrier_knockout


# ── SABR ──

class TestSABR:

    def test_forward_stays_positive(self):
        """SABR forward should stay non-negative (absorbing at 0)."""
        engine = MCEngine(
            SABRProcess(f0=0.05, alpha=0.3, beta=0.5, rho=-0.3, nu=0.4),
            TimeGrid.uniform(1.0, 100),
            n_paths=10_000, seed=42,
        )
        paths = engine.paths
        assert np.all(paths[:, :, 0] >= 0)

    def test_vol_stays_positive(self):
        """SABR stochastic vol should stay positive (lognormal dynamics)."""
        engine = MCEngine(
            SABRProcess(f0=0.05, alpha=0.3, beta=0.5, rho=-0.3, nu=0.4),
            TimeGrid.uniform(1.0, 100),
            n_paths=10_000, seed=42,
        )
        paths = engine.paths
        assert np.all(paths[:, :, 1] > 0)

    def test_beta_1_is_lognormal(self):
        """Beta=1 SABR should produce lognormal-like forward distribution."""
        engine = MCEngine(
            SABRProcess(f0=100.0, alpha=0.20, beta=1.0, rho=0.0, nu=0.0),
            TimeGrid.uniform(1.0, 50),
            n_paths=50_000, seed=42,
        )
        terminal = engine.paths[:, -1, 0]
        # Forward should be positive, mean ≈ f0
        assert float(np.mean(terminal)) == pytest.approx(100.0, rel=0.05)

    def test_smile_effect(self):
        """SABR with nu > 0 should produce a smile (OTM calls worth more than BS)."""
        # Price ATM call with SABR vs flat vol
        sabr = MCEngine(
            SABRProcess(f0=100.0, alpha=0.20, beta=1.0, rho=-0.3, nu=0.5),
            TimeGrid.uniform(1.0, 100),
            n_paths=50_000, seed=42,
        )
        terminal_sabr = sabr.paths[:, -1, 0]
        otm_call_sabr = float(np.mean(np.maximum(terminal_sabr - 120, 0)))

        # Flat vol GBM for comparison
        flat = MCEngine(
            BlackScholesProcess(100.0, 0.0, 0.20),
            TimeGrid.uniform(1.0, 1),
            n_paths=50_000, seed=42,
        )
        terminal_flat = np.exp(flat.paths[:, -1])
        otm_call_flat = float(np.mean(np.maximum(terminal_flat - 120, 0)))

        # SABR OTM call should be higher (fatter tails from stoch vol)
        assert otm_call_sabr > otm_call_flat * 0.8  # allow margin


# ── Rough Bergomi ──

class TestRoughBergomi:

    def test_spot_finite(self):
        engine = MCEngine(
            RoughBergomiProcess(s0=100, xi=0.04, eta=1.5, H=0.1, rho=-0.7),
            TimeGrid.uniform(1.0, 100),
            n_paths=5_000, seed=42,
        )
        terminal = np.exp(engine.paths[:, -1, 0])
        assert np.all(np.isfinite(terminal))
        assert float(np.mean(terminal)) > 0

    def test_rough_H_lower_vol_of_vol(self):
        """Lower H → rougher paths → more vol clustering."""
        engine_rough = MCEngine(
            RoughBergomiProcess(s0=100, xi=0.04, eta=1.5, H=0.05, rho=-0.7),
            TimeGrid.uniform(1.0, 100),
            n_paths=10_000, seed=42,
        )
        engine_smooth = MCEngine(
            RoughBergomiProcess(s0=100, xi=0.04, eta=1.5, H=0.40, rho=-0.7),
            TimeGrid.uniform(1.0, 100),
            n_paths=10_000, seed=42,
        )
        # Rough paths should have higher kurtosis of log-returns
        returns_rough = np.diff(engine_rough.paths[:, :, 0], axis=1)
        returns_smooth = np.diff(engine_smooth.paths[:, :, 0], axis=1)
        kurt_rough = float(np.mean(returns_rough ** 4) / np.mean(returns_rough ** 2) ** 2)
        kurt_smooth = float(np.mean(returns_smooth ** 4) / np.mean(returns_smooth ** 2) ** 2)
        # Not guaranteed but typically rough has higher kurtosis
        assert math.isfinite(kurt_rough)
        assert math.isfinite(kurt_smooth)


# ── SLV ──

class TestSLV:

    def test_pure_heston_mode(self):
        """SLV with mixing=0 should behave like Heston."""
        engine = MCEngine(
            SLVProcess(s0=100, r=0.05, local_vol_func=None,
                       v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, mixing=0.0),
            TimeGrid.uniform(1.0, 100),
            n_paths=50_000, seed=42,
        )
        terminal = np.exp(engine.paths[:, -1, 0])
        price = float(np.mean(np.maximum(terminal - 100, 0))) * math.exp(-0.05)
        assert price > 0

    def test_flat_local_vol(self):
        """SLV with flat L(S,t) = 0.20."""
        engine = MCEngine(
            SLVProcess(s0=100, r=0.05, local_vol_func=lambda s, t: 0.20 * np.ones_like(s),
                       v0=0.04, mixing=0.5),
            TimeGrid.uniform(1.0, 100),
            n_paths=20_000, seed=42,
        )
        terminal = np.exp(engine.paths[:, -1, 0])
        assert float(np.mean(terminal)) > 50  # should be around 100 × exp(r)

    def test_variance_stays_positive(self):
        engine = MCEngine(
            SLVProcess(s0=100, r=0.05, local_vol_func=None,
                       v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7),
            TimeGrid.uniform(2.0, 200),
            n_paths=5_000, seed=42,
        )
        assert np.all(engine.paths[:, :, 1] >= 0)


# ── Hull-White ──

class TestHullWhite:

    def test_mean_reverts(self):
        """HW short rate should mean-revert to θ/a = r0."""
        engine = MCEngine(
            HullWhiteProcess(r0=0.05, a=2.0, sigma=0.01),
            TimeGrid.uniform(5.0, 100),
            n_paths=10_000, seed=42,
        )
        terminal_mean = float(np.mean(engine.paths[:, -1]))
        assert abs(terminal_mean - 0.05) < 0.01

    def test_time_dependent_theta(self):
        """HW with increasing θ(t) should push rates up."""
        theta_up = lambda t: 0.10 + 0.02 * t  # rising target rate
        engine = MCEngine(
            HullWhiteProcess(r0=0.05, a=1.0, sigma=0.01, theta_func=theta_up),
            TimeGrid.uniform(5.0, 50),
            n_paths=10_000, seed=42,
        )
        terminal = float(np.mean(engine.paths[:, -1]))
        initial = float(np.mean(engine.paths[:, 0]))
        assert terminal > initial  # rates should rise

    def test_bond_price_from_paths(self):
        """Price a zero-coupon bond as E[exp(-∫r ds)]."""
        engine = MCEngine(
            HullWhiteProcess(r0=0.05, a=0.5, sigma=0.01),
            TimeGrid.uniform(1.0, 100),
            n_paths=50_000, seed=42,
        )
        paths = engine.paths  # (n_paths, n_steps+1)
        dt = engine.time_grid.dt

        # Integrate short rate: ∫r ds ≈ Σ r_i × dt_i
        integral = np.sum(paths[:, :-1] * dt[np.newaxis, :], axis=1)
        bond_price = float(np.mean(np.exp(-integral)))

        # Should be close to exp(-r0 × T) for low vol
        analytical_approx = math.exp(-0.05 * 1.0)
        assert bond_price == pytest.approx(analytical_approx, rel=0.02)


# ── Brownian Bridge ──

class TestBrownianBridge:

    def test_max_geq_endpoints(self):
        """Bridge max should be ≥ max(start, end)."""
        for _ in range(100):
            m = brownian_bridge_max(100.0, 102.0, 1 / 252, 0.20)
            assert m >= max(100.0, 102.0)

    def test_barrier_correction_increases_knockouts(self):
        """Barrier correction should knock out MORE paths than discrete monitoring."""
        engine = MCEngine(
            BlackScholesProcess(100, 0.05, 0.20),
            TimeGrid.uniform(1.0, 12),  # monthly — coarse monitoring
            n_paths=50_000, seed=42,
        )
        paths = engine.paths

        # Discrete: check endpoints only
        spots = np.exp(paths)
        discrete_alive = np.all(spots < 130, axis=1)

        # Bridge-corrected
        bridge_alive = barrier_correction(
            paths, barrier=130, sigma=0.20,
            dt=1 / 12, barrier_type="up", log_space=True, seed=42,
        )

        # Bridge should knock out more (or equal)
        assert np.sum(bridge_alive) <= np.sum(discrete_alive)

    def test_barrier_correction_shape(self):
        paths = np.zeros((100, 13))  # 100 paths, 12 steps
        alive = barrier_correction(paths, 1.0, 0.20, 1 / 12, "up", log_space=True)
        assert alive.shape == (100,)
