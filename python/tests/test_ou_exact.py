"""Tests for OU process exact step."""

import pytest
import math
import numpy as np

from pricebook.models.mc_engine import MCEngine, TimeGrid
from pricebook.models.mc_processes import OUProcess


class TestOUExactStep:
    def test_mean_reversion(self):
        """Long-run mean should converge to theta."""
        proc = OUProcess(x0=0.0, kappa=2.0, theta=0.05, sigma=0.01)
        tg = TimeGrid.uniform(10.0, 100)
        e = MCEngine(proc, tg, n_paths=10_000, seed=42)
        paths = e.generate_paths()
        terminal_mean = float(paths[:, -1].mean())
        assert terminal_mean == pytest.approx(0.05, abs=0.005)

    def test_stationary_variance(self):
        """Stationary variance should be σ²/(2κ)."""
        kappa, sigma = 2.0, 0.10
        proc = OUProcess(x0=0.05, kappa=kappa, theta=0.05, sigma=sigma)
        tg = TimeGrid.uniform(20.0, 200)
        e = MCEngine(proc, tg, n_paths=20_000, seed=42)
        paths = e.generate_paths()
        terminal_var = float(paths[:, -1].var())
        expected_var = sigma**2 / (2 * kappa)
        assert terminal_var == pytest.approx(expected_var, rel=0.15)

    def test_deterministic_at_zero_vol(self):
        """σ=0 → deterministic mean reversion."""
        proc = OUProcess(x0=0.10, kappa=1.0, theta=0.05, sigma=0.0)
        tg = TimeGrid.uniform(5.0, 50)
        e = MCEngine(proc, tg, n_paths=100, seed=42)
        paths = e.generate_paths()
        # X(T) = X0*exp(-κT) + θ(1-exp(-κT))
        expected = 0.10 * math.exp(-5) + 0.05 * (1 - math.exp(-5))
        assert float(paths[0, -1]) == pytest.approx(expected, abs=0.001)

    def test_reproducible(self):
        proc1 = OUProcess(0.0, 2.0, 0.05, 0.01)
        proc2 = OUProcess(0.0, 2.0, 0.05, 0.01)
        tg = TimeGrid.uniform(1.0, 10)
        e1 = MCEngine(proc1, tg, n_paths=100, seed=42)
        e2 = MCEngine(proc2, tg, n_paths=100, seed=42)
        np.testing.assert_array_equal(e1.generate_paths(), e2.generate_paths())
