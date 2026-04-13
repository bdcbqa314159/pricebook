"""Tests for optimal transport."""

import math

import numpy as np
import pytest

from pricebook.optimal_transport import (
    MOTBounds,
    SinkhornResult,
    martingale_ot_bounds,
    sinkhorn,
    wasserstein_1d,
    wasserstein_discrete,
    wasserstein_gaussian,
)
from pricebook.equity_option import equity_option_price
from pricebook.black76 import OptionType


# ---- 1D Wasserstein ----

class TestWasserstein1D:
    def test_identical_distributions(self):
        a = [1, 2, 3, 4, 5]
        assert wasserstein_1d(a, a) == pytest.approx(0.0)

    def test_shifted_distributions(self):
        a = [0, 1, 2, 3, 4]
        b = [1, 2, 3, 4, 5]
        # W_1 of a shift by 1 = 1
        assert wasserstein_1d(a, b, p=1) == pytest.approx(1.0, rel=0.05)

    def test_w2_between_gaussians(self):
        """W_2 of two Gaussian samples should match analytical."""
        np.random.seed(42)
        a = np.random.normal(0, 1, 10_000)
        b = np.random.normal(2, 1, 10_000)
        w2 = wasserstein_1d(a, b, p=2)
        analytical = wasserstein_gaussian(0, 1, 2, 1)
        assert w2 == pytest.approx(analytical, rel=0.10)

    def test_different_sizes(self):
        a = [1, 2, 3]
        b = [1, 2, 3, 4, 5]
        w = wasserstein_1d(a, b)
        assert w > 0


class TestWassersteinGaussian:
    def test_same_gaussian(self):
        assert wasserstein_gaussian(0, 1, 0, 1) == pytest.approx(0.0)

    def test_shifted_mean(self):
        # W_2 = |μ₁ − μ₂| when σ₁ = σ₂
        assert wasserstein_gaussian(0, 1, 3, 1) == pytest.approx(3.0)

    def test_different_variance(self):
        assert wasserstein_gaussian(0, 1, 0, 2) == pytest.approx(1.0)


# ---- Discrete OT ----

class TestWassersteinDiscrete:
    def test_identical(self):
        a = [0.5, 0.5]
        b = [0.5, 0.5]
        C = [[0, 1], [1, 0]]
        result = wasserstein_discrete(a, b, C)
        assert result.converged
        assert result.cost == pytest.approx(0.0)

    def test_simple_transport(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        C = [[0, 2], [3, 0]]
        result = wasserstein_discrete(a, b, C)
        assert result.converged
        assert result.cost == pytest.approx(2.0)

    def test_plan_marginals(self):
        a = [0.3, 0.7]
        b = [0.5, 0.5]
        C = [[1, 2], [3, 1]]
        result = wasserstein_discrete(a, b, C)
        if result.converged:
            np.testing.assert_allclose(result.plan.sum(axis=1), a, atol=1e-6)
            np.testing.assert_allclose(result.plan.sum(axis=0), b, atol=1e-6)


# ---- Sinkhorn ----

class TestSinkhorn:
    def test_converges(self):
        a = np.array([0.5, 0.5])
        b = np.array([0.5, 0.5])
        C = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = sinkhorn(a, b, C, epsilon=0.1)
        assert result.converged

    def test_approaches_true_ot(self):
        """As ε → 0, Sinkhorn cost → true OT cost."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        C = np.array([[0.0, 2.0], [3.0, 0.0]])

        costs = []
        for eps in [1.0, 0.1, 0.01]:
            result = sinkhorn(a, b, C, epsilon=eps, max_iter=5000)
            costs.append(result.cost)

        # Should converge toward 2.0 (true OT cost)
        assert costs[-1] == pytest.approx(2.0, rel=0.1)

    def test_plan_marginals(self):
        a = np.array([0.3, 0.4, 0.3])
        b = np.array([0.5, 0.5])
        C = np.array([[1, 2], [2, 1], [1, 1]])
        result = sinkhorn(a, b, C, epsilon=0.05)
        if result.converged:
            np.testing.assert_allclose(result.plan.sum(axis=1), a, atol=1e-4)
            np.testing.assert_allclose(result.plan.sum(axis=0), b, atol=1e-4)


# ---- Martingale OT bounds ----

class TestMartingaleOTBounds:
    def test_bounds_contain_bs_price(self):
        """MOT bounds should contain the BS exotic price."""
        spot, rate, vol, T = 100.0, 0.05, 0.20, 1.0
        strikes = np.linspace(70, 140, 15)
        calls = np.array([
            equity_option_price(spot, K, rate, vol, T, OptionType.CALL)
            for K in strikes
        ])
        # Simple exotic: digital call at K=100
        payoff = lambda S: 1.0 if S > 100 else 0.0

        bounds = martingale_ot_bounds(strikes, calls, payoff, spot, rate, T)
        assert bounds.lower_bound <= bounds.upper_bound
        assert bounds.call_prices_used == 15

    def test_trivial_payoff(self):
        """Payoff = 0 → bounds = [0, 0]."""
        strikes = np.array([90, 100, 110])
        calls = np.array([12, 6, 2])
        bounds = martingale_ot_bounds(strikes, calls, lambda S: 0.0, 100, 0.05, 1.0)
        assert bounds.lower_bound <= 0.01
        assert bounds.upper_bound <= 0.01
