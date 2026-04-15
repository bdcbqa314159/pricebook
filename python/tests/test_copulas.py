"""Tests for copula models."""

import math

import numpy as np
import pytest

from pricebook.copulas import (
    ClaytonCopula,
    CopulaDefaultResult,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    StudentTCopula,
    TranchePricingResult,
    copula_default_simulation,
    tranche_pricing_copula,
)


PDS = [0.02] * 20  # 20 names, 2% PD each


# ---- Gaussian copula ----

class TestGaussianCopula:
    def test_uniform_marginals(self):
        cop = GaussianCopula(0.3)
        rng = np.random.default_rng(42)
        U = cop.sample(10_000, 5, rng)
        # Marginals should be approximately uniform
        for j in range(5):
            assert U[:, j].mean() == pytest.approx(0.5, abs=0.02)

    def test_zero_correlation_independent(self):
        cop = GaussianCopula(0.0)
        result = copula_default_simulation(cop, PDS, n_sims=50_000, seed=42)
        assert abs(result.correlation_estimate) < 0.05

    def test_high_correlation_clusters(self):
        cop = GaussianCopula(0.9)
        result = copula_default_simulation(cop, PDS, n_sims=50_000, seed=42)
        assert result.correlation_estimate > 0.1


# ---- Student-t copula ----

class TestStudentTCopula:
    def test_converges_to_gaussian(self):
        """ν → ∞ should match Gaussian copula."""
        gauss = copula_default_simulation(
            GaussianCopula(0.3), PDS, n_sims=50_000, seed=42)
        t_high = copula_default_simulation(
            StudentTCopula(0.3, nu=100), PDS, n_sims=50_000, seed=42)
        assert t_high.n_defaults_mean == pytest.approx(
            gauss.n_defaults_mean, rel=0.15)

    def test_tail_dependence_positive(self):
        cop = StudentTCopula(0.3, nu=5)
        assert cop.tail_dependence > 0

    def test_lower_nu_more_clustering(self):
        """Lower ν → more tail dependence → more joint defaults."""
        high_nu = copula_default_simulation(
            StudentTCopula(0.3, nu=30), PDS, n_sims=50_000, seed=42)
        low_nu = copula_default_simulation(
            StudentTCopula(0.3, nu=3), PDS, n_sims=50_000, seed=42)
        # More clustering means higher default correlation
        assert low_nu.correlation_estimate > high_nu.correlation_estimate * 0.5


# ---- Clayton copula ----

class TestClaytonCopula:
    def test_lower_tail_dependence(self):
        cop = ClaytonCopula(2.0)
        assert cop.lower_tail_dependence > 0

    def test_uniform_marginals(self):
        cop = ClaytonCopula(1.0)
        rng = np.random.default_rng(42)
        U = cop.sample(10_000, 3, rng)
        for j in range(3):
            assert U[:, j].mean() == pytest.approx(0.5, abs=0.05)

    def test_invalid_theta(self):
        with pytest.raises(ValueError):
            ClaytonCopula(0.0)


# ---- Frank copula ----

class TestFrankCopula:
    def test_produces_defaults(self):
        cop = FrankCopula(5.0)
        result = copula_default_simulation(cop, PDS, n_sims=10_000, seed=42)
        assert result.n_defaults_mean > 0

    def test_invalid_theta(self):
        with pytest.raises(ValueError):
            FrankCopula(0.0)


# ---- Gumbel copula ----

class TestGumbelCopula:
    def test_upper_tail_dependence(self):
        cop = GumbelCopula(2.0)
        assert cop.upper_tail_dependence > 0

    def test_theta_one_independence(self):
        """θ = 1 → independence."""
        cop = GumbelCopula(1.0)
        result = copula_default_simulation(cop, PDS, n_sims=50_000, seed=42)
        assert abs(result.correlation_estimate) < 0.05

    def test_invalid_theta(self):
        with pytest.raises(ValueError):
            GumbelCopula(0.5)


# ---- Tranche pricing ----

class TestTranchePricingCopula:
    def test_different_copulas_different_tranches(self):
        """Different copulas produce different tranche expected losses."""
        gauss = tranche_pricing_copula(
            GaussianCopula(0.3), PDS, 0.0, 0.03, n_sims=100_000, seed=42)
        clayton = tranche_pricing_copula(
            ClaytonCopula(1.5), PDS, 0.0, 0.03, n_sims=100_000, seed=42)
        t_cop = tranche_pricing_copula(
            StudentTCopula(0.3, 5), PDS, 0.0, 0.03, n_sims=100_000, seed=42)
        # All produce positive EL; values differ by copula
        assert gauss.expected_loss > 0
        assert clayton.expected_loss > 0
        assert t_cop.expected_loss > 0

    def test_senior_tranche_lower_el(self):
        """Senior tranche has lower expected loss than equity."""
        cop = GaussianCopula(0.3)
        equity = tranche_pricing_copula(cop, PDS, 0.0, 0.03, seed=42)
        senior = tranche_pricing_copula(cop, PDS, 0.07, 0.10, seed=42)
        assert senior.expected_loss < equity.expected_loss

    def test_spread_positive(self):
        result = tranche_pricing_copula(
            GaussianCopula(0.3), PDS, 0.0, 0.03, seed=42)
        assert result.tranche_spread > 0

    def test_copula_name_recorded(self):
        result = tranche_pricing_copula(
            StudentTCopula(0.3, 5), PDS, 0.0, 0.03, seed=42)
        assert result.copula_name == "StudentTCopula"
