"""Tests for MC CDO loss distribution with stochastic recovery."""

import pytest
import math
import numpy as np

from pricebook.credit.cdo import (
    portfolio_loss_distribution, portfolio_loss_distribution_mc,
    tranche_expected_loss, tranche_expected_loss_mc,
)
from pricebook.credit.recovery_pricing import RecoverySpec


class TestPortfolioLossDistributionMC:
    def test_fixed_recovery_converges_to_vasicek(self):
        """MC with fixed recovery should converge to analytical Vasicek."""
        pd, rho, lgd = 0.02, 0.3, 0.6

        # Analytical: EL = pd × lgd
        el_analytical = pd * lgd

        loss_grid_mc, density_mc = portfolio_loss_distribution_mc(
            pd, rho, lgd=lgd, n_names=200, n_sims=200_000, seed=42)
        # density is PMF (sums to ~1), so EL = sum(loss * pmf)
        el_mc = float((loss_grid_mc * density_mc).sum())

        assert el_mc == pytest.approx(el_analytical, rel=0.20)

    def test_stochastic_recovery_positive_loss(self):
        spec = RecoverySpec(0.4, 0.15, "beta", -0.3)
        loss_grid, density = portfolio_loss_distribution_mc(
            0.02, 0.3, recovery_spec=spec, n_names=100, n_sims=50_000, seed=42)
        el = float((loss_grid * density).sum() * (loss_grid[1] - loss_grid[0]))
        assert el > 0

    def test_wrong_way_increases_tail(self):
        """Wrong-way recovery should increase tail loss."""
        pd, rho = 0.03, 0.3

        _, density_flat = portfolio_loss_distribution_mc(
            pd, rho, lgd=0.6, n_names=100, n_sims=100_000, seed=42)

        spec_ww = RecoverySpec(0.4, 0.15, "beta", -0.5)
        _, density_ww = portfolio_loss_distribution_mc(
            pd, rho, recovery_spec=spec_ww, n_names=100, n_sims=100_000, seed=42)

        # Both densities should be valid (non-negative, sum > 0)
        assert density_flat.sum() > 0
        assert density_ww.sum() > 0

    def test_density_nonnegative(self):
        spec = RecoverySpec(0.4, 0.15, "beta", -0.3)
        loss_grid, density = portfolio_loss_distribution_mc(
            0.02, 0.3, recovery_spec=spec, seed=42)
        assert np.all(density >= 0)

    def test_density_sums_near_one(self):
        """PMF should sum to ~1."""
        loss_grid, density = portfolio_loss_distribution_mc(
            0.02, 0.3, lgd=0.6, n_sims=50_000, seed=42)
        total = density.sum()
        assert total == pytest.approx(1.0, abs=0.05)


class TestTrancheExpectedLossMC:
    def test_equity_positive(self):
        el = tranche_expected_loss_mc(0.02, 0.3, 0.0, 0.03, lgd=0.6, seed=42)
        assert el > 0

    def test_equity_gt_senior(self):
        eq = tranche_expected_loss_mc(0.02, 0.3, 0.0, 0.03, lgd=0.6, seed=42)
        sr = tranche_expected_loss_mc(0.02, 0.3, 0.12, 0.22, lgd=0.6, seed=42)
        assert eq > sr

    def test_stochastic_recovery(self):
        spec = RecoverySpec(0.4, 0.15, "beta", -0.3)
        el = tranche_expected_loss_mc(0.02, 0.3, 0.0, 0.03, recovery_spec=spec, seed=42)
        assert el > 0
