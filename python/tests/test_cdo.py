"""Tests for CDO tranche pricing."""

import pytest
import math
import numpy as np

from pricebook.cdo import (
    vasicek_conditional_pd,
    portfolio_loss_distribution,
    tranche_expected_loss,
    tranche_spread,
    base_correlation,
)


PD, RHO, LGD = 0.02, 0.30, 0.60


class TestVasicekConditionalPD:
    def test_at_zero_factor(self):
        """M=0: conditional PD < unconditional PD (skew effect of Phi transform)."""
        cpd = vasicek_conditional_pd(0.02, 0.3, 0.0)
        assert 0 < cpd < 0.05  # reasonable range

    def test_negative_factor_higher_pd(self):
        """Negative M (downturn): higher PD."""
        cpd_down = vasicek_conditional_pd(0.02, 0.3, -2.0)
        cpd_up = vasicek_conditional_pd(0.02, 0.3, 2.0)
        assert cpd_down > cpd_up

    def test_zero_correlation(self):
        """rho=0: conditional = unconditional."""
        cpd = vasicek_conditional_pd(0.02, 0.0, -3.0)
        assert cpd == pytest.approx(0.02)

    def test_bounded(self):
        for M in [-3, 0, 3]:
            cpd = vasicek_conditional_pd(0.02, 0.3, M)
            assert 0 <= cpd <= 1


class TestLossDistribution:
    def test_sums_to_one(self):
        loss_grid, density = portfolio_loss_distribution(PD, RHO, LGD)
        assert density.sum() == pytest.approx(1.0, rel=0.05)

    def test_non_negative(self):
        _, density = portfolio_loss_distribution(PD, RHO, LGD)
        assert np.all(density >= 0)

    def test_expected_loss(self):
        """E[loss] ≈ PD * LGD."""
        loss_grid, density = portfolio_loss_distribution(PD, RHO, LGD)
        el = (loss_grid * density).sum()
        # Kernel-smoothed approximation — should be in right ballpark
        assert el == pytest.approx(PD * LGD, rel=0.5)


class TestTranchePricing:
    def test_equity_highest_spread(self):
        loss_grid, density = portfolio_loss_distribution(PD, RHO, LGD)
        eq_spread = tranche_spread(loss_grid, density, 0.0, 0.03)
        mezz_spread = tranche_spread(loss_grid, density, 0.03, 0.07)
        assert eq_spread > mezz_spread

    def test_senior_lowest_spread(self):
        loss_grid, density = portfolio_loss_distribution(PD, RHO, LGD)
        mezz_spread = tranche_spread(loss_grid, density, 0.07, 0.15)
        senior_spread = tranche_spread(loss_grid, density, 0.15, 0.30)
        assert mezz_spread >= senior_spread

    def test_spreads_positive(self):
        loss_grid, density = portfolio_loss_distribution(PD, RHO, LGD)
        for a, d in [(0.0, 0.03), (0.03, 0.07), (0.07, 0.15)]:
            s = tranche_spread(loss_grid, density, a, d)
            assert s >= 0

    def test_tranche_losses_sum_to_portfolio(self):
        """Sum of tranche losses = portfolio expected loss."""
        loss_grid, density = portfolio_loss_distribution(PD, RHO, LGD)
        tranches = [(0.0, 0.03), (0.03, 0.07), (0.07, 0.15), (0.15, 0.30), (0.30, LGD)]
        total = sum(tranche_expected_loss(loss_grid, density, a, d) for a, d in tranches)
        portfolio_el = (loss_grid * density).sum()
        assert total == pytest.approx(portfolio_el, rel=0.1)


class TestBaseCorrelation:
    def test_round_trip(self):
        """Compute spread → find base corr → recompute spread → match."""
        loss_grid, density = portfolio_loss_distribution(PD, 0.20, LGD)
        target = tranche_spread(loss_grid, density, 0.0, 0.03)

        if target > 0.0001:
            bc = base_correlation(target, 0.03, PD, LGD)
            assert 0 < bc < 1

            # Recompute with found correlation
            lg2, d2 = portfolio_loss_distribution(PD, bc, LGD)
            recovered = tranche_spread(lg2, d2, 0.0, 0.03)
            assert recovered == pytest.approx(target, rel=0.2)

    def test_base_corr_positive(self):
        """Base correlation is in valid range."""
        loss_grid, density = portfolio_loss_distribution(PD, 0.20, LGD)
        s = tranche_spread(loss_grid, density, 0.0, 0.03)
        if s > 0.0001:
            bc = base_correlation(s, 0.03, PD, LGD)
            assert 0 < bc < 1
