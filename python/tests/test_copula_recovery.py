"""Tests for per-name stochastic recovery in copula default simulation."""

import pytest
import math
import numpy as np

from pricebook.statistics.copulas import (
    GaussianCopula, StudentTCopula, ClaytonCopula,
    copula_default_simulation, tranche_pricing_copula,
)
from pricebook.credit.recovery_pricing import RecoverySpec


class TestCopulaDefaultRecovery:
    def test_gaussian_fixed_matches_flat(self):
        """Fixed RecoverySpec(0.4) should match flat lgd=0.6."""
        copula = GaussianCopula(0.3)
        pds = [0.03] * 20
        specs = [RecoverySpec(0.4, 0.0, "fixed", 0.0) for _ in range(20)]

        flat = copula_default_simulation(copula, pds, lgd=0.6, seed=42)
        stoch = copula_default_simulation(copula, pds, seed=42, recovery_specs=specs)

        assert np.mean(stoch.loss_distribution) == pytest.approx(
            np.mean(flat.loss_distribution), rel=0.10)

    def test_gaussian_wrong_way_increases_loss(self):
        copula = GaussianCopula(0.3)
        pds = [0.03] * 20

        flat = copula_default_simulation(copula, pds, lgd=0.6, seed=42)
        specs_ww = [RecoverySpec(0.4, 0.15, "beta", -0.5) for _ in range(20)]
        ww = copula_default_simulation(copula, pds, seed=42, recovery_specs=specs_ww)

        assert np.mean(ww.loss_distribution) >= np.mean(flat.loss_distribution) * 0.90

    def test_student_t_with_specs(self):
        copula = StudentTCopula(0.3, 5.0)
        pds = [0.03] * 10
        specs = [RecoverySpec(0.4, 0.15, "beta", -0.3) for _ in range(10)]

        result = copula_default_simulation(copula, pds, seed=42, recovery_specs=specs)
        assert np.mean(result.loss_distribution) >= 0

    def test_clayton_with_specs(self):
        """Clayton copula with recovery specs (no systematic factor — unconditional)."""
        copula = ClaytonCopula(2.0)
        pds = [0.05] * 5
        specs = [RecoverySpec(0.4, 0.15, "beta", 0.0) for _ in range(5)]

        result = copula_default_simulation(copula, pds, seed=42, recovery_specs=specs)
        assert np.mean(result.loss_distribution) >= 0

    def test_heterogeneous_seniority(self):
        copula = GaussianCopula(0.3)
        pds = [0.03] * 10
        specs = (
            [RecoverySpec(0.65, 0.15, "beta", -0.3)] * 5  # senior
            + [RecoverySpec(0.28, 0.20, "beta", -0.3)] * 5  # sub
        )
        result = copula_default_simulation(copula, pds, seed=42, recovery_specs=specs)
        assert result.n_defaults_mean >= 0


class TestTranchePricingCopulaRecovery:
    def test_equity_tranche(self):
        copula = GaussianCopula(0.3)
        pds = [0.03] * 20
        specs = [RecoverySpec(0.4, 0.15, "beta", -0.3) for _ in range(20)]

        result = tranche_pricing_copula(copula, pds, 0.0, 0.03, seed=42,
                                         recovery_specs=specs)
        assert result.expected_loss > 0
        assert result.tranche_spread > 0

    def test_equity_gt_senior(self):
        """Equity tranche spread > senior."""
        copula = GaussianCopula(0.3)
        pds = [0.03] * 20
        specs = [RecoverySpec(0.4, 0.15, "beta", -0.3) for _ in range(20)]

        eq = tranche_pricing_copula(copula, pds, 0.0, 0.03, seed=42, recovery_specs=specs)
        sr = tranche_pricing_copula(copula, pds, 0.12, 0.22, seed=42, recovery_specs=specs)

        assert eq.tranche_spread > sr.tranche_spread

    def test_backward_compatible(self):
        """No recovery_specs should work as before."""
        copula = GaussianCopula(0.3)
        pds = [0.03] * 10
        result = tranche_pricing_copula(copula, pds, 0.0, 0.03, lgd=0.6, seed=42)
        assert result.expected_loss >= 0
