"""MC migration wave 4: final 6 files (copula, single-step, Greeks)."""

from __future__ import annotations

import math

import numpy as np
import pytest


class TestCDSSwaptionMigration:
    def test_pedersen_via_engine(self):
        from pricebook.cds_swaption import PedersenCDSSwaption, pedersen_price_mc_via_engine
        model = PedersenCDSSwaption(flat_hazard=0.02, spread_vol=0.40, flat_rate=0.05)
        result = pedersen_price_mc_via_engine(
            model, expiry=1.0, cds_maturity=5.0,
            strike_spread=0.02, n_paths=10_000,
        )
        assert result.premium >= 0

    def test_stochastic_intensity_via_engine(self):
        from pricebook.cds_swaption import StochasticIntensitySwaption, stochastic_intensity_swaption_via_engine
        model = StochasticIntensitySwaption(kappa=1.0, theta=0.02, xi=0.1)
        result = stochastic_intensity_swaption_via_engine(
            model, expiry=1.0, cds_maturity=5.0,
            strike_spread=0.02, n_paths=5_000,
        )
        assert result.premium >= 0


class TestCLNMigration:
    def test_cln_via_engine_import(self):
        from pricebook.cln import cln_price_stochastic_recovery_via_engine
        assert callable(cln_price_stochastic_recovery_via_engine)

    def test_basket_cln_via_engine_import(self):
        from pricebook.cln import basket_cln_price_mc_via_engine
        assert callable(basket_cln_price_mc_via_engine)


class TestCopulasMigration:
    def test_copula_default_via_engine(self):
        from pricebook.copulas import GaussianCopula, copula_default_simulation_via_engine
        copula = GaussianCopula(rho=0.3)
        result = copula_default_simulation_via_engine(
            copula, [0.02, 0.03, 0.01], n_sims=5_000,
        )
        assert result.n_defaults_mean >= 0

    def test_tranche_pricing_via_engine(self):
        from pricebook.copulas import GaussianCopula, tranche_pricing_copula_via_engine
        copula = GaussianCopula(rho=0.3)
        result = tranche_pricing_copula_via_engine(
            copula, [0.02] * 100, attach=0.03, detach=0.07,
            n_sims=10_000,
        )
        assert 0 <= result.expected_loss <= 1


class TestMCGreeksMigration:
    def test_pathwise_delta_via_engine(self):
        from pricebook.mc_greeks import pathwise_delta_via_engine
        result = pathwise_delta_via_engine(100, 100, 0.05, 0.20, 1.0, n_paths=10_000)
        assert 0 < result.value < 1

    def test_lr_delta_via_engine(self):
        from pricebook.mc_greeks import likelihood_ratio_delta_via_engine
        result = likelihood_ratio_delta_via_engine(100, 100, 0.05, 0.20, 1.0, n_paths=10_000)
        assert 0 < result.value < 1


class TestRecoveryPricingMigration:
    def test_correlated_default_recovery_via_engine(self):
        from pricebook.recovery_pricing import RecoverySpec, correlated_default_recovery_via_engine
        spec = RecoverySpec(mean=0.4, std=0.15, correlation_to_default=-0.3)
        result = correlated_default_recovery_via_engine(0.05, spec, n_sims=10_000)
        assert 0 < result.expected_loss < 1


class TestTailRiskMigration:
    def test_tail_risk_pricing_via_engine(self):
        from pricebook.tail_risk import tail_risk_pricing_via_engine
        result = tail_risk_pricing_via_engine(
            100, 70, 0.05, 1.0, n_paths=10_000,
        )
        assert result.deep_otm_put_price >= 0
        assert 0 <= result.tail_probability <= 1
