"""MC migration wave 3: final 12 files with inline Euler loops."""

from __future__ import annotations

import math

import numpy as np
import pytest


class TestDividendAdvancedMigration:
    def test_buhler_via_engine(self):
        from pricebook.equity.dividend_advanced import BuhlerStochasticDividend
        model = BuhlerStochasticDividend(r=0.05, q0=0.02, kappa_q=1.0,
                                         theta_q=0.02, xi_q=0.01, sigma_s=0.20)
        result = model.simulate_via_engine(100, 1.0, n_paths=2_000)
        assert result.spot_paths.shape[0] == 2_000
        assert result.mean_terminal_spot > 0


class TestHazardRateModelsMigration:
    def test_hw_hazard_via_engine(self):
        from pricebook.hazard_rate_models import HWHazardRate, hw_hazard_simulate_via_engine
        model = HWHazardRate(a=0.5, sigma=0.01)
        result = hw_hazard_simulate_via_engine(model, 0.02, 5.0, n_paths=2_000)
        assert result.lambda_paths.shape == (2_000, 101)
        assert 0 < result.survival_mc < 1

    def test_bk_hazard_via_engine(self):
        from pricebook.hazard_rate_models import BKHazardRate, bk_hazard_simulate_via_engine
        model = BKHazardRate(a=0.5, sigma=0.10)
        result = bk_hazard_simulate_via_engine(model, 0.02, 5.0, n_paths=2_000)
        assert 0 < result.survival_mc < 1

    def test_cirpp_hazard_via_engine(self):
        from pricebook.hazard_rate_models import CIRPlusPlus, cirpp_hazard_simulate_via_engine
        model = CIRPlusPlus(kappa=1.0, theta=0.02, xi=0.05)
        result = cirpp_hazard_simulate_via_engine(model, 0.02, 5.0, n_paths=2_000)
        assert 0 < result.survival_mc < 1

    def test_two_factor_via_engine(self):
        from pricebook.hazard_rate_models import TwoFactorIntensity, two_factor_simulate_via_engine
        model = TwoFactorIntensity(a1=0.1, sigma1=0.01, a2=1.0, sigma2=0.005)
        result = two_factor_simulate_via_engine(model, 5.0, n_paths=2_000)
        assert 0 < result.survival_mc < 1


class TestSpecialProcessMigration:
    def test_cir_via_engine(self):
        from pricebook.models.special_process import cir_sample_via_engine
        paths = cir_sample_via_engine(2.0, 0.04, 0.3, 0.04, 5.0, 50, 2_000)
        assert paths.shape == (2_000, 51)
        assert np.all(paths >= 0)

    def test_ou_via_engine(self):
        from pricebook.models.special_process import ou_sample_via_engine
        paths = ou_sample_via_engine(1.0, 0.0, 0.5, 0.0, 5.0, 50, 2_000)
        assert paths.shape == (2_000, 51)


class TestStochasticCorrelationMigration:
    def test_cir_correlation_via_engine(self):
        from pricebook.models.stochastic_correlation import CIRCorrelation, cir_correlation_simulate_via_engine
        model = CIRCorrelation(rho0=0.5, kappa=2.0, theta=0.3, sigma=0.2)
        result = cir_correlation_simulate_via_engine(model, 1.0, n_paths=2_000)
        assert result.rho_paths.shape[0] == 2_000
        assert -1 < result.mean_terminal_rho < 1

    def test_two_asset_stoch_corr_via_engine(self):
        from pricebook.models.stochastic_correlation import CIRCorrelation, simulate_two_asset_stoch_corr_via_engine
        model = CIRCorrelation(rho0=0.5, kappa=2.0, theta=0.3, sigma=0.2)
        result = simulate_two_asset_stoch_corr_via_engine(
            100, 50, 0.05, 0.02, 0.01, 0.20, 0.25, model, 1.0, n_paths=1_000)
        assert result.spot1_paths.shape[0] == 1_000


class TestVolTermStructureMigration:
    def test_bergomi_2f_via_engine(self):
        from pricebook.options.vol_term_structure import Bergomi2Factor, bergomi_2f_simulate_via_engine
        model = Bergomi2Factor(xi0=0.04, eta1=0.5, eta2=0.3)
        result = bergomi_2f_simulate_via_engine(model, 1.0, n_paths=2_000)
        assert result.vol_paths.shape[0] == 2_000
        assert result.mean_terminal_vol > 0


class TestVolVolDerivativesMigration:
    def test_vix_option_via_engine(self):
        from pricebook.options.vol_vol_derivatives import vix_option_price_via_engine
        result = vix_option_price_via_engine(
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3,
            T_option=0.25, T_variance=30/365, strike_vol=0.20,
            n_paths=5_000,
        )
        assert result.price > 0
        assert result.forward_vix > 0


class TestWeatherDerivativesMigration:
    def test_seasonal_ou_via_engine(self):
        from pricebook.options.weather_derivatives import SeasonalOUTemperature, seasonal_ou_simulate_via_engine
        model = SeasonalOUTemperature()
        result = seasonal_ou_simulate_via_engine(model, 90, n_paths=500)
        assert result.temperatures.shape == (500, 90)


class TestMultiAssetLocalVolMigration:
    def test_multi_asset_slv_via_engine(self):
        from pricebook.options.multi_asset_local_vol import multi_asset_slv_simulate_via_engine
        result = multi_asset_slv_simulate_via_engine(
            100, 50, 0.05, 0.02, 0.01, 0.20, 0.25,
            0.04, 2.0, 0.04, 0.3, 0.5, -0.3,
            1.0, n_paths=1_000,
        )
        assert result.mean_terminal_1 > 0
        assert result.mean_terminal_2 > 0


class TestRiskyFloatingMCMigration:
    def test_via_engine_import(self):
        from pricebook.risky_floating_mc import price_risky_frn_mc_via_engine
        assert callable(price_risky_frn_mc_via_engine)


class TestRoughEquityMigration:
    def test_rbergomi_equity_via_engine(self):
        from pricebook.models.rough_equity import rBergomiEquity, rbergomi_equity_simulate_via_engine
        model = rBergomiEquity(xi0=0.04, eta=1.5, H=0.1)
        result = rbergomi_equity_simulate_via_engine(
            model, 100, 0.05, 0.02, 1.0, n_paths=1_000, n_steps=50,
        )
        assert result.spot_paths.shape[0] == 1_000
        assert result.mean_terminal > 0


class TestLMMAdvancedMigration:
    def test_sabr_lmm_via_engine(self):
        from pricebook.models.lmm_advanced import SABRLMM, sabr_lmm_simulate_via_engine
        model = SABRLMM([0.05] * 5)
        result = sabr_lmm_simulate_via_engine(model, 2.0, n_steps=50, n_paths=1_000)
        assert result.forward_paths.shape == (1_000, 51, 5)


class TestFXSLVCalibrationMigration:
    def test_slv_barrier_via_engine(self):
        from pricebook.fx.fx_slv_calibration import LeverageFunction, slv_barrier_price_via_engine
        lev = LeverageFunction(np.array([0.0, 0.5, 1.0]),
                                np.array([0.5, 1.0, 2.0]),
                                np.array([[1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0],
                                          [1.0, 1.0, 1.0]]),
                                method="particle")
        result = slv_barrier_price_via_engine(
            1.10, 1.10, 1.20, 0.03, 0.01, lev,
            0.01, 2.0, 0.01, 0.1, -0.3, 1.0,
            n_paths=1_000, n_steps=50,
        )
        assert result.price >= 0
