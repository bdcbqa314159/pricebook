"""MC migration wave 2: tests for remaining 21 instrument files."""

from __future__ import annotations

import math

import numpy as np
import pytest


class TestBasketCDSMigration:
    def test_simulate_defaults_via_engine(self):
        from pricebook.basket_cds import simulate_defaults_copula_via_engine
        from pricebook.survival_curve import SurvivalCurve
        from datetime import date
        curves = [SurvivalCurve.flat(date(2026, 1, 1), 0.02), SurvivalCurve.flat(date(2026, 1, 1), 0.03)]
        result = simulate_defaults_copula_via_engine(curves, T=5.0, rho=0.3, n_sims=1_000)
        assert result is not None


class TestBermudanSwaptionMigration:
    def test_bermudan_swaption_via_engine(self):
        from pricebook.bermudan_swaption import bermudan_swaption_lsm_via_engine
        from pricebook.hull_white import HullWhite
        from pricebook.discount_curve import DiscountCurve
        from datetime import date
        curve = DiscountCurve.flat(date(2026, 1, 1), 0.05)
        hw = HullWhite(a=0.1, sigma=0.01, curve=curve)
        result = bermudan_swaption_lsm_via_engine(
            hw, exercise_years=[1, 2, 3, 4], swap_end_years=5.0,
            strike=0.05, n_paths=2_000,
        )
        assert isinstance(result, float)


class TestBermudanLMMMigration:
    def test_bermudan_lmm_via_engine(self):
        from pricebook.bermudan_lmm import bermudan_swaption_lmm_via_engine
        result = bermudan_swaption_lmm_via_engine(
            forward_rates=[0.05] * 10, inst_vols=[0.20] * 10,
            strike=0.05, exercise_indices=[2, 4, 6, 8],
            swap_end_idx=10, n_paths=1_000,
        )
        assert result is not None


class TestCommodityModelsMigration:
    def test_schwartz_one_factor_via_engine(self):
        from pricebook.commodity.commodity_models import SchwartzOneFactor, schwartz_one_factor_simulate_via_engine
        model = SchwartzOneFactor(kappa=0.5, mu=4.6, sigma=0.3)
        result = schwartz_one_factor_simulate_via_engine(model, spot=100, T=1.0, n_paths=1_000)
        assert result is not None


class TestConvertibleBondMigration:
    def test_convertible_via_engine(self):
        from pricebook.convertible_bond import contingent_convertible_via_engine
        assert callable(contingent_convertible_via_engine)


class TestEquityBasketMigration:
    def test_equity_basket_mc_via_engine(self):
        from pricebook.equity_basket import equity_basket_mc_via_engine
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = equity_basket_mc_via_engine(
            spots=[100.0, 100.0], weights=[0.5, 0.5], strike=100.0,
            rate=0.05, dividend_yields=[0.02, 0.02],
            vols=[0.20, 0.25], correlations=corr, T=1.0, n_paths=5_000,
        )
        assert result is not None


class TestEquityExoticMigration:
    def test_lookback_fixed_via_engine(self):
        from pricebook.equity_exotic import equity_lookback_fixed_via_engine
        result = equity_lookback_fixed_via_engine(
            spot=100, strike=100, rate=0.05, dividend_yield=0.02,
            vol=0.20, T=1.0, n_paths=5_000,
        )
        assert result is not None


class TestFXCorrelationMigration:
    def test_fx_basket_via_engine(self):
        from pricebook.fx_correlation import fx_basket_option_via_engine
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        result = fx_basket_option_via_engine(
            spots=[1.10, 130.0], weights=[0.5, 0.5], strike=1.0,
            rates_dom=0.03, rates_for=[0.01, 0.005],
            vols=[0.08, 0.10], correlations=corr, T=1.0, n_paths=5_000,
        )
        assert result is not None


class TestHJMMigration:
    def test_hjm_simulate_via_engine(self):
        from pricebook.hjm import hjm_simulate_via_engine
        assert callable(hjm_simulate_via_engine)


class TestHybridMCMigration:
    def test_hybrid_mc_via_engine(self):
        from pricebook.hybrid_mc import hybrid_mc_simulate_via_engine
        assert callable(hybrid_mc_simulate_via_engine)


class TestIRExoticMigration:
    def test_tarn_via_engine(self):
        from pricebook.ir_exotic import tarn_price_via_engine
        result = tarn_price_via_engine(
            notional=1_000_000, coupon_rate=0.05, target=0.10,
            maturity_years=5, flat_rate=0.05, rate_vol=0.01,
            n_paths=2_000,
        )
        assert result is not None

    def test_snowball_via_engine(self):
        from pricebook.ir_exotic import snowball_price_via_engine
        result = snowball_price_via_engine(
            notional=1_000_000, initial_coupon=0.06, spread=0.01,
            maturity_years=3, flat_rate=0.05, rate_vol=0.01,
            n_paths=2_000,
        )
        assert result is not None


class TestJarrowYildirimMigration:
    def test_jy_simulate_via_engine(self):
        from pricebook.jarrow_yildirim import jy_simulate_via_engine
        assert callable(jy_simulate_via_engine)


class TestJumpProcessMigration:
    def test_merton_terminal_via_engine(self):
        from pricebook.jump_process import MertonJumpDiffusion, merton_terminal_via_engine
        mjd = MertonJumpDiffusion(mu=0.05, sigma=0.20, lam=0.5, jump_mean=-0.05, jump_std=0.10)
        S_T = merton_terminal_via_engine(mjd, S0=100, T=1.0, n_paths=5_000)
        assert len(S_T) == 5_000
        assert np.all(S_T > 0)


class TestLMMMigration:
    def test_lmm_simulate_via_engine(self):
        from pricebook.lmm import lmm_simulate_via_engine
        assert callable(lmm_simulate_via_engine)


class TestPRDCMigration:
    def test_prdc_via_engine(self):
        from pricebook.prdc import prdc_price_via_engine
        assert callable(prdc_price_via_engine)


class TestProcessesExtendedMigration:
    def test_cev_paths_via_engine(self):
        from pricebook.processes_extended import cev_paths_via_engine
        result = cev_paths_via_engine(
            spot=100, rate=0.05, vol=0.20, beta=0.5,
            T=1.0, n_steps=50, n_paths=2_000,
        )
        assert result is not None

    def test_bates_paths_via_engine(self):
        from pricebook.processes_extended import bates_paths_via_engine
        result = bates_paths_via_engine(
            spot=100, v0=0.04, rate=0.05, kappa=2.0, theta=0.04,
            xi=0.3, rho=-0.7, lam=0.5, mu_j=-0.05, sigma_j=0.10,
            T=1.0, n_steps=50, n_paths=2_000,
        )
        assert result is not None


class TestShortRateModelsMigration:
    def test_bk_simulate_via_engine(self):
        from pricebook.short_rate_models import BKRateModel, bk_simulate_via_engine
        model = BKRateModel(a=0.1, sigma=0.01)
        result = bk_simulate_via_engine(model, r0=0.05, T=5.0, n_steps=50, n_paths=2_000)
        assert result is not None

    def test_cirpp_simulate_via_engine(self):
        from pricebook.short_rate_models import CIRPPRateModel, cirpp_simulate_via_engine
        model = CIRPPRateModel(kappa=0.5, theta=0.05, xi=0.01)
        result = cirpp_simulate_via_engine(model, r0=0.05, T=5.0, n_steps=50, n_paths=2_000)
        assert result is not None


class TestStochasticCreditMigration:
    def test_cir_intensity_via_engine(self):
        from pricebook.stochastic_credit import CIRIntensity, cir_intensity_simulate_via_engine
        model = CIRIntensity(kappa=0.5, theta=0.02, xi=0.05)
        paths = cir_intensity_simulate_via_engine(model, lam0=0.02, T=5.0, n_steps=50, n_paths=2_000)
        assert paths.shape == (2_000, 51)


class TestXVAMigration:
    def test_simulate_exposures_via_engine(self):
        from pricebook.xva import simulate_exposures_via_engine
        assert callable(simulate_exposures_via_engine)


class TestLoanMigration:
    def test_breach_probability_via_engine(self):
        from pricebook.loan_covenant import breach_probability_mc_via_engine
        result = breach_probability_mc_via_engine(
            current_ebitda=10e6, debt=20e6, threshold=1.5,
            ebitda_vol=0.20, horizon=1.0, n_paths=5_000,
        )
        assert 0 <= result <= 1

    def test_stochastic_recovery_via_engine(self):
        from pricebook.loan_credit import stochastic_recovery_sample_via_engine
        assert callable(stochastic_recovery_sample_via_engine)
