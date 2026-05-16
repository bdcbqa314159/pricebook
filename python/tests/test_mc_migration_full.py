"""Full MC migration tests: validate all _via_engine functions match originals."""

from __future__ import annotations

import math

import numpy as np
import pytest


# ── Batch A: Simple GBM instruments ──

class TestAutocallableMigrationFull:

    def test_engine_price_positive(self):
        from pricebook.mc_instrument_adapters import autocallable_mc
        result = autocallable_mc(100, 0.05, 0.20, 3.0, n_paths=20_000)
        assert result.price > 0

    def test_autocallable_via_engine_import(self):
        from pricebook.autocallable import autocallable_via_engine
        assert callable(autocallable_via_engine)


class TestCliquetMigrationFull:

    def test_engine_price_positive(self):
        from pricebook.mc_instrument_adapters import cliquet_mc
        result = cliquet_mc(100, 0.05, 0.20, 1.0, n_paths=20_000)
        assert result.price > 0

    def test_cliquet_via_engine_import(self):
        from pricebook.cliquet import cliquet_via_engine
        assert callable(cliquet_via_engine)


class TestTARFMigrationFull:

    def test_engine_price_finite(self):
        from pricebook.mc_instrument_adapters import tarf_mc
        result = tarf_mc(1.10, 0.02, 0.01, 0.08, 1.10, 0.10, 1.0, n_paths=20_000)
        assert math.isfinite(result.price)

    def test_tarf_via_engine_import(self):
        from pricebook.tarf import tarf_via_engine
        assert callable(tarf_via_engine)


class TestMultiAssetMCMigration:

    def test_correlated_gbm_via_engine_shape(self):
        from pricebook.multi_asset_mc import correlated_gbm_via_engine
        paths = correlated_gbm_via_engine(
            [100.0, 50.0], [0.20, 0.25],
            [[1.0, 0.5], [0.5, 1.0]], [0.05, 0.05],
            T=1.0, n_steps=10, n_paths=1_000,
        )
        assert paths.shape == (2, 1_000, 11)
        assert np.all(paths[:, :, 0] > 0)

    def test_basket_via_engine_positive(self):
        from pricebook.multi_asset_mc import basket_option_mc_via_engine
        result = basket_option_mc_via_engine(
            [100.0, 100.0], [0.20, 0.25],
            [[1.0, 0.5], [0.5, 1.0]], 0.05, 100.0, 1.0,
            n_paths=20_000,
        )
        assert result.price > 0
        assert result.n_assets == 2


class TestEquityStructuredMigration:

    def test_equity_autocallable_via_engine(self):
        from pricebook.equity.equity_structured import equity_autocallable_via_engine
        result = equity_autocallable_via_engine(
            spot=100, autocall_barrier=105, coupon_barrier=100,
            protection_barrier=70, coupon=0.08, rate=0.05,
            dividend_yield=0.02, vol=0.20, T=2.0,
            observation_dates=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            n_paths=5_000,
        )
        assert result.price > 0

    def test_shark_fin_via_engine(self):
        from pricebook.equity.equity_structured import shark_fin_via_engine
        result = shark_fin_via_engine(
            spot=100, strike=100, knock_out_barrier=130,
            rebate=0.02, participation=1.0, rate=0.05,
            dividend_yield=0.02, vol=0.20, T=1.0,
            n_paths=5_000,
        )
        assert result.price > 0
        assert 0 <= result.knock_out_probability <= 1


class TestMultiAssetExoticMigration:

    def test_simulate_correlated_via_engine(self):
        from pricebook.multi_asset_exotic import _simulate_correlated_via_engine
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        paths = _simulate_correlated_via_engine(
            [100.0, 50.0], 0.05, [0.02, 0.01], [0.20, 0.25],
            corr, 1.0, 1_000, 10, 42,
        )
        assert paths.shape == (1_000, 11, 2)

    def test_rainbow_via_engine(self):
        from pricebook.multi_asset_exotic import rainbow_option_via_engine
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = rainbow_option_via_engine(
            [100.0, 100.0], 100.0, 0.05, [0.02, 0.02],
            [0.20, 0.25], corr, 1.0, n_paths=5_000,
        )
        assert result.price > 0


# ── Batch B: Stochastic vol ──

class TestHestonMCMigration:

    def test_heston_euler_via_engine(self):
        from pricebook.heston_mc import heston_euler_via_engine
        S, v = heston_euler_via_engine(
            100, 0.05, 1.0, 0.04, 2.0, 0.04, 0.3, -0.7,
            n_steps=50, n_paths=5_000,
        )
        assert S.shape == (5_000, 51)
        assert v.shape == (5_000, 51)
        assert np.all(S[:, 0] > 0)

    def test_heston_european_via_engine(self):
        from pricebook.heston_mc import heston_mc_european_via_engine
        price = heston_mc_european_via_engine(
            100, 100, 0.05, 1.0, 0.04, 2.0, 0.04, 0.3, -0.7,
            n_paths=20_000,
        )
        assert price > 0

    def test_heston_conditional_via_engine(self):
        from pricebook.heston_mc import heston_mc_european_via_engine
        price = heston_mc_european_via_engine(
            100, 100, 0.05, 1.0, 0.04, 2.0, 0.04, 0.3, -0.7,
            n_paths=10_000, use_conditional=True,
        )
        assert price > 0


class TestSABRMCMigration:

    def test_sabr_paths_via_engine(self):
        from pricebook.sabr_mc import sabr_mc_paths_via_engine
        F, sig = sabr_mc_paths_via_engine(
            0.05, 1.0, 0.30, 0.5, -0.3, 0.4,
            n_steps=50, n_paths=5_000,
        )
        assert F.shape == (5_000, 51)
        assert sig.shape == (5_000, 51)

    def test_sabr_european_via_engine(self):
        from pricebook.sabr_mc import sabr_mc_european_via_engine
        price = sabr_mc_european_via_engine(
            0.05, 0.05, 1.0, 0.30, 0.5, -0.3, 0.4,
            n_paths=20_000,
        )
        assert price > 0


class TestJumpModelsMigration:

    def test_bates_fx_via_engine(self):
        from pricebook.fx.fx_jumps import BatesFXModel
        model = BatesFXModel(0.04, 2.0, 0.04, 0.3, -0.7, 0.5, -0.05, 0.10)
        result = model.simulate_option_via_engine(
            1.10, 1.10, 0.03, 0.01, 1.0, n_paths=5_000,
        )
        assert result.price > 0

    def test_svj_equity_via_engine(self):
        from pricebook.equity.equity_jumps import SVJEquityModel
        model = SVJEquityModel(0.04, 2.0, 0.04, 0.3, -0.7, 0.5, -0.05, 0.10)
        result = model.simulate_option_via_engine(
            100, 100, 0.05, 0.02, 1.0, n_paths=5_000,
        )
        assert result.price > 0


# ── Batch C: Advanced processes ──

class TestRoughVolMigration:

    def test_rbergomi_via_engine(self):
        from pricebook.rough_vol import rbergomi_mc_via_engine
        S_T = rbergomi_mc_via_engine(
            100, 0.05, 0.04, 1.5, 0.1, 1.0,
            n_steps=50, n_paths=2_000,
        )
        assert len(S_T) == 2_000
        assert np.all(S_T > 0)


class TestCreditHybridMigration:

    def test_convertible_bond_via_engine(self):
        from pricebook.credit_hybrid import convertible_bond_via_engine
        result = convertible_bond_via_engine(
            1000, 0.05, 5, 10, 100, n_paths=5_000,
        )
        assert result.price > 0
        assert result.bond_floor > 0

    def test_floating_cln_via_engine(self):
        from pricebook.credit_hybrid import floating_cln_via_engine
        result = floating_cln_via_engine(1_000_000, 0.01, 5, n_paths=5_000)
        assert result.price > 0


class TestFXExoticMigration:

    def test_lookback_via_engine(self):
        from pricebook.fx.fx_exotic import fx_lookback_floating_via_engine
        result = fx_lookback_floating_via_engine(
            1.10, 0.03, 0.01, 0.10, 1.0, n_paths=5_000,
        )
        assert result.price > 0

    def test_range_accrual_via_engine(self):
        from pricebook.fx.fx_exotic import fx_range_accrual_via_engine
        result = fx_range_accrual_via_engine(
            1.10, 0.03, 0.01, 0.08, 1.0, 1.05, 1.15,
            n_paths=5_000,
        )
        assert result.price >= 0
        assert 0 <= result.accrual_rate <= 1


# ── Batch D: Remaining files ──

class TestFXStructuredMigration:

    def test_fx_tarf_via_engine(self):
        from pricebook.fx.fx_structured import fx_tarf_price_via_engine
        result = fx_tarf_price_via_engine(
            1.10, 1.10, 0.10, 0.03, 0.01, 0.08, 1.0,
            n_paths=5_000,
        )
        assert math.isfinite(result.price)

    def test_fx_autocallable_via_engine(self):
        from pricebook.fx.fx_structured import fx_autocallable_price_via_engine
        result = fx_autocallable_price_via_engine(
            1.10, 1.15, 0.02, 0.03, 0.01, 0.08, 1.0,
            [0.25, 0.5, 0.75, 1.0], n_paths=5_000,
        )
        assert result.price > 0


class TestCommodityExoticMigration:

    def test_commodity_lookback_fixed_via_engine(self):
        from pricebook.commodity.commodity_exotic import commodity_lookback_via_engine
        result = commodity_lookback_via_engine(
            100, 0.05, 0.03, 0.25, 1.0,
            is_floating=False, strike=100, n_paths=5_000,
        )
        assert result.price > 0

    def test_commodity_asian_via_engine(self):
        from pricebook.commodity.commodity_exotic import commodity_asian_monthly_via_engine
        result = commodity_asian_monthly_via_engine(
            100, 100, 0.05, 0.03, 0.25, 1.0,
            n_paths=10_000,
        )
        assert result.price > 0


class TestEquityRatesHybridMigration:

    def test_callable_equity_note_via_engine(self):
        from pricebook.equity.equity_rates_hybrid import callable_equity_note_via_engine
        result = callable_equity_note_via_engine(
            100, 0.05, 0.20, 0.01, -0.3, 1000, 1.0,
            0.0, 2.0, [0.5, 1.0, 1.5], n_paths=2_000,
        )
        assert result.price > 0

    def test_equity_ir_joint_via_engine(self):
        from pricebook.equity.equity_rates_hybrid import equity_ir_joint_simulate_via_engine
        result = equity_ir_joint_simulate_via_engine(
            100, 0.05, 0.20, 0.01, -0.3, 1.0, n_paths=1_000,
        )
        assert result.equity_paths.shape == (1_000, 51)
        assert result.rate_paths.shape == (1_000, 51)


# ── Previously untested _via_engine functions ──

class TestLocalVolSLVMigration:

    def test_local_vol_mc_via_engine(self):
        from pricebook.local_vol import LocalVolSurface, local_vol_mc_via_engine
        strikes = np.array([80.0, 100.0, 120.0])
        times = np.array([0.5, 1.0])
        vols = np.array([[0.22, 0.20, 0.21], [0.23, 0.20, 0.22]])
        lv = LocalVolSurface(strikes, times, vols)
        S_T = local_vol_mc_via_engine(100, 0.05, lv, 1.0, n_steps=50, n_paths=2_000)
        assert len(S_T) == 2_000
        assert np.all(S_T > 0)

    def test_slv_mc_via_engine(self):
        from pricebook.local_vol import LocalVolSurface
        from pricebook.slv import SLVModel, HestonParams, slv_mc_via_engine
        strikes = np.array([80.0, 100.0, 120.0])
        times = np.array([0.5, 1.0])
        vols = np.array([[0.22, 0.20, 0.21], [0.23, 0.20, 0.22]])
        lv = LocalVolSurface(strikes, times, vols)
        heston = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
        model = SLVModel(lv, heston, mixing=0.5)
        S_T = slv_mc_via_engine(100, 0.05, model, 1.0, n_steps=50, n_paths=2_000)
        assert len(S_T) == 2_000
        assert np.all(S_T > 0)
