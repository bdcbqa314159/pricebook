"""Tests for multi-factor HJM and LIBOR Market Model."""

import math
import pytest
import numpy as np

from pricebook.lmm import MultiFactorHJM, LMM
from pricebook.hjm import HJMModel
from pricebook.black76 import black76_price, OptionType


TENORS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
FLAT_RATE = 0.05
F0 = [FLAT_RATE] * len(TENORS)


# ---------------------------------------------------------------------------
# Multi-factor HJM
# ---------------------------------------------------------------------------


class TestMultiFactorHJM:
    def test_paths_shape(self):
        hjm = MultiFactorHJM(F0, TENORS)
        paths = hjm.simulate(T=1.0, n_steps=10, n_paths=100)
        assert paths.shape == (100, 11, len(TENORS))

    def test_initial_curve_preserved(self):
        hjm = MultiFactorHJM(F0, TENORS)
        paths = hjm.simulate(T=1.0, n_steps=10, n_paths=50)
        for p in range(50):
            np.testing.assert_array_almost_equal(paths[p, 0, :], F0)

    def test_single_factor_matches_hjm(self):
        """Single-factor multi-HJM should match existing HJMModel."""
        vol = 0.01
        mf = MultiFactorHJM(F0, TENORS, vol_funcs=[lambda t, x: vol])
        sf = HJMModel(F0, TENORS, constant_vol=vol)

        mf_paths = mf.simulate(T=2.0, n_steps=20, n_paths=5000, seed=42)
        sf_paths = sf.simulate(T=2.0, n_steps=20, n_paths=5000, seed=42)

        # Mean forward curves should be similar
        mf_mean = mf_paths[:, -1, :].mean(axis=0)
        sf_mean = sf_paths[:, -1, :].mean(axis=0)
        # Same seed but different implementations — compare qualitatively
        assert mf_mean[0] == pytest.approx(sf_mean[0], rel=0.15)

    def test_two_factors_more_volatile(self):
        """Two factors should produce more variance than one."""
        one_f = MultiFactorHJM(F0, TENORS, vol_funcs=[lambda t, x: 0.01])
        two_f = MultiFactorHJM(F0, TENORS, vol_funcs=[
            lambda t, x: 0.01, lambda t, x: 0.005,
        ])

        p1 = one_f.simulate(T=2.0, n_steps=20, n_paths=5000, seed=1)
        p2 = two_f.simulate(T=2.0, n_steps=20, n_paths=5000, seed=1)

        var1 = p1[:, -1, 0].var()
        var2 = p2[:, -1, 0].var()
        assert var2 > var1

    def test_discount_factors_decreasing(self):
        hjm = MultiFactorHJM(F0, TENORS)
        paths = hjm.simulate(T=5.0, n_steps=50, n_paths=1000)
        df = hjm.discount_factors(paths, dt=5.0/50)
        mean_df = df.mean(axis=0)
        # Discount factors should be decreasing on average
        assert all(mean_df[i] >= mean_df[i+1] - 0.01 for i in range(len(mean_df)-1))

    def test_three_factors(self):
        """Level + slope + curvature."""
        hjm = MultiFactorHJM(F0, TENORS, vol_funcs=[
            lambda t, x: 0.01,                           # level
            lambda t, x: 0.005 * math.exp(-0.5 * x),    # slope
            lambda t, x: 0.003 * x * math.exp(-0.5 * x),  # curvature
        ])
        paths = hjm.simulate(T=2.0, n_steps=20, n_paths=1000)
        assert paths.shape == (1000, 21, len(TENORS))


# ---------------------------------------------------------------------------
# LIBOR Market Model
# ---------------------------------------------------------------------------


LMM_TENORS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
LMM_RATES = [0.05] * len(LMM_TENORS)
LMM_VOLS = [0.20] * len(LMM_TENORS)


class TestLMM:
    def test_simulate_shape(self):
        lmm = LMM(LMM_RATES, LMM_TENORS, LMM_VOLS)
        L = lmm.simulate(n_paths=100)
        assert L.shape == (100, len(LMM_TENORS))

    def test_rates_positive(self):
        lmm = LMM(LMM_RATES, LMM_TENORS, LMM_VOLS)
        L = lmm.simulate(n_paths=1000)
        assert np.all(L >= 0)

    def test_mean_near_initial(self):
        """Mean forward rate should be near initial (martingale-ish)."""
        lmm = LMM(LMM_RATES, LMM_TENORS, LMM_VOLS)
        L = lmm.simulate(n_paths=50_000)
        mean_L = L.mean(axis=0)
        # Lognormal drift means E[L] slightly above L0
        for i in range(len(LMM_RATES)):
            assert mean_L[i] == pytest.approx(LMM_RATES[i], rel=0.20)

    def test_caplet_positive(self):
        lmm = LMM(LMM_RATES, LMM_TENORS, LMM_VOLS)
        price = lmm.caplet_price(fixing_idx=3, strike=0.05, df=0.95, n_paths=50_000)
        assert price > 0

    def test_caplet_matches_black(self):
        """LMM caplet should be close to Black caplet price."""
        tau = 0.25
        L0 = 0.05
        vol = 0.20
        K = 0.05
        T = 1.0
        df = math.exp(-0.05 * (T + tau))

        # Black caplet: tau * df * Black76(L0, K, vol, T)
        black_price = tau * df * black76_price(L0, K, vol, T, df=1.0, option_type=OptionType.CALL)

        lmm = LMM([L0] * 4, [0.25, 0.5, 0.75, 1.0], [vol] * 4, tau=tau)
        mc_price = lmm.caplet_price(fixing_idx=3, strike=K, df=df, n_paths=100_000)

        assert mc_price == pytest.approx(black_price, rel=0.15)


class TestRebonatoApprox:
    def test_positive(self):
        vol = LMM.rebonato_swaption_vol(
            np.array(LMM_VOLS), np.array(LMM_RATES), tau=0.25, T_expiry=1.0,
        )
        assert vol > 0

    def test_scales_with_input_vol(self):
        """Doubling forward vols should roughly double swaption vol."""
        vol1 = LMM.rebonato_swaption_vol(
            np.array([0.10] * 4), np.array([0.05] * 4), tau=0.25, T_expiry=1.0,
        )
        vol2 = LMM.rebonato_swaption_vol(
            np.array([0.20] * 4), np.array([0.05] * 4), tau=0.25, T_expiry=1.0,
        )
        assert vol2 == pytest.approx(2.0 * vol1, rel=0.01)
