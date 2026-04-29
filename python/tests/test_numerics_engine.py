"""Tests for Layer D numerics: multi-asset MC, pathwise Greeks."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.local_vol import calibrate_dupire, LocalVolSurface
from pricebook.multi_asset_mc import CorrelatedGBM, basket_option_mc, worst_of_mc
from pricebook.sabr import calibrate_sabr_smile, sabr_implied_vol
from pricebook.pathwise_greeks import pathwise_asian_delta, pathwise_european_delta
from pricebook.black76 import OptionType
from pricebook.gbm import GBMGenerator
from pricebook.rng import PseudoRandom


# ---- D1: Dupire calibration ----

class TestDupireCalibration:

    def test_flat_vol_roundtrip(self):
        """Flat implied vol → flat local vol."""
        strikes = [80, 90, 100, 110, 120]
        times = [0.25, 0.50, 1.0]
        flat = 0.20
        vols = [[flat] * 5 for _ in range(3)]
        lv = calibrate_dupire(spot=100, rate=0.03, strikes=strikes, times=times,
                               implied_vols=vols)
        # Local vol should be close to flat for interior points
        mid_vol = lv.vol(100, 0.5)
        assert mid_vol == pytest.approx(flat, abs=0.02)

    def test_smile_produces_different_local_vol(self):
        """Non-flat smile → non-constant local vol."""
        strikes = [80, 90, 100, 110, 120]
        times = [0.25, 0.50, 1.0]
        vols = [[0.30, 0.25, 0.20, 0.22, 0.25]] * 3  # skew
        lv = calibrate_dupire(spot=100, rate=0.03, strikes=strikes, times=times,
                               implied_vols=vols)
        # Local vol at low strike should differ from high strike
        v_low = lv.vol(85, 0.5)
        v_high = lv.vol(115, 0.5)
        assert v_low != pytest.approx(v_high, abs=0.01)

    def test_returns_local_vol_surface(self):
        strikes = [90, 100, 110]
        times = [0.5, 1.0]
        vols = [[0.20, 0.20, 0.20], [0.20, 0.20, 0.20]]
        lv = calibrate_dupire(spot=100, rate=0.03, strikes=strikes, times=times,
                               implied_vols=vols)
        assert isinstance(lv, LocalVolSurface)


# ---- D2: SABR calibration ----

class TestSABRCalibration:

    def test_basic_calibration(self):
        """Calibrate SABR to a synthetic smile."""
        forward = 100.0
        strikes = [85, 90, 95, 100, 105, 110, 115]
        # Generate synthetic vols from known SABR params
        true_alpha, true_beta, true_rho, true_nu = 0.20, 0.5, -0.3, 0.4
        T = 1.0
        market_vols = [sabr_implied_vol(forward, k, T, true_alpha, true_beta, true_rho, true_nu)
                       for k in strikes]

        result = calibrate_sabr_smile(forward, strikes, market_vols, T, beta=true_beta)
        assert result["rmse"] * 10_000 < 0.5  # sub-0.5bp RMSE
        assert result["alpha"] == pytest.approx(true_alpha, abs=0.01)
        assert result["rho"] == pytest.approx(true_rho, abs=0.05)
        assert result["nu"] == pytest.approx(true_nu, abs=0.05)

    def test_reprice_errors(self):
        forward = 100.0
        strikes = [90, 95, 100, 105, 110]
        T = 1.0
        market_vols = [sabr_implied_vol(forward, k, T, 0.20, 0.5, -0.2, 0.3)
                       for k in strikes]
        result = calibrate_sabr_smile(forward, strikes, market_vols, T)
        assert result["max_error_bp"] < 1.0  # sub-1bp

    def test_returns_all_fields(self):
        forward = 100.0
        strikes = [95, 100, 105]
        market_vols = [0.22, 0.20, 0.21]
        result = calibrate_sabr_smile(forward, strikes, market_vols, T=1.0)
        assert "alpha" in result
        assert "rho" in result
        assert "nu" in result
        assert "reprice_errors_bp" in result


# ---- D3: CorrelatedGBM ----

class TestCorrelatedGBM:

    def test_single_asset(self):
        gen = CorrelatedGBM(spots=[100], vols=[0.20], corr_matrix=[[1.0]], rates=0.03)
        paths = gen.generate(T=1.0, n_steps=50, n_paths=1000)
        assert paths.shape == (1, 1000, 51)
        assert paths[0, :, 0].mean() == pytest.approx(100.0)

    def test_two_assets(self):
        gen = CorrelatedGBM(
            spots=[100, 50], vols=[0.20, 0.30],
            corr_matrix=[[1.0, 0.5], [0.5, 1.0]], rates=0.03)
        paths = gen.generate(T=1.0, n_steps=50, n_paths=10_000)
        assert paths.shape == (2, 10_000, 51)

    def test_correlation_preserved(self):
        """Generated returns should have correlation close to input."""
        rho = 0.7
        gen = CorrelatedGBM(
            spots=[100, 100], vols=[0.20, 0.20],
            corr_matrix=[[1.0, rho], [rho, 1.0]], rates=0.03)
        paths = gen.generate(T=1.0, n_steps=1, n_paths=100_000)
        # Log-returns
        r1 = np.log(paths[0, :, 1] / paths[0, :, 0])
        r2 = np.log(paths[1, :, 1] / paths[1, :, 0])
        empirical_corr = np.corrcoef(r1, r2)[0, 1]
        assert empirical_corr == pytest.approx(rho, abs=0.03)

    def test_marginal_distribution(self):
        """Each asset's terminal should be lognormal with correct mean."""
        gen = CorrelatedGBM(spots=[100], vols=[0.20], corr_matrix=[[1.0]], rates=0.05)
        terminals = gen.terminal(T=1.0, n_paths=200_000)
        expected_mean = 100 * math.exp(0.05)  # E[S(T)] = S₀ exp(r T)
        assert terminals.mean() == pytest.approx(expected_mean, rel=0.01)

    def test_non_pd_raises(self):
        with pytest.raises(ValueError, match="positive definite"):
            CorrelatedGBM(spots=[100, 100], vols=[0.2, 0.2],
                          corr_matrix=[[1.0, 1.5], [1.5, 1.0]])

    def test_mismatched_dims_raises(self):
        with pytest.raises(ValueError):
            CorrelatedGBM(spots=[100, 100], vols=[0.2],
                          corr_matrix=[[1.0, 0.5], [0.5, 1.0]])


class TestBasketOption:

    def test_call_price(self):
        gen = CorrelatedGBM(
            spots=[100, 100], vols=[0.20, 0.25],
            corr_matrix=[[1.0, 0.5], [0.5, 1.0]], rates=0.03)
        r = basket_option_mc(gen, strike=100, T=1.0, n_paths=50_000)
        assert math.isfinite(r.price)
        assert r.price > 0
        assert r.n_assets == 2

    def test_higher_corr_higher_price(self):
        """Higher correlation → higher basket option price (less diversification)."""
        gen_low = CorrelatedGBM(
            spots=[100, 100], vols=[0.20, 0.20],
            corr_matrix=[[1.0, 0.1], [0.1, 1.0]], rates=0.03)
        gen_high = CorrelatedGBM(
            spots=[100, 100], vols=[0.20, 0.20],
            corr_matrix=[[1.0, 0.9], [0.9, 1.0]], rates=0.03)
        r_low = basket_option_mc(gen_low, strike=100, T=1.0, n_paths=50_000)
        r_high = basket_option_mc(gen_high, strike=100, T=1.0, n_paths=50_000)
        assert r_high.price > r_low.price


class TestWorstOf:

    def test_probability(self):
        gen = CorrelatedGBM(
            spots=[100, 100], vols=[0.25, 0.30],
            corr_matrix=[[1.0, 0.5], [0.5, 1.0]], rates=0.03)
        prob = worst_of_mc(gen, T=1.0, barrier=0.70, n_paths=50_000)
        assert 0 <= prob <= 1

    def test_higher_vol_more_breaches(self):
        gen_low = CorrelatedGBM(
            spots=[100], vols=[0.10], corr_matrix=[[1.0]], rates=0.03)
        gen_high = CorrelatedGBM(
            spots=[100], vols=[0.50], corr_matrix=[[1.0]], rates=0.03)
        p_low = worst_of_mc(gen_low, T=1.0, barrier=0.70, n_paths=50_000)
        p_high = worst_of_mc(gen_high, T=1.0, barrier=0.70, n_paths=50_000)
        assert p_high > p_low


# ---- Pathwise Greeks ----

class TestPathwiseGreeks:

    def test_asian_delta_positive(self):
        """Call delta should be positive."""
        gen = GBMGenerator(spot=100, rate=0.03, vol=0.20)
        rng = PseudoRandom(seed=42)
        paths = gen.generate(T=1.0, n_steps=12, n_paths=100_000, rng=rng)
        delta = pathwise_asian_delta(paths, strike=100, spot=100, rate=0.03, T=1.0)
        assert delta > 0

    def test_asian_delta_vs_bump(self):
        """Pathwise delta should match bump-and-reprice within MC error."""
        gen = GBMGenerator(spot=100, rate=0.03, vol=0.20)
        rng = PseudoRandom(seed=42)
        paths = gen.generate(T=1.0, n_steps=12, n_paths=200_000, rng=rng)
        pw_delta = pathwise_asian_delta(paths, strike=100, spot=100, rate=0.03, T=1.0)

        # Bump-and-reprice
        df = math.exp(-0.03)
        avg = paths[:, 1:].mean(axis=1)
        base_price = float(df * np.maximum(avg - 100, 0).mean())

        bump = 1.0
        gen_up = GBMGenerator(spot=101, rate=0.03, vol=0.20)
        paths_up = gen_up.generate(T=1.0, n_steps=12, n_paths=200_000, rng=PseudoRandom(seed=42))
        avg_up = paths_up[:, 1:].mean(axis=1)
        up_price = float(df * np.maximum(avg_up - 100, 0).mean())

        bump_delta = (up_price - base_price) / bump
        assert pw_delta == pytest.approx(bump_delta, abs=0.02)

    def test_european_delta(self):
        gen = GBMGenerator(spot=100, rate=0.03, vol=0.20)
        rng = PseudoRandom(seed=42)
        paths = gen.generate(T=1.0, n_steps=1, n_paths=200_000, rng=rng)
        terminals = paths[:, -1]
        delta = pathwise_european_delta(terminals, strike=100, spot=100, rate=0.03, T=1.0)
        # ATM call delta ≈ 0.5
        assert 0.4 < delta < 0.7

    def test_put_delta_negative(self):
        gen = GBMGenerator(spot=100, rate=0.03, vol=0.20)
        rng = PseudoRandom(seed=42)
        paths = gen.generate(T=1.0, n_steps=1, n_paths=100_000, rng=rng)
        terminals = paths[:, -1]
        delta = pathwise_european_delta(terminals, strike=100, spot=100, rate=0.03, T=1.0,
                                         option_type=OptionType.PUT)
        assert delta < 0
