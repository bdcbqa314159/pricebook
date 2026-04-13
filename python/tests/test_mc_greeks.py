"""Tests for MC Greeks, optimal MLMC, LSM improvements, dual bound."""

import math

import numpy as np
import pytest

from pricebook.equity_option import equity_delta, equity_vega
from pricebook.mc_greeks import (
    MCGreekResult,
    OptimalMLMCResult,
    dual_upper_bound,
    likelihood_ratio_delta,
    likelihood_ratio_vega,
    lsm_with_basis,
    optimal_mlmc,
    pathwise_delta,
    pathwise_vega,
)


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0


# ---- Pathwise Greeks ----

class TestPathwiseGreeks:
    def test_delta_matches_bs(self):
        """IPA delta matches BS analytical delta."""
        result = pathwise_delta(SPOT, STRIKE, RATE, VOL, T,
                                n_paths=200_000, seed=42)
        bs_delta = equity_delta(SPOT, STRIKE, RATE, VOL, T)
        assert result.value == pytest.approx(bs_delta, rel=0.05)
        assert result.method == "pathwise_delta"

    def test_vega_matches_bs(self):
        """IPA vega matches BS analytical vega."""
        result = pathwise_vega(SPOT, STRIKE, RATE, VOL, T,
                               n_paths=200_000, seed=42)
        bs_vega = equity_vega(SPOT, STRIKE, RATE, VOL, T)
        assert result.value == pytest.approx(bs_vega, rel=0.05)

    def test_delta_between_zero_and_one(self):
        result = pathwise_delta(SPOT, STRIKE, RATE, VOL, T, seed=42)
        assert 0 < result.value < 1

    def test_std_error_reported(self):
        result = pathwise_delta(SPOT, STRIKE, RATE, VOL, T, seed=42)
        assert result.std_error > 0
        assert result.n_paths == 100_000


# ---- Likelihood ratio Greeks ----

class TestLikelihoodRatioGreeks:
    def test_lr_delta_matches_bs(self):
        """LR delta matches BS delta (higher variance than IPA)."""
        result = likelihood_ratio_delta(SPOT, STRIKE, RATE, VOL, T,
                                        n_paths=500_000, seed=42)
        bs_delta = equity_delta(SPOT, STRIKE, RATE, VOL, T)
        assert result.value == pytest.approx(bs_delta, rel=0.10)

    def test_lr_vega_matches_bs(self):
        result = likelihood_ratio_vega(SPOT, STRIKE, RATE, VOL, T,
                                       n_paths=500_000, seed=42)
        bs_vega = equity_vega(SPOT, STRIKE, RATE, VOL, T)
        assert result.value == pytest.approx(bs_vega, rel=0.10)

    def test_lr_higher_variance_than_pathwise(self):
        """LR method should have higher std error than pathwise for calls."""
        ipa = pathwise_delta(SPOT, STRIKE, RATE, VOL, T,
                             n_paths=100_000, seed=42)
        lr = likelihood_ratio_delta(SPOT, STRIKE, RATE, VOL, T,
                                    n_paths=100_000, seed=42)
        assert lr.std_error > ipa.std_error


# ---- Optimal MLMC ----

class TestOptimalMLMC:
    def _simulate_level(self, level, n_paths, seed):
        """GBM MLMC level simulator."""
        rng = np.random.default_rng(seed)
        n_steps = 2 ** level
        dt = T / n_steps

        S = np.full(n_paths, SPOT)
        for _ in range(n_steps):
            dW = rng.standard_normal(n_paths) * math.sqrt(dt)
            S = S * np.exp((RATE - 0.5 * VOL**2) * dt + VOL * dW)

        df = math.exp(-RATE * T)
        fine = df * np.maximum(S - STRIKE, 0.0)

        if level == 0:
            return fine, None

        # Coarse: same BM but half the steps
        rng2 = np.random.default_rng(seed)
        n_coarse = n_steps // 2
        dt_c = T / n_coarse
        S_c = np.full(n_paths, SPOT)
        for _ in range(n_coarse):
            dW = rng2.standard_normal(n_paths) * math.sqrt(dt_c)
            S_c = S_c * np.exp((RATE - 0.5 * VOL**2) * dt_c + VOL * dW)

        coarse = df * np.maximum(S_c - STRIKE, 0.0)
        return fine, coarse

    def test_produces_result(self):
        result = optimal_mlmc(
            payoff=None,
            simulate_level=self._simulate_level,
            T=T, epsilon=0.5, L_max=5, n_pilot=500, seed=42,
        )
        assert result.price > 0
        assert result.n_levels >= 1
        assert len(result.paths_per_level) == result.n_levels

    def test_price_reasonable(self):
        from pricebook.equity_option import equity_option_price
        from pricebook.black76 import OptionType
        bs = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)
        result = optimal_mlmc(None, self._simulate_level,
                              T, epsilon=0.5, L_max=5, n_pilot=1000, seed=42)
        assert result.price == pytest.approx(bs, rel=0.15)


# ---- LSM with orthogonal bases ----

class TestLSMWithBasis:
    def _gbm_paths(self, n_paths=10_000, n_steps=50):
        rng = np.random.default_rng(42)
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = SPOT
        for i in range(n_steps):
            dW = rng.standard_normal(n_paths) * math.sqrt(dt)
            paths[:, i + 1] = paths[:, i] * np.exp(
                (RATE - 0.5 * VOL**2) * dt + VOL * dW
            )
        return paths

    def test_laguerre_positive_price(self):
        paths = self._gbm_paths()
        price = lsm_with_basis(paths, STRIKE, RATE, T,
                               basis_type="laguerre", degree=3)
        assert price > 0

    def test_chebyshev_positive_price(self):
        paths = self._gbm_paths()
        price = lsm_with_basis(paths, STRIKE, RATE, T,
                               basis_type="chebyshev", degree=3)
        assert price > 0

    def test_american_put_exceeds_european(self):
        from pricebook.equity_option import equity_option_price
        from pricebook.black76 import OptionType
        euro = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)
        paths = self._gbm_paths()
        american = lsm_with_basis(paths, STRIKE, RATE, T,
                                  basis_type="laguerre", is_call=False)
        assert american >= euro * 0.95  # allow MC noise


# ---- Dual upper bound ----

class TestDualUpperBound:
    def test_upper_exceeds_lsm(self):
        """Upper bound ≥ LSM lower bound."""
        rng = np.random.default_rng(42)
        n_paths, n_steps = 10_000, 50
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = SPOT
        for i in range(n_steps):
            dW = rng.standard_normal(n_paths) * math.sqrt(dt)
            paths[:, i + 1] = paths[:, i] * np.exp(
                (RATE - 0.5 * VOL**2) * dt + VOL * dW
            )

        lsm = lsm_with_basis(paths, STRIKE, RATE, T, is_call=False)
        upper = dual_upper_bound(paths, STRIKE, RATE, T, lsm, is_call=False)
        assert upper >= lsm * 0.95  # allow small MC noise

    def test_bracket_contains_european(self):
        """LSM ≤ upper, and European ≤ upper."""
        from pricebook.equity_option import equity_option_price
        from pricebook.black76 import OptionType
        euro = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT)

        rng = np.random.default_rng(42)
        paths = np.zeros((10_000, 51))
        paths[:, 0] = SPOT
        dt = T / 50
        for i in range(50):
            dW = rng.standard_normal(10_000) * math.sqrt(dt)
            paths[:, i + 1] = paths[:, i] * np.exp(
                (RATE - 0.5 * VOL**2) * dt + VOL * dW
            )

        lsm = lsm_with_basis(paths, STRIKE, RATE, T, is_call=False)
        upper = dual_upper_bound(paths, STRIKE, RATE, T, lsm, is_call=False)
        assert upper >= euro * 0.90
