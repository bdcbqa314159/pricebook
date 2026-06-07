"""Tests for pricebook.options.american_multi_asset."""

import pytest

from pricebook.options.american_multi_asset import (
    american_spread_option,
    american_basket_option,
    american_best_of,
    american_worst_of_put,
)

N_PATHS = 5000
SEED = 42
T = 1.0
R = 0.05
S1, S2 = 100.0, 100.0
VOL1, VOL2 = 0.20, 0.20
RHO = 0.5


class TestAmericanSpreadOption:
    def test_price_positive(self):
        res = american_spread_option(
            S1=110.0, S2=100.0, strike=5.0,
            vol1=VOL1, vol2=VOL2, rho=RHO,
            T=T, r=R, n_paths=N_PATHS, seed=SEED,
        )
        assert res.price > 0

    def test_result_has_european_price(self):
        res = american_spread_option(
            S1=110.0, S2=100.0, strike=5.0,
            vol1=VOL1, vol2=VOL2, rho=RHO,
            T=T, r=R, n_paths=N_PATHS, seed=SEED,
        )
        assert res.european_price > 0

    def test_american_ge_european(self):
        res = american_spread_option(
            S1=110.0, S2=100.0, strike=5.0,
            vol1=VOL1, vol2=VOL2, rho=RHO,
            T=T, r=R, n_paths=N_PATHS, seed=SEED,
        )
        assert res.price >= res.european_price - 1e-3  # allow small MC noise


class TestAmericanBasketOption:
    def test_price_positive(self):
        res = american_basket_option(
            spots=[S1, S2],
            vols=[VOL1, VOL2],
            correlations=[[1.0, RHO], [RHO, 1.0]],
            weights=[0.5, 0.5],
            strike=100.0,
            T=T, r=R,
            option_type="call",
            n_paths=N_PATHS, seed=SEED,
        )
        assert res.price > 0

    def test_price_ge_european(self):
        res = american_basket_option(
            spots=[S1, S2],
            vols=[VOL1, VOL2],
            correlations=[[1.0, RHO], [RHO, 1.0]],
            weights=[0.5, 0.5],
            strike=100.0,
            T=T, r=R,
            option_type="call",
            n_paths=N_PATHS, seed=SEED,
        )
        assert res.price >= res.european_price - 1e-3

    def test_delta_length(self):
        res = american_basket_option(
            spots=[S1, S2],
            vols=[VOL1, VOL2],
            correlations=[[1.0, RHO], [RHO, 1.0]],
            weights=[0.5, 0.5],
            strike=100.0,
            T=T, r=R,
            n_paths=N_PATHS, seed=SEED,
        )
        assert len(res.delta) == 2


class TestAmericanBestOf:
    def test_price_positive(self):
        res = american_best_of(
            S1=S1, S2=S2, vol1=VOL1, vol2=VOL2, rho=RHO,
            T=T, r=R,
            n_paths=N_PATHS, seed=SEED,
        )
        assert res.price > 0

    def test_price_ge_single_asset(self):
        """Best-of price >= max(S1, S2) discounted (lower bound)."""
        res = american_best_of(
            S1=S1, S2=S2, vol1=VOL1, vol2=VOL2, rho=RHO,
            T=T, r=R,
            n_paths=N_PATHS, seed=SEED,
        )
        import math
        lower_bound = max(S1, S2) * math.exp(-R * T)
        assert res.price >= lower_bound * 0.90  # allow 10% slack for MC


class TestAmericanWorstOfPut:
    def test_price_positive(self):
        res = american_worst_of_put(
            S1=S1, S2=S2, strike=110.0,
            vol1=VOL1, vol2=VOL2, rho=RHO,
            T=T, r=R,
            n_paths=N_PATHS, seed=SEED,
        )
        assert res.price > 0

    def test_american_ge_european(self):
        res = american_worst_of_put(
            S1=S1, S2=S2, strike=110.0,
            vol1=VOL1, vol2=VOL2, rho=RHO,
            T=T, r=R,
            n_paths=N_PATHS, seed=SEED,
        )
        assert res.price >= res.european_price - 1e-3
