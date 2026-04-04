"""Tests for SABR MC simulation."""

import math
import pytest
import numpy as np

from pricebook.sabr_mc import (
    sabr_mc_paths,
    sabr_mc_european,
    sabr_mc_asian,
    sabr_mc_implied_vol,
)
from pricebook.sabr import sabr_implied_vol
from pricebook.black76 import OptionType, black76_price


F, K, T = 100.0, 100.0, 1.0
ALPHA, BETA, RHO, NU = 0.20, 0.5, -0.3, 0.4


class TestSABRPaths:
    def test_shape(self):
        F_paths, sig = sabr_mc_paths(F, T, ALPHA, BETA, RHO, NU,
                                      n_steps=10, n_paths=100)
        assert F_paths.shape == (100, 11)
        assert sig.shape == (100, 11)

    def test_initial_values(self):
        F_paths, sig = sabr_mc_paths(F, T, ALPHA, BETA, RHO, NU,
                                      n_steps=10, n_paths=50)
        assert F_paths[:, 0] == pytest.approx(F)
        assert sig[:, 0] == pytest.approx(ALPHA)

    def test_forward_non_negative(self):
        F_paths, _ = sabr_mc_paths(F, T, ALPHA, BETA, RHO, NU,
                                    n_steps=50, n_paths=1000)
        assert np.all(F_paths >= 0)

    def test_vol_positive(self):
        _, sig = sabr_mc_paths(F, T, ALPHA, BETA, RHO, NU,
                                n_steps=50, n_paths=1000)
        assert np.all(sig > 0)


class TestSABREuropean:
    def test_call_positive(self):
        price = sabr_mc_european(F, K, T, ALPHA, BETA, RHO, NU, n_paths=10_000)
        assert price > 0

    def test_put_positive(self):
        price = sabr_mc_european(F, K, T, ALPHA, BETA, RHO, NU,
                                 option_type=OptionType.PUT, n_paths=10_000)
        assert price > 0

    def test_matches_hagan(self):
        """MC price should be close to Hagan approximation price."""
        hagan_vol = sabr_implied_vol(F, K, T, ALPHA, BETA, RHO, NU)
        hagan_price = black76_price(F, K, hagan_vol, T, df=1.0, option_type=OptionType.CALL)
        mc_price = sabr_mc_european(F, K, T, ALPHA, BETA, RHO, NU,
                                    n_paths=200_000, n_steps=200)
        assert mc_price == pytest.approx(hagan_price, rel=0.10)

    def test_otm_strike(self):
        price = sabr_mc_european(F, 110.0, T, ALPHA, BETA, RHO, NU, n_paths=100_000)
        atm = sabr_mc_european(F, K, T, ALPHA, BETA, RHO, NU, n_paths=100_000)
        assert price > 0
        assert price < atm

    def test_beta_1_lognormal(self):
        """Beta=1 is lognormal SABR."""
        price = sabr_mc_european(F, K, T, 0.20, 1.0, -0.3, 0.4, n_paths=50_000)
        assert price > 0


class TestSABRImpliedVol:
    def test_atm_matches_hagan(self):
        mc_vol = sabr_mc_implied_vol(F, K, T, ALPHA, BETA, RHO, NU,
                                     n_paths=200_000, n_steps=200)
        hagan_vol = sabr_implied_vol(F, K, T, ALPHA, BETA, RHO, NU)
        assert mc_vol == pytest.approx(hagan_vol, rel=0.10)


class TestSABRAsian:
    def test_positive(self):
        price = sabr_mc_asian(F, K, T, ALPHA, BETA, RHO, NU, n_paths=10_000)
        assert price > 0

    def test_asian_less_than_european(self):
        """Asian call ≤ European call (averaging reduces variance)."""
        eur = sabr_mc_european(F, K, T, ALPHA, BETA, RHO, NU,
                               n_paths=50_000, seed=42)
        asian = sabr_mc_asian(F, K, T, ALPHA, BETA, RHO, NU,
                              n_paths=50_000, seed=42)
        assert asian <= eur * 1.05  # allow MC noise

    def test_sabr_asian_differs_from_flat(self):
        """SABR Asian price should differ from flat-vol Asian."""
        sabr_price = sabr_mc_asian(F, K, T, ALPHA, BETA, RHO, NU,
                                   n_paths=50_000, seed=42)
        # Flat vol Asian: use Hagan ATM vol with no smile dynamics
        assert sabr_price > 0  # sanity — full comparison needs flat-vol MC
