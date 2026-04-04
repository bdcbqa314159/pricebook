"""Tests for Heston MC simulation: Euler, QE, European, barrier."""

import math
import pytest

from pricebook.heston_mc import (
    heston_euler,
    heston_qe,
    heston_mc_european,
    heston_mc_barrier,
)
from pricebook.heston import heston_price
from pricebook.black76 import OptionType


# Standard Heston params
S, K, R, T = 100.0, 100.0, 0.05, 1.0
V0, KAPPA, THETA, XI, RHO = 0.04, 2.0, 0.04, 0.3, -0.7

# Semi-analytical reference
BS_HESTON_CALL = heston_price(S, K, R, T, V0, KAPPA, THETA, XI, RHO)


class TestEulerScheme:
    def test_paths_shape(self):
        S_paths, v_paths = heston_euler(S, R, T, V0, KAPPA, THETA, XI, RHO,
                                         n_steps=10, n_paths=100)
        assert S_paths.shape == (100, 11)
        assert v_paths.shape == (100, 11)

    def test_initial_values(self):
        S_paths, v_paths = heston_euler(S, R, T, V0, KAPPA, THETA, XI, RHO,
                                         n_steps=10, n_paths=50)
        assert S_paths[:, 0] == pytest.approx(S)
        assert v_paths[:, 0] == pytest.approx(V0)

    def test_variance_non_negative(self):
        _, v = heston_euler(S, R, T, V0, KAPPA, THETA, XI, RHO,
                            n_steps=50, n_paths=1000)
        assert np.all(v >= 0)

    def test_european_matches_analytical(self):
        price = heston_mc_european(S, K, R, T, V0, KAPPA, THETA, XI, RHO,
                                   scheme="euler", n_paths=100_000, n_steps=200)
        assert price == pytest.approx(BS_HESTON_CALL, rel=0.10)


class TestQEScheme:
    def test_paths_shape(self):
        S_paths, v_paths = heston_qe(S, R, T, V0, KAPPA, THETA, XI, RHO,
                                      n_steps=10, n_paths=100)
        assert S_paths.shape == (100, 11)
        assert v_paths.shape == (100, 11)

    def test_variance_non_negative(self):
        _, v = heston_qe(S, R, T, V0, KAPPA, THETA, XI, RHO,
                          n_steps=50, n_paths=1000)
        assert np.all(v >= 0)

    def test_european_matches_analytical(self):
        price = heston_mc_european(S, K, R, T, V0, KAPPA, THETA, XI, RHO,
                                   scheme="qe", n_paths=100_000, n_steps=100)
        assert price == pytest.approx(BS_HESTON_CALL, rel=0.05)

    def test_qe_accurate_with_few_steps(self):
        """QE should be accurate even with coarse time stepping."""
        qe = heston_mc_european(S, K, R, T, V0, KAPPA, THETA, XI, RHO,
                                scheme="qe", n_paths=100_000, n_steps=20, seed=1)
        assert qe == pytest.approx(BS_HESTON_CALL, rel=0.10)


class TestHestonMCEuropean:
    def test_call_positive(self):
        price = heston_mc_european(S, K, R, T, V0, KAPPA, THETA, XI, RHO,
                                   n_paths=10_000)
        assert price > 0

    def test_put_positive(self):
        price = heston_mc_european(S, K, R, T, V0, KAPPA, THETA, XI, RHO,
                                   option_type=OptionType.PUT, n_paths=10_000)
        assert price > 0

    def test_put_call_parity(self):
        call = heston_mc_european(S, K, R, T, V0, KAPPA, THETA, XI, RHO,
                                  n_paths=100_000, seed=42)
        put = heston_mc_european(S, K, R, T, V0, KAPPA, THETA, XI, RHO,
                                 option_type=OptionType.PUT, n_paths=100_000, seed=42)
        parity = call - put - (S - K * math.exp(-R * T))
        assert parity == pytest.approx(0.0, abs=0.5)


class TestHestonMCBarrier:
    def test_knockout_leq_vanilla(self):
        vanilla = heston_mc_european(S, K, R, T, V0, KAPPA, THETA, XI, RHO,
                                     n_paths=50_000)
        ko = heston_mc_barrier(S, K, 80.0, R, T, V0, KAPPA, THETA, XI, RHO,
                               is_up=False, is_knock_in=False,
                               n_paths=50_000)
        assert ko <= vanilla * 1.05  # allow MC noise

    def test_knockin_positive(self):
        ki = heston_mc_barrier(S, K, 80.0, R, T, V0, KAPPA, THETA, XI, RHO,
                               is_up=False, is_knock_in=True,
                               n_paths=10_000)
        assert ki >= 0

    def test_in_out_parity(self):
        """knock-in + knock-out ≈ vanilla."""
        ko = heston_mc_barrier(S, K, 80.0, R, T, V0, KAPPA, THETA, XI, RHO,
                               is_up=False, is_knock_in=False,
                               n_paths=50_000, seed=99)
        ki = heston_mc_barrier(S, K, 80.0, R, T, V0, KAPPA, THETA, XI, RHO,
                               is_up=False, is_knock_in=True,
                               n_paths=50_000, seed=99)
        vanilla = heston_mc_european(S, K, R, T, V0, KAPPA, THETA, XI, RHO,
                                     n_paths=50_000, seed=99)
        assert ko + ki == pytest.approx(vanilla, rel=0.05)


import numpy as np  # noqa: E402 (used in test assertions)
