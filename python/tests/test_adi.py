"""Tests for ADI 2D PDE solver (Heston)."""

import pytest
import math

from pricebook.adi import heston_pde
from pricebook.heston import heston_price
from pricebook.equity_option import equity_option_price
from pricebook.black76 import OptionType


SPOT, STRIKE, RATE, T = 100.0, 100.0, 0.05, 1.0
V0, KAPPA, THETA, XI, RHO = 0.04, 2.0, 0.04, 0.3, -0.7


class TestHestonPDE:
    def test_call_positive(self):
        p = heston_pde(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, XI, RHO)
        assert p > 0

    def test_put_positive(self):
        p = heston_pde(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, XI, RHO,
                       OptionType.PUT)
        assert p > 0

    def test_matches_semi_analytical(self):
        """Heston PDE ≈ semi-analytical (Gauss-Legendre)."""
        ref = heston_price(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, XI, RHO)
        pde = heston_pde(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, XI, RHO,
                         n_x=100, n_v=50, n_time=150)
        assert pde == pytest.approx(ref, rel=0.20)

    def test_put_matches_semi_analytical(self):
        ref = heston_price(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, XI, RHO,
                           OptionType.PUT)
        pde = heston_pde(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, XI, RHO,
                         OptionType.PUT, n_x=100, n_v=50, n_time=150)
        assert pde == pytest.approx(ref, rel=0.20)

    def test_zero_xi_matches_bs(self):
        """Zero vol-of-vol → Black-Scholes."""
        vol = math.sqrt(V0)
        bs = equity_option_price(SPOT, STRIKE, RATE, vol, T, OptionType.CALL)
        pde = heston_pde(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, 0.001, 0.0,
                         n_x=100, n_v=50, n_time=150)
        assert pde == pytest.approx(bs, rel=0.05)

    def test_otm_call(self):
        ref = heston_price(SPOT, 120.0, RATE, T, V0, KAPPA, THETA, XI, RHO)
        pde = heston_pde(SPOT, 120.0, RATE, T, V0, KAPPA, THETA, XI, RHO,
                         n_x=100, n_v=50, n_time=150)
        assert pde == pytest.approx(ref, rel=0.10)

    def test_higher_v0_higher_price(self):
        p_low = heston_pde(SPOT, STRIKE, RATE, T, 0.01, KAPPA, THETA, XI, RHO)
        p_high = heston_pde(SPOT, STRIKE, RATE, T, 0.09, KAPPA, THETA, XI, RHO)
        assert p_high > p_low

    def test_with_dividend(self):
        p = heston_pde(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, XI, RHO,
                       div_yield=0.02)
        assert p > 0

    def test_convergence(self):
        """Finer grid → closer to semi-analytical."""
        ref = heston_price(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, XI, RHO)
        coarse = heston_pde(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, XI, RHO,
                            n_x=40, n_v=20, n_time=50)
        fine = heston_pde(SPOT, STRIKE, RATE, T, V0, KAPPA, THETA, XI, RHO,
                          n_x=100, n_v=50, n_time=150)
        assert abs(fine - ref) < abs(coarse - ref)
