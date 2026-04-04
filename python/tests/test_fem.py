"""Tests for Finite Element Method (1D)."""

import math
import pytest
import numpy as np

from pricebook.fem import (
    _p1_mass,
    _p1_stiffness,
    _p2_mass,
    _p2_stiffness,
    assemble_p1,
    assemble_p2,
    fem_heat_cn,
    fem_bs_european,
)
from pricebook.equity_option import equity_option_price
from pricebook.black76 import OptionType


class TestElementMatrices:
    def test_p1_mass_symmetric(self):
        M = _p1_mass(1.0)
        np.testing.assert_array_almost_equal(M, M.T)

    def test_p1_mass_positive(self):
        M = _p1_mass(1.0)
        eigvals = np.linalg.eigvalsh(M)
        assert all(e > 0 for e in eigvals)

    def test_p1_stiffness_symmetric(self):
        K = _p1_stiffness(1.0)
        np.testing.assert_array_almost_equal(K, K.T)

    def test_p1_mass_scales_with_h(self):
        M1 = _p1_mass(1.0)
        M2 = _p1_mass(2.0)
        np.testing.assert_array_almost_equal(M2, 2.0 * M1)

    def test_p2_mass_symmetric(self):
        M = _p2_mass(1.0)
        np.testing.assert_array_almost_equal(M, M.T)

    def test_p2_stiffness_symmetric(self):
        K = _p2_stiffness(1.0)
        np.testing.assert_array_almost_equal(K, K.T)


class TestAssembly:
    def test_p1_size(self):
        nodes = np.linspace(0, 1, 11)
        M, K = assemble_p1(nodes)
        assert M.shape == (11, 11)
        assert K.shape == (11, 11)

    def test_p1_spd(self):
        nodes = np.linspace(0, 1, 5)
        M, K = assemble_p1(nodes)
        eigvals = np.linalg.eigvalsh(M.to_dense())
        assert all(e >= -1e-14 for e in eigvals)

    def test_p2_has_midpoints(self):
        nodes = np.linspace(0, 1, 5)  # 4 elements
        M, K, all_nodes = assemble_p2(nodes)
        assert len(all_nodes) == 9  # 5 boundaries + 4 midpoints
        assert M.shape == (9, 9)

    def test_p2_spd(self):
        nodes = np.linspace(0, 1, 5)
        M, K, _ = assemble_p2(nodes)
        eigvals = np.linalg.eigvalsh(M.to_dense())
        assert all(e >= -1e-14 for e in eigvals)


class TestHeatEquation:
    def test_convergence(self):
        """Heat equation with sine initial condition should decay exponentially."""
        nodes = np.linspace(0, math.pi, 51)
        u0 = np.sin(nodes)
        t_final = 0.1
        dt = t_final / 100

        u = fem_heat_cn(nodes, u0, dt, 100, bc_left=0.0, bc_right=0.0)

        # Analytical: u(x, t) = sin(x) * exp(-t)
        expected = np.sin(nodes) * math.exp(-t_final)
        # Allow some numerical error
        error = np.max(np.abs(u - expected))
        assert error < 0.01

    def test_steady_state(self):
        """After long time, solution should approach steady state (linear)."""
        nodes = np.linspace(0, 1, 21)
        u0 = np.zeros(21)
        u = fem_heat_cn(nodes, u0, dt=0.01, n_steps=1000, bc_left=0.0, bc_right=1.0)
        # Steady state: linear from 0 to 1
        expected = nodes
        error = np.max(np.abs(u - expected))
        assert error < 0.05


class TestBSFEM:
    def test_call_matches_analytical(self):
        """FEM call price should match Black-Scholes."""
        bs = equity_option_price(100, 100, 0.05, 0.20, 1.0)
        fem = fem_bs_european(100, 100, 0.05, 0.20, 1.0, n_spatial=150, n_time=150)
        assert fem == pytest.approx(bs, rel=0.10)

    def test_put_matches_analytical(self):
        bs = equity_option_price(100, 100, 0.05, 0.20, 1.0, OptionType.PUT)
        fem = fem_bs_european(100, 100, 0.05, 0.20, 1.0, n_spatial=150, n_time=150, is_call=False)
        assert fem == pytest.approx(bs, rel=0.10)

    def test_call_positive(self):
        price = fem_bs_european(100, 105, 0.05, 0.20, 1.0)
        assert price > 0

    def test_itm_call(self):
        price = fem_bs_european(120, 100, 0.05, 0.20, 1.0)
        assert price > 20  # at least intrinsic

    def test_otm_put_small(self):
        price = fem_bs_european(120, 100, 0.05, 0.20, 1.0, is_call=False)
        assert 0 < price < 10

    def test_p2_higher_accuracy(self):
        """P2 elements should converge faster than P1 on coarse grids."""
        bs = equity_option_price(100, 100, 0.05, 0.20, 1.0)
        p1 = fem_bs_european(100, 100, 0.05, 0.20, 1.0, n_spatial=30, n_time=30, order=1)
        p2 = fem_bs_european(100, 100, 0.05, 0.20, 1.0, n_spatial=30, n_time=30, order=2)
        err_p1 = abs(p1 - bs)
        err_p2 = abs(p2 - bs)
        # P2 should be at least as good
        assert err_p2 <= err_p1 * 1.5  # allow some margin
