"""Tests for PDE solver framework."""

import math
import pytest
import numpy as np
from scipy.stats import norm

from pricebook.numerical._pde import (
    PDESolver1D, PDEMethod, PDEResult, GridType, BoundaryCondition,
    solve_bs_pde, solve_pde_with_vega, build_grid, extract_greeks,
)


def _bs_call(S, K, T, vol, r):
    """Analytical Black-Scholes call for comparison."""
    d1 = (math.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def _bs_delta(S, K, T, vol, r):
    d1 = (math.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * math.sqrt(T))
    return norm.cdf(d1)


# ═══════════════════════════════════════════════════════════════
# Grid builders
# ═══════════════════════════════════════════════════════════════

class TestGrids:
    def test_uniform(self):
        g = build_grid(0, 200, 100, GridType.UNIFORM)
        assert len(g) == 100
        assert abs(g[1] - g[0] - 2.02) < 0.1  # uniform spacing

    def test_log(self):
        g = build_grid(1, 200, 100, GridType.LOG)
        assert g[0] > 0
        # Log grid: more points at low values
        low_count = np.sum(g < 50)
        high_count = np.sum(g > 150)
        assert low_count > high_count

    def test_sinh(self):
        g = build_grid(0, 200, 100, GridType.SINH, concentration_point=100)
        # Sinh: concentrated near strike
        near_strike = np.sum(np.abs(g - 100) < 20)
        assert near_strike > 20

    def test_chebyshev(self):
        g = build_grid(0, 200, 50, GridType.CHEBYSHEV)
        assert len(g) == 50


# ═══════════════════════════════════════════════════════════════
# Greeks extraction
# ═══════════════════════════════════════════════════════════════

class TestGreeks:
    def test_extract_from_linear(self):
        """Linear payoff: delta = 1, gamma = 0."""
        grid = np.linspace(80, 120, 100)
        values = grid - 100  # linear payoff above 100
        g = extract_greeks(grid, values, 100.0)
        assert abs(g["delta"] - 1.0) < 0.01
        assert abs(g["gamma"]) < 0.01

    def test_extract_price(self):
        grid = np.linspace(80, 120, 100)
        values = np.maximum(grid - 100, 0)
        g = extract_greeks(grid, values, 110.0)
        assert abs(g["price"] - 10.0) < 0.5


# ═══════════════════════════════════════════════════════════════
# 1D PDE: Method selection
# ═══════════════════════════════════════════════════════════════

class TestMethodSelection:
    @pytest.mark.parametrize("method", [
        PDEMethod.EXPLICIT, PDEMethod.IMPLICIT, PDEMethod.CRANK_NICOLSON,
    ])
    def test_all_methods_price_call(self, method):
        """Core methods should produce a reasonable European call price."""
        n_time = 200 if method != PDEMethod.EXPLICIT else 5000
        result = solve_bs_pde(100, 100, 1.0, 0.25, 0.04,
                               method=method, n_space=200, n_time=n_time)
        bs = _bs_call(100, 100, 1.0, 0.25, 0.04)
        assert abs(result.price - bs) < 1.0


class TestCrankNicolson:
    def test_atm_call(self):
        result = solve_bs_pde(100, 100, 1.0, 0.25, 0.04)
        bs = _bs_call(100, 100, 1.0, 0.25, 0.04)
        assert abs(result.price - bs) < 0.5

    def test_itm_call(self):
        result = solve_bs_pde(120, 100, 1.0, 0.25, 0.04)
        bs = _bs_call(120, 100, 1.0, 0.25, 0.04)
        assert abs(result.price - bs) < 1.0

    def test_otm_call(self):
        result = solve_bs_pde(80, 100, 1.0, 0.25, 0.04)
        bs = _bs_call(80, 100, 1.0, 0.25, 0.04)
        assert abs(result.price - bs) < 0.5

    def test_put(self):
        result = solve_bs_pde(100, 100, 1.0, 0.25, 0.04, is_call=False)
        bs_call = _bs_call(100, 100, 1.0, 0.25, 0.04)
        bs_put = bs_call - 100 + 100 * math.exp(-0.04)  # put-call parity
        assert abs(result.price - bs_put) < 0.5


class TestAmericanPDE:
    def test_american_put_above_european(self):
        """American put ≥ European put."""
        eu = solve_bs_pde(100, 100, 1.0, 0.25, 0.04, is_call=False, is_american=False)
        am = solve_bs_pde(100, 100, 1.0, 0.25, 0.04, is_call=False, is_american=True)
        assert am.price >= eu.price - 0.01

    def test_american_call_equals_european(self):
        """American call = European call (no dividends)."""
        eu = solve_bs_pde(100, 100, 1.0, 0.25, 0.04, is_call=True, is_american=False)
        am = solve_bs_pde(100, 100, 1.0, 0.25, 0.04, is_call=True, is_american=True)
        assert abs(am.price - eu.price) < 0.5


class TestGreeksPDE:
    def test_delta_positive_call(self):
        result = solve_bs_pde(100, 100, 1.0, 0.25, 0.04)
        assert 0 < result.delta < 1

    def test_delta_matches_bs(self):
        result = solve_bs_pde(100, 100, 1.0, 0.25, 0.04, n_space=300, n_time=300)
        bs_delta = _bs_delta(100, 100, 1.0, 0.25, 0.04)
        assert abs(result.delta - bs_delta) < 0.05

    def test_gamma_positive(self):
        result = solve_bs_pde(100, 100, 1.0, 0.25, 0.04)
        assert result.gamma > 0

    def test_vega(self):
        result = solve_pde_with_vega(100, 100, 1.0, 0.25, 0.04)
        assert result.vega is not None
        assert result.vega > 0


class TestGridTypes:
    def test_uniform_grid_accurate(self):
        result = solve_bs_pde(100, 100, 1.0, 0.25, 0.04, grid_type=GridType.UNIFORM)
        bs = _bs_call(100, 100, 1.0, 0.25, 0.04)
        assert abs(result.price - bs) < 0.5

    def test_all_grids_produce_result(self):
        """All grid types should produce a valid result (not necessarily accurate)."""
        for gt in [GridType.UNIFORM, GridType.LOG, GridType.SINH]:
            result = solve_bs_pde(100, 100, 1.0, 0.25, 0.04, grid_type=gt)
            assert result.price > 0


class TestSerialization:
    def test_to_dict(self):
        result = solve_bs_pde(100, 100, 1.0, 0.25, 0.04)
        d = result.to_dict()
        assert "price" in d
        assert "delta" in d
        assert "gamma" in d
        assert "method" in d


class TestClassAPI:
    def test_reusable(self):
        solver = PDESolver1D(PDEMethod.CRANK_NICOLSON, n_space=150, n_time=150)
        r1 = solver.solve(100, 100, 1.0, 0.25, 0.04)
        r2 = solver.solve(100, 110, 0.5, 0.30, 0.03)
        assert r1.price != r2.price
