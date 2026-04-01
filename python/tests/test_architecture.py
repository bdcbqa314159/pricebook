"""Tests for protocols, results, and registry."""

import pytest
import math

from pricebook.protocols import (
    VolSurface,
    SolverResult,
    QuadratureResult,
    MCResult,
)
from pricebook.results import TreeResult, PDEResult
from pricebook.registry import (
    get_solver,
    get_integrator,
    get_tree_european,
    get_tree_american,
    get_pde_pricer,
    get_mc_pricer,
    list_solvers,
    list_integrators,
    list_mc_pricers,
)
from pricebook.vol_surface import FlatVol
from pricebook.vol_smile import VolSmile
from pricebook.vol_surface_strike import VolSurfaceStrike
from pricebook.swaption_vol import SwaptionVolSurface
from pricebook.fx_vol_surface import FXVolSurface, FXVolQuote
from pricebook.black76 import OptionType
from pricebook.equity_option import equity_option_price
from datetime import date


SPOT, STRIKE, RATE, VOL, T = 100.0, 100.0, 0.05, 0.20, 1.0
BS = equity_option_price(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL)


class TestVolSurfaceProtocol:
    """All vol surface types satisfy the VolSurface protocol."""

    def test_flat_vol(self):
        assert isinstance(FlatVol(0.20), VolSurface)

    def test_vol_smile(self):
        smile = VolSmile([90, 100, 110], [0.22, 0.20, 0.22])
        # VolSmile.vol takes (strike) not (expiry, strike), so not a match
        # This is expected — VolSmile is a building block, not a surface

    def test_vol_surface_strike(self):
        smile = VolSmile([90, 100, 110], [0.22, 0.20, 0.22])
        s = VolSurfaceStrike(date(2024, 1, 15), [date(2025, 1, 15)], [smile])
        assert isinstance(s, VolSurface)


class TestResultTypes:
    def test_solver_result(self):
        r = SolverResult(root=1.0, iterations=5, converged=True, function_value=0.0)
        assert r.converged

    def test_quadrature_result(self):
        r = QuadratureResult(value=3.14, error_estimate=1e-10, n_evaluations=16)
        assert r.n_evaluations == 16

    def test_mc_result(self):
        r = MCResult(price=10.5, std_error=0.1, n_paths=100000)
        assert r.n_paths == 100000

    def test_tree_result(self):
        r = TreeResult(price=10.5, delta=0.6, gamma=0.02, n_steps=500, method="crr")
        assert r.method == "crr"

    def test_pde_result(self):
        r = PDEResult(price=10.5, delta=0.6, n_spot=200, n_time=200, scheme="cn")
        assert r.scheme == "cn"


class TestSolverRegistry:
    def test_list_solvers(self):
        names = list_solvers()
        assert "newton" in names
        assert "brent" in names
        assert "halley" in names
        assert "itp" in names

    def test_get_newton(self):
        solver = get_solver("newton")
        r = solver(lambda x: x**2 - 4, lambda x: 2*x, x0=3.0)
        assert r.root == pytest.approx(2.0, abs=1e-10)

    def test_get_brent(self):
        solver = get_solver("brent")
        root = solver(lambda x: x**2 - 4, 0, 3)
        assert root == pytest.approx(2.0, abs=1e-10)

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown solver"):
            get_solver("nonexistent")


class TestIntegratorRegistry:
    def test_list_integrators(self):
        names = list_integrators()
        assert "gauss_legendre" in names
        assert "adaptive_simpson" in names

    def test_get_gauss_legendre(self):
        integrator = get_integrator("gauss_legendre")
        r = integrator(math.sin, 0, math.pi, n=16)
        assert r.value == pytest.approx(2.0, abs=1e-10)

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown integrator"):
            get_integrator("nonexistent")


class TestTreeRegistry:
    def test_binomial_european(self):
        pricer = get_tree_european("binomial")
        p = pricer(SPOT, STRIKE, RATE, VOL, T, 200, OptionType.CALL)
        assert p == pytest.approx(BS, rel=0.005)

    def test_trinomial_european(self):
        pricer = get_tree_european("trinomial")
        p = pricer(SPOT, STRIKE, RATE, VOL, T, 200, OptionType.CALL)
        assert p == pytest.approx(BS, rel=0.005)

    def test_binomial_american(self):
        pricer = get_tree_american("binomial")
        p = pricer(SPOT, STRIKE, RATE, VOL, T, 200, OptionType.PUT)
        assert p > 0

    def test_trinomial_american(self):
        pricer = get_tree_american("trinomial")
        p = pricer(SPOT, STRIKE, RATE, VOL, T, 200, OptionType.PUT)
        assert p > 0

    def test_trees_agree(self):
        """Both trees give approximately the same price."""
        bin_p = get_tree_european("binomial")(SPOT, STRIKE, RATE, VOL, T, 300, OptionType.CALL)
        tri_p = get_tree_european("trinomial")(SPOT, STRIKE, RATE, VOL, T, 300, OptionType.CALL)
        assert bin_p == pytest.approx(tri_p, rel=0.005)


class TestPDERegistry:
    def test_european(self):
        pricer = get_pde_pricer("european")
        p = pricer(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, n_spot=200, n_time=200)
        assert p == pytest.approx(BS, rel=0.005)

    def test_american(self):
        pricer = get_pde_pricer("american")
        p = pricer(SPOT, STRIKE, RATE, VOL, T, OptionType.PUT, n_spot=200, n_time=200)
        assert p > 0


class TestMCRegistry:
    def test_list_mc(self):
        names = list_mc_pricers()
        assert "european" in names
        assert "stratified" in names
        assert "lsm" in names
        assert "mlmc" in names

    def test_european(self):
        pricer = get_mc_pricer("european")
        r = pricer(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, n_paths=50_000)
        assert r.price == pytest.approx(BS, rel=0.05)

    def test_stratified(self):
        pricer = get_mc_pricer("stratified")
        r = pricer(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, n_paths=50_000)
        assert r.price == pytest.approx(BS, rel=0.05)

    def test_all_mc_agree(self):
        """All MC methods give approximately the same European price."""
        for name in ["european", "stratified", "importance"]:
            pricer = get_mc_pricer(name)
            r = pricer(SPOT, STRIKE, RATE, VOL, T, OptionType.CALL, n_paths=100_000)
            assert r.price == pytest.approx(BS, rel=0.03), f"{name} disagrees"
