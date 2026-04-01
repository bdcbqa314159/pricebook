"""Tests for optimization toolkit."""

import pytest
import math
import numpy as np

from pricebook.optimization import minimize, minimize_least_squares, OptimizerResult
from pricebook.registry import get_optimizer, list_optimizers


def rosenbrock(x):
    """Rosenbrock function: minimum at (1, 1), f=0."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def rosenbrock_grad(x):
    return np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
        200 * (x[1] - x[0]**2),
    ])


def quadratic(x):
    """Simple quadratic: minimum at (0, 0), f=0."""
    return x[0]**2 + x[1]**2


class TestNelderMead:
    def test_quadratic(self):
        r = minimize(quadratic, x0=[5.0, 3.0], method="nelder_mead")
        assert r.converged
        assert r.fun < 1e-10
        assert np.allclose(r.x, [0, 0], atol=1e-4)

    def test_rosenbrock(self):
        r = minimize(rosenbrock, x0=[0.0, 0.0], method="nelder_mead", maxiter=5000)
        assert r.fun < 1e-6
        assert np.allclose(r.x, [1, 1], atol=0.01)

    def test_returns_result(self):
        r = minimize(quadratic, x0=[1.0, 1.0])
        assert isinstance(r, OptimizerResult)
        assert r.method == "nelder_mead"
        assert r.n_evaluations > 0


class TestBFGS:
    def test_rosenbrock(self):
        r = minimize(rosenbrock, x0=[0.0, 0.0], method="bfgs",
                     gradient=rosenbrock_grad)
        assert r.converged
        assert np.allclose(r.x, [1, 1], atol=0.01)

    def test_quadratic(self):
        r = minimize(quadratic, x0=[5.0, 3.0], method="bfgs",
                     gradient=lambda x: 2 * np.array(x))
        assert r.converged
        assert r.fun < 1e-10


class TestLBFGSB:
    def test_bounded(self):
        r = minimize(quadratic, x0=[5.0, 3.0], method="l_bfgs_b",
                     bounds=[(-10, 10), (-10, 10)])
        assert r.converged
        assert r.fun < 1e-10

    def test_bounds_active(self):
        """Minimum at (0,0) but bounded away → lands on boundary."""
        r = minimize(quadratic, x0=[5.0, 3.0], method="l_bfgs_b",
                     bounds=[(1, 10), (1, 10)])
        assert np.allclose(r.x, [1, 1], atol=0.01)


class TestDifferentialEvolution:
    def test_rosenbrock(self):
        r = minimize(rosenbrock, x0=[0.0, 0.0], method="differential_evolution",
                     bounds=[(-5, 5), (-5, 5)])
        assert r.fun < 1e-6
        assert np.allclose(r.x, [1, 1], atol=0.01)

    def test_requires_bounds(self):
        with pytest.raises(ValueError, match="bounds required"):
            minimize(quadratic, x0=[1.0, 1.0], method="differential_evolution")


class TestBasinHopping:
    def test_quadratic(self):
        r = minimize(quadratic, x0=[5.0, 3.0], method="basin_hopping", maxiter=50)
        assert r.fun < 1e-6


class TestLeastSquares:
    def test_linear_fit(self):
        """Fit y = a*x + b to data."""
        x_data = np.array([1, 2, 3, 4, 5], dtype=float)
        y_data = np.array([2.1, 3.9, 6.1, 7.9, 10.1])

        def residuals(params):
            a, b = params
            return a * x_data + b - y_data

        r = minimize_least_squares(residuals, x0=[0.0, 0.0])
        assert r.converged
        assert r.x[0] == pytest.approx(2.0, abs=0.1)  # slope ≈ 2
        assert r.x[1] == pytest.approx(0.0, abs=0.5)   # intercept ≈ 0

    def test_returns_result(self):
        r = minimize_least_squares(lambda x: x, x0=[1.0])
        assert isinstance(r, OptimizerResult)
        assert r.method == "lm"


class TestRegistry:
    def test_list_optimizers(self):
        names = list_optimizers()
        assert "nelder_mead" in names
        assert "bfgs" in names
        assert "differential_evolution" in names

    def test_get_nelder_mead(self):
        opt = get_optimizer("nelder_mead")
        r = opt(quadratic, x0=[5.0, 3.0])
        assert r.converged

    def test_get_bfgs(self):
        opt = get_optimizer("bfgs")
        r = opt(quadratic, x0=[5.0, 3.0], gradient=lambda x: 2*np.array(x))
        assert r.converged

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown optimizer"):
            get_optimizer("nonexistent")

    def test_all_optimizers_find_minimum(self):
        for name in ["nelder_mead", "bfgs"]:
            opt = get_optimizer(name)
            kwargs = {}
            if name == "bfgs":
                kwargs["gradient"] = lambda x: 2 * np.array(x)
            r = opt(quadratic, x0=[5.0, 3.0], **kwargs)
            assert r.fun < 1e-6, f"{name} failed"
