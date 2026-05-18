"""Tests for distribution theory: Schwartz distributions, Sobolev, Green's, Feynman-Kac."""
import math
import pytest
import numpy as np
from pricebook.statistics.distribution_theory import (
    Distribution, dirac_delta, heaviside_dist, regular_distribution,
    sobolev_norm, greens_function_heat, greens_function_bs,
    feynman_kac_pde, feynman_kac_verify,
)


class TestDiracDelta:
    def test_sifting_property(self):
        delta = dirac_delta()
        assert abs(delta(math.sin) - math.sin(0.0)) < 1e-10

    def test_shifted_delta(self):
        delta_2 = dirac_delta(2.0)
        assert abs(delta_2(math.sin) - math.sin(2.0)) < 1e-10

    def test_delta_on_polynomial(self):
        delta = dirac_delta(1.0)
        phi = lambda x: x**2 + 3*x + 1
        assert abs(delta(phi) - 5.0) < 1e-10


class TestHeaviside:
    def test_heaviside_integral(self):
        H = heaviside_dist()
        phi = lambda x: math.exp(-x**2)
        result = H(phi)
        assert abs(result - math.sqrt(math.pi)/2) < 0.05

    def test_derivative_of_heaviside(self):
        H = heaviside_dist()
        H_prime = H.derivative()
        phi = lambda x: math.exp(-x**2)
        result = H_prime(phi)
        assert abs(result - 1.0) < 0.15


class TestRegularDistribution:
    def test_l2_function(self):
        f_dist = regular_distribution(lambda x: x**2)
        phi = lambda x: math.exp(-x**2)
        result = f_dist(phi)
        assert abs(result - math.sqrt(math.pi)/2) < 0.1


class TestSobolevNorm:
    def test_smooth_function(self):
        x = np.linspace(-5, 5, 1000)
        f = np.exp(-x**2)
        dx = x[1] - x[0]
        result = sobolev_norm(f, dx, s=1.0)
        assert result.h0_norm > 0
        assert result.h1_norm >= result.h0_norm

    def test_l2_norm(self):
        x = np.linspace(-5, 5, 1000)
        f = np.exp(-x**2)
        dx = x[1] - x[0]
        result = sobolev_norm(f, dx, s=0.0)
        expected = (math.pi / 2) ** 0.25
        assert abs(result.h0_norm - expected) < 0.1


class TestGreensFunction:
    def test_heat_kernel_integrates_to_one(self):
        x = np.linspace(-10, 10, 1000)
        dx = x[1] - x[0]
        G = np.array([greens_function_heat(xi, 1.0) for xi in x])
        assert abs(np.sum(G) * dx - 1.0) < 0.01

    def test_heat_kernel_peak(self):
        expected = 1.0 / math.sqrt(4 * math.pi)
        assert abs(greens_function_heat(0, 1.0, 1.0) - expected) < 1e-10

    def test_bs_kernel_positive(self):
        assert greens_function_bs(100, 100, 0.2, 1.0, 0.05) > 0


class TestFeynmanKac:
    def test_pde_coefficients(self):
        result = feynman_kac_pde(
            drift=lambda x: 0.05 * x,
            diffusion=lambda x: 0.2 * x,
            rate=0.05,
        )
        # a = 0.5 * sigma^2, b = mu, c = -r
        assert hasattr(result, "drift_coeff")
        assert hasattr(result, "diffusion_coeff")
        assert hasattr(result, "killing_rate")

    def test_verify_matching(self):
        result = feynman_kac_verify(pde_value=10.5, mc_value=10.3, tol=0.05)
        assert hasattr(result, "passed")
        assert result.passed  # within tolerance

    def test_verify_mismatch(self):
        result = feynman_kac_verify(pde_value=10.0, mc_value=15.0, tol=0.01)
        assert not result.passed
