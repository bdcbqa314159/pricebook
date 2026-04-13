"""Tests for distribution theory, Sobolev norms, Green's functions, Feynman-Kac."""

import math

import numpy as np
import pytest

from pricebook.distribution_theory import (
    Distribution,
    FeynmanKacPDE,
    FeynmanKacVerification,
    SobolevNormResult,
    dirac_delta,
    feynman_kac_pde,
    feynman_kac_verify,
    greens_function_bs,
    greens_function_heat,
    heaviside_dist,
    regular_distribution,
    sobolev_norm,
)


# ---- Distributions ----

class TestDistributions:
    def test_dirac_delta(self):
        """⟨δ, φ⟩ = φ(0)."""
        delta = dirac_delta(0.0)
        assert delta(lambda x: x**2 + 3) == pytest.approx(3.0)

    def test_dirac_shifted(self):
        """⟨δ_{2}, φ⟩ = φ(2)."""
        delta = dirac_delta(2.0)
        assert delta(lambda x: x**2) == pytest.approx(4.0)

    def test_heaviside_positive(self):
        """⟨H, φ⟩ > 0 for positive φ."""
        H = heaviside_dist(0.0)
        result = H(lambda x: math.exp(-x**2))
        assert result > 0

    def test_dirac_derivative_is_negative_dphi(self):
        """⟨δ', φ⟩ = −φ'(0). For φ(x) = x², φ'(0) = 0 → ⟨δ', φ⟩ ≈ 0."""
        delta = dirac_delta(0.0)
        delta_prime = delta.derivative()
        result = delta_prime(lambda x: x**2)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_dirac_derivative_nonzero(self):
        """⟨δ', φ⟩ = −φ'(0). For φ(x) = x, φ'(0) = 1 → ⟨δ', φ⟩ = −1."""
        delta = dirac_delta(0.0)
        delta_prime = delta.derivative()
        result = delta_prime(lambda x: x)
        assert result == pytest.approx(-1.0, abs=1e-3)

    def test_regular_distribution(self):
        """⟨T_f, φ⟩ = ∫ f φ dx for f(x) = 1 on [0,1], φ(x) = 1."""
        f = lambda x: 1.0 if 0 <= x <= 1 else 0.0
        T = regular_distribution(f)
        result = T(lambda x: 1.0)
        assert result == pytest.approx(1.0, rel=0.05)


# ---- Sobolev norms ----

class TestSobolevNorm:
    def test_constant_function(self):
        """Constant: H⁰ = |c|√L, H¹ = same (derivative = 0)."""
        f = np.ones(100) * 2.0
        dx = 0.1
        result = sobolev_norm(f, dx)
        assert result.h0_norm > 0
        # H¹ ≈ H⁰ for constant (derivative ≈ 0)
        assert result.h1_norm == pytest.approx(result.h0_norm, rel=0.01)

    def test_smooth_vs_kinked(self):
        """Smooth function has smaller H¹ norm than kinked for same L²."""
        x = np.linspace(-1, 1, 200)
        dx = x[1] - x[0]
        smooth = np.sin(math.pi * x)
        kinked = np.abs(x)  # kink at 0
        s_result = sobolev_norm(smooth, dx)
        k_result = sobolev_norm(kinked, dx)
        # Kinked has larger derivative → larger H¹
        assert k_result.h1_norm / k_result.h0_norm > s_result.h1_norm / s_result.h0_norm * 0.5


# ---- Green's functions ----

class TestGreensFunction:
    def test_heat_kernel_integrates_to_one(self):
        """∫ G(x,t) dx = 1 for any t > 0."""
        t = 0.5
        x = np.linspace(-10, 10, 1000)
        dx = x[1] - x[0]
        G = np.array([greens_function_heat(xi, t) for xi in x])
        integral = np.sum(G) * dx
        assert integral == pytest.approx(1.0, abs=0.01)

    def test_heat_kernel_peaks_at_zero(self):
        G_0 = greens_function_heat(0.0, 1.0)
        G_1 = greens_function_heat(1.0, 1.0)
        assert G_0 > G_1

    def test_bs_greens_integrates_to_one(self):
        """BS transition density integrates to 1."""
        S = 100.0
        K_grid = np.linspace(10, 300, 500)
        dK = K_grid[1] - K_grid[0]
        G = np.array([greens_function_bs(S, K, 0.05, 0.20, 1.0) for K in K_grid])
        integral = np.sum(G) * dK
        assert integral == pytest.approx(1.0, abs=0.05)

    def test_bs_greens_is_lognormal(self):
        """Peak near the forward: S exp(rT)."""
        S, r, T = 100, 0.05, 1.0
        fwd = S * math.exp(r * T)
        G_fwd = greens_function_bs(S, fwd, r, 0.20, T)
        G_far = greens_function_bs(S, 2 * fwd, r, 0.20, T)
        assert G_fwd > G_far

    def test_zero_time(self):
        assert greens_function_heat(1.0, 0.0) == 0.0
        assert greens_function_bs(100, 100, 0.05, 0.20, 0.0) == 0.0


# ---- Feynman-Kac ----

class TestFeynmanKac:
    def test_gbm_pde_coefficients(self):
        """GBM: μ(S)=rS, σ(S)=σS → PDE: rS u' + 0.5σ²S² u'' − ru = 0."""
        r, vol = 0.05, 0.20
        pde = feynman_kac_pde(
            drift=lambda x: r * x,
            diffusion=lambda x: vol * x,
            rate=r,
        )
        # At S=100: drift_coeff = 5, diff_coeff = 0.5×0.04×10000 = 200
        assert pde.drift_coeff(100) == pytest.approx(5.0)
        assert pde.diffusion_coeff(100) == pytest.approx(200.0)
        assert pde.killing_rate == r

    def test_ou_pde_coefficients(self):
        """OU: μ(x)=κ(θ−x), σ(x)=σ."""
        kappa, theta, sigma = 2.0, 0.04, 0.3
        pde = feynman_kac_pde(
            drift=lambda x: kappa * (theta - x),
            diffusion=lambda x: sigma,
        )
        assert pde.drift_coeff(0.1) == pytest.approx(2.0 * (0.04 - 0.1))
        assert pde.diffusion_coeff(0.1) == pytest.approx(0.5 * 0.09)

    def test_feynman_kac_verify_passes(self):
        result = feynman_kac_verify(10.45, 10.50, tol=0.05)
        assert result.passed

    def test_feynman_kac_verify_fails(self):
        result = feynman_kac_verify(10.45, 12.00, tol=0.05)
        assert not result.passed

    def test_bs_pde_mc_consistency(self):
        """BS PDE price should match MC price (the Feynman-Kac theorem)."""
        from pricebook.equity_option import equity_option_price
        from pricebook.black76 import OptionType
        from pricebook.finite_difference import fd_european

        spot, strike, rate, vol, T = 100, 100, 0.05, 0.20, 1.0
        bs = equity_option_price(spot, strike, rate, vol, T, OptionType.CALL)
        fd = fd_european(spot, strike, rate, vol, T, OptionType.CALL)

        result = feynman_kac_verify(fd, bs, tol=0.02)
        assert result.passed
