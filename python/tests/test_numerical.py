"""Tests for pricebook.numerical — comprehensive numerical methods."""

import math

import numpy as np
import pytest

from pricebook.numerical import (
    # Distributions
    Normal, StudentT, LogNormal, Uniform, Exponential,
    # Linear algebra
    qr, cholesky, expm, gmres, sylvester, cond, is_positive_definite,
    # ODE
    euler, rk4, rk45, bdf,
    # Optimisation
    minimize, linprog, qp, bisection, find_root,
    proximal_gradient, projection_simplex, soft_threshold,
    # Quadrature
    gauss_jacobi, tanh_sinh, clenshaw_curtis,
    # Interpolation
    bilinear, rbf_interpolate,
    # MC
    qe_heston_step,
    # Trees
    binomial_2d,
    # Fourier
    fractional_fft, hilbert_transform, wavelet_transform,
    # Distribution theory
    dirac_delta, SchwartzTestFunction, sobolev_norm,
)


# ═══════════════════════════════════════════════════════════════
# Distributions
# ═══════════════════════════════════════════════════════════════

class TestNormal:
    def test_cdf_symmetry(self):
        n = Normal()
        assert n.cdf(0) == pytest.approx(0.5)
        assert n.cdf(1.96) == pytest.approx(0.975, abs=0.001)

    def test_pdf_peak(self):
        assert Normal().pdf(0) == pytest.approx(1 / math.sqrt(2 * math.pi), abs=1e-10)

    def test_ppf_roundtrip(self):
        n = Normal(mu=5, sigma=2)
        assert n.ppf(n.cdf(7.0)) == pytest.approx(7.0, abs=1e-6)

    def test_vectorised(self):
        r = Normal().cdf(np.array([0, 1, 2]))
        assert len(r) == 3

    def test_rvs_shape(self):
        assert len(Normal().rvs(100)) == 100


class TestStudentT:
    def test_cdf_symmetric(self):
        assert StudentT(5).cdf(0) == pytest.approx(0.5, abs=1e-10)

    def test_heavier_tails(self):
        # t(3) has heavier tails than normal
        assert StudentT(3).cdf(-3) > Normal().cdf(-3)


class TestLogNormal:
    def test_cdf_at_median(self):
        assert LogNormal(0, 1).cdf(1.0) == pytest.approx(0.5, abs=1e-10)

    def test_mean(self):
        assert LogNormal(0, 0.5).mean() == pytest.approx(math.exp(0.125), abs=1e-6)


# ═══════════════════════════════════════════════════════════════
# Linear Algebra
# ═══════════════════════════════════════════════════════════════

class TestLinAlg:
    def test_qr_decomposition(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        r = qr(A)
        assert np.allclose(r.Q @ r.R, A, atol=1e-12)

    def test_cholesky(self):
        S = np.array([[4, 2], [2, 3]], dtype=float)
        L = cholesky(S)
        assert np.allclose(L @ L.T, S, atol=1e-12)

    def test_expm_identity(self):
        assert np.allclose(expm(np.zeros((2, 2))), np.eye(2), atol=1e-12)

    def test_gmres_converges(self):
        A = np.eye(10) * 5 + np.random.default_rng(42).normal(0, 0.1, (10, 10))
        b = np.ones(10)
        r = gmres(A, b)
        assert r.converged
        assert np.allclose(A @ r.x, b, atol=1e-6)

    def test_sylvester(self):
        A = np.diag([1, 2])
        B = np.diag([3, 4])
        C = np.eye(2)
        X = sylvester(A, B, C)
        assert np.allclose(A @ X + X @ B, C, atol=1e-10)

    def test_is_pd(self):
        assert is_positive_definite(np.eye(3))
        assert not is_positive_definite(np.array([[-1, 0], [0, 1]]))


# ═══════════════════════════════════════════════════════════════
# ODE
# ═══════════════════════════════════════════════════════════════

class TestODE:
    def test_euler_decay(self):
        r = euler(lambda t, y: -y, (0, 5), 1.0, n_steps=10000)
        assert r.y[-1, 0] == pytest.approx(math.exp(-5), abs=1e-3)

    def test_rk4_accuracy(self):
        r = rk4(lambda t, y: -y, (0, 5), 1.0, n_steps=100)
        assert r.y[-1, 0] == pytest.approx(math.exp(-5), abs=1e-8)

    def test_rk45_adaptive(self):
        r = rk45(lambda t, y: -y, (0, 5), 1.0, tol=1e-10)
        assert r.success
        assert r.y[-1, 0] == pytest.approx(math.exp(-5), abs=1e-9)


# ═══════════════════════════════════════════════════════════════
# Optimisation
# ═══════════════════════════════════════════════════════════════

class TestOptimize:
    def test_minimize_rosenbrock(self):
        def rosenbrock(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        r = minimize(rosenbrock, [0, 0], method="bfgs")
        assert r.converged
        assert abs(r.x[0] - 1.0) < 0.01
        assert abs(r.x[1] - 1.0) < 0.01

    def test_linprog(self):
        # min -x-2y s.t. x+y<=4, x<=3, x,y>=0
        r = linprog(np.array([-1, -2]),
                    A_ub=np.array([[1, 1], [1, 0]]),
                    b_ub=np.array([4, 3]),
                    bounds=[(0, None), (0, None)])
        assert r.converged
        assert r.fun == pytest.approx(-8.0, abs=0.1)

    def test_projection_simplex(self):
        x = np.array([0.5, 0.3, 0.8])
        p = projection_simplex(x)
        assert np.sum(p) == pytest.approx(1.0, abs=1e-10)
        assert np.all(p >= 0)

    def test_bisection(self):
        r = bisection(lambda x: x ** 2 - 2, 0, 2)
        assert r.converged
        assert r.root == pytest.approx(math.sqrt(2), abs=1e-10)


# ═══════════════════════════════════════════════════════════════
# Quadrature
# ═══════════════════════════════════════════════════════════════

class TestQuadrature:
    def test_gauss_jacobi_legendre(self):
        # alpha=beta=0 → Gauss-Legendre
        r = gauss_jacobi(lambda x: x ** 2, n=8, a=0, b=1)
        assert r.value == pytest.approx(1.0 / 3, abs=1e-10)

    def test_clenshaw_curtis(self):
        r = clenshaw_curtis(lambda x: x ** 2, a=0, b=1, n=16)
        assert r.value == pytest.approx(1.0 / 3, abs=1e-6)


# ═══════════════════════════════════════════════════════════════
# Interpolation
# ═══════════════════════════════════════════════════════════════

class TestInterpolation:
    def test_bilinear(self):
        xs = np.array([0, 1, 2], dtype=float)
        ys = np.array([0, 1, 2], dtype=float)
        zs = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=float)
        assert bilinear(0.5, 0.5, xs, ys, zs) == pytest.approx(1.0, abs=0.1)

    def test_rbf(self):
        centers = np.array([[0], [1], [2]], dtype=float)
        values = np.array([0, 1, 0], dtype=float)
        rbf = rbf_interpolate(centers, values, kernel="gaussian")
        result = rbf.evaluate(np.array([[1.0]]))
        assert result[0] == pytest.approx(1.0, abs=0.1)


# ═══════════════════════════════════════════════════════════════
# MC
# ═══════════════════════════════════════════════════════════════

class TestMC:
    def test_qe_heston_positive(self):
        v = np.array([0.04, 0.01, 0.10])
        v_next = qe_heston_step(v, kappa=2, theta=0.04, xi=0.3, dt=0.01)
        assert np.all(v_next >= 0)

    def test_qe_heston_mean_reversion(self):
        v = np.full(10000, 0.10)
        for _ in range(100):
            v = qe_heston_step(v, kappa=5, theta=0.04, xi=0.3, dt=0.01)
        assert v.mean() == pytest.approx(0.04, abs=0.01)


# ═══════════════════════════════════════════════════════════════
# Trees
# ═══════════════════════════════════════════════════════════════

class TestTrees:
    def test_2d_binomial_positive(self):
        r = binomial_2d(100, 100, 5, 0.05, 0.2, 0.2, 0.5, 1.0, n_steps=30)
        assert r.price > 0

    def test_2d_spread_put_positive(self):
        r = binomial_2d(100, 100, 5, 0.05, 0.2, 0.2, 0.5, 1.0,
                        n_steps=30, payoff_type="spread_put")
        assert r.price >= 0


# ═══════════════════════════════════════════════════════════════
# Fourier
# ═══════════════════════════════════════════════════════════════

class TestFourier:
    def test_fractional_fft_standard(self):
        x = np.array([1, 0, 0, 0], dtype=float)
        # alpha=1 should give standard DFT (all ones for delta)
        r = fractional_fft(x, 1.0)
        assert r[0] == pytest.approx(1.0, abs=0.1)

    def test_wavelet_haar(self):
        x = np.ones(16)
        w = wavelet_transform(x, levels=2, wavelet="haar")
        assert w.levels == 2
        assert len(w.coefficients) == 16


# ═══════════════════════════════════════════════════════════════
# Distribution Theory
# ═══════════════════════════════════════════════════════════════

class TestDistributionTheory:
    def test_dirac_delta(self):
        phi = SchwartzTestFunction(sigma=1.0)
        delta = dirac_delta(0.0)
        assert delta(phi) == pytest.approx(phi(0), abs=1e-10)

    def test_sobolev_l2(self):
        f = np.sin(np.linspace(0, 2 * math.pi, 256))
        s = sobolev_norm(f, dx=2 * math.pi / 256, s=0)
        assert s.h0 > 0

    def test_dirac_at_nonzero(self):
        delta = dirac_delta(2.0)
        assert delta(lambda x: x ** 2) == pytest.approx(4.0, abs=1e-10)
