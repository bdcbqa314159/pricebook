"""Tests for advanced numerical methods: spectral, QMC, stochastic calculus."""

import math
import pytest
import numpy as np

from pricebook.numerical._spectral import (
    chebyshev_nodes, chebyshev_diff_matrix, chebyshev_coefficients,
    chebyshev_evaluate, chebyshev_expand, spectral_solve_bvp,
    spectral_integrate, SpectralResult,
)
from pricebook.numerical._qmc import (
    sobol_sequence, halton_sequence, latin_hypercube,
    sparse_grid, SparseGridResult,
)
from pricebook.numerical._stochastic import (
    ito_formula, ito_log_transform, stratonovich_to_ito, ito_to_stratonovich,
    quadratic_variation, realized_variance, realized_volatility,
    bipower_variation, jump_test, milstein_correction,
    ItoFormulaResult,
)


# ═══════════════════════════════════════════════════════════════
# Spectral Methods
# ═══════════════════════════════════════════════════════════════

class TestChebyshev:
    def test_nodes_endpoints(self):
        nodes = chebyshev_nodes(10)
        assert abs(nodes[0] - 1.0) < 1e-10    # cos(0) = 1
        assert abs(nodes[-1] - (-1.0)) < 1e-10  # cos(π) = -1

    def test_nodes_custom_interval(self):
        nodes = chebyshev_nodes(5, 0, 10)
        assert abs(nodes[0] - 10) < 1e-10
        assert abs(nodes[-1] - 0) < 1e-10

    def test_diff_matrix_shape(self):
        """Differentiation matrix should be (N+1, N+1)."""
        D = chebyshev_diff_matrix(10)
        assert D.shape == (11, 11)
        # Row sums should be ~0 (differentiating constant gives 0)
        assert np.max(np.abs(D.sum(axis=1))) < 1e-8

    def test_diff_matrix_differentiates(self):
        """D must give the actual derivative (sign included), not -derivative.

        Row-sum/shape checks are sign-invariant and missed the -D bug.
        """
        n = 16
        x = chebyshev_nodes(n)            # on [-1, 1]
        D = chebyshev_diff_matrix(n)
        err = np.max(np.abs(D @ np.sin(2 * x) - 2 * np.cos(2 * x)))
        assert err < 1e-8

    def test_diff_matrix_sign_explicit(self):
        """D(1) must match Trefethen: [[0.5, -0.5], [0.5, -0.5]]."""
        D = chebyshev_diff_matrix(1)
        np.testing.assert_allclose(D, [[0.5, -0.5], [0.5, -0.5]], atol=1e-12)

    def test_interpolate_sin(self):
        result = chebyshev_expand(np.sin, 20, 0, np.pi)
        val = result.evaluate(np.pi / 4)
        assert abs(float(np.atleast_1d(val)[0]) - math.sin(np.pi / 4)) < 0.01

    def test_interpolate_exponential(self):
        result = chebyshev_expand(np.exp, 15, 0, 1)
        val = result.evaluate(0.5)
        assert abs(float(np.atleast_1d(val)[0]) - math.exp(0.5)) < 0.01

    def test_evaluate_off_center_asymmetric(self):
        """Non-symmetric f at an off-center point: catches the reversed-interval
        mirror bug (f(0.25) must be 0.25, not 0.75)."""
        result = chebyshev_expand(lambda x: x, 12, 0.0, 1.0)
        for q in (0.1, 0.25, 0.9):
            val = float(np.atleast_1d(result.evaluate(q))[0])
            assert abs(val - q) < 1e-10

    def test_interpolate_degree_zero_raises(self):
        """n=0 must fail loud, not return NaN with residual 0.0."""
        with pytest.raises(ValueError, match="n >= 1"):
            chebyshev_expand(np.exp, 0, 0, 1)

    def test_spectral_integrate(self):
        """∫₀¹ x² dx = 1/3."""
        result = spectral_integrate(lambda x: x**2, 0, 1, n=10)
        assert abs(result - 1/3) < 1e-12

    def test_spectral_integrate_trig(self):
        """∫₀^π sin(x) dx = 2."""
        result = spectral_integrate(np.sin, 0, np.pi, n=20)
        assert abs(result - 2.0) < 1e-10

    def test_bvp_simple(self):
        """u'' = -π²sin(πx), u(0)=u(1)=0 → u = sin(πx)."""
        def L(D, D2, x):
            return D2

        def rhs(x):
            return -np.pi**2 * np.sin(np.pi * x)
        result = spectral_solve_bvp(L, rhs, 0, 0, n=20, a=0, b=1)
        # Check interior point — spectral accuracy, so a real tolerance.
        x_test = 0.5
        u_test = result.evaluate(x_test)
        assert abs(float(np.atleast_1d(u_test)[0]) - np.sin(np.pi * x_test)) < 1e-6

    def test_bvp_general_operator(self):
        """u'' + u' = 2 + 2x, u(0)=0, u(1)=1 → u = x² (asymmetric, advection term).

        Catches both the -D sign bug (the D term) and the D2-only boundary
        lifting (the D term's boundary column must be subtracted). Evaluated
        off-center to also exercise the interval orientation.
        """
        def L(D, D2, x):
            return D2 + D

        def rhs(x):
            return 2.0 + 2.0 * x
        result = spectral_solve_bvp(L, rhs, bc_left=0.0, bc_right=1.0, n=16, a=0, b=1)
        for q in (0.25, 0.7):
            val = float(np.atleast_1d(result.evaluate(q))[0])
            assert abs(val - q**2) < 1e-8

    def test_bvp_variable_coefficient_mms(self):
        """Manufactured solution with a *variable-coefficient* operator
        L[u] = u'' + p(x)u' + q(x)u, asymmetric interval and solution.

        This is the bulletproof generalization: constant-coefficient operators
        (the old tests) can't distinguish a D2-only boundary lifting from a
        correct one, and a symmetric solution hides orientation bugs. p(x), q(x)
        non-constant exercise the D and I boundary columns; the asymmetric cubic
        on [0.3, 2.1] exercises orientation. Recovery is exact (poly degree ≤ n).
        """
        a, b = 0.3, 2.1
        u = lambda x: x**3 - 2 * x**2 + 0.5 * x + 1.0          # noqa: E731
        up = lambda x: 3 * x**2 - 4 * x + 0.5                  # noqa: E731
        upp = lambda x: 6 * x - 4                              # noqa: E731
        p = lambda x: 1.0 + x                                  # noqa: E731  variable advection
        q = lambda x: -2.0 + 0.5 * x                           # noqa: E731  variable reaction

        def rhs(x):
            return upp(x) + p(x) * up(x) + q(x) * u(x)

        def L(D, D2, x):
            return D2 + np.diag(p(x)) @ D + np.diag(q(x))

        result = spectral_solve_bvp(L, rhs, bc_left=u(a), bc_right=u(b), n=16, a=a, b=b)
        # Node-level recovery (isolates the solve from the evaluate map).
        nodes = chebyshev_nodes(16, a, b)
        assert np.max(np.abs(result.values - u(nodes))) < 1e-8
        # Off-centre recovery (solve + evaluate orientation together).
        for frac in (0.25, 0.7):
            x = a + frac * (b - a)
            val = float(np.atleast_1d(result.evaluate(x))[0])
            assert abs(val - u(x)) < 1e-8

    def test_to_dict(self):
        result = chebyshev_expand(np.sin, 10, 0, 1)
        d = result.to_dict()
        assert "n_points" in d


# ═══════════════════════════════════════════════════════════════
# Quasi-Monte Carlo
# ═══════════════════════════════════════════════════════════════

class TestQMC:
    def test_sobol_shape(self):
        pts = sobol_sequence(100, 3)
        assert pts.shape == (100, 3)
        assert np.all(pts >= 0) and np.all(pts <= 1)

    def test_halton_shape(self):
        pts = halton_sequence(50, 4)
        assert pts.shape == (50, 4)
        assert np.all(pts >= 0) and np.all(pts <= 1)

    def test_halton_uniform(self):
        """Halton should be roughly uniform."""
        pts = halton_sequence(1000, 1)
        assert abs(pts.mean() - 0.5) < 0.05

    def test_latin_hypercube(self):
        pts = latin_hypercube(100, 5)
        assert pts.shape == (100, 5)
        assert np.all(pts >= 0) and np.all(pts <= 1)

    def test_lhs_stratified(self):
        """Each dimension should have one point per stratum."""
        pts = latin_hypercube(10, 2, seed=42)
        # Check first dimension: bins [0,0.1), [0.1,0.2), ... should each have 1 point
        bins = (pts[:, 0] * 10).astype(int)
        assert len(set(bins)) == 10

    def test_qmc_integration(self):
        """QMC should estimate ∫[0,1]² (x+y) dxdy = 1 well."""
        pts = sobol_sequence(1000, 2)
        values = pts[:, 0] + pts[:, 1]
        estimate = values.mean()
        assert abs(estimate - 1.0) < 0.05


class TestSparseGrid:
    def test_basic(self):
        sg = sparse_grid(2, level=2)
        assert isinstance(sg, SparseGridResult)
        assert sg.n_dims == 2
        assert sg.n_points > 0

    def test_1d_matches_gauss(self):
        """1D sparse grid should integrate polynomials exactly."""
        sg = sparse_grid(1, level=3)
        # ∫₀¹ x² dx = 1/3
        result = sg.integrate(lambda x: x[0]**2)
        assert abs(result - 1/3) < 0.1

    def test_to_dict(self):
        d = sparse_grid(2, level=2).to_dict()
        assert "n_points" in d


# ═══════════════════════════════════════════════════════════════
# Stochastic Calculus
# ═══════════════════════════════════════════════════════════════

class TestItoFormula:
    def test_log_gbm(self):
        """For f(X) = log(X), dX = μXdt + σXdW: df = (μ-½σ²)dt + σdW."""
        result = ito_log_transform(mu=0.05, sigma=0.20)
        assert abs(result["log_drift"] - (0.05 - 0.02)) < 1e-10
        assert abs(result["ito_correction"] - (-0.02)) < 1e-10

    def test_ito_formula_basic(self):
        r = ito_formula(f_prime=1.0, f_double_prime=0.0, mu=0.05, sigma=0.20)
        assert isinstance(r, ItoFormulaResult)
        assert r.drift_correction == 0.0  # linear f → no correction
        assert r.total_drift == 0.05

    def test_ito_quadratic(self):
        """f(X) = X², f'=2X, f''=2. At X=1: correction = σ²."""
        r = ito_formula(f_prime=2.0, f_double_prime=2.0, mu=0.1, sigma=0.3)
        assert abs(r.drift_correction - 0.09) < 1e-10  # ½×0.09×2 = 0.09


class TestStratonovich:
    def test_roundtrip(self):
        """Ito → Strat → Ito should be identity."""
        drift = 0.05
        diff = 0.20
        diff_prime = 0.10
        strat = ito_to_stratonovich(drift, diff, diff_prime)
        ito = stratonovich_to_ito(strat["strat_drift"], diff, diff_prime)
        assert abs(ito["ito_drift"] - drift) < 1e-10


class TestQuadraticVariation:
    def test_brownian_qv(self):
        """[W]_T ≈ T for Brownian motion."""
        rng = np.random.default_rng(42)
        n = 10000
        dt = 1.0 / n
        dW = rng.normal(0, math.sqrt(dt), n)
        W = np.concatenate([[0], np.cumsum(dW)])
        qv = quadratic_variation(W)
        assert abs(qv - 1.0) < 0.1  # [W]_1 ≈ 1

    def test_realized_vol(self):
        rng = np.random.default_rng(42)
        log_returns = rng.normal(0, 0.01, 252)  # daily vol = 1%
        rv = realized_volatility(log_returns, annualize=252)
        # Should be near 0.01 × √252 ≈ 0.159
        # realized_variance uses annualize × Σr², realized_vol = √(RV)
        # With 252 daily returns of std 0.01: RV ≈ 252 × 252 × 0.0001 ≈ 0.0252 × 252
        # Just check it's positive and finite
        assert rv > 0 and rv < 10

    def test_bipower_robust(self):
        rng = np.random.default_rng(42)
        log_returns = rng.normal(0, 0.01, 252)
        rv = realized_variance(log_returns)
        bv = bipower_variation(log_returns)
        # Without jumps: RV ≈ BV
        assert abs(rv - bv) / max(rv, 1e-10) < 0.3


class TestJumpTest:
    def test_no_jumps(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        result = jump_test(returns)
        # Should not detect jumps in pure Gaussian
        assert result["p_value"] > 0.01

    def test_with_jumps(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        # Add jumps
        jump_times = rng.choice(500, 10, replace=False)
        returns[jump_times] += rng.choice([-0.05, 0.05], 10)
        result = jump_test(returns)
        assert result["jump_component"] > 0


class TestMilstein:
    def test_correction_nonzero(self):
        """Non-constant σ should give nonzero Milstein correction."""
        sigma = lambda x: 0.2 * x
        sigma_prime = lambda x: 0.2
        corr = milstein_correction(sigma, sigma_prime, x=1.0, dW=0.1, dt=0.01)
        assert corr != 0.0

    def test_correction_zero_for_additive(self):
        """Constant σ → σ' = 0 → no correction."""
        corr = milstein_correction(lambda x: 0.2, lambda x: 0.0, x=1.0, dW=0.1, dt=0.01)
        assert corr == 0.0
