"""Tests for backlog closure: HV ADI, Strang MC, SDP, sparse Jacobian."""

import pytest
import math
import numpy as np


class TestHVADI:
    def test_hv_heston_call(self):
        from pricebook.models.hundsdorfer_verwer import hv_adi_heston
        price = hv_adi_heston(100, 100, 0.04, 1.0, 0.04, 2.0, 0.04, 0.3, -0.7,
                               n_x=40, n_v=20, n_time=50)
        assert price > 0

    def test_hv_vs_cs(self):
        """HV and Craig-Sneyd should give similar prices."""
        from pricebook.models.hundsdorfer_verwer import hv_adi_heston
        from pricebook.models.adi import heston_pde
        from pricebook.models.black76 import OptionType
        hv = hv_adi_heston(100, 100, 0.04, 1.0, 0.04, 2.0, 0.04, 0.3, -0.7,
                            n_x=40, n_v=20, n_time=50)
        cs = heston_pde(100, 100, 0.04, 1.0, 0.04, 2.0, 0.04, 0.3, -0.7,
                         OptionType.CALL, n_x=40, n_v=20, n_time=50)
        assert hv == pytest.approx(cs, rel=0.15)

    def test_hv_put(self):
        from pricebook.models.hundsdorfer_verwer import hv_adi_heston
        price = hv_adi_heston(100, 100, 0.04, 1.0, 0.04, 2.0, 0.04, 0.3, -0.7,
                               is_call=False, n_x=40, n_v=20, n_time=50)
        assert price > 0


class TestStrangMC:
    def test_merton_strang(self):
        from pricebook.models.sde_strang import strang_merton_mc
        r = strang_merton_mc(100, 100, 0.04, 0.20, 1.0, n_paths=50_000)
        assert r.price > 0
        assert r.stderr > 0

    def test_strang_vs_no_jumps(self):
        """Zero jump intensity → should match BS approximately."""
        from pricebook.models.sde_strang import strang_merton_mc
        from pricebook.models.black76 import black76_price, OptionType
        r = strang_merton_mc(100, 100, 0.04, 0.20, 1.0, jump_intensity=0.0, n_paths=100_000)
        bs = black76_price(100 * math.exp(0.04), 100, 0.20, 1.0, math.exp(-0.04), OptionType.CALL)
        assert r.price == pytest.approx(bs, rel=0.05)

    def test_bates_strang(self):
        from pricebook.models.sde_strang import strang_bates_mc
        r = strang_bates_mc(100, 100, 0.04, 1.0, 0.04, 2.0, 0.04, 0.3, -0.7,
                             n_paths=50_000)
        assert r.price > 0


class TestSDP:
    def test_nearest_psd(self):
        from pricebook.numerical.sdp import nearest_psd
        M = np.array([[1, 2], [2, 1]])  # not PSD (eigenvalues 3, -1)
        P = nearest_psd(M)
        eigvals = np.linalg.eigvalsh(P)
        assert all(e >= -1e-10 for e in eigvals)

    def test_nearest_correlation(self):
        from pricebook.numerical.sdp import nearest_correlation_sdp
        # Start with invalid correlation (not PSD)
        A = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.5], [0.9, -0.5, 1.0]])
        r = nearest_correlation_sdp(A)
        assert r.feasible
        # Check unit diagonal
        assert all(abs(r.X[i, i] - 1.0) < 1e-6 for i in range(3))

    def test_factor_covariance(self):
        from pricebook.numerical.sdp import factor_covariance_bounds
        B = np.array([[1.0, 0.5], [0.8, 0.3], [0.6, 0.7]])
        Sigma_F = np.array([[0.04, 0.01], [0.01, 0.02]])
        idio = np.array([0.01, 0.015, 0.008])
        r = factor_covariance_bounds(B, Sigma_F, idio)
        assert r["n_assets"] == 3
        assert r["n_factors"] == 2
        assert r["min_variance_vol"] > 0


class TestSparseJacobian:
    def test_tridiagonal(self):
        from pricebook.numerical.sparse_jacobian import banded_jacobian
        # Tridiagonal system: f_i = x_{i-1} - 2x_i + x_{i+1}
        def f(x):
            n = len(x)
            result = np.zeros(n)
            for i in range(n):
                result[i] = -2 * x[i]
                if i > 0:
                    result[i] += x[i - 1]
                if i < n - 1:
                    result[i] += x[i + 1]
            return result

        x = np.ones(20)
        r = banded_jacobian(f, x, bandwidth=1)
        assert r.n_colors == 3  # tridiagonal needs 3 colours
        assert r.n_evaluations == 4  # 1 base + 3 perturbations
        # Check J is correct (should be tridiagonal with -2 on diagonal)
        assert r.J[5, 5] == pytest.approx(-2.0, abs=0.01)
        assert r.J[5, 4] == pytest.approx(1.0, abs=0.01)
        assert r.J[5, 6] == pytest.approx(1.0, abs=0.01)

    def test_detect_sparsity(self):
        from pricebook.numerical.sparse_jacobian import detect_sparsity
        def f(x):
            return np.array([x[0] + x[1], x[1] + x[2], x[2]])
        sparsity = detect_sparsity(f, np.ones(3))
        assert sparsity[0, 0] == True   # f0 depends on x0
        assert sparsity[0, 2] == False  # f0 does NOT depend on x2
        assert sparsity[2, 0] == False  # f2 does NOT depend on x0

    def test_speedup(self):
        from pricebook.numerical.sparse_jacobian import banded_jacobian
        def f(x):
            n = len(x)
            result = np.zeros(n)
            for i in range(n):
                result[i] = x[i]**2
                if i > 0:
                    result[i] += x[i-1]
            return result
        x = np.ones(100)
        r = banded_jacobian(f, x, bandwidth=1)
        assert r.n_evaluations < 10  # much less than 101
        assert r.sparsity_ratio < 0.1
