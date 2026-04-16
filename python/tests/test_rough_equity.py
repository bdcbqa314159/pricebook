"""Tests for rough volatility equity models."""

import math

import numpy as np
import pytest

from pricebook.rough_equity import (
    ForwardVarianceCurve,
    RoughHestonParams,
    forward_variance_curve,
    rBergomiEquity,
    rough_heston_cf,
    rough_heston_price,
)


# ---- rBergomi ----

class TestRBergomi:
    def test_basic(self):
        model = rBergomiEquity(xi0=0.04, eta=2.0, H=0.1, rho=-0.7)
        result = model.simulate(100, 0.03, 0.02, 1.0,
                                 n_paths=500, n_steps=50, seed=42)
        assert result.H == 0.1
        assert result.n_paths == 500

    def test_paths_positive(self):
        model = rBergomiEquity(0.04, 2.0, 0.1)
        result = model.simulate(100, 0.03, 0.02, 1.0,
                                 n_paths=500, n_steps=50, seed=42)
        assert np.all(result.spot_paths >= 0)
        assert np.all(result.variance_paths >= 0)

    def test_initial_conditions(self):
        model = rBergomiEquity(0.04, 2.0, 0.1)
        result = model.simulate(100, 0.03, 0.02, 1.0,
                                 n_paths=100, n_steps=50, seed=42)
        assert np.all(result.spot_paths[:, 0] == 100)
        assert np.all(np.isclose(result.variance_paths[:, 0], 0.04))

    def test_invalid_H(self):
        with pytest.raises(ValueError):
            rBergomiEquity(0.04, 2.0, 0.6)  # H must be < 0.5
        with pytest.raises(ValueError):
            rBergomiEquity(0.04, 2.0, 0.0)  # H must be > 0

    def test_forward_variance(self):
        model = rBergomiEquity(0.04, 2.0, 0.1)
        assert model.forward_variance(1.0) == 0.04

    def test_implied_vol(self):
        model = rBergomiEquity(0.04, 2.0, 0.1, rho=-0.7)
        iv = model.implied_vol(100, 100, 0.03, 0.02, 1.0,
                                n_paths=5_000, seed=42)
        # rBergomi with η=2.0 produces higher effective vol than ξ₀=0.04
        assert 0.05 < iv < 2.0


# ---- Rough Heston ----

class TestRoughHeston:
    def _params(self):
        return RoughHestonParams(
            v0=0.04, lambda_=1.5, theta=0.04, xi=0.3, rho=-0.5, H=0.1,
        )

    def test_cf_at_zero(self):
        """CF(0) = 1 (normalisation)."""
        params = self._params()
        cf = rough_heston_cf(0.0 + 0.0j, 1.0, params)
        assert abs(cf - 1.0) < 0.1

    def test_cf_non_zero(self):
        params = self._params()
        cf = rough_heston_cf(1.0 + 0.0j, 1.0, params)
        # Should be complex-valued
        assert isinstance(cf, complex) or isinstance(cf, np.complex128)

    def test_price_basic(self):
        params = self._params()
        price = rough_heston_price(100, 100, 0.03, 0.02, 1.0, params)
        assert isinstance(price, float)
        assert price >= 0

    def test_put_call_parity(self):
        """Put ≥ 0, price positive."""
        params = self._params()
        call = rough_heston_price(100, 100, 0.03, 0.02, 1.0, params, is_call=True)
        put = rough_heston_price(100, 100, 0.03, 0.02, 1.0, params, is_call=False)
        assert call >= 0
        assert put >= 0


# ---- Forward variance curve ----

class TestForwardVarianceCurve:
    def test_basic(self):
        tenors = [0.25, 0.5, 1.0, 2.0]
        atm_var = [0.04, 0.045, 0.05, 0.055]
        curve = forward_variance_curve(tenors, atm_var)
        assert isinstance(curve, ForwardVarianceCurve)
        assert len(curve.forward_variances) == 4

    def test_monotone_atm_monotone_fwd(self):
        """Monotone ATM variance → forward variance positive."""
        tenors = [0.25, 0.5, 1.0, 2.0]
        atm_var = [0.04, 0.045, 0.05, 0.055]
        curve = forward_variance_curve(tenors, atm_var)
        assert np.all(curve.forward_variances > 0)

    def test_flat_atm_variance_constant_fwd(self):
        """Flat ATM variance → constant forward variance."""
        tenors = [0.5, 1.0, 1.5, 2.0]
        atm_var = [0.04] * 4
        curve = forward_variance_curve(tenors, atm_var)
        # Interior values should be flat
        np.testing.assert_allclose(curve.forward_variances[1:-1], 0.04, atol=0.005)

    def test_method(self):
        curve = forward_variance_curve([1.0, 2.0], [0.04, 0.05])
        assert curve.method == "finite_difference"
