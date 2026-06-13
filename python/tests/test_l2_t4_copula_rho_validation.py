"""Regression for L2 Wave-2 audit — `GaussianCopula` and `StudentTCopula`
constructors had no validation on `rho`.

Pre-fix:
- `rho > 1`: ``math.sqrt(1 - rho)`` raised opaque ``ValueError: math
  domain error`` deep inside the sample() call.
- `rho < 0`: ``math.sqrt(rho)`` raised the same domain error.
- `rho = NaN`: slipped past everything via IEEE 754 (NaN comparisons
  are False) and silently produced an all-NaN sample array (since
  ``norm.cdf(NaN) = NaN``).
- `StudentTCopula(nu<=0)`: produced a degenerate Student-t.

Post-fix both constructors validate rho ∈ [0, 1] and (for StudentT)
nu > 0, raising ``ValueError`` upfront with a diagnostic message.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.statistics.copulas import GaussianCopula, StudentTCopula


class TestGaussianCopulaRho:
    def test_rho_above_one_raises(self):
        with pytest.raises(ValueError, match="rho must be in"):
            GaussianCopula(rho=1.5)

    def test_rho_below_zero_raises(self):
        with pytest.raises(ValueError, match="rho must be in"):
            GaussianCopula(rho=-0.5)

    def test_rho_nan_raises(self):
        with pytest.raises(ValueError, match="NaN slipped"):
            GaussianCopula(rho=float("nan"))

    def test_valid_rho_works(self):
        c = GaussianCopula(rho=0.3)
        u = c.sample(100, 5, np.random.default_rng(42))
        assert u.shape == (100, 5)
        assert ((u >= 0) & (u <= 1)).all()


class TestStudentTCopulaRhoNu:
    def test_rho_out_of_range_raises(self):
        with pytest.raises(ValueError, match="rho must be in"):
            StudentTCopula(rho=1.2, nu=5.0)

    def test_nu_zero_raises(self):
        with pytest.raises(ValueError, match="nu must be > 0"):
            StudentTCopula(rho=0.3, nu=0.0)

    def test_nu_negative_raises(self):
        with pytest.raises(ValueError, match="nu must be > 0"):
            StudentTCopula(rho=0.3, nu=-1.0)

    def test_nan_in_either_raises(self):
        with pytest.raises(ValueError, match="NaN slipped"):
            StudentTCopula(rho=float("nan"), nu=5.0)
        with pytest.raises(ValueError, match="NaN slipped"):
            StudentTCopula(rho=0.3, nu=float("nan"))

    def test_valid_params_work(self):
        c = StudentTCopula(rho=0.5, nu=4.0)
        u = c.sample(100, 3, np.random.default_rng(42))
        assert u.shape == (100, 3)
        assert ((u >= 0) & (u <= 1)).all()
