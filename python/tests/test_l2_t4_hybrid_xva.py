"""Regression for L2 phase-2 audit of `risk.hybrid_xva`:

Pre-fix `hybrid_cva` and `hybrid_fva` omitted the discount factor
``D(0, t_i)`` in their summations.  Both produced *future-valued*
XVA — the textbook formulas require present-valuing each time-step's
exposure before summing.  For a 10y trade with rates ~5%, the pre-fix
overstated CVA/FVA by ~30%-40%.

Fix: added optional ``discount_factors`` parameter to both.  When
supplied, multiplies EPE/EE by ``D(0, t_i)`` element-wise before
summing.  Backwards compatible — omitting it preserves the old behaviour
(now documented as requiring pre-discounted exposure inputs).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.risk.hybrid_xva import hybrid_cva, hybrid_fva


class TestCVADiscounting:
    def test_discount_factor_reduces_cva(self):
        """With finite rates, discounted CVA < undiscounted CVA."""
        n_paths, n_steps_p1 = 200, 51
        rng = np.random.default_rng(42)
        # Positive expected exposure paths.
        exp = np.abs(rng.standard_normal((n_paths, n_steps_p1)) * 100)
        pd = np.linspace(0, 0.02, n_steps_p1)

        # Without discounting (pre-fix behaviour, treats as PV).
        no_df = hybrid_cva(exp, pd, recovery=0.4)
        # With discount factors at 5% rate over 10 years.
        times = np.linspace(0, 10, n_steps_p1)
        df = np.exp(-0.05 * times)
        with_df = hybrid_cva(exp, pd, recovery=0.4, discount_factors=df)

        assert with_df.cva < no_df.cva
        # Ratio should be in the 0.7-0.95 range (avg DF over 10y at 5% ≈ 0.78).
        assert 0.6 < with_df.cva / no_df.cva < 0.95

    def test_df_shape_validation(self):
        exp = np.zeros((10, 21))
        pd = np.zeros(21)
        with pytest.raises(ValueError, match="discount_factors"):
            hybrid_cva(exp, pd, discount_factors=np.ones(10))  # wrong shape

    def test_df_one_unchanged(self):
        """All discount factors = 1 → result matches no-discounting."""
        exp = np.array([[10.0, 20.0, 30.0], [5.0, 10.0, 15.0]])
        pd = np.array([0.0, 0.01, 0.02])
        no_df = hybrid_cva(exp, pd, recovery=0.4)
        with_ones = hybrid_cva(exp, pd, recovery=0.4,
                                discount_factors=np.ones(3))
        assert with_ones.cva == pytest.approx(no_df.cva, abs=1e-12)


class TestFVADiscounting:
    def test_discount_factor_reduces_fva(self):
        n_paths, n_steps = 200, 50
        rng = np.random.default_rng(42)
        exp = np.abs(rng.standard_normal((n_paths, n_steps)) * 100)

        no_df = hybrid_fva(exp, funding_spread_bps=100, dt=0.2)
        times = np.arange(n_steps) * 0.2
        df = np.exp(-0.05 * times)
        with_df = hybrid_fva(exp, funding_spread_bps=100, dt=0.2,
                              discount_factors=df)

        assert with_df.fva < no_df.fva

    def test_fva_df_shape_validation(self):
        exp = np.zeros((10, 21))
        with pytest.raises(ValueError, match="shape"):
            hybrid_fva(exp, 50, 0.1, discount_factors=np.ones(5))


class TestNoChangeWhenAlreadyDiscounted:
    """If user pre-discounts upstream and omits the param, behaviour unchanged."""

    def test_cva_unchanged_no_df(self):
        exp = np.array([[10.0, 20.0]])
        pd = np.array([0.0, 0.01])
        r = hybrid_cva(exp, pd, recovery=0.4)
        # CVA = 0.6 · sum(EPE * marginal_pd) = 0.6 · (10·0 + 20·0.01) = 0.12.
        assert r.cva == pytest.approx(0.12, abs=1e-12)
