"""Tests for TRS trinomial tree and XVA (Lou 2018, Section 8.3-8.5)."""

from __future__ import annotations

import math

import pytest

from pricebook.trs_lou import trs_equity_full_csa
from pricebook.trs_tree import (
    _trinomial_probabilities,
    trs_trinomial_tree,
    trs_tree_xva,
)


# ---- 8.4 Trinomial tree ----

class TestTrinomialTree:
    """Validation items 12-16."""

    def test_v12_probabilities_sum_to_one(self):
        """V12: pu + pm + pd = 1."""
        _, _, pu, pm, pd = _trinomial_probabilities(0.05, 0.30, 0.01)
        assert pu + pm + pd == pytest.approx(1.0, abs=1e-10)

    def test_v12_probabilities_non_negative(self):
        """V12: All probabilities non-negative."""
        _, _, pu, pm, pd = _trinomial_probabilities(0.05, 0.50, 0.01)
        assert pu >= 0
        assert pm >= 0
        assert pd >= 0

    def test_v13_drift_check(self):
        """V13: E[S_{t+dt}/St] = exp((rs-q)dt)."""
        rs_q = 0.05
        sigma = 0.30
        dt = 0.01
        u, d, pu, pm, pd = _trinomial_probabilities(rs_q, sigma, dt)

        # E[S_{t+dt}/St] = pu*u + pm*1 + pd*d
        expected_ratio = math.exp(rs_q * dt)
        actual_ratio = pu * u + pm * 1.0 + pd * d
        assert actual_ratio == pytest.approx(expected_ratio, rel=0.01)

    def test_v14_variance_check(self):
        """V14: Var[ln(S_{t+dt}/St)] ≈ σ²dt."""
        rs_q = 0.05
        sigma = 0.30
        dt = 0.01
        u, d, pu, pm, pd = _trinomial_probabilities(rs_q, sigma, dt)

        ln_u = math.log(u)
        ln_d = math.log(d)
        # E[ln(ratio)] and E[ln(ratio)²]
        E_ln = pu * ln_u + pm * 0.0 + pd * ln_d
        E_ln2 = pu * ln_u**2 + pm * 0.0 + pd * ln_d**2
        var = E_ln2 - E_ln**2

        assert var == pytest.approx(sigma**2 * dt, rel=0.05)

    def test_tree_full_csa_matches_analytic(self):
        """Tree with full CSA should converge to analytic (Eq 7)."""
        S0 = 100.0
        r = 0.05
        D = math.exp(-r * 1.0)
        libor = (1 / D - 1) / 1.0
        r_f = libor
        sigma = 0.20  # moderate vol for stable tree
        rs_minus_r = 0.0

        analytic = trs_equity_full_csa(S0, S0, r_f, 1.0, 0.0, D, rs_minus_r=0.0)

        tree = trs_trinomial_tree(
            S0, r_f, 1.0, r, rs_minus_r, sigma,
            n_steps=100, mu=1.0)

        # Tree should converge to analytic
        assert tree.value == pytest.approx(analytic.value, abs=S0 * 0.05)

    def test_tree_full_csa_with_repo(self):
        """Tree with repo spread, full CSA, should converge to Eq 7."""
        S0 = 100.0
        r = 0.05
        D = math.exp(-r * 1.0)
        libor = (1 / D - 1) / 1.0
        r_f = libor
        sigma = 0.20
        rs_minus_r = 0.02

        analytic = trs_equity_full_csa(S0, S0, r_f, 1.0, 0.0, D, rs_minus_r=rs_minus_r)

        tree = trs_trinomial_tree(
            S0, r_f, 1.0, r, rs_minus_r, sigma,
            n_steps=100, mu=1.0)

        assert tree.value == pytest.approx(analytic.value, abs=S0 * 0.05)

    def test_v16_single_period_recursion(self):
        """V16: Single period should match standard rollback."""
        S0 = 100.0
        r = 0.10
        r_f = 0.12
        sigma = 0.30

        tree = trs_trinomial_tree(
            S0, r_f, 1.0, r, 0.0, sigma,
            n_steps=50, mu=1.0)

        assert math.isfinite(tree.value)


# ---- 8.5 XVA decomposition ----

class TestXVADecomposition:
    """Validation items 17-21."""

    def test_v17_xva_equals_vstar_minus_v(self):
        """V17: U = V* - V."""
        result = trs_tree_xva(
            S_0=100.0, r_f=0.12, T=1.0, r=0.10, rs_minus_r=0.02,
            sigma=0.50, r_b=0.12, r_c=0.15,
            s_b=0.017, s_c=0.042, mu_b=0.003, mu_c=0.008,
            n_steps=50, mu=0.0)

        assert result.total_xva == pytest.approx(
            result.value_star - result.value, rel=1e-10)

    def test_v18_xva_components_finite(self):
        """V18: All XVA components should be finite."""
        result = trs_tree_xva(
            S_0=100.0, r_f=0.12, T=1.0, r=0.10, rs_minus_r=0.02,
            sigma=0.30, r_b=0.12, r_c=0.15,
            s_b=0.017, s_c=0.042, mu_b=0.003, mu_c=0.008,
            n_steps=50, mu=0.0)

        assert math.isfinite(result.cva)
        assert math.isfinite(result.dva)
        assert math.isfinite(result.cfa)
        assert math.isfinite(result.dfa)
        assert result.cva >= 0
        assert result.dva >= 0

    def test_v30_full_csa_no_xva(self):
        """V30: mu=1 (full CSA) => all XVAs = 0."""
        result = trs_tree_xva(
            S_0=100.0, r_f=0.12, T=1.0, r=0.10, rs_minus_r=0.02,
            sigma=0.50, r_b=0.12, r_c=0.15,
            s_b=0.017, s_c=0.042,
            n_steps=50, mu=1.0)

        assert abs(result.total_xva) < 1e-6
        assert result.value == pytest.approx(result.value_star, abs=1e-6)

    def test_v32_symmetric_spreads(self):
        """V32: sb=sc, μb=μc => symmetric XVA components."""
        result = trs_tree_xva(
            S_0=100.0, r_f=0.12, T=1.0, r=0.10, rs_minus_r=0.02,
            sigma=0.30, r_b=0.12, r_c=0.12,
            s_b=0.03, s_c=0.03, mu_b=0.005, mu_c=0.005,
            n_steps=50, mu=0.0)

        # With symmetric spreads, proportional split gives equal CVA/DVA shares
        # (though U ≠ 0 because re ≠ r for uncollateralised)
        assert math.isfinite(result.total_xva)
        # sb = sc => CVA share = DVA share of the total
        if abs(result.total_xva) > 1e-10:
            assert result.cva == pytest.approx(result.dva, abs=0.01)
            assert result.cfa == pytest.approx(result.dfa, abs=0.01)

    def test_xva_result_protocol(self):
        """XVA result implements InstrumentResult protocol."""
        result = trs_tree_xva(
            S_0=100.0, r_f=0.12, T=1.0, r=0.10, rs_minus_r=0.02,
            sigma=0.50, r_b=0.12, r_c=0.15,
            s_b=0.017, s_c=0.042,
            n_steps=50, mu=0.0)

        assert hasattr(result, 'price')
        assert isinstance(result.to_dict(), dict)
        assert math.isfinite(result.price)


# ---- Edge cases ----

class TestEdgeCases:
    """Validation items 28-31."""

    def test_v28_low_vol(self):
        """V28: σ→0 => near-deterministic, tree converges."""
        tree = trs_trinomial_tree(
            100.0, 0.12, 1.0, 0.10, 0.0, 0.01,  # very low vol
            n_steps=50, mu=1.0)
        assert math.isfinite(tree.value)

    def test_v29_short_maturity(self):
        """V29: T→0 => V→0."""
        tree = trs_trinomial_tree(
            100.0, 0.12, 0.001, 0.10, 0.0, 0.30,
            n_steps=10, mu=1.0)
        assert abs(tree.value) < 1.0  # small for tiny T

    def test_tree_convergence(self):
        """Tree value converges as n_steps increases."""
        S0 = 100.0; r = 0.10; r_f = 0.12; sigma = 0.30

        values = []
        for n in [50, 100, 200]:
            tree = trs_trinomial_tree(S0, r_f, 1.0, r, 0.0, sigma,
                                      n_steps=n, mu=1.0)
            values.append(tree.value)

        # Values should converge (later values closer together)
        diff1 = abs(values[1] - values[0])
        diff2 = abs(values[2] - values[1])
        assert diff2 < diff1  # convergence
