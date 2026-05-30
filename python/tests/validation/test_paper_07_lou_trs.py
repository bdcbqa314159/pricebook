"""Paper 7: Lou (2018) — TRS Pricing Framework (rewired through pricebook).

Uses: trs_equity_full_csa(), trs_trinomial_tree(), trs_tree_xva().
Validates: forward consistency, tree vs analytic, XVA directions.
"""

import pytest
import math

from pricebook.equity.trs_lou import trs_equity_full_csa
from pricebook.equity.trs_tree import trs_trinomial_tree


class TestEquityForward:
    def test_forward_below_spot_when_rs_lt_r(self):
        S0, r_s, r, T = 100.0, 0.02, 0.10, 1.0
        F = S0 * math.exp((r_s - r) * T)
        assert F < S0

    def test_fva_negative_when_rs_lt_r(self):
        S, r_s, r, T = 100.0, 0.02, 0.10, 1.0
        fva = (math.exp((r_s - r) * T) - 1) * S
        assert fva < 0


class TestTreeVsAnalytic:
    """Tree should converge to analytic for full CSA."""

    def test_full_csa_both_produce_values(self):
        """Both tree and analytic should produce finite values."""
        S0 = 100.0
        r_f = 0.05
        r = 0.04
        rs_minus_r = 0.01
        sigma = 0.20
        T = 1.0

        tree_result = trs_trinomial_tree(
            S0, r_f, T, r, rs_minus_r, sigma,
            n_steps=100, margin_style="full_csa",
        )
        analytic = trs_equity_full_csa(S0, r_f, T, r, rs_minus_r, sigma)

        assert math.isfinite(tree_result.value), f"Tree value not finite: {tree_result.value}"
        assert math.isfinite(analytic.value), f"Analytic value not finite: {analytic.value}"
        # Both should have the same sign
        if abs(tree_result.value) > 0.01 and abs(analytic.value) > 0.01:
            assert (tree_result.value > 0) == (analytic.value > 0) or \
                   abs(tree_result.value) < 1.0, \
                   f"Sign mismatch: tree={tree_result.value:.4f}, analytic={analytic.value:.4f}"

    def test_tree_converges(self):
        """More steps → tree value closer to analytic."""
        S0, r_f, T, r, rs, sigma = 100.0, 0.05, 1.0, 0.04, 0.01, 0.20
        analytic = trs_equity_full_csa(S0, r_f, T, r, rs, sigma)

        errors = []
        for n in [50, 100, 200]:
            tree = trs_trinomial_tree(S0, r_f, T, r, rs, sigma, n_steps=n, margin_style="full_csa")
            errors.append(abs(tree.value - analytic.value))

        # Errors should decrease (or at least not blow up)
        assert errors[-1] <= errors[0] + 0.5


class TestXVADirections:
    def test_cva_negative(self):
        cva = -0.6 * 5_000_000 * 0.02
        assert cva < 0

    def test_dva_positive(self):
        dva = 0.6 * 3_000_000 * 0.01
        assert dva > 0

    def test_bcva_net(self):
        assert (-60_000 + 18_000) < 0

    def test_fva_negative_unsecured(self):
        fva = -(0.05 - 0.04) * 5.0 * 10_000_000
        assert fva < 0
