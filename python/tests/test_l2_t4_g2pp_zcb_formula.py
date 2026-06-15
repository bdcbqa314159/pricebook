"""Regression for L2 T4 audit of the G2++ ZCB analytical formula:

Brigo-Mercurio (2nd ed., eq. 4.10) gives:

    P(t, T) = (P^M(T) / P^M(t)) · exp(-B_a(T-t)·x - B_b(T-t)·y
                                       + 0.5·[V(t, T) - V(0, T) + V(0, t)])

For stationary OU integrated variance, V(t, T) = V(0, T-t).  At t=0 with
x=y=0 the formula collapses to P^M(T) exactly.

Pre-fix all three internal implementations had the exponent as
``0.5·[V(0, T) − V(0, t)]`` — missing the ``− V(0, T) + V(0, t) + V(t, T)``
correction.  At t=0 this gave ``P^M(T) · exp(0.5 V(0, T))``, an
~exp(0.5·V(T)) bias (0.25% for T=5y, σ₁=0.01).  The bug propagated to:

* ``G2PlusPlus.zcb_price`` (vasicek.py) — used in tests
* ``G2PPTree.zcb_price`` (g2pp_tree.py) — used at terminal time-step
  of the swaption tree and at every exercise-date node in
  ``bermudan_swaption_g2pp_tree``
* ``_zcb_path`` (bermudan_swaption_g2pp.py) — used by the LSM pricer

``g2pp_swaption_price`` (g2pp_calibration.py) already uses the correct
formula via the A_i factor under the T-forward measure (line 322,
``exp(0.5 · (V_alpha_i - V_0_i + V_0_alpha))``) — so the Jamshidian
analytical pricer is unaffected, and after the fix the tree converges
to it.

Fix (T4-G2T2): use ``0.5·[V(t, T) − V(0, T) + V(0, t)]`` in all three.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from pricebook.core.day_count import date_from_year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.g2pp_calibration import g2pp_swaption_price
from pricebook.models.g2pp_tree import G2PPTree, g2pp_european_swaption_tree
from pricebook.models.vasicek import G2PlusPlus


REF = date(2026, 1, 15)


@pytest.fixture
def flat_curve() -> DiscountCurve:
    return DiscountCurve.flat(REF, 0.04)


@pytest.fixture
def g2pp(flat_curve) -> G2PlusPlus:
    return G2PlusPlus(a=0.1, b=0.05, sigma1=0.01, sigma2=0.008,
                       rho=-0.3, curve=flat_curve)


class TestZCBAtRootEqualsMarket:
    """At t=0 with x=y=0 the model ZCB must equal P_market(T) exactly."""

    def test_g2pp_zcb_at_origin_equals_curve(self, g2pp, flat_curve):
        T = 5.0
        model = g2pp.zcb_price(0.0, 0.0, T)
        mkt = flat_curve.df(date_from_year_fraction(REF, T))
        # Exact match (no V(T) bias).  Pre-fix: ~exp(0.5·V(0,T)) ≈
        # 1.0025 multiplicative error for these params.
        assert model == pytest.approx(mkt, rel=1e-10)

    def test_tree_zcb_at_root_equals_curve(self, g2pp, flat_curve):
        T = 5.0
        tree = G2PPTree(g2pp, T=T, n_steps=30)
        model = tree.zcb_price(0, tree.j_max_x, tree.j_max_y, T)
        mkt = flat_curve.df(date_from_year_fraction(REF, T))
        assert model == pytest.approx(mkt, rel=1e-10)


class TestTreeBackwardMatchesAnalytical:
    """The tree's analytical helper and its backward-induction price for
    a constant terminal must agree to within lattice discretisation."""

    def test_pzero_t_via_backward_matches_analytical_zcb(self, g2pp):
        T = 5.0
        tree = G2PPTree(g2pp, T=T, n_steps=30)
        terminal = np.ones((tree.n_x, tree.n_y))
        p_back = tree.backward_induction(terminal, option_func=None)
        p_ana = tree.zcb_price(0, tree.j_max_x, tree.j_max_y, T)
        assert p_back == pytest.approx(p_ana, rel=1e-3)


class TestSwaptionTreeConvergesToJamshidian:
    """With the ZCB formula correct in both the tree and the Jamshidian
    analytical, the tree-priced European swaption must converge to the
    closed-form price as n_steps → ∞.  Pre-fix the discrepancy was ~13%
    on a vanilla 1y/2y swaption — driven by the ZCB formula bug at the
    terminal slice of the tree."""

    def test_swaption_tree_within_10pct_of_jamshidian(self, g2pp, flat_curve):
        tree_result = g2pp_european_swaption_tree(
            g2pp, expiry_years=1.0, swap_end_years=3.0,
            strike=0.04, is_payer=True, n_steps=100, swap_freq=2,
        )
        ana = g2pp_swaption_price(
            a=g2pp.a, b=g2pp.b,
            sigma1=g2pp.sigma1, sigma2=g2pp.sigma2,
            rho=g2pp.rho, curve=flat_curve,
            expiry_years=1.0, tenor_years=2.0,
            strike=0.04, is_payer=True,
        )
        rel_err = abs(tree_result.price - ana) / max(ana, 1e-6)
        # 10% tolerance: lattice + 1st-order correlation correction
        # (the joint p clamping when rho ≠ 0).  Pre-fix was 13% — the
        # extra exp(0.5·V) factor in tree.zcb_price biased the swap PV
        # at expiry, propagating into the swaption value.
        assert rel_err < 0.10, (
            f"tree={tree_result.price:.6f}, ana={ana:.6f}, rel_err={rel_err:.4f}"
        )
