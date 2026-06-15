"""Regression for L2 T4 audit of `models.g2pp_tree.G2PPTree._fwd_rate`:

Pre-fix the instantaneous-forward helper used a finite difference with
``eps = 1e-5`` years (~ 8 seconds).  Because the curve's ``df`` API
takes a ``datetime.date`` and ``date_from_year_fraction`` rounds the
input year-fraction to a day, the two evaluation points ``t ± eps``
round either to the same date (so the helper returns 0) or to dates
one day apart (so the helper returns the true forward × (1/365) / (2·eps)
≈ 137× over-stated).

For a 30-step, T=5y tree on a flat 4% curve this gave φ(t) ≈ 0 at 27 of
31 grid times and φ(t) ≈ 5.48 at the 4 other times — so every node
discounted by exp(-5.48·dt) ≈ 0.40 on four steps, yielding a tree
discount factor P(0,5) ≈ 0.026 vs the curve value 0.819.  All G2++
tree-based products (bonds, swaptions priced via :func:`
g2pp_european_swaption_tree`, callable / puttable variants) were
catastrophically mis-priced as a result.  The existing G2PP tests
hid the defect with a 50% tolerance.

Fix (T4-G2T1): delegate to :meth:`DiscountCurve.instantaneous_forward`,
which uses a stable one-day step (consistent with the day-rounded
``df`` lookup).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from pricebook.core.day_count import date_from_year_fraction
from pricebook.core.discount_curve import DiscountCurve
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


class TestPhiSmooth:
    """φ(t) must be smooth — no spurious 0/5.48 alternation."""

    def test_phi_close_to_short_rate_on_flat_curve(self, g2pp):
        """On a flat 4% curve, φ(t) ≈ 0.04 + O(σ²) for all grid times.

        Pre-fix: half the grid had φ ≈ 0, the other half φ ≈ 5.48.
        """
        tree = G2PPTree(g2pp, T=5.0, n_steps=30)
        for t in tree.times:
            phi = tree._phi(t)
            assert 0.03 < phi < 0.05, (
                f"φ({t:.4f}) = {phi:.4f} — expected ≈ 0.04 + O(σ²).  "
                f"Pre-fix this would alternate between ≈ 0 and ≈ 5.48."
            )


class TestTreeFitsCurve:
    """P_tree(0, T) computed by backward induction with terminal=1 must
    match the input curve's P(0, T).  Pre-fix this differed by 30×."""

    def test_pzero_t_matches_curve(self, g2pp, flat_curve):
        tree = G2PPTree(g2pp, T=5.0, n_steps=30)
        terminal = np.ones((tree.n_x, tree.n_y))
        p_tree = tree.backward_induction(terminal, option_func=None)
        p_curve = flat_curve.df(date_from_year_fraction(REF, 5.0))
        # 1% tolerance: lattice discretisation + 1st-order correlation
        # correction.  Pre-fix was off by ~97% (0.026 vs 0.819).
        assert p_tree == pytest.approx(p_curve, rel=1e-2), (
            f"P_tree(0,5) = {p_tree:.6f}, P_curve(0,5) = {p_curve:.6f}"
        )

    def test_pzero_t_matches_curve_short_horizon(self, g2pp, flat_curve):
        """Same parity at a 1-year horizon."""
        tree = G2PPTree(g2pp, T=1.0, n_steps=20)
        terminal = np.ones((tree.n_x, tree.n_y))
        p_tree = tree.backward_induction(terminal, option_func=None)
        p_curve = flat_curve.df(date_from_year_fraction(REF, 1.0))
        assert p_tree == pytest.approx(p_curve, rel=1e-2)


class TestSwaptionTreeMatchesAnalytical:
    """Tight cross-check vs Jamshidian — pre-fix the existing
    ``test_european_swaption_tree_close_to_analytical`` used a 50%
    tolerance which hid the defect."""

    def test_swaption_tree_within_15pct_of_analytical(self, g2pp, flat_curve):
        from pricebook.models.g2pp_calibration import g2pp_swaption_price

        result = g2pp_european_swaption_tree(
            g2pp, expiry_years=1.0, swap_end_years=3.0,
            strike=0.04, is_payer=True, n_steps=50, swap_freq=2,
        )
        analytical = g2pp_swaption_price(
            a=g2pp.a, b=g2pp.b,
            sigma1=g2pp.sigma1, sigma2=g2pp.sigma2,
            rho=g2pp.rho, curve=flat_curve,
            expiry_years=1.0, tenor_years=2.0,
            strike=0.04, is_payer=True,
        )
        rel_err = abs(result.price - analytical) / max(analytical, 1e-6)
        # 15% tolerance: lattice discretisation + 1st-order correlation
        # correction (the joint p clamping when rho ≠ 0 biases the
        # distribution).  The existing test uses 50% which masked the
        # phi(t) defect — this tightens it ~3× to catch any future
        # regression of the same class.
        assert rel_err < 0.15, (
            f"tree={result.price:.6f}, analytical={analytical:.6f}, "
            f"rel_err={rel_err:.4f}"
        )
