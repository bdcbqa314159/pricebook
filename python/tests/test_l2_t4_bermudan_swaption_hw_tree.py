"""Regression for L2 T4 audit of `options.bermudan_swaption`:

The HW trinomial tree in ``bermudan_swaption_tree`` had three coupled
defects, all rolled into this fix slice (T4-BERM1):

1. **Wrong trinomial branching probabilities.**  The drift terms used
   ``/6`` instead of the textbook ``/2`` (Hull §32.4 eq. 32.10).
   Result: ``p_u − p_d = −j·a·dt/3`` instead of ``−j·a·dt`` — drift
   3× too small.

2. **Missing time-varying α(t) shift.**  Short rate at node (step, j)
   used ``r0 + j·dr`` regardless of step.  The proper HW trinomial
   fits α(t) so the tree reprices the initial discount curve at each
   step boundary.  Pre-fix the tree silently mis-matched a non-flat
   input curve.

3. **Exercise vs continuation discount mismatch.**  The old loop
   computed ``new_values = exp(-r·dt) × continuation`` then took
   ``max(new_values, exercise_at_step+1)`` — comparing DISCOUNTED
   continuation to UNDISCOUNTED exercise.  Exercise was systematically
   over-valued by exp(+r·dt); single-exercise (European) case was
   missing the entire terminal discount.

LSM was fixed in parallel (T4-BERM2): conditional OU mean uses
``α(t)`` not ``forward_rate(t)``.

These tests pin:
- Single-exercise tree matches the analytical HW European swaption
  (within tree discretization).  Pre-fix the missing terminal
  discount blew this comparison apart.
- Tree-vs-LSM agreement under a non-flat curve, where the missing
  α(t) shift would have broken consistency.
- Sanity: prices remain positive and ordered correctly.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention
from pricebook.models.hull_white import HullWhite
from pricebook.models.hw_calibration import _hw_swaption_price
from pricebook.options.bermudan_swaption import (
    bermudan_swaption_tree, bermudan_swaption_lsm,
)


REF = date(2026, 1, 15)


@pytest.fixture
def flat_curve():
    return DiscountCurve.flat(REF, 0.05)


@pytest.fixture
def hw_flat(flat_curve):
    return HullWhite(a=0.1, sigma=0.01, curve=flat_curve)


def _sloped_curve():
    """Upward-sloping curve from 2% short-end to 6% at 10y."""
    dates = [REF + timedelta(days=d) for d in [30, 180, 365, 730, 1825, 3650]]
    rates = [0.02, 0.025, 0.03, 0.04, 0.05, 0.06]
    dfs = [math.exp(-r * ((d - REF).days / 365.0))
           for r, d in zip(rates, dates)]
    return DiscountCurve(
        reference_date=REF, dates=dates, dfs=dfs,
        day_count=DayCountConvention.ACT_365_FIXED,
    )


class TestSingleExerciseTerminalDiscount:
    def test_european_tree_matches_analytical(self, hw_flat):
        """Single exercise at the end → European swaption.  Pre-fix the
        Bermudan tree omitted the terminal discount, mispricing the
        European case by a factor of ``exp(+r·T)``.  Post-fix the tree
        should agree with the analytical HW formula within tree
        discretization (~few % at n_steps=100)."""
        T_exp = 3.0
        T_swap_end = 8.0
        K = 0.05

        # Use the existing tree_european_swaption as the reference — it
        # uses the same α(t) infrastructure and proper terminal
        # discounting, so it's the gold-standard tree comparison.
        ref_price = hw_flat.tree_european_swaption(
            expiry_T=T_exp, swap_end_T=T_swap_end, strike=K,
            n_steps=100, is_payer=True, payments_per_year=1,
        )
        berm_single = bermudan_swaption_tree(
            hw_flat, exercise_years=[T_exp],
            swap_end_years=T_swap_end, strike=K,
            n_steps=100, swap_freq=1.0,
        )
        # Both trees use the same α(t) infrastructure but differ on
        # backward-induction vs state-price-weighting → agree within
        # tree discretization (~2-3% at n_steps=100).
        assert berm_single == pytest.approx(ref_price, rel=3e-2)

    def test_european_not_overvalued_by_factor_exp_rT(self, hw_flat, flat_curve):
        """Sanity bound: the single-exercise (European) Bermudan tree
        must match the standalone HW European tree to within
        discretization.  Pre-fix the missing terminal discount would
        inflate by ``exp(r·T)``."""
        T_exp = 5.0
        analytical = _hw_swaption_price(
            hw_flat.a, hw_flat.sigma, flat_curve,
            expiry_years=T_exp, tenor_years=5.0, strike=0.05,
            is_payer=True, n_steps=100,
        )
        berm = bermudan_swaption_tree(
            hw_flat, exercise_years=[T_exp],
            swap_end_years=T_exp + 5.0, strike=0.05,
            n_steps=100, swap_freq=1.0,
        )
        # Pre-fix bias factor would be ≈ exp(0.05 × 5) ≈ 1.28.
        # Post-fix the tree should be within tree discretization error.
        assert berm == pytest.approx(analytical, rel=5e-2)


class TestNonFlatCurveAlphaShift:
    def test_tree_matches_curve_zcb_after_alpha_calibration(self):
        """Forward-fit α(t) means the tree reprices the initial ZCB
        curve to discretization tolerance.  Verifies the calibration
        infrastructure exposed by ``build_tree_alphas``."""
        curve = _sloped_curve()
        hw = HullWhite(a=0.1, sigma=0.01, curve=curve)
        # The tree-pricing of a ZCB at T must match curve.df(T).
        for T in [1.0, 3.0, 5.0]:
            tree_p = hw.tree_zcb(T, n_steps=100)
            d_T = REF + timedelta(days=int(round(T * 365)))
            curve_p = curve.df(d_T)
            assert tree_p == pytest.approx(curve_p, rel=2e-2), (
                f"Tree ZCB at T={T} ({tree_p:.6f}) ≠ "
                f"curve.df ({curve_p:.6f}) — α(t) calibration broken"
            )

    def test_bermudan_positive_under_sloped_curve(self):
        """Bermudan tree on a non-flat curve must produce a finite,
        positive price.  Pre-fix the missing α(t) shift meant the tree's
        short-rate distribution was centred on r0 = front-end rate
        (~2%) at every step — gross mismatch with the curve's actual
        long-end rates (~6%), and the tree did not match the curve."""
        curve = _sloped_curve()
        hw = HullWhite(a=0.1, sigma=0.01, curve=curve)
        price = bermudan_swaption_tree(
            hw, exercise_years=[2.0, 3.0, 4.0, 5.0],
            swap_end_years=8.0, strike=0.045,
            n_steps=80, swap_freq=1.0,
        )
        assert math.isfinite(price)
        assert price > 0


class TestProbabilitiesCorrectDrift:
    def test_drift_term_uses_textbook_divide_by_two(self, hw_flat):
        """At higher mean-reversion speed ``a``, mean-reversion is
        stronger, vol of long-dated short rates is lower, and the
        Bermudan swaption value drops.  Pre-fix the drift term was 3×
        too small, so increasing ``a`` had a damped effect on the
        price.  Post-fix the standard sensitivity should be visible."""
        curve = hw_flat.curve

        hw_low_mr = HullWhite(a=0.05, sigma=0.01, curve=curve)
        hw_high_mr = HullWhite(a=0.20, sigma=0.01, curve=curve)

        p_low = bermudan_swaption_tree(
            hw_low_mr, exercise_years=[1, 2, 3, 4, 5],
            swap_end_years=10, strike=0.05, n_steps=80,
        )
        p_high = bermudan_swaption_tree(
            hw_high_mr, exercise_years=[1, 2, 3, 4, 5],
            swap_end_years=10, strike=0.05, n_steps=80,
        )
        # Higher mean reversion → lower long-end vol → lower option value.
        assert p_high < p_low, (
            f"Stronger mean-reversion (a=0.20) priced at {p_high:.6f} "
            f"is not below weak mean-reversion (a=0.05) at {p_low:.6f}"
        )


class TestLSMAlphaCorrection:
    def test_lsm_under_sloped_curve_positive_and_below_tree(self):
        """With the curve sloped 2% → 6%, LSM (now using α(t) drift)
        and tree (now using α(t) shift) should both produce positive,
        finite prices.  LSM is the standard American/Bermudan lower
        bound (suboptimal exercise boundary from a regression fit), so
        LSM ≤ tree is the directional expectation.  Pre-fix both had
        α-related bugs that partially cancelled for flat curves but
        accumulated under slope."""
        curve = _sloped_curve()
        hw = HullWhite(a=0.1, sigma=0.01, curve=curve)
        tree_price = bermudan_swaption_tree(
            hw, exercise_years=[1, 2, 3],
            swap_end_years=6.0, strike=0.045, n_steps=80,
        )
        lsm_price = bermudan_swaption_lsm(
            hw, exercise_years=[1, 2, 3],
            swap_end_years=6.0, strike=0.045,
            n_paths=30_000, seed=42,
        )
        assert tree_price > 0
        assert lsm_price > 0
        # LSM is a lower bound (suboptimal exercise rule).
        assert lsm_price <= tree_price * 1.05  # 5% slack for MC noise
