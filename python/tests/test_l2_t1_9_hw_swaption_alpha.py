"""Regression for L2 Tier-1 T1.9 — Hull-White tree swaption centres expiry
nodes on α(T_expiry), not r(0).

Pre-fix, `HullWhite.tree_european_swaption` set the rate at each expiry-time
node as `r_j = r0 + j*dr` where `r0 = f(0,0)`.  This biased every swaption
with a non-flat forward curve, because the short rate at the expiry nodes
should be centred on `α(T_expiry) ≈ f(0, T_expiry) + (σ²/(2a²))·(1−e^{−aT})²`
(Brigo-Mercurio eq. 3.34), not on today's short rate.

The post-fix test compares the tree price against the analytical Jamshidian
formula on a steeply rising curve.  Pre-fix the disagreement was many vols
wide; post-fix the agreement is within a few percent of price.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricebook.core.day_count import (
    DayCountConvention,
    date_from_year_fraction,
    year_fraction,
)
from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.hull_white import HullWhite


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _sloped_curve(slope_bp_per_year: float = 50.0, ref_year: int = 2024) -> DiscountCurve:
    """Linearly rising zero curve from 1% at t=0 to 1% + slope*T at t=10y."""
    from datetime import date
    ref = date(ref_year, 1, 1)
    tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    rates = [0.01 + (slope_bp_per_year * 1e-4) * t for t in tenors]
    dfs = [math.exp(-r * t) for r, t in zip(rates, tenors)]
    dates = [date_from_year_fraction(ref, t) for t in tenors]
    return DiscountCurve(ref, dates, dfs, day_count=DayCountConvention.ACT_365_FIXED)


def _jamshidian_payer_swaption(
    hw: HullWhite, expiry_T: float, swap_end_T: float, strike: float,
) -> float:
    """Analytical Hull-White European payer-swaption price (Jamshidian decomposition).

    Assumes annual coupons at expiry+1, expiry+2, ..., swap_end (matching the
    `tree_european_swaption` convention).
    """
    a, sigma = hw.a, hw.sigma
    ref = hw.curve.reference_date
    n = int(round(swap_end_T - expiry_T))
    pay_times = [expiry_T + k for k in range(1, n + 1)]
    # coupons c_i: K for i<n, 1+K for i=n (unit-notional swap).
    c = [strike] * (n - 1) + [1.0 + strike]

    P0_expiry = hw.curve.df(date_from_year_fraction(ref, expiry_T))
    P0_pay = [hw.curve.df(date_from_year_fraction(ref, t)) for t in pay_times]

    # Find r* such that Σ c_i · P(expiry, T_i; r*) = 1.
    def swap_minus_one(r: float) -> float:
        return sum(ci * hw.zcb_price(expiry_T, ti, r)
                   for ci, ti in zip(c, pay_times)) - 1.0

    # Bracket r*: r=−1 gives huge value (>1), r=+1 gives ≈0.
    lo, hi = -0.5, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if swap_minus_one(mid) > 0:
            lo = mid
        else:
            hi = mid
    r_star = 0.5 * (lo + hi)

    # Strike on each bond option.
    K_i = [hw.zcb_price(expiry_T, ti, r_star) for ti in pay_times]

    # Black-Scholes ZBP under HW.
    from pricebook.models.black76 import _norm_cdf

    total = 0.0
    for ci, ti, P0_i, K_bond in zip(c, pay_times, P0_pay, K_i):
        # σ_p for the (T_e, T_i) bond.
        tau = ti - expiry_T
        if a > 1e-12:
            B_tau = (1.0 - math.exp(-a * tau)) / a
            sigma_p = sigma * B_tau * math.sqrt(
                (1.0 - math.exp(-2.0 * a * expiry_T)) / (2.0 * a)
            )
        else:
            sigma_p = sigma * tau * math.sqrt(expiry_T)

        if sigma_p <= 1e-14 or P0_i <= 0 or K_bond <= 0:
            zbp = max(K_bond * P0_expiry - P0_i, 0.0)
        else:
            h = math.log(P0_i / (K_bond * P0_expiry)) / sigma_p + 0.5 * sigma_p
            zbp = K_bond * P0_expiry * _norm_cdf(-h + sigma_p) - P0_i * _norm_cdf(-h)
        total += ci * zbp

    return total


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestHWSwaptionAlpha:
    def test_alpha_formula_at_zero_equals_short_rate(self):
        """α(0) = f(0,0) + 0 = today's short rate (sanity check)."""
        curve = _sloped_curve(slope_bp_per_year=0.0)  # flat 1%
        hw = HullWhite(a=0.03, sigma=0.01, curve=curve)
        # α(0) = f(0,0) since 1-e^0 = 0.
        assert abs(hw._alpha(0.0) - hw._forward_rate(0.0)) < 1e-12

    def test_alpha_grows_with_forward_curve(self):
        """On a steeply rising curve, α(T) >> α(0)."""
        curve = _sloped_curve(slope_bp_per_year=100.0)  # 1% → 11% at 10y
        hw = HullWhite(a=0.03, sigma=0.005, curve=curve)
        alpha_0 = hw._alpha(0.0)
        alpha_5 = hw._alpha(5.0)
        # On a curve with 100bp/yr slope, α(5y) ≈ 1% + 500bp + small convex tail.
        assert alpha_5 - alpha_0 > 0.04, (
            f"α(5y) − α(0) = {alpha_5 - alpha_0:.4f}, expected > 0.04 (400bp)"
        )

    def test_tree_swaption_atm_matches_jamshidian_flat_curve(self):
        """On a flat curve, pre-fix and post-fix differ minimally (α(T)≈α(0)).
        This test confirms agreement with Jamshidian on the easy case."""
        curve = _sloped_curve(slope_bp_per_year=0.0)  # flat 1%
        hw = HullWhite(a=0.05, sigma=0.005, curve=curve)
        # ATM strike: forward swap rate at 2y into 3y.
        ref = curve.reference_date
        P_expiry = curve.df(date_from_year_fraction(ref, 2.0))
        P_end = curve.df(date_from_year_fraction(ref, 5.0))
        annuity = sum(curve.df(date_from_year_fraction(ref, 2.0 + k))
                      for k in range(1, 4))
        fwd_rate = (P_expiry - P_end) / annuity

        tree_price = hw.tree_european_swaption(
            expiry_T=2.0, swap_end_T=5.0, strike=fwd_rate, n_steps=200,
        )
        jam_price = _jamshidian_payer_swaption(hw, 2.0, 5.0, fwd_rate)

        rel_err = abs(tree_price - jam_price) / jam_price
        assert rel_err < 0.05, (
            f"Tree vs Jamshidian (flat curve, ATM 2y5y): "
            f"tree={tree_price:.5f}, jam={jam_price:.5f}, rel={rel_err:.3f}"
        )

    def test_tree_swaption_matches_jamshidian_steep_curve(self):
        """The pre-fix discriminator: on a steep curve the rate-centring bug
        is large.  Post-fix, the tree should track Jamshidian within a few
        percent for moderate n_steps."""
        curve = _sloped_curve(slope_bp_per_year=80.0)  # 1% → 9% at 10y
        hw = HullWhite(a=0.05, sigma=0.005, curve=curve)
        # ATM strike: forward swap rate at 3y into 4y.
        ref = curve.reference_date
        P_expiry = curve.df(date_from_year_fraction(ref, 3.0))
        P_end = curve.df(date_from_year_fraction(ref, 7.0))
        annuity = sum(curve.df(date_from_year_fraction(ref, 3.0 + k))
                      for k in range(1, 5))
        fwd_rate = (P_expiry - P_end) / annuity

        tree_price = hw.tree_european_swaption(
            expiry_T=3.0, swap_end_T=7.0, strike=fwd_rate, n_steps=200,
        )
        jam_price = _jamshidian_payer_swaption(hw, 3.0, 7.0, fwd_rate)

        rel_err = abs(tree_price - jam_price) / jam_price
        # Pre-fix this was off by >20% on a 80bp/yr slope; post-fix should be
        # well under 10% for ATM with 200 steps.
        assert rel_err < 0.10, (
            f"Tree vs Jamshidian (80bp slope, ATM 3y4y): "
            f"tree={tree_price:.5f}, jam={jam_price:.5f}, rel={rel_err:.3f}"
        )

    def test_tree_swaption_deep_itm_recovers_intrinsic(self):
        """Deep ITM payer should price near annuity·(fwd − K) — pre-fix this
        recovered intrinsic from a wrong-centred rate (effectively pricing
        as-if the rate at expiry was r(0)), so on a steep curve the deep-ITM
        intrinsic was far from the true forward-swap intrinsic."""
        curve = _sloped_curve(slope_bp_per_year=50.0)
        hw = HullWhite(a=0.05, sigma=0.003, curve=curve)  # low vol → near-intrinsic
        ref = curve.reference_date
        P_expiry = curve.df(date_from_year_fraction(ref, 3.0))
        P_end = curve.df(date_from_year_fraction(ref, 6.0))
        annuity = sum(curve.df(date_from_year_fraction(ref, 3.0 + k))
                      for k in range(1, 4))
        fwd_rate = (P_expiry - P_end) / annuity

        # Strike 200bp below fwd — deeply ITM.
        K = fwd_rate - 0.02
        intrinsic = annuity * (fwd_rate - K)
        tree_price = hw.tree_european_swaption(
            expiry_T=3.0, swap_end_T=6.0, strike=K, n_steps=200,
        )

        # With very low vol the swaption value ≈ intrinsic (a few % over from
        # time value).  Pre-fix on a 50bp/yr curve, the price would be biased
        # by ≈ annuity · 150bp ≈ 0.04, dwarfing intrinsic for tight strikes.
        rel = abs(tree_price - intrinsic) / intrinsic
        assert rel < 0.15, (
            f"Deep-ITM tree vs intrinsic: tree={tree_price:.5f}, "
            f"intrinsic={intrinsic:.5f}, rel={rel:.3f}"
        )
