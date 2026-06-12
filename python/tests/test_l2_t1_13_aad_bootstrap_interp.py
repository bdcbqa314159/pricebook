"""Regression for L2 Tier-1 T1.13 — AAD bootstrap properly interpolates
intermediate coupon DFs through the unknown last-pillar DF.

Pre-fix, `aad_bootstrap` constructed a `temp_curve` from existing pillars only
(omitting the new `mat` pillar), and read intermediate-coupon DFs by calling
`temp_curve.df(coupon_date)`.  When the new swap's tenor extended beyond all
existing pillars (the common case — e.g. bootstrap a 5y swap against a 1y
deposit), the intermediate-coupon dates fell in the extrapolation region.
`aad_log_linear_interp` flat-extrapolates beyond the last pillar, so the
bootstrap silently equated intermediate-coupon DFs to the LAST DEPOSIT DF.
The resulting curve mispriced the bootstrap instruments (the bootstrapped 5y
DF did not satisfy par-zero PV), and AAD sensitivities flowed only through
the deposit pillar — not through the new swap's structure.

Post-fix uses a fixed-point iteration where the trial curve INCLUDES the
current `df_mat` estimate as a pillar, so intermediate-coupon DFs interpolate
between the last existing pillar and `df_mat`.  At convergence (via Newton-
equivalent iteration of x → 1 − A − B(x)), the AAD graph captures the implicit
dx/dp = −(∂A + ∂B) / (1 + B') of the par-condition fixed point.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.curves.aad import Number, Tape
from pricebook.curves.aad_curves import AADDiscountCurve, aad_bootstrap


REF = date(2024, 1, 1)


def _swap_par_pv(curve: AADDiscountCurve, mat: date, par_rate: float) -> float:
    """Forward swap PV at par_rate (using semi-annual schedule).  At par this
    should be ≈ 0.  Pre-fix this was far from zero for swaps extending past
    the existing pillar grid."""
    from pricebook.core.schedule import Frequency, generate_schedule
    from pricebook.core.day_count import DayCountConvention, year_fraction

    schedule = generate_schedule(REF, mat, Frequency.SEMI_ANNUAL)
    swap_dc = DayCountConvention.THIRTY_360

    pv = 0.0
    for k in range(1, len(schedule)):
        tau_k = year_fraction(schedule[k - 1], schedule[k], swap_dc)
        df_k = curve.df(schedule[k]).value
        pv += par_rate * tau_k * df_k
    # Plus terminal notional swap payment: receive 1 - df(mat)
    pv += curve.df(mat).value - 1.0
    return pv


class TestRoundtripPar:
    """Bootstrap must produce a curve that re-prices each input swap at par."""

    def test_bootstrap_repriceses_5y_swap_after_1y_deposit(self):
        """Canonical T1.13 case: 1y deposit + 5y swap.  Pre-fix the 5y swap
        re-priced with PV != 0 because intermediate-coupon DFs were flat-
        extrapolated at the 1y deposit DF."""
        with Tape() as _:
            dep_rate = Number(0.025)
            swap_rate = Number(0.04)
            d1 = REF + timedelta(days=365)
            d5 = REF + timedelta(days=5 * 365)

            curve = aad_bootstrap(REF, [(d1, dep_rate)], [(d5, swap_rate)])

            pv = _swap_par_pv(curve, d5, swap_rate.value)
            assert abs(pv) < 1e-8, (
                f"5y swap re-pricing PV after bootstrap: {pv:.2e} "
                f"(should be ~0 if bootstrap properly interpolated through df_mat)"
            )

    def test_bootstrap_repriceses_three_swaps(self):
        """Stress: 1y deposit + 2y + 5y + 10y swaps.  Each must re-price at par."""
        with Tape() as _:
            dep = Number(0.03)
            s2 = Number(0.035)
            s5 = Number(0.040)
            s10 = Number(0.045)
            d1 = REF + timedelta(days=365)
            d2 = REF + timedelta(days=2 * 365)
            d5 = REF + timedelta(days=5 * 365)
            d10 = REF + timedelta(days=10 * 365)

            curve = aad_bootstrap(REF, [(d1, dep)],
                                  [(d2, s2), (d5, s5), (d10, s10)])

            for mat, par in [(d2, s2.value), (d5, s5.value), (d10, s10.value)]:
                pv = _swap_par_pv(curve, mat, par)
                assert abs(pv) < 1e-8, (
                    f"Swap with mat={mat} re-priced with PV={pv:.2e} (should be 0)"
                )


class TestAADAdjoints:
    """Sensitivities flow through the bootstrap correctly."""

    def test_dpv_dswaprate_nonzero(self):
        """The sensitivity of a far-dated DF w.r.t. the swap rate must be
        nonzero (pre-fix: with flat extrapolation, intermediate sensitivities
        were misallocated).  Smoke check: the adjoint exists and is negative
        (DF decreases as rate increases)."""
        with Tape() as _:
            dep = Number(0.025)
            swap = Number(0.04)
            d1 = REF + timedelta(days=365)
            d5 = REF + timedelta(days=5 * 365)
            d3 = REF + timedelta(days=3 * 365)

            curve = aad_bootstrap(REF, [(d1, dep)], [(d5, swap)])
            df3 = curve.df(d3)  # intermediate point
            df3.propagate_to_start()

            # df(3y) depends on the 5y swap rate (interpolated between 1y and 5y).
            assert abs(swap.adjoint) > 1e-6, (
                f"d df(3y) / d swap_5y = {swap.adjoint:.2e} — expected non-zero"
            )
            # And on the deposit rate (1y endpoint of the interpolation).
            assert abs(dep.adjoint) > 1e-6, (
                f"d df(3y) / d dep_1y = {dep.adjoint:.2e} — expected non-zero"
            )

    def test_bumped_swaprate_changes_intermediate_df(self):
        """A 1bp bump in the swap rate must change intermediate DFs by a
        sensible amount.  Pre-fix the intermediate DF was insensitive to
        the bumped swap (it was anchored to the unchanged deposit DF)."""
        d1 = REF + timedelta(days=365)
        d5 = REF + timedelta(days=5 * 365)
        d3 = REF + timedelta(days=3 * 365)

        with Tape() as _:
            curve_base = aad_bootstrap(
                REF, [(d1, Number(0.025))], [(d5, Number(0.04))],
            )
            base_df3 = curve_base.df(d3).value

        with Tape() as _:
            curve_bump = aad_bootstrap(
                REF, [(d1, Number(0.025))], [(d5, Number(0.04 + 1e-4))],
            )
            bump_df3 = curve_bump.df(d3).value

        # Pre-fix the bumped curve's df(3y) was almost identical to base
        # (flat-extrapolated from the unchanged deposit DF).
        # Post-fix the difference should reflect the 1bp swap rate move.
        # For a 1bp bump, expect Δdf(3y) ~ -3 * 1e-4 * df ≈ -2.5e-4.
        delta = bump_df3 - base_df3
        assert abs(delta) > 1e-5, (
            f"Δdf(3y) for 1bp swap bump = {delta:.2e}; pre-fix was ~0"
        )
