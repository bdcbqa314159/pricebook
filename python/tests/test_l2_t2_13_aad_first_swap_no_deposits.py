"""Regression for L2 Tier-2 T2.13 — `aad_bootstrap` handles the
no-deposits case correctly (first swap pillars from zero).

Pre-fix `aad_bootstrap` had:

    df_k = temp_curve.df(schedule[k]) if temp_curve else Number(1.0)

When `deposit_quotes` was empty, `temp_curve` was None and the bootstrap
silently assumed `df_k = 1.0` (i.e. zero rate) over the ENTIRE pre-maturity
period of the first swap.  The bootstrap then produced a `df_mat` value
that didn't actually satisfy the par condition.

T2.13 is subsumed by T1.13 (v0.911): the fixed-point AAD iteration with
the trial curve including `df_mat` as a pillar produces correct
interpolation between t=0 (df=1) and t=mat (df=df_mat) — the standard
log-linear convention.  This file locks that in as an explicit regression.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.schedule import Frequency, generate_schedule
from pricebook.curves.aad import Number, Tape
from pricebook.curves.aad_curves import aad_bootstrap


REF = date(2024, 1, 1)


def _swap_par_pv(curve, mat, par_rate, freq=Frequency.SEMI_ANNUAL,
                 dc=DayCountConvention.THIRTY_360):
    schedule = generate_schedule(REF, mat, freq)
    pv = 0.0
    for k in range(1, len(schedule)):
        tau = year_fraction(schedule[k - 1], schedule[k], dc)
        pv += par_rate * tau * curve.df(schedule[k]).value
    pv += curve.df(mat).value - 1.0
    return pv


class TestNoDeposits:
    def test_single_swap_no_deposits_repriceses_at_par(self):
        """The canonical T2.13 case: bootstrap a 5y swap with NO deposits.
        Re-priced par PV must be ≈ 0 (pre-fix it was far from zero)."""
        d5 = REF + timedelta(days=5 * 365)
        with Tape() as _:
            sw = Number(0.04)
            curve = aad_bootstrap(REF, deposit_quotes=[], swap_quotes=[(d5, sw)])
            pv = _swap_par_pv(curve, d5, sw.value)
            assert abs(pv) < 1e-8, (
                f"5y swap re-pricing PV: {pv:.2e} (must be ~0 — pre-fix "
                f"the no-deposit case assumed df=1 over pre-maturity)"
            )

    def test_multiple_swaps_no_deposits(self):
        """Multiple swaps with no deposits: all must re-price at par."""
        d2 = REF + timedelta(days=2 * 365)
        d5 = REF + timedelta(days=5 * 365)
        d10 = REF + timedelta(days=10 * 365)
        with Tape() as _:
            s2 = Number(0.030)
            s5 = Number(0.040)
            s10 = Number(0.045)
            curve = aad_bootstrap(
                REF, deposit_quotes=[],
                swap_quotes=[(d2, s2), (d5, s5), (d10, s10)],
            )
            for mat, par in [(d2, s2.value), (d5, s5.value), (d10, s10.value)]:
                pv = _swap_par_pv(curve, mat, par)
                assert abs(pv) < 1e-8, (
                    f"Swap mat={mat}: PV={pv:.2e}"
                )

    def test_aad_adjoint_flows_through_no_deposit_first_swap(self):
        """The whole point of AAD: sensitivities must flow.  Pre-fix the
        no-deposit case had `df_k = Number(1.0)` (a constant Number with
        no AAD links), so the bootstrap's df_mat had no dependency on the
        swap rate."""
        d5 = REF + timedelta(days=5 * 365)
        d3 = REF + timedelta(days=3 * 365)
        with Tape() as _:
            sw = Number(0.04)
            curve = aad_bootstrap(REF, [], [(d5, sw)])
            df3 = curve.df(d3)
            df3.propagate_to_start()
            # The 3y DF must respond to the 5y swap rate (interpolated
            # between t=0 and t=5y in log space).
            assert abs(sw.adjoint) > 1e-6, (
                f"d df(3y) / d swap_5y = {sw.adjoint:.6e} — adjoint did "
                f"not flow through the no-deposit bootstrap"
            )
            # Sign check: higher swap rate → lower DF.
            assert sw.adjoint < 0
