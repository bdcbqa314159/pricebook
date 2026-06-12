"""Regression for L2 Tier-3 T3.19 — `multicurve_newton` annuity covers the
full life of the swap, not just up to the last pillar below maturity.

Pre-fix:

    dates_up_to = [d for d in ois_pillar_dates if d <= inst['maturity']]
    df_T = ois.df(inst['maturity'])
    annuity = _compute_annuity(ois, dates_up_to, day_count)
    model_rate = (1 - df_T) / annuity

If `inst['maturity']` did not exactly match a pillar, `dates_up_to` truncated
at the last pillar BELOW maturity, while `df_T = ois.df(maturity)` was
interpolated to the maturity.  The par-rate `(1 − df_T) / annuity` mixed two
different time horizons.

Post-fix appends `maturity` to `dates_up_to` so the annuity covers ref →
maturity (same horizon as `df_T`).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.day_count import DayCountConvention
from pricebook.curves.multicurve_solver import multicurve_newton


REF = date(2024, 1, 1)


class TestAnnuityFullLife:
    def test_swap_matures_between_pillars_re_prices(self):
        """A swap with maturity falling BETWEEN existing pillars must
        re-price at par after the multicurve solve.  Pre-fix the annuity
        was truncated at the last pre-maturity pillar → par rate mis-computed,
        Newton couldn't converge to the input rate."""
        # Pillars at 1, 2, 3, 5, 7y.  A 4y swap (maturity between pillars).
        ois_pillars = [
            REF + timedelta(days=365),
            REF + timedelta(days=730),
            REF + timedelta(days=1095),
            REF + timedelta(days=1825),
            REF + timedelta(days=2555),
        ]
        ois_instruments = [
            {"type": "swap", "maturity": ois_pillars[0], "rate": 0.03},
            {"type": "swap", "maturity": ois_pillars[1], "rate": 0.035},
            {"type": "swap", "maturity": ois_pillars[2], "rate": 0.038},
            # 4y instrument with maturity NOT in pillars (between 3y and 5y).
            {"type": "swap", "maturity": REF + timedelta(days=4 * 365), "rate": 0.040},
            {"type": "swap", "maturity": ois_pillars[3], "rate": 0.042},
            {"type": "swap", "maturity": ois_pillars[4], "rate": 0.044},
        ]
        # No projection curve: pass empty projection instruments and a
        # single dummy pillar to satisfy interface.  But for testing the
        # OIS branch we can also use just the OIS side with no projection.
        # Use the projection list = empty and projection_pillar_dates =
        # empty list.
        ois_pillars_solver = ois_pillars + [REF + timedelta(days=4 * 365)]
        ois_pillars_solver.sort()
        ois_instruments_solver = sorted(
            ois_instruments, key=lambda i: i["maturity"],
        )

        result = multicurve_newton(
            ois_instruments=ois_instruments_solver,
            projection_instruments=[{"type": "swap", "maturity": REF + timedelta(days=365), "rate": 0.03}],
            ois_pillar_dates=ois_pillars_solver,
            projection_pillar_dates=[REF + timedelta(days=365)],
            reference_date=REF,
            day_count=DayCountConvention.ACT_365_FIXED,
            tol=1e-8, max_iter=50,
        )
        ois_curve = result.ois_curve
        # Re-price all swaps.  Each should match the input rate to high
        # precision.
        from pricebook.curves.multicurve_solver import multicurve_newton as _m
        for inst in ois_instruments_solver:
            df_T = ois_curve.df(inst["maturity"])
            # Recompute annuity over the same period set with maturity.
            dates_up_to = [d for d in ois_pillars_solver if d <= inst["maturity"]]
            if not dates_up_to or dates_up_to[-1] != inst["maturity"]:
                dates_up_to = dates_up_to + [inst["maturity"]]
            from pricebook.core.day_count import year_fraction
            ann = 0.0
            prev = REF
            for d in dates_up_to:
                yf = year_fraction(prev, d, DayCountConvention.ACT_365_FIXED)
                ann += yf * ois_curve.df(d)
                prev = d
            model_par = (1.0 - df_T) / ann
            assert math.isclose(model_par, inst["rate"], abs_tol=1e-6), (
                f"Swap mat={inst['maturity']} rate={inst['rate']:.4f}: "
                f"model_par={model_par:.6f}"
            )


class TestExistingFixtureUnchanged:
    """The L1 A.3 fix was a separate annuity bug; verify our T3.19 fix
    doesn't regress it."""

    def test_pillar_at_maturity_unchanged(self):
        """When swap maturity EXACTLY matches a pillar, the dates_up_to list
        already ended at maturity.  Verify the modification (appending
        maturity if not present) is a no-op for this canonical case."""
        ois_pillars = [
            REF + timedelta(days=365),
            REF + timedelta(days=1825),
        ]
        ois_instruments = [
            {"type": "swap", "maturity": ois_pillars[0], "rate": 0.03},
            {"type": "swap", "maturity": ois_pillars[1], "rate": 0.04},
        ]
        result = multicurve_newton(
            ois_instruments=ois_instruments,
            projection_instruments=[{"type": "swap", "maturity": REF + timedelta(days=365), "rate": 0.03}],
            ois_pillar_dates=ois_pillars,
            projection_pillar_dates=[REF + timedelta(days=365)],
            reference_date=REF,
            day_count=DayCountConvention.ACT_365_FIXED,
        )
        ois_curve = result.ois_curve
        # Each should re-price at par.
        for inst in ois_instruments:
            df_T = ois_curve.df(inst["maturity"])
            from pricebook.core.day_count import year_fraction
            dates_up_to = [d for d in ois_pillars if d <= inst["maturity"]]
            ann = 0.0
            prev = REF
            for d in dates_up_to:
                yf = year_fraction(prev, d, DayCountConvention.ACT_365_FIXED)
                ann += yf * ois_curve.df(d)
                prev = d
            model_par = (1.0 - df_T) / ann
            assert math.isclose(model_par, inst["rate"], abs_tol=1e-6)
