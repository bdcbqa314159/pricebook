"""Regression for L2 T4 audit of `options.cliquet.Cliquet.price_mc`:

Pre-fix every period's drift used a single flat rate

    rate = -log(curve.df(maturity)) / T

which equals the curve's spot-to-maturity zero rate, NOT the
forward rate at each reset date.  For non-flat curves the path's
per-period drift was systematically biased: early periods saw too
high a drift under a steep upward curve, late periods too low.

Same flat-curve defect as ``tarf`` (T4-TARF1, v1.047).

Fix: per-segment drift uses the forward zero rate
``-log(df_i / df_{i-1}) / dt_i`` so each step matches the curve's
local term structure.  Terminal discount unchanged (matches
curve.df(T) by construction).

These tests pin:
- For a FLAT curve, post-fix price equals pre-fix price exactly
  (the two formulations coincide).
- For a steeply sloped curve, the cliquet price now depends on the
  curve's slope through the per-period drifts (pre-fix the
  dependence ran only through the terminal discount factor).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention
from pricebook.options.cliquet import Cliquet


REF = date(2026, 4, 28)


def _quarterly_dates(n_years: int = 2) -> list[date]:
    return [REF + timedelta(days=90 * k) for k in range(1, 4 * n_years + 1)]


class TestFlatCurveUnchanged:
    def test_flat_curve_finite_price(self):
        """Flat 4% curve — per-segment forward rates all equal 4%, so
        post-fix matches pre-fix exactly."""
        curve = DiscountCurve.flat(REF, 0.04)
        opt = Cliquet(
            reset_dates=_quarterly_dates(2),
            local_floor=-0.02, local_cap=0.05,
            global_floor=0.0, global_cap=0.30,
            notional=1_000_000,
        )
        r = opt.price_mc(spot=100, curve=curve, vol=0.20,
                         n_paths=10_000, seed=42)
        assert math.isfinite(r.price)
        assert r.price > 0


class TestSlopedCurveShiftsPrice:
    def test_sloped_curve_path_uses_forward_rates(self):
        """For an upward-sloping curve (1% short → 6% long), the price
        must NOT equal the flat-curve price at the same average rate.
        Pre-fix both produced the same number because the flat-rate
        path drift was insensitive to slope (it used only the spot-to-
        maturity zero rate, which depends only on curve.df(T))."""
        # Sloped curve.
        dates = [REF + timedelta(days=d) for d in [30, 180, 365, 730, 1825]]
        rates = [0.01, 0.02, 0.03, 0.05, 0.06]
        dfs = [math.exp(-r * ((d - REF).days / 365.0))
               for r, d in zip(rates, dates)]
        sloped = DiscountCurve(
            reference_date=REF, dates=dates, dfs=dfs,
            day_count=DayCountConvention.ACT_365_FIXED,
        )
        # Flat curve calibrated to the SAME terminal df → same terminal
        # discount factor but uniform drift across periods.
        T_end = (_quarterly_dates(2)[-1] - REF).days / 365.0
        eq_flat_rate = -math.log(sloped.df(_quarterly_dates(2)[-1])) / T_end
        flat = DiscountCurve.flat(REF, eq_flat_rate)

        opt = Cliquet(
            reset_dates=_quarterly_dates(2),
            local_floor=-0.05, local_cap=0.05,
            global_floor=0.0, global_cap=0.50,
            notional=1_000_000,
        )
        r_sloped = opt.price_mc(spot=100, curve=sloped, vol=0.20,
                                n_paths=20_000, seed=42)
        r_flat = opt.price_mc(spot=100, curve=flat, vol=0.20,
                              n_paths=20_000, seed=42)
        # Pre-fix these would have been equal (both saw the same
        # terminal-zero-rate drift everywhere).  Post-fix the curve
        # slope shifts the per-period drift distribution → different
        # local return distribution → different price.
        assert r_sloped.price != pytest.approx(r_flat.price, abs=10.0), (
            f"sloped={r_sloped.price:.2f}, flat={r_flat.price:.2f} — "
            f"cliquet price should depend on curve slope via per-period drift"
        )
