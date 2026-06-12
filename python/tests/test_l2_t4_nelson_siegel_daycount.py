"""Regression for L2 Wave-2 audit — Nelson-Siegel / Svensson day-count mismatch.

Pre-fix `ns_discount_curve` and `svensson_discount_curve` constructed pillar
dates via `date_from_year_fraction(ref, t)`, which uses 365.25 days/yr (the
Julian year).  The resulting `DiscountCurve` interprets those dates via its
default ACT/365 Fixed day-count (365.0 days/yr).  This mismatch means the
stored discount factor `df = exp(-y(t) * t)` is read back at a different
year-fraction t' = round(t * 365.25) / 365.0 = t * 1.000685, giving an
implied zero rate y' = y * t/t' ≠ y.

For y = 5%:
- pre-fix 10y: y' = 4.9973% (0.27 bp off)
- pre-fix 30y: y' = 4.9963% (0.37 bp off)

Post-fix the date is built with 365.0 days/yr (matching DiscountCurve's
ACT/365 Fixed), so for integer-year tenors the round-trip is exact.
Non-integer tenors (0.25y, 0.5y) retain a small residual error driven by
rounding parameterised t to whole days — unavoidable with date-based
pillars.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.curves.nelson_siegel import (
    nelson_siegel_yield,
    svensson_yield,
    ns_discount_curve,
    svensson_discount_curve,
)


REF = date(2024, 1, 1)


def _pillar_date(t: float) -> date:
    return REF + timedelta(days=int(round(t * 365.0)))


class TestNSDiscountCurveRoundTrip:
    @pytest.mark.parametrize("t", [1, 2, 5, 7, 10, 15, 20, 30])
    def test_integer_year_tenors_round_trip_exactly(self, t):
        """At integer-year tenors, the curve's implied zero rate equals
        the NS yield to machine precision (post-fix)."""
        curve = ns_discount_curve(REF, 0.05, -0.01, 0.005, 2.0)
        d = _pillar_date(t)
        df = curve.df(d)
        yf = (d - REF).days / 365.0
        implied_y = -math.log(df) / yf
        ns_y = nelson_siegel_yield(t, 0.05, -0.01, 0.005, 2.0)
        assert implied_y == pytest.approx(ns_y, abs=1e-9), \
            f"t={t}: implied={implied_y}, NS={ns_y}, diff={(implied_y - ns_y) * 1e4:.2f} bp"


class TestSvenssonDiscountCurveRoundTrip:
    @pytest.mark.parametrize("t", [1, 5, 10, 30])
    def test_integer_year_tenors_round_trip_exactly(self, t):
        curve = svensson_discount_curve(REF, 0.05, -0.01, 0.005, 2.0, 0.003, 5.0)
        d = _pillar_date(t)
        df = curve.df(d)
        yf = (d - REF).days / 365.0
        implied_y = -math.log(df) / yf
        sv_y = svensson_yield(t, 0.05, -0.01, 0.005, 2.0, 0.003, 5.0)
        assert implied_y == pytest.approx(sv_y, abs=1e-9), \
            f"t={t}: implied={implied_y}, Svensson={sv_y}, diff={(implied_y - sv_y) * 1e4:.2f} bp"


class TestLongTenorErrorBelowOneBp:
    def test_30y_5pct_implied_within_0p1_bp(self):
        """The headline pre-fix error: 30y at 5% used to drift by 0.4 bp.
        Post-fix it is exact (well within 0.1 bp)."""
        curve = ns_discount_curve(REF, 0.05, 0.0, 0.0, 1.0)
        d = _pillar_date(30)
        df = curve.df(d)
        yf = (d - REF).days / 365.0
        implied = -math.log(df) / yf
        assert abs(implied - 0.05) * 1e4 < 0.1, \
            f"30y drift {(implied - 0.05) * 1e4:+.2f} bp exceeds 0.1 bp"
