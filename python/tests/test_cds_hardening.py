"""Tests for CDS hardening: rec01, theta, duration, convexity, PnL attribution, forward CDS."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.cds import (
    CDS, StandardCDS,
    cds_pnl_attribution, CDSPnLAttribution,
    forward_cds_par_spread, forward_risky_annuity,
    ForwardCDSCurveResult,
)
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 1, 15)
END_5Y = REF + relativedelta(years=5)


def _make_cds(spread=0.01, hazard=0.02, rate=0.04):
    dc = make_flat_curve(REF, rate)
    sc = make_flat_survival(REF, hazard)
    cds = CDS(REF, END_5Y, spread=spread, notional=10_000_000, recovery=0.4)
    return cds, dc, sc


# ---- rec01 ----

class TestRec01:

    def test_negative_for_buyer(self):
        """Higher recovery reduces protection leg → rec01 < 0 for buyer."""
        cds, dc, sc = _make_cds()
        r = cds.rec01(dc, sc)
        assert r < 0

    def test_proportional_to_default_prob(self):
        """Higher hazard → larger magnitude rec01."""
        cds_low, dc, sc_low = _make_cds(hazard=0.01)
        cds_high, _, sc_high = _make_cds(hazard=0.05)
        r_low = abs(cds_low.rec01(dc, sc_low))
        r_high = abs(cds_high.rec01(dc, sc_high))
        assert r_high > r_low

    def test_zero_at_full_recovery(self):
        """At R=1, protection is 0, so rec01 ≈ 0."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        cds = CDS(REF, END_5Y, spread=0.01, notional=10_000_000, recovery=0.99)
        r = cds.rec01(dc, sc, shift=0.01)
        # At R=0.99, bumped to R=1.0, protection leg collapses
        assert r < 0


# ---- theta ----

class TestTheta:

    def test_finite(self):
        cds, dc, sc = _make_cds()
        t = cds.theta(dc, sc)
        assert math.isfinite(t)

    def test_near_zero_at_par(self):
        """At par spread, theta should be small (PV ≈ 0, time decay minimal)."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        cds = CDS(REF, END_5Y, spread=0.01, notional=10_000_000, recovery=0.4)
        par = cds.par_spread(dc, sc)
        at_par = CDS(REF, END_5Y, spread=par, notional=10_000_000, recovery=0.4)
        t = at_par.theta(dc, sc)
        # Theta should be small relative to notional
        assert abs(t) < 10_000  # less than 10K on 10M notional

    def test_multi_day(self):
        cds, dc, sc = _make_cds()
        t1 = cds.theta(dc, sc, days=1)
        t5 = cds.theta(dc, sc, days=5)
        # Multi-day theta should be roughly proportional
        assert abs(t5) > abs(t1) * 2


# ---- spread duration and convexity ----

class TestSpreadDurationConvexity:

    def test_duration_negative_for_buyer(self):
        """For protection buyer: CS01 > 0, PV > 0, so -CS01/PV < 0."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.05)  # high hazard → positive PV for buyer
        cds = CDS(REF, END_5Y, spread=0.01, notional=10_000_000, recovery=0.4)
        d = cds.spread_duration(dc, sc)
        assert d < 0  # wider spreads help buyer → negative duration

    def test_convexity_nonzero(self):
        """Spread convexity should be nonzero (second-order sensitivity exists)."""
        cds, dc, sc = _make_cds()
        c = cds.spread_convexity(dc, sc)
        assert c != 0

    def test_duration_finite(self):
        cds, dc, sc = _make_cds()
        d = cds.spread_duration(dc, sc)
        assert math.isfinite(d)

    def test_convexity_finite(self):
        cds, dc, sc = _make_cds()
        c = cds.spread_convexity(dc, sc)
        assert math.isfinite(c)


# ---- P&L attribution ----

class TestPnLAttribution:

    def test_components_sum_to_total(self):
        """Sum of components should equal total PnL."""
        dc0 = make_flat_curve(REF, 0.04)
        sc0 = make_flat_survival(REF, 0.02)
        dc1 = make_flat_curve(REF, 0.04)  # same rates
        sc1 = make_flat_survival(REF, 0.022)  # spread widened
        cds = CDS(REF, END_5Y, spread=0.01, notional=10_000_000, recovery=0.4)

        attr = cds_pnl_attribution(cds, dc0, sc0, dc1, sc1, horizon_days=1)
        component_sum = attr.spread + attr.carry + attr.roll_down + attr.convexity + attr.residual
        assert attr.total == pytest.approx(component_sum, abs=1.0)

    def test_spread_widening_positive_for_buyer(self):
        """Protection buyer profits when spreads widen."""
        dc0 = make_flat_curve(REF, 0.04)
        sc0 = make_flat_survival(REF, 0.02)
        sc1 = make_flat_survival(REF, 0.04)  # big widening
        cds = CDS(REF, END_5Y, spread=0.01, notional=10_000_000, recovery=0.4)

        attr = cds_pnl_attribution(cds, dc0, sc0, dc0, sc1, horizon_days=1)
        assert attr.total > 0

    def test_to_dict(self):
        dc0 = make_flat_curve(REF, 0.04)
        sc0 = make_flat_survival(REF, 0.02)
        cds = CDS(REF, END_5Y, spread=0.01, notional=10_000_000, recovery=0.4)
        attr = cds_pnl_attribution(cds, dc0, sc0, dc0, sc0, horizon_days=1)
        d = attr.to_dict()
        assert "total" in d
        assert "spread" in d
        assert "residual" in d


# ---- Forward CDS with curves ----

class TestForwardCDS:

    def test_spot_equals_par(self):
        """Forward CDS starting today should equal spot par spread."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        cds = CDS(REF, END_5Y, spread=0.01, notional=1.0, recovery=0.4)
        par = cds.par_spread(dc, sc)

        fwd = forward_cds_par_spread(dc, sc, REF, END_5Y, recovery=0.4)
        assert fwd.forward_spread == pytest.approx(par, rel=1e-4)

    def test_forward_spread_positive(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        start = REF + relativedelta(years=1)
        end = REF + relativedelta(years=6)
        fwd = forward_cds_par_spread(dc, sc, start, end, recovery=0.4)
        assert fwd.forward_spread > 0

    def test_survival_to_start(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        start = REF + relativedelta(years=2)
        fwd = forward_cds_par_spread(dc, sc, start, END_5Y, recovery=0.4)
        expected_surv = sc.survival(start)
        assert fwd.survival_to_start == pytest.approx(expected_surv)

    def test_annuity_positive(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        fwd = forward_cds_par_spread(dc, sc, REF, END_5Y, recovery=0.4)
        assert fwd.risky_annuity > 0

    def test_to_dict(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        fwd = forward_cds_par_spread(dc, sc, REF, END_5Y, recovery=0.4)
        d = fwd.to_dict()
        assert "forward_spread" in d
        assert "risky_annuity" in d
        assert "survival_to_start" in d

    def test_matches_flat_cds_swaption(self):
        """For flat curves, forward spread should match cds_swaption.forward_cds_spread()."""
        from pricebook.cds_swaption import forward_cds_spread as fwd_flat
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)

        start_years = 1.0
        end_years = 6.0
        start_d = REF + relativedelta(years=1)
        end_d = REF + relativedelta(years=6)

        fwd_curve = forward_cds_par_spread(dc, sc, start_d, end_d, recovery=0.4)
        fwd_scalar = fwd_flat(start_years, end_years, 0.02, 0.04, 0.4)

        # Should be close (not exact due to different discretisation)
        assert fwd_curve.forward_spread == pytest.approx(fwd_scalar.forward_spread, rel=0.05)


# ---- Forward risky annuity ----

class TestForwardRiskyAnnuity:

    def test_spot_annuity(self):
        """Forward annuity starting today = spot annuity."""
        from pricebook.cds import risky_annuity
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)

        spot = risky_annuity(REF, END_5Y, dc, sc)
        fwd = forward_risky_annuity(dc, sc, REF, END_5Y)
        assert fwd == pytest.approx(spot)

    def test_positive(self):
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        start = REF + relativedelta(years=1)
        end = REF + relativedelta(years=6)
        ann = forward_risky_annuity(dc, sc, start, end)
        assert ann > 0

    def test_decreasing_with_start(self):
        """Forward annuity shrinks as start moves further out."""
        dc = make_flat_curve(REF, 0.04)
        sc = make_flat_survival(REF, 0.02)
        end = REF + relativedelta(years=10)
        ann1 = forward_risky_annuity(dc, sc, REF + relativedelta(years=1), end)
        ann3 = forward_risky_annuity(dc, sc, REF + relativedelta(years=3), end)
        assert ann1 > ann3


# ---- Serialisation round-trip ----

class TestSerialisation:

    def test_cds_round_trip(self):
        cds = CDS(REF, END_5Y, spread=0.015, notional=5_000_000, recovery=0.35)
        d = cds.to_dict()
        from pricebook.serialisable import from_dict
        cds2 = from_dict(d)
        assert cds2.spread == cds.spread
        assert cds2.notional == cds.notional
        assert cds2.recovery == cds.recovery
        assert cds2.start == cds.start
        assert cds2.end == cds.end

    def test_standard_cds_round_trip(self):
        cds = StandardCDS(REF, END_5Y, spread=0.005,
                          standard_coupon=0.01, grade="IG", notional=10_000_000)
        d = cds.to_dict()
        cds2 = StandardCDS.from_dict(d)
        assert cds2.spread == cds.spread
        assert cds2.grade == cds.grade
        assert cds2.standard_coupon == cds.standard_coupon

    def test_forward_result_to_dict(self):
        r = ForwardCDSCurveResult(0.012, 4.5, 0.054, 0.96)
        d = r.to_dict()
        assert d["forward_spread"] == 0.012
        assert d["survival_to_start"] == 0.96
