"""Tests for IBORCurve: conventions, bootstrap, forward rates, fixings."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from pricebook.bootstrap import bootstrap_forward_curve
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.ibor_curve import (
    IBORCurve,
    IBORConventions,
    bootstrap_ibor,
    EURIBOR_3M_CONVENTIONS,
    EURIBOR_6M_CONVENTIONS,
    TIBOR_3M_CONVENTIONS,
)
from pricebook.ois import bootstrap_ois
from pricebook.rate_index import EURIBOR_3M, EURIBOR_6M, TIBOR_3M
from pricebook.schedule import Frequency, generate_schedule

REF = date(2026, 4, 27)


def _ois_curve():
    """Flat OIS at 3%."""
    return DiscountCurve.flat(REF, 0.03)


def _ois_from_rates():
    """OIS curve from par rates (more realistic)."""
    rates = [
        (REF + timedelta(days=365), 0.030),
        (REF + timedelta(days=730), 0.031),
        (REF + timedelta(days=1825), 0.033),
        (REF + timedelta(days=3650), 0.035),
    ]
    return bootstrap_ois(REF, rates)


def _euribor_3m_swaps():
    """EURIBOR 3M swap par rates (above OIS by ~10-20bp)."""
    return [
        (REF + timedelta(days=365), 0.032),
        (REF + timedelta(days=730), 0.033),
        (REF + timedelta(days=1095), 0.034),
        (REF + timedelta(days=1825), 0.035),
        (REF + timedelta(days=3650), 0.037),
    ]


def _euribor_3m_deposits():
    """Short-end deposits for EURIBOR 3M."""
    return [
        (REF + timedelta(days=91), 0.031),
    ]


# ---- Conventions ----

class TestIBORConventions:

    def test_euribor_3m_conventions(self):
        c = EURIBOR_3M_CONVENTIONS
        assert c.index is EURIBOR_3M
        assert c.float_frequency == Frequency.QUARTERLY
        assert c.float_day_count == DayCountConvention.ACT_360
        assert c.fixed_frequency == Frequency.ANNUAL
        assert c.fixed_day_count == DayCountConvention.THIRTY_360
        assert c.tenor_months == 3
        assert c.currency == "EUR"
        assert c.name == "EURIBOR_3M"

    def test_euribor_6m_conventions(self):
        c = EURIBOR_6M_CONVENTIONS
        assert c.float_frequency == Frequency.SEMI_ANNUAL
        assert c.tenor_months == 6

    def test_tibor_conventions(self):
        c = TIBOR_3M_CONVENTIONS
        assert c.float_day_count == DayCountConvention.ACT_365_FIXED
        assert c.fixed_frequency == Frequency.SEMI_ANNUAL
        assert c.currency == "JPY"

    def test_frozen(self):
        """Conventions should be immutable."""
        with pytest.raises(AttributeError):
            EURIBOR_3M_CONVENTIONS.spot_lag = 3


# ---- IBORCurve basic ----

class TestIBORCurve:

    def test_construction(self):
        ois = _ois_curve()
        proj = DiscountCurve.flat(REF, 0.035)
        ibor = IBORCurve(proj, EURIBOR_3M_CONVENTIONS, ois)
        assert ibor.reference_date == REF
        assert ibor.index is EURIBOR_3M
        assert ibor.tenor_months == 3
        assert ibor.discount_curve is ois
        assert ibor.projection_curve is proj

    def test_forward_rate_flat(self):
        """Flat curve → constant forward rate."""
        proj = DiscountCurve.flat(REF, 0.035)
        ibor = IBORCurve(proj, EURIBOR_3M_CONVENTIONS)
        d1 = REF + timedelta(days=365)
        d2 = REF + timedelta(days=456)  # ~3M later
        fwd = ibor.forward_rate(d1, d2)
        # For flat CC rate, simply-compounded forward ≈ rate (small difference)
        assert fwd == pytest.approx(0.035, abs=0.002)

    def test_forward_rate_consistency(self):
        """forward_rate uses float_day_count (ACT/360 for EURIBOR)."""
        proj = DiscountCurve.flat(REF, 0.035)
        ibor = IBORCurve(proj, EURIBOR_3M_CONVENTIONS)
        d1 = REF + timedelta(days=365)
        d2 = REF + timedelta(days=456)
        fwd = ibor.forward_rate(d1, d2)
        # Manual: (df1/df2 - 1) / tau_ACT360
        df1 = proj.df(d1)
        df2 = proj.df(d2)
        tau = year_fraction(d1, d2, DayCountConvention.ACT_360)
        expected = (df1 / df2 - 1.0) / tau
        assert fwd == pytest.approx(expected)

    def test_fixing(self):
        """fixing() computes forward rate for the index tenor."""
        proj = DiscountCurve.flat(REF, 0.035)
        ibor = IBORCurve(proj, EURIBOR_3M_CONVENTIONS)
        fix = ibor.fixing(REF + timedelta(days=100))
        assert math.isfinite(fix)
        assert fix > 0

    def test_df_delegates(self):
        proj = DiscountCurve.flat(REF, 0.035)
        ibor = IBORCurve(proj, EURIBOR_3M_CONVENTIONS)
        d = REF + timedelta(days=365)
        assert ibor.df(d) == pytest.approx(proj.df(d))

    def test_bumped(self):
        proj = DiscountCurve.flat(REF, 0.035)
        ibor = IBORCurve(proj, EURIBOR_3M_CONVENTIONS)
        bumped = ibor.bumped(0.001)
        d = REF + timedelta(days=365)
        assert bumped.zero_rate(d) > ibor.zero_rate(d)
        assert bumped.conventions is ibor.conventions


# ---- Bootstrap ----

class TestBootstrapIBOR:

    def test_basic_bootstrap(self):
        """Bootstrap EURIBOR 3M from deposits + swaps with OIS discounting."""
        ois = _ois_curve()
        ibor = bootstrap_ibor(
            REF, EURIBOR_3M_CONVENTIONS, ois,
            deposits=_euribor_3m_deposits(),
            swaps=_euribor_3m_swaps(),
        )
        assert ibor.reference_date == REF
        assert ibor.conventions is EURIBOR_3M_CONVENTIONS

    def test_round_trip_swaps(self):
        """Each input swap reprices at par under dual-curve."""
        ois = _ois_curve()
        swaps = _euribor_3m_swaps()
        ibor = bootstrap_ibor(REF, EURIBOR_3M_CONVENTIONS, ois, swaps=swaps)
        conv = EURIBOR_3M_CONVENTIONS

        for mat, par_rate in swaps:
            fixed_sched = generate_schedule(REF, mat, conv.fixed_frequency)
            float_sched = generate_schedule(REF, mat, conv.float_frequency)

            pv_fixed = 0.0
            for i in range(1, len(fixed_sched)):
                yf = year_fraction(fixed_sched[i-1], fixed_sched[i], conv.fixed_day_count)
                pv_fixed += par_rate * yf * ois.df(fixed_sched[i])

            pv_float = 0.0
            for i in range(1, len(float_sched)):
                d1, d2 = float_sched[i-1], float_sched[i]
                fwd = ibor.forward_rate(d1, d2)
                yf = year_fraction(d1, d2, conv.float_day_count)
                pv_float += fwd * yf * ois.df(d2)

            assert pv_fixed == pytest.approx(pv_float, abs=1e-8), \
                f"Swap {mat} failed: PV_fixed={pv_fixed:.10f}, PV_float={pv_float:.10f}"

    def test_forward_rates_positive(self):
        """All forward rates should be positive."""
        ois = _ois_curve()
        ibor = bootstrap_ibor(REF, EURIBOR_3M_CONVENTIONS, ois, swaps=_euribor_3m_swaps())
        for m in range(3, 120, 3):
            d1 = REF + timedelta(days=m * 30)
            d2 = REF + timedelta(days=(m + 3) * 30)
            assert ibor.forward_rate(d1, d2) > 0

    def test_ibor_above_ois(self):
        """IBOR forwards should be above OIS forwards (positive basis)."""
        ois = _ois_curve()
        ibor = bootstrap_ibor(REF, EURIBOR_3M_CONVENTIONS, ois, swaps=_euribor_3m_swaps())
        d1 = REF + timedelta(days=365)
        d2 = REF + timedelta(days=456)
        ibor_fwd = ibor.forward_rate(d1, d2)
        ois_fwd = ois.forward_rate(d1, d2)
        assert ibor_fwd > ois_fwd

    def test_euribor_6m_bootstrap(self):
        """Bootstrap EURIBOR 6M (semi-annual float, annual fixed)."""
        ois = _ois_curve()
        swaps_6m = [
            (REF + timedelta(days=365), 0.033),
            (REF + timedelta(days=730), 0.034),
            (REF + timedelta(days=1825), 0.036),
        ]
        ibor = bootstrap_ibor(REF, EURIBOR_6M_CONVENTIONS, ois, swaps=swaps_6m)
        assert ibor.tenor_months == 6

        # Round-trip for 6M conventions
        conv = EURIBOR_6M_CONVENTIONS
        for mat, par_rate in swaps_6m:
            fixed_sched = generate_schedule(REF, mat, conv.fixed_frequency)
            float_sched = generate_schedule(REF, mat, conv.float_frequency)

            pv_fixed = sum(
                par_rate * year_fraction(fixed_sched[i-1], fixed_sched[i], conv.fixed_day_count)
                * ois.df(fixed_sched[i])
                for i in range(1, len(fixed_sched))
            )
            pv_float = sum(
                ibor.forward_rate(float_sched[i-1], float_sched[i])
                * year_fraction(float_sched[i-1], float_sched[i], conv.float_day_count)
                * ois.df(float_sched[i])
                for i in range(1, len(float_sched))
            )
            assert pv_fixed == pytest.approx(pv_float, abs=1e-8)

    def test_with_realistic_ois(self):
        """Bootstrap with OIS curve from par rates (not flat)."""
        ois = _ois_from_rates()
        ibor = bootstrap_ibor(REF, EURIBOR_3M_CONVENTIONS, ois, swaps=_euribor_3m_swaps())
        d1 = REF + timedelta(days=365)
        d2 = REF + timedelta(days=456)
        fwd = ibor.forward_rate(d1, d2)
        assert math.isfinite(fwd)
        assert fwd > 0

    def test_deposits_only(self):
        """Short curve from deposits only."""
        ois = _ois_curve()
        deps = [
            (REF + timedelta(days=30), 0.030),
            (REF + timedelta(days=91), 0.031),
            (REF + timedelta(days=182), 0.032),
        ]
        ibor = bootstrap_ibor(REF, EURIBOR_3M_CONVENTIONS, ois, deposits=deps)
        assert ibor.df(REF + timedelta(days=91)) < 1.0

    def test_no_instruments_raises(self):
        ois = _ois_curve()
        with pytest.raises(ValueError, match="At least one instrument"):
            bootstrap_ibor(REF, EURIBOR_3M_CONVENTIONS, ois)

    def test_deposits_plus_swaps(self):
        """Combined deposits + swaps bootstrap."""
        ois = _ois_curve()
        ibor = bootstrap_ibor(
            REF, EURIBOR_3M_CONVENTIONS, ois,
            deposits=_euribor_3m_deposits(),
            swaps=_euribor_3m_swaps(),
        )
        # Deposit round-trip
        dep_mat, dep_rate = _euribor_3m_deposits()[0]
        tau = year_fraction(REF, dep_mat, EURIBOR_3M_CONVENTIONS.float_day_count)
        model_rate = (1.0 / ibor.df(dep_mat) - 1.0) / tau
        assert model_rate == pytest.approx(dep_rate, abs=1e-6)
