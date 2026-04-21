"""Tests for FI convention hardening (FI1-FI10)."""

from datetime import date, timedelta

import pytest

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.schedule import Frequency


# ---- FI1: ACT/ACT ICMA ----

class TestActActICMA:
    def test_full_semi_annual_period(self):
        """Full semi-annual coupon period = 0.5 years exactly."""
        yf = year_fraction(
            date(2026, 1, 15), date(2026, 7, 15),
            DayCountConvention.ACT_ACT_ICMA,
            ref_start=date(2026, 1, 15), ref_end=date(2026, 7, 15),
            frequency=2,
        )
        assert yf == pytest.approx(0.5, rel=1e-10)

    def test_half_period(self):
        """Half of a semi-annual period = 0.25 years."""
        yf = year_fraction(
            date(2026, 1, 15), date(2026, 4, 15),
            DayCountConvention.ACT_ACT_ICMA,
            ref_start=date(2026, 1, 15), ref_end=date(2026, 7, 15),
            frequency=2,
        )
        # 90 days out of 181 total days in period, freq=2
        # yf = 90 / (181 * 2) = 0.2486
        expected = 90 / (181 * 2)
        assert yf == pytest.approx(expected, rel=1e-10)

    def test_quarterly_full_period(self):
        """Full quarterly period = 0.25 years."""
        yf = year_fraction(
            date(2026, 1, 15), date(2026, 4, 15),
            DayCountConvention.ACT_ACT_ICMA,
            ref_start=date(2026, 1, 15), ref_end=date(2026, 4, 15),
            frequency=4,
        )
        assert yf == pytest.approx(0.25, rel=1e-10)

    def test_annual_full_period(self):
        """Full annual period = 1.0 year."""
        yf = year_fraction(
            date(2026, 1, 15), date(2027, 1, 15),
            DayCountConvention.ACT_ACT_ICMA,
            ref_start=date(2026, 1, 15), ref_end=date(2027, 1, 15),
            frequency=1,
        )
        assert yf == pytest.approx(1.0, rel=1e-10)

    def test_differs_from_isda(self):
        """ICMA and ISDA should give different results for same period."""
        args = (date(2026, 1, 15), date(2026, 7, 15))
        icma = year_fraction(*args, DayCountConvention.ACT_ACT_ICMA,
                             ref_start=args[0], ref_end=args[1], frequency=2)
        isda = year_fraction(*args, DayCountConvention.ACT_ACT_ISDA)
        # ICMA = exactly 0.5, ISDA = 181/365 ≈ 0.4959
        assert icma == pytest.approx(0.5)
        assert isda != pytest.approx(0.5)

    def test_fallback_without_ref_dates(self):
        """Without ref dates, falls back to ACT/365F."""
        yf = year_fraction(
            date(2026, 1, 15), date(2026, 7, 15),
            DayCountConvention.ACT_ACT_ICMA,
        )
        expected = 181 / 365.0
        assert yf == pytest.approx(expected, rel=1e-10)
