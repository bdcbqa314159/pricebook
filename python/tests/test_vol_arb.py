"""Tests for vol surface arbitrage detection and enforcement."""

import math
import pytest
from datetime import date

from pricebook.vol_arb import (
    CalendarArbViolation,
    ButterflyArbViolation,
    SurfaceArbReport,
    check_surface_arbitrage,
    detect_butterfly_arb,
    detect_calendar_arb,
    enforce_no_butterfly_arb,
    enforce_no_calendar_arb,
)


REF = date(2024, 1, 15)


# ---- Step 1: no-arb detection ----

class TestCalendarArb:
    def test_no_arb_increasing_total_var(self):
        expiries = [date(2024, 4, 15), date(2024, 7, 15), date(2025, 1, 15)]
        vols = [0.20, 0.20, 0.20]  # flat vol → total var increases with T
        violations = detect_calendar_arb(expiries, vols, REF)
        assert violations == []

    def test_detect_arb(self):
        """Step 1 test: detect a planted arbitrage."""
        expiries = [date(2024, 4, 15), date(2024, 7, 15)]
        # Short-term vol so high that σ²T₁ > σ²T₂
        vols = [0.50, 0.15]
        violations = detect_calendar_arb(expiries, vols, REF)
        assert len(violations) == 1
        assert violations[0].short_total_var > violations[0].long_total_var

    def test_borderline_no_violation(self):
        expiries = [date(2024, 4, 15), date(2024, 7, 15)]
        # Exactly flat total variance → no violation
        t1 = (expiries[0] - REF).days / 365.0
        t2 = (expiries[1] - REF).days / 365.0
        v1 = 0.20
        v2 = v1 * math.sqrt(t1 / t2)
        violations = detect_calendar_arb(expiries, [v1, v2], REF)
        assert violations == []


class TestButterflyArb:
    def test_no_arb_convex(self):
        strikes = [90.0, 100.0, 110.0]
        prices = [12.0, 5.0, 1.5]  # convex (decreasing, concave shape OK)
        violations = detect_butterfly_arb(strikes, prices)
        assert violations == []

    def test_detect_arb(self):
        """Step 1 test: detect a planted butterfly arbitrage."""
        strikes = [90.0, 100.0, 110.0]
        # Mid-price too high → not convex
        prices = [10.0, 9.0, 2.0]
        # Interpolated at K=100: w=(110-100)/(110-90)=0.5, cap=0.5×10+0.5×2=6
        # 9 > 6 → violation
        violations = detect_butterfly_arb(strikes, prices)
        assert len(violations) == 1

    def test_five_strikes(self):
        strikes = [80, 90, 100, 110, 120]
        prices = [20.0, 12.0, 15.0, 4.0, 1.0]  # spike at 100 → arb
        violations = detect_butterfly_arb(strikes, prices)
        assert len(violations) > 0


class TestCombinedCheck:
    def test_arb_free_surface(self):
        expiries = [date(2024, 4, 15), date(2024, 7, 15)]
        vols = [0.20, 0.20]
        report = check_surface_arbitrage(expiries, vols, REF)
        assert report.is_arb_free is True

    def test_calendar_arb_detected(self):
        expiries = [date(2024, 4, 15), date(2024, 7, 15)]
        vols = [0.50, 0.10]
        report = check_surface_arbitrage(expiries, vols, REF)
        assert report.is_arb_free is False
        assert len(report.calendar_violations) > 0


# ---- Step 2: arb-free construction ----

class TestEnforceNoCalendarArb:
    def test_fixes_violation(self):
        """Step 2 test: fitted surface passes all no-arb checks."""
        expiries = [date(2024, 4, 15), date(2024, 7, 15)]
        vols = [0.50, 0.10]
        adjusted = enforce_no_calendar_arb(expiries, vols, REF)
        violations = detect_calendar_arb(expiries, adjusted, REF)
        assert violations == []

    def test_no_change_when_clean(self):
        expiries = [date(2024, 4, 15), date(2024, 7, 15)]
        vols = [0.20, 0.20]
        adjusted = enforce_no_calendar_arb(expiries, vols, REF)
        assert adjusted == pytest.approx(vols)

    def test_multi_expiry(self):
        expiries = [date(2024, 4, 15), date(2024, 7, 15), date(2025, 1, 15)]
        vols = [0.40, 0.10, 0.15]  # first→second is arb
        adjusted = enforce_no_calendar_arb(expiries, vols, REF)
        violations = detect_calendar_arb(expiries, adjusted, REF)
        assert violations == []


class TestEnforceNoButterflyArb:
    def test_fixes_violation(self):
        strikes = [90.0, 100.0, 110.0]
        prices = [10.0, 9.0, 2.0]  # mid too high
        adjusted = enforce_no_butterfly_arb(strikes, prices)
        violations = detect_butterfly_arb(strikes, adjusted)
        assert violations == []
        # Mid should be capped at interpolation
        assert adjusted[1] <= 0.5 * adjusted[0] + 0.5 * adjusted[2] + 1e-10

    def test_no_change_when_convex(self):
        strikes = [90.0, 100.0, 110.0]
        prices = [12.0, 5.0, 1.5]
        adjusted = enforce_no_butterfly_arb(strikes, prices)
        assert adjusted == pytest.approx(prices)
