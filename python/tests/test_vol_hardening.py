"""Tests for vol infrastructure hardening (VH5-VH12)."""

from datetime import date

import pytest

from pricebook.vol_surface import (
    FlatVol, VolTermStructure,
    check_calendar_arbitrage, check_butterfly_arbitrage, validate_vol_surface,
)
from pricebook.vol_smile import VolSmile
from pricebook.vol_surface_strike import VolSurfaceStrike


# ---- VH5: VolSurface.bumped() ----

class TestFlatVolBumped:
    def test_bump_up(self):
        fv = FlatVol(0.20)
        bumped = fv.bumped(0.01)
        assert bumped.vol() == pytest.approx(0.21)

    def test_bump_down(self):
        fv = FlatVol(0.20)
        bumped = fv.bumped(-0.05)
        assert bumped.vol() == pytest.approx(0.15)

    def test_bump_floor_at_zero(self):
        fv = FlatVol(0.02)
        bumped = fv.bumped(-0.10)
        assert bumped.vol() == 0.0


class TestVolTermStructureBumped:
    def test_bump_shifts_all(self):
        ref = date(2026, 4, 21)
        vts = VolTermStructure(ref, [date(2027, 4, 21), date(2028, 4, 21)], [0.20, 0.22])
        bumped = vts.bumped(0.01)
        assert bumped.vol(date(2027, 4, 21)) == pytest.approx(0.21, rel=1e-4)
        assert bumped.vol(date(2028, 4, 21)) == pytest.approx(0.23, rel=1e-4)


class TestVolSmileBumped:
    def test_bump_shifts_all(self):
        smile = VolSmile([90, 95, 100, 105, 110], [0.25, 0.22, 0.20, 0.22, 0.25])
        bumped = smile.bumped(0.02)
        assert bumped.vol(100) == pytest.approx(0.22)
        assert bumped.vol(90) == pytest.approx(0.27)


class TestVolSurfaceStrikeBumped:
    def test_bump_shifts_surface(self):
        ref = date(2026, 4, 21)
        smile1 = VolSmile([90, 100, 110], [0.25, 0.20, 0.25])
        smile2 = VolSmile([90, 100, 110], [0.27, 0.22, 0.27])
        surface = VolSurfaceStrike(ref, [date(2027, 4, 21), date(2028, 4, 21)],
                                   [smile1, smile2])
        bumped = surface.bumped(0.01)
        assert bumped.vol(date(2027, 4, 21), 100) == pytest.approx(0.21, rel=1e-4)


# ---- VH6: Arbitrage checks ----

class TestCalendarArbitrage:
    def test_no_violation_monotone(self):
        """Increasing total variance → no violation."""
        violations = check_calendar_arbitrage([0.5, 1.0, 2.0], [0.20, 0.20, 0.20])
        assert len(violations) == 0

    def test_violation_decreasing_total_var(self):
        """Decreasing total variance → calendar arbitrage."""
        # σ²T: 0.04*0.5=0.02, 0.01*1.0=0.01 → decreasing
        violations = check_calendar_arbitrage([0.5, 1.0], [0.20, 0.10])
        assert len(violations) == 1

    def test_flat_vol_no_violation(self):
        violations = check_calendar_arbitrage([0.25, 0.5, 1.0, 2.0], [0.20] * 4)
        assert len(violations) == 0


class TestButterflyArbitrage:
    def test_convex_prices_no_violation(self):
        """Convex call prices → no butterfly arbitrage."""
        # Call prices should be convex in strike
        strikes = [90, 95, 100, 105, 110]
        prices = [12.0, 8.0, 5.0, 3.0, 2.0]  # convex
        violations = check_butterfly_arbitrage(strikes, prices)
        assert len(violations) == 0

    def test_non_convex_violation(self):
        """Non-convex call prices → butterfly arbitrage."""
        strikes = [90, 95, 100, 105, 110]
        prices = [12.0, 8.0, 5.0, 6.0, 2.0]  # bump at 105 → non-convex
        violations = check_butterfly_arbitrage(strikes, prices)
        assert len(violations) > 0


class TestValidateVolSurface:
    def test_clean_surface(self):
        result = validate_vol_surface([0.5, 1.0, 2.0], [0.20, 0.20, 0.20])
        assert result.is_arbitrage_free
        assert result.total_variance_monotone

    def test_dirty_surface(self):
        result = validate_vol_surface([0.5, 1.0], [0.20, 0.10])
        assert not result.is_arbitrage_free
