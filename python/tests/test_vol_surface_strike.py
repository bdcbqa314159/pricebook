"""Tests for VolSurfaceStrike."""

import pytest
from datetime import date

from pricebook.vol_smile import VolSmile
from pricebook.vol_surface_strike import VolSurfaceStrike


REF = date(2024, 1, 15)
EXP1 = date(2024, 7, 15)  # ~6M
EXP2 = date(2025, 1, 15)  # ~1Y


def _smile(atm: float, skew: float = 0.05):
    """Helper: symmetric smile around ATM with given skew."""
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    vols = [atm + 2 * skew, atm + skew, atm, atm + skew, atm + 2 * skew]
    return VolSmile(strikes, vols)


class TestConstruction:
    def test_basic(self):
        s = VolSurfaceStrike(REF, [EXP1], [_smile(0.20)])
        assert s.reference_date == REF

    def test_mismatched_raises(self):
        with pytest.raises(ValueError, match="same length"):
            VolSurfaceStrike(REF, [EXP1, EXP2], [_smile(0.20)])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            VolSurfaceStrike(REF, [], [])


class TestSingleExpiry:
    def test_at_pillar(self):
        s = VolSurfaceStrike(REF, [EXP1], [_smile(0.20)])
        assert s.vol(EXP1, 100.0) == pytest.approx(0.20, abs=1e-10)

    def test_any_date_returns_same_smile(self):
        s = VolSurfaceStrike(REF, [EXP1], [_smile(0.20)])
        v1 = s.vol(date(2024, 3, 15), 100.0)
        v2 = s.vol(date(2025, 6, 15), 100.0)
        assert v1 == pytest.approx(v2)


class TestTwoExpiries:
    @pytest.fixture
    def surface(self):
        return VolSurfaceStrike(
            REF,
            [EXP1, EXP2],
            [_smile(0.20), _smile(0.24)],
        )

    def test_at_first_expiry(self, surface):
        assert surface.vol(EXP1, 100.0) == pytest.approx(0.20, abs=1e-10)

    def test_at_second_expiry(self, surface):
        assert surface.vol(EXP2, 100.0) == pytest.approx(0.24, abs=1e-10)

    def test_midpoint_expiry(self, surface):
        """Midpoint between 6M (0.20) and 1Y (0.24) ≈ 0.22."""
        mid = date(2024, 10, 15)
        v = surface.vol(mid, 100.0)
        assert v == pytest.approx(0.22, abs=0.01)

    def test_strike_dimension_works(self, surface):
        """OTM strike should have higher vol (smile)."""
        v_atm = surface.vol(EXP1, 100.0)
        v_otm = surface.vol(EXP1, 110.0)
        assert v_otm > v_atm

    def test_interpolates_smile_at_intermediate_expiry(self, surface):
        """Smile shape should blend between expiries."""
        mid = date(2024, 10, 15)
        v_atm = surface.vol(mid, 100.0)
        v_wing = surface.vol(mid, 110.0)
        assert v_wing > v_atm


class TestExtrapolation:
    @pytest.fixture
    def surface(self):
        return VolSurfaceStrike(
            REF,
            [EXP1, EXP2],
            [_smile(0.20), _smile(0.24)],
        )

    def test_before_first(self, surface):
        v = surface.vol(date(2024, 2, 15), 100.0)
        assert v == pytest.approx(0.20, abs=1e-10)

    def test_after_last(self, surface):
        v = surface.vol(date(2026, 1, 15), 100.0)
        assert v == pytest.approx(0.24, abs=1e-10)


class TestVolInterface:
    def test_vol_without_strike(self):
        """vol(expiry) with no strike should still work."""
        s = VolSurfaceStrike(REF, [EXP1], [_smile(0.20)])
        v = s.vol(EXP1)
        assert v > 0

    def test_compatible_with_swaption(self):
        """Can be used wherever vol(expiry, strike) is expected."""
        s = VolSurfaceStrike(REF, [EXP1], [_smile(0.20)])
        v = s.vol(EXP1, 0.03)  # swaption-style strike
        assert v > 0
