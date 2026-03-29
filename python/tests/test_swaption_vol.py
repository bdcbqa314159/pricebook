"""Tests for SwaptionVolSurface."""

import pytest
from datetime import date

from pricebook.swaption_vol import SwaptionVolSurface


@pytest.fixture
def grid_surface():
    """2x3 vol grid: 2 expiries × 3 tenors."""
    return SwaptionVolSurface(
        reference_date=date(2024, 1, 15),
        expiries=[date(2025, 1, 15), date(2026, 1, 15)],
        tenors=[2.0, 5.0, 10.0],
        vols=[
            [0.20, 0.18, 0.16],  # 1Y expiry
            [0.22, 0.20, 0.18],  # 2Y expiry
        ],
    )


class TestConstruction:
    def test_basic(self, grid_surface):
        assert grid_surface.reference_date == date(2024, 1, 15)

    def test_single_point(self):
        s = SwaptionVolSurface(
            reference_date=date(2024, 1, 15),
            expiries=[date(2025, 1, 15)],
            tenors=[5.0],
            vols=[[0.20]],
        )
        assert s.vol_expiry_tenor(date(2025, 1, 15), 5.0) == pytest.approx(0.20)

    def test_mismatched_rows_raises(self):
        with pytest.raises(ValueError, match="rows"):
            SwaptionVolSurface(
                reference_date=date(2024, 1, 15),
                expiries=[date(2025, 1, 15)],
                tenors=[5.0],
                vols=[[0.20], [0.22]],  # 2 rows for 1 expiry
            )

    def test_mismatched_cols_raises(self):
        with pytest.raises(ValueError, match="columns"):
            SwaptionVolSurface(
                reference_date=date(2024, 1, 15),
                expiries=[date(2025, 1, 15)],
                tenors=[2.0, 5.0],
                vols=[[0.20]],  # 1 col for 2 tenors
            )

    def test_negative_vol_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            SwaptionVolSurface(
                reference_date=date(2024, 1, 15),
                expiries=[date(2025, 1, 15)],
                tenors=[5.0],
                vols=[[-0.01]],
            )

    def test_empty_expiries_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            SwaptionVolSurface(
                reference_date=date(2024, 1, 15),
                expiries=[],
                tenors=[5.0],
                vols=[],
            )


class TestAtPillars:
    def test_exact_pillar(self, grid_surface):
        """Vol at exact grid point."""
        v = grid_surface.vol_expiry_tenor(date(2025, 1, 15), 5.0)
        assert v == pytest.approx(0.18)

    def test_exact_pillar_corner(self, grid_surface):
        v = grid_surface.vol_expiry_tenor(date(2026, 1, 15), 10.0)
        assert v == pytest.approx(0.18)


class TestInterpolation:
    def test_interp_expiry_only(self, grid_surface):
        """Midpoint between two expiries at exact tenor."""
        v = grid_surface.vol_expiry_tenor(date(2025, 7, 15), 5.0)
        # Midpoint between 0.18 (1Y) and 0.20 (2Y) ≈ 0.19
        assert v == pytest.approx(0.19, abs=0.005)

    def test_interp_tenor_only(self, grid_surface):
        """Midpoint between two tenors at exact expiry."""
        v = grid_surface.vol_expiry_tenor(date(2025, 1, 15), 3.5)
        # Midpoint between 0.20 (2Y tenor) and 0.18 (5Y tenor) ≈ 0.19
        assert v == pytest.approx(0.19, abs=0.005)

    def test_bilinear_center(self, grid_surface):
        """Bilinear at the center of a grid cell."""
        v = grid_surface.vol_expiry_tenor(date(2025, 7, 15), 3.5)
        # Average of 4 corners: (0.20+0.18+0.22+0.20)/4 = 0.20
        assert v == pytest.approx(0.20, abs=0.005)


class TestExtrapolation:
    def test_flat_before_first_expiry(self, grid_surface):
        """Before first expiry: flat extrapolation."""
        v = grid_surface.vol_expiry_tenor(date(2024, 7, 15), 5.0)
        assert v == pytest.approx(0.18)  # same as first expiry, tenor=5

    def test_flat_after_last_expiry(self, grid_surface):
        """After last expiry: flat extrapolation."""
        v = grid_surface.vol_expiry_tenor(date(2030, 1, 15), 5.0)
        assert v == pytest.approx(0.20)  # same as last expiry, tenor=5

    def test_flat_before_first_tenor(self, grid_surface):
        """Below first tenor: flat extrapolation."""
        v = grid_surface.vol_expiry_tenor(date(2025, 1, 15), 0.5)
        assert v == pytest.approx(0.20)  # same as first tenor

    def test_flat_after_last_tenor(self, grid_surface):
        """Above last tenor: flat extrapolation."""
        v = grid_surface.vol_expiry_tenor(date(2025, 1, 15), 30.0)
        assert v == pytest.approx(0.16)  # same as last tenor


class TestVolInterface:
    def test_vol_method_compatible(self, grid_surface):
        """vol(expiry, strike) interface works for swaption pricing."""
        v = grid_surface.vol(date(2025, 1, 15), 0.03)
        assert v > 0

    def test_vol_strike_ignored(self, grid_surface):
        """ATM surface: strike doesn't affect the result."""
        v1 = grid_surface.vol(date(2025, 1, 15), 0.01)
        v2 = grid_surface.vol(date(2025, 1, 15), 0.10)
        assert v1 == pytest.approx(v2)

    def test_single_expiry_surface(self):
        """Single-expiry surface still works with vol() interface."""
        s = SwaptionVolSurface(
            reference_date=date(2024, 1, 15),
            expiries=[date(2025, 1, 15)],
            tenors=[2.0, 5.0, 10.0],
            vols=[[0.20, 0.18, 0.16]],
        )
        v = s.vol(date(2025, 7, 15), 0.03)
        assert v > 0
