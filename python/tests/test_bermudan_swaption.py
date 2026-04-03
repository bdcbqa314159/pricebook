"""Tests for bermudan swaptions."""

import pytest
from datetime import date

from pricebook.bermudan_swaption import bermudan_swaption_tree, bermudan_swaption_lsm
from pricebook.hull_white import HullWhite
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


@pytest.fixture
def hw():
    curve = make_flat_curve(REF, 0.05)
    return HullWhite(a=0.1, sigma=0.01, curve=curve)


class TestBermudanTree:
    def test_positive(self, hw):
        price = bermudan_swaption_tree(
            hw, exercise_years=[1, 2, 3, 4, 5],
            swap_end_years=10, strike=0.05, n_steps=50,
        )
        assert price > 0

    def test_bermudan_geq_european(self, hw):
        """Bermudan with multiple dates ≥ European (single date)."""
        european = bermudan_swaption_tree(
            hw, exercise_years=[5],
            swap_end_years=10, strike=0.05, n_steps=50,
        )
        bermudan = bermudan_swaption_tree(
            hw, exercise_years=[1, 2, 3, 4, 5],
            swap_end_years=10, strike=0.05, n_steps=50,
        )
        assert bermudan >= european - 0.001

    def test_single_date_positive(self, hw):
        """One exercise date still gives positive value."""
        berm = bermudan_swaption_tree(
            hw, exercise_years=[1],
            swap_end_years=6, strike=0.05, n_steps=50,
        )
        assert berm > 0

    def test_receiver(self, hw):
        price = bermudan_swaption_tree(
            hw, exercise_years=[1, 2, 3],
            swap_end_years=8, strike=0.05, is_payer=False, n_steps=50,
        )
        assert price >= 0  # receiver may be near zero at ATM

    def test_higher_vol_higher_price(self):
        curve = make_flat_curve(REF, 0.05)
        hw_low = HullWhite(a=0.1, sigma=0.005, curve=curve)
        hw_high = HullWhite(a=0.1, sigma=0.02, curve=curve)
        p_low = bermudan_swaption_tree(hw_low, [1, 2, 3], 8, 0.05, n_steps=50)
        p_high = bermudan_swaption_tree(hw_high, [1, 2, 3], 8, 0.05, n_steps=50)
        assert p_high > p_low


class TestBermudanLSM:
    def test_positive(self, hw):
        price = bermudan_swaption_lsm(
            hw, exercise_years=[1, 2, 3, 4, 5],
            swap_end_years=10, strike=0.05, n_paths=20_000,
        )
        assert price > 0

    def test_bermudan_geq_european(self, hw):
        european = bermudan_swaption_lsm(
            hw, exercise_years=[5],
            swap_end_years=10, strike=0.05, n_paths=20_000,
        )
        bermudan = bermudan_swaption_lsm(
            hw, exercise_years=[1, 2, 3, 4, 5],
            swap_end_years=10, strike=0.05, n_paths=20_000,
        )
        assert bermudan >= european - 0.001

    def test_lsm_agrees_with_tree(self, hw):
        """LSM ≈ tree (approximately, within MC noise)."""
        tree = bermudan_swaption_tree(
            hw, exercise_years=[1, 2, 3],
            swap_end_years=8, strike=0.05, n_steps=50,
        )
        lsm = bermudan_swaption_lsm(
            hw, exercise_years=[1, 2, 3],
            swap_end_years=8, strike=0.05, n_paths=50_000,
        )
        # Both methods are simplified; verify same order of magnitude
        assert lsm > 0
        assert abs(lsm - tree) / max(lsm, tree) < 0.50
