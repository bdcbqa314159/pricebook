"""Tests for interest rate swap."""

import pytest
from datetime import date

from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.schedule import Frequency
from pricebook.day_count import DayCountConvention
from tests.conftest import make_flat_curve


class TestParRate:
    """Par rate makes PV = 0."""

    def test_par_rate_zeroes_pv(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)

        # First find the par rate
        swap_dummy = InterestRateSwap(
            ref, date(2026, 1, 15), fixed_rate=0.0,
            fixed_frequency=Frequency.SEMI_ANNUAL,
            float_frequency=Frequency.QUARTERLY,
        )
        par = swap_dummy.par_rate(curve)

        # Build a swap at par rate — PV should be ~0
        swap_at_par = InterestRateSwap(
            ref, date(2026, 1, 15), fixed_rate=par,
            fixed_frequency=Frequency.SEMI_ANNUAL,
            float_frequency=Frequency.QUARTERLY,
        )
        assert swap_at_par.pv(curve) == pytest.approx(0.0, abs=1.0)

    def test_par_rate_positive(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        swap = InterestRateSwap(
            ref, date(2029, 1, 15), fixed_rate=0.0,
            fixed_frequency=Frequency.SEMI_ANNUAL,
            float_frequency=Frequency.QUARTERLY,
        )
        assert swap.par_rate(curve) > 0

    def test_par_rate_close_to_curve_rate(self):
        """On a flat curve the par rate should be close to the curve rate."""
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.04)
        swap = InterestRateSwap(
            ref, date(2029, 1, 15), fixed_rate=0.0,
            fixed_frequency=Frequency.SEMI_ANNUAL,
            float_frequency=Frequency.QUARTERLY,
        )
        par = swap.par_rate(curve)
        # Not exactly equal due to compounding/day count differences,
        # but should be in the same ballpark
        assert abs(par - 0.04) < 0.005


class TestDirection:
    """Payer vs receiver."""

    def test_payer_and_receiver_opposite_sign(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        payer = InterestRateSwap(
            ref, date(2026, 1, 15), fixed_rate=0.04,
            direction=SwapDirection.PAYER,
        )
        receiver = InterestRateSwap(
            ref, date(2026, 1, 15), fixed_rate=0.04,
            direction=SwapDirection.RECEIVER,
        )
        assert payer.pv(curve) == pytest.approx(-receiver.pv(curve), rel=1e-10)

    def test_payer_positive_when_fixed_below_market(self):
        """Pay a low fixed rate, receive floating at market — should be positive."""
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        swap = InterestRateSwap(
            ref, date(2026, 1, 15), fixed_rate=0.01,
            direction=SwapDirection.PAYER,
        )
        assert swap.pv(curve) > 0

    def test_payer_negative_when_fixed_above_market(self):
        """Pay a high fixed rate, receive floating at market — should be negative."""
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        swap = InterestRateSwap(
            ref, date(2026, 1, 15), fixed_rate=0.10,
            direction=SwapDirection.PAYER,
        )
        assert swap.pv(curve) < 0


class TestPV:
    """General PV properties."""

    def test_pv_scales_with_notional(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        swap1 = InterestRateSwap(
            ref, date(2026, 1, 15), fixed_rate=0.04, notional=1_000_000.0,
        )
        swap2 = InterestRateSwap(
            ref, date(2026, 1, 15), fixed_rate=0.04, notional=2_000_000.0,
        )
        assert swap2.pv(curve) == pytest.approx(2 * swap1.pv(curve), rel=1e-10)

    def test_pv_at_par_rate_is_zero(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        swap = InterestRateSwap(
            ref, date(2029, 1, 15), fixed_rate=0.0,
        )
        par = swap.par_rate(curve)
        swap_par = InterestRateSwap(
            ref, date(2029, 1, 15), fixed_rate=par,
        )
        assert swap_par.pv(curve) == pytest.approx(0.0, abs=1.0)

    def test_longer_maturity_higher_sensitivity(self):
        """A 5Y swap should have more PV sensitivity than a 1Y swap for the same off-market rate."""
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        swap_1y = InterestRateSwap(ref, date(2025, 1, 15), fixed_rate=0.03)
        swap_5y = InterestRateSwap(ref, date(2029, 1, 15), fixed_rate=0.03)
        assert abs(swap_5y.pv(curve)) > abs(swap_1y.pv(curve))


class TestSpread:
    """Floating leg spread."""

    def test_spread_shifts_pv(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        swap_no_spread = InterestRateSwap(
            ref, date(2026, 1, 15), fixed_rate=0.05, spread=0.0,
        )
        swap_with_spread = InterestRateSwap(
            ref, date(2026, 1, 15), fixed_rate=0.05, spread=0.01,
        )
        # Positive spread on floating leg increases payer PV
        assert swap_with_spread.pv(curve) > swap_no_spread.pv(curve)


class TestDualCurve:
    """Dual-curve swap pricing."""

    def test_single_curve_equivalent(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        swap = InterestRateSwap(ref, date(2026, 1, 15), fixed_rate=0.04)
        assert swap.pv(curve) == pytest.approx(swap.pv(curve, projection_curve=curve), rel=1e-10)

    def test_par_rate_dual_curve(self):
        """Par rate at dual-curve zeroes PV."""
        ref = date(2024, 1, 15)
        discount = make_flat_curve(ref, rate=0.03)
        projection = make_flat_curve(ref, rate=0.05)
        swap = InterestRateSwap(ref, date(2026, 1, 15), fixed_rate=0.0)
        par = swap.par_rate(discount, projection_curve=projection)
        swap_par = InterestRateSwap(ref, date(2026, 1, 15), fixed_rate=par)
        assert swap_par.pv(discount, projection_curve=projection) == pytest.approx(0.0, abs=1.0)

    def test_dual_curve_par_differs_from_single(self):
        ref = date(2024, 1, 15)
        discount = make_flat_curve(ref, rate=0.03)
        projection = make_flat_curve(ref, rate=0.05)
        swap = InterestRateSwap(ref, date(2026, 1, 15), fixed_rate=0.0)
        par_single = swap.par_rate(discount)
        par_dual = swap.par_rate(discount, projection_curve=projection)
        assert par_single != pytest.approx(par_dual, rel=1e-3)
