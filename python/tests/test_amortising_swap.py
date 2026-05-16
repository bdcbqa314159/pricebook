"""Tests for amortising, accreting, and roller-coaster swaps via InterestRateSwap."""

import pytest
from datetime import date

from pricebook.fixed_income.swap import InterestRateSwap
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.schedule import Frequency


REF = date(2024, 1, 15)
END = date(2029, 1, 15)


def _flat_curve():
    return DiscountCurve.flat(REF, 0.05)


class TestAmortising:
    def test_bullet_is_special_case(self):
        """Constant notional = bullet swap."""
        curve = _flat_curve()
        freq = Frequency.SEMI_ANNUAL
        amort = InterestRateSwap(
            REF, END, 0.05,
            notional=[1_000_000] * 10,
            fixed_frequency=freq, float_frequency=freq,
        )
        bullet = InterestRateSwap(
            REF, END, 0.05, notional=1_000_000,
            fixed_frequency=freq, float_frequency=freq,
        )
        assert amort.pv(curve) == pytest.approx(bullet.pv(curve), abs=5000)

    def test_amortising_dv01_less_than_bullet(self):
        curve = _flat_curve()
        amort = InterestRateSwap.amortising(REF, END, 0.05, 1_000_000)
        bullet = InterestRateSwap(REF, END, 0.05, notional=1_000_000)
        assert abs(amort.dv01(curve)) < abs(bullet.dv01(curve))

    def test_amortising_notional_decreases(self):
        swap = InterestRateSwap.amortising(REF, END, 0.05, 1_000_000)
        assert swap.notional_schedule[0] > swap.notional_schedule[-1]
        assert swap.notional_schedule[-1] >= 0

    def test_par_rate(self):
        curve = _flat_curve()
        swap = InterestRateSwap.amortising(REF, END, 0.05, 1_000_000)
        par = swap.par_rate(curve)
        swap2 = InterestRateSwap.amortising(REF, END, par, 1_000_000)
        assert swap2.pv(curve) == pytest.approx(0.0, abs=100)


class TestAccreting:
    def test_notional_increases(self):
        swap = InterestRateSwap.accreting(REF, END, 0.05, 500_000, 2_000_000)
        assert swap.notional_schedule[0] < swap.notional_schedule[-1]

    def test_pv(self):
        curve = _flat_curve()
        swap = InterestRateSwap.accreting(REF, END, 0.05, 500_000, 2_000_000)
        assert isinstance(swap.pv(curve), float)


class TestRollerCoaster:
    def test_arbitrary_schedule(self):
        curve = _flat_curve()
        notionals = [1e6, 2e6, 3e6, 2e6, 1e6, 500_000, 1e6, 2e6, 1.5e6, 1e6]
        swap = InterestRateSwap.roller_coaster(REF, END, 0.05, notionals)
        assert isinstance(swap.pv(curve), float)

    def test_serialisation_roundtrip(self):
        swap = InterestRateSwap.amortising(REF, END, 0.04, 1_000_000)
        d = swap.to_dict()
        swap2 = InterestRateSwap.from_dict(d)
        assert len(swap2.notional_schedule) == len(swap.notional_schedule)
