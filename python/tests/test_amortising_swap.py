"""Tests for amortising, accreting, and roller-coaster swaps."""

import pytest
from datetime import date

from pricebook.amortising_swap import AmortisingSwap
from pricebook.swap import InterestRateSwap
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency


REF = date(2024, 1, 15)


def _flat_curve():
    return DiscountCurve.flat(REF, 0.05)


class TestAmortising:
    def test_bullet_is_special_case(self):
        """Constant notional = bullet swap (same frequency)."""
        curve = _flat_curve()
        freq = Frequency.SEMI_ANNUAL
        amort = AmortisingSwap(
            REF, date(2029, 1, 15), 0.05,
            notional_schedule=[1_000_000] * 20,
            frequency=freq,
        )
        bullet = InterestRateSwap(
            REF, date(2029, 1, 15), 0.05, notional=1_000_000,
            fixed_frequency=freq, float_frequency=freq,
        )
        # Both should have similar PV (small difference due to leg structure)
        assert amort.pv(curve) == pytest.approx(bullet.pv(curve), abs=5000)

    def test_amortising_dv01_less_than_bullet(self):
        curve = _flat_curve()
        amort = AmortisingSwap.amortising(
            REF, date(2029, 1, 15), 0.05, 1_000_000,
        )
        bullet = AmortisingSwap(
            REF, date(2029, 1, 15), 0.05,
            notional_schedule=[1_000_000] * 20,
        )
        assert abs(amort.dv01(curve)) < abs(bullet.dv01(curve))

    def test_amortising_notional_decreases(self):
        swap = AmortisingSwap.amortising(
            REF, date(2029, 1, 15), 0.05, 1_000_000,
        )
        assert swap.notionals[0] > swap.notionals[-1]
        assert swap.notionals[-1] >= 0

    def test_par_rate(self):
        curve = _flat_curve()
        swap = AmortisingSwap.amortising(
            REF, date(2029, 1, 15), 0.05, 1_000_000,
        )
        par = swap.par_rate(curve)
        swap2 = AmortisingSwap.amortising(
            REF, date(2029, 1, 15), par, 1_000_000,
        )
        assert swap2.pv(curve) == pytest.approx(0.0, abs=100)


class TestAccreting:
    def test_notional_increases(self):
        swap = AmortisingSwap.accreting(
            REF, date(2029, 1, 15), 0.05,
            initial_notional=500_000, final_notional=2_000_000,
        )
        assert swap.notionals[0] < swap.notionals[-1]

    def test_pv(self):
        curve = _flat_curve()
        swap = AmortisingSwap.accreting(
            REF, date(2029, 1, 15), 0.05,
            initial_notional=500_000, final_notional=2_000_000,
        )
        pv = swap.pv(curve)
        assert isinstance(pv, float)


class TestRollerCoaster:
    def test_arbitrary_schedule(self):
        curve = _flat_curve()
        notionals = [1e6, 2e6, 3e6, 2e6, 1e6, 500_000, 1e6, 2e6, 1.5e6, 1e6]
        swap = AmortisingSwap(
            REF, date(2029, 1, 15), 0.05, notionals,
        )
        pv = swap.pv(curve)
        assert isinstance(pv, float)

    def test_wal(self):
        swap = AmortisingSwap.amortising(
            REF, date(2029, 1, 15), 0.05, 1_000_000,
        )
        wal = swap.weighted_average_life
        assert 0 < wal < 5.0  # should be less than maturity

    def test_average_notional(self):
        swap = AmortisingSwap.amortising(
            REF, date(2029, 1, 15), 0.05, 1_000_000,
        )
        assert swap.average_notional > 0
        assert swap.average_notional < 1_000_000  # amortising → avg < initial
