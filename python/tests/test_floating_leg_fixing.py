"""Tests for floating leg fixing management (FX1-FX6)."""

from datetime import date, timedelta

import pytest

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.floating_leg import FloatingCashflow, FloatingLeg
from pricebook.schedule import Frequency, StubType


# ---- Helpers ----

def _flat_curve(ref: date, rate: float = 0.04, years: int = 12) -> DiscountCurve:
    """Build a flat discount curve for testing."""
    import math
    dates = [ref + timedelta(days=365 * i) for i in range(1, years + 1)]
    dfs = [math.exp(-rate * i) for i in range(1, years + 1)]
    return DiscountCurve(ref, dates, dfs)


# ---- FX1: Payment Delay ----

class TestPaymentDelay:
    def test_default_zero_delay(self):
        """Default: payment_date == accrual_end (backward compatible)."""
        leg = FloatingLeg(
            date(2026, 4, 21), date(2027, 4, 21), Frequency.QUARTERLY,
        )
        for cf in leg.cashflows:
            assert cf.payment_date == cf.accrual_end

    def test_two_day_delay(self):
        """T+2 payment delay: payment_date = accrual_end + 2 days."""
        leg = FloatingLeg(
            date(2026, 4, 21), date(2027, 4, 21), Frequency.QUARTERLY,
            payment_delay_days=2,
        )
        for cf in leg.cashflows:
            assert cf.payment_date == cf.accrual_end + timedelta(days=2)

    def test_delay_affects_pv(self):
        """Payment delay reduces PV (later payment = more discounting)."""
        ref = date(2026, 4, 21)
        curve = _flat_curve(ref)
        kwargs = dict(
            start=ref, end=date(2031, 4, 21), frequency=Frequency.QUARTERLY,
            notional=1_000_000, spread=0.01,
        )
        pv_no_delay = FloatingLeg(**kwargs, payment_delay_days=0).pv(curve)
        pv_with_delay = FloatingLeg(**kwargs, payment_delay_days=2).pv(curve)
        # Later payment → more discounting → lower PV
        assert pv_with_delay < pv_no_delay

    def test_five_day_delay(self):
        """Larger delay = larger PV impact."""
        ref = date(2026, 4, 21)
        curve = _flat_curve(ref)
        kwargs = dict(
            start=ref, end=date(2031, 4, 21), frequency=Frequency.QUARTERLY,
            notional=1_000_000, spread=0.01,
        )
        pv_2d = FloatingLeg(**kwargs, payment_delay_days=2).pv(curve)
        pv_5d = FloatingLeg(**kwargs, payment_delay_days=5).pv(curve)
        assert pv_5d < pv_2d

    def test_negative_delay_raises(self):
        with pytest.raises(ValueError, match="payment_delay_days"):
            FloatingLeg(
                date(2026, 4, 21), date(2027, 4, 21), Frequency.QUARTERLY,
                payment_delay_days=-1,
            )

    def test_zero_delay_backward_compatible(self):
        """Zero delay produces same PV as old code (no delay parameter)."""
        ref = date(2026, 4, 21)
        curve = _flat_curve(ref)
        leg = FloatingLeg(
            start=ref, end=date(2028, 4, 21), frequency=Frequency.QUARTERLY,
            notional=1_000_000, spread=0.005, payment_delay_days=0,
        )
        # Manually compute PV the old way (payment_date = accrual_end)
        proj = curve
        expected = sum(
            cf.amount(proj) * curve.df(cf.accrual_end)
            for cf in leg.cashflows
        )
        assert leg.pv(curve) == pytest.approx(expected, rel=1e-12)
