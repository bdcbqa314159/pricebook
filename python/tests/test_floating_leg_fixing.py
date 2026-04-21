"""Tests for floating leg fixing management (FX1-FX6)."""

from datetime import date, timedelta

import pytest

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.fixings import FixingsStore
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


# ---- FX2: FixingsStore Integration ----

class TestFixingsIntegration:
    def test_past_periods_use_fixings(self):
        """Past accrual periods should use stored fixings, not forward curve."""
        # Curve reference date is mid-leg, so some periods are in the past
        ref = date(2026, 10, 21)
        curve = _flat_curve(ref, rate=0.04)

        leg = FloatingLeg(
            start=date(2026, 4, 21), end=date(2027, 4, 21),
            frequency=Frequency.QUARTERLY,
            notional=1_000_000,
        )

        store = FixingsStore()
        # Set a fixing for the first period (starts Apr 21, ends ~Jul 21)
        store.set("SOFR", date(2026, 4, 21), 0.05)
        # Set a fixing for the second period (starts ~Jul 21, ends ~Oct 21)
        store.set("SOFR", date(2026, 7, 21), 0.045)

        pv_with_fixings = leg.pv(curve, fixings=store, rate_name="SOFR")
        pv_without_fixings = leg.pv(curve)

        # They should differ because fixings (5%, 4.5%) differ from forward (~4%)
        assert pv_with_fixings != pytest.approx(pv_without_fixings, rel=1e-4)

    def test_future_periods_ignore_fixings(self):
        """Future periods should still use forward curve even if fixings exist."""
        ref = date(2026, 4, 21)
        curve = _flat_curve(ref, rate=0.04)

        leg = FloatingLeg(
            start=ref, end=date(2027, 4, 21),
            frequency=Frequency.QUARTERLY,
            notional=1_000_000,
        )

        store = FixingsStore()
        # Store a fixing for a future date — should be ignored
        store.set("SOFR", date(2026, 7, 21), 0.10)

        pv_with_fixings = leg.pv(curve, fixings=store, rate_name="SOFR")
        pv_without_fixings = leg.pv(curve)

        # All periods are future → fixings should be ignored → same PV
        assert pv_with_fixings == pytest.approx(pv_without_fixings, rel=1e-12)

    def test_no_fixings_store_backward_compat(self):
        """Without fixings store, PV is unchanged (backward compat)."""
        ref = date(2026, 10, 21)
        curve = _flat_curve(ref, rate=0.04)
        leg = FloatingLeg(
            start=date(2026, 4, 21), end=date(2027, 4, 21),
            frequency=Frequency.QUARTERLY,
        )
        # No fixings → all periods use forward curve
        pv1 = leg.pv(curve)
        pv2 = leg.pv(curve, fixings=None, rate_name=None)
        assert pv1 == pytest.approx(pv2, rel=1e-14)

    def test_missing_fixing_falls_back_to_curve(self):
        """If a past period has no fixing stored, fall back to forward rate."""
        ref = date(2026, 10, 21)
        curve = _flat_curve(ref, rate=0.04)

        leg = FloatingLeg(
            start=date(2026, 4, 21), end=date(2027, 4, 21),
            frequency=Frequency.QUARTERLY,
        )

        # Empty store — no fixings at all
        store = FixingsStore()

        pv_with_empty = leg.pv(curve, fixings=store, rate_name="SOFR")
        pv_without = leg.pv(curve)

        # Empty store → all fall back to curve → same PV
        assert pv_with_empty == pytest.approx(pv_without, rel=1e-12)

    def test_fixing_value_used_correctly(self):
        """Verify the actual amount uses fixing rate + spread."""
        ref = date(2027, 4, 21)  # After entire leg
        curve = _flat_curve(ref, rate=0.04)

        leg = FloatingLeg(
            start=date(2026, 4, 21), end=date(2026, 10, 21),
            frequency=Frequency.QUARTERLY,
            notional=1_000_000,
            spread=0.005,
        )

        store = FixingsStore()
        fixing_rate = 0.045
        store.set("SOFR", leg.cashflows[0].accrual_start, fixing_rate)
        store.set("SOFR", leg.cashflows[1].accrual_start, fixing_rate)

        # All periods are past (ref = 2027), all use fixings
        pv = leg.pv(curve, fixings=store, rate_name="SOFR")

        # Manually compute expected PV
        expected = 0.0
        for cf in leg.cashflows:
            amount = cf.notional * (fixing_rate + cf.spread) * cf.year_frac
            expected += amount * curve.df(cf.payment_date)

        assert pv == pytest.approx(expected, rel=1e-10)
