"""Tests for floating leg fixing management (FX1-FX6)."""

from datetime import date, timedelta

import pytest

from pricebook.calendar import USSettlementCalendar
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.fixings import FixingsStore
from pricebook.floating_leg import FloatingCashflow, FloatingLeg
from pricebook.rate_index import (
    CompoundingMethod, RateIndex, get_rate_index, all_rate_indices,
    overnight_indices, indices_for_currency,
)
from pricebook.schedule import Frequency, StubType, generate_schedule


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


# ---- FX3: Observation Shift ----

class TestObservationShift:
    def test_default_zero_shift(self):
        """Default: fixing_date == accrual_start (no shift)."""
        leg = FloatingLeg(
            date(2026, 4, 21), date(2027, 4, 21), Frequency.QUARTERLY,
        )
        for cf in leg.cashflows:
            assert cf.fixing_date == cf.accrual_start

    def test_two_day_shift(self):
        """SOFR-style T-2: fixing_date = accrual_start - 2 days."""
        leg = FloatingLeg(
            date(2026, 4, 21), date(2027, 4, 21), Frequency.QUARTERLY,
            observation_shift_days=2,
        )
        for cf in leg.cashflows:
            assert cf.fixing_date == cf.accrual_start - timedelta(days=2)

    def test_shift_affects_fixing_lookup(self):
        """With observation shift, fixings must be stored at fixing_date, not accrual_start."""
        ref = date(2027, 4, 21)  # After entire leg
        curve = _flat_curve(ref, rate=0.04)

        leg = FloatingLeg(
            start=date(2026, 4, 21), end=date(2026, 10, 21),
            frequency=Frequency.QUARTERLY,
            notional=1_000_000,
            observation_shift_days=2,
        )

        store = FixingsStore()
        fixing_rate = 0.05

        # Store at fixing_date (shifted), NOT accrual_start
        for cf in leg.cashflows:
            store.set("SOFR", cf.fixing_date, fixing_rate)

        pv = leg.pv(curve, fixings=store, rate_name="SOFR")

        # Should use the fixings
        expected = 0.0
        for cf in leg.cashflows:
            amount = cf.notional * (fixing_rate + cf.spread) * cf.year_frac
            expected += amount * curve.df(cf.payment_date)
        assert pv == pytest.approx(expected, rel=1e-10)

    def test_fixing_at_accrual_start_not_found_with_shift(self):
        """With shift, a fixing stored at accrual_start won't be found (it looks at fixing_date)."""
        ref = date(2027, 4, 21)
        curve = _flat_curve(ref, rate=0.04)

        leg = FloatingLeg(
            start=date(2026, 4, 21), end=date(2026, 10, 21),
            frequency=Frequency.QUARTERLY,
            notional=1_000_000,
            observation_shift_days=2,
        )

        store = FixingsStore()
        # Store at accrual_start — wrong date for shifted leg
        for cf in leg.cashflows:
            store.set("SOFR", cf.accrual_start, 0.10)

        pv_with = leg.pv(curve, fixings=store, rate_name="SOFR")
        pv_without = leg.pv(curve)

        # Fixings at accrual_start won't match fixing_date → falls back to curve → same PV
        assert pv_with == pytest.approx(pv_without, rel=1e-12)

    def test_negative_shift_raises(self):
        with pytest.raises(ValueError, match="observation_shift_days"):
            FloatingLeg(
                date(2026, 4, 21), date(2027, 4, 21), Frequency.QUARTERLY,
                observation_shift_days=-1,
            )


# ---- FX4: Back Stubs ----

class TestBackStubs:
    def test_short_back_stub(self):
        """SHORT_BACK: regular periods from start, short final period."""
        # 13 months → 4 quarterly + 1 short month
        sched = generate_schedule(
            date(2026, 1, 15), date(2027, 2, 15), Frequency.QUARTERLY,
            stub=StubType.SHORT_BACK,
        )
        assert sched[0] == date(2026, 1, 15)
        assert sched[-1] == date(2027, 2, 15)
        # Regular periods: Jan→Apr→Jul→Oct→Jan, then short Jan→Feb
        assert sched[1] == date(2026, 4, 15)
        assert sched[2] == date(2026, 7, 15)
        assert sched[3] == date(2026, 10, 15)
        assert sched[4] == date(2027, 1, 15)
        assert len(sched) == 6  # 5 periods

    def test_long_back_stub(self):
        """LONG_BACK: short final stub merged into previous period."""
        # 13 months quarterly → 4 regular + 1-month stub → merge → 3 regular + 1 long
        sched = generate_schedule(
            date(2026, 1, 15), date(2027, 2, 15), Frequency.QUARTERLY,
            stub=StubType.LONG_BACK,
        )
        assert sched[0] == date(2026, 1, 15)
        assert sched[-1] == date(2027, 2, 15)
        # The short 1-month stub should be merged with previous period
        # Regular: Jan→Apr→Jul→Oct, then long Oct→Feb (4 months)
        assert len(sched) == 5  # 4 periods instead of 5

    def test_exact_division_no_stub(self):
        """Exact division: no stub regardless of stub type."""
        for stub in [StubType.SHORT_BACK, StubType.LONG_BACK,
                     StubType.SHORT_FRONT, StubType.LONG_FRONT]:
            sched = generate_schedule(
                date(2026, 1, 15), date(2027, 1, 15), Frequency.QUARTERLY,
                stub=stub,
            )
            assert len(sched) == 5  # 4 periods, exactly quarterly
            assert sched[0] == date(2026, 1, 15)
            assert sched[-1] == date(2027, 1, 15)

    def test_short_front_vs_short_back_different(self):
        """Front and back stubs produce different intermediate dates for non-exact periods."""
        start = date(2026, 1, 10)
        end = date(2027, 2, 15)  # Not exactly N quarters from start
        front = generate_schedule(start, end, Frequency.QUARTERLY, stub=StubType.SHORT_FRONT)
        back = generate_schedule(start, end, Frequency.QUARTERLY, stub=StubType.SHORT_BACK)
        # Both share start and end
        assert front[0] == back[0] == start
        assert front[-1] == back[-1] == end
        # But intermediate dates differ (front rolls from end, back rolls from start)
        if len(front) > 2 and len(back) > 2:
            assert front[1] != back[1]

    def test_floating_leg_with_back_stub(self):
        """FloatingLeg works with back stub type."""
        ref = date(2026, 1, 15)
        curve = _flat_curve(ref, rate=0.04)
        leg = FloatingLeg(
            start=ref, end=date(2027, 2, 15), frequency=Frequency.QUARTERLY,
            stub=StubType.SHORT_BACK,
        )
        pv = leg.pv(curve)
        assert pv != 0.0  # Sanity: non-zero PV


# ---- FX5: RateIndex Registry ----

class TestRateIndexRegistry:
    def test_sofr(self):
        sofr = get_rate_index("SOFR")
        assert sofr.currency == "USD"
        assert sofr.is_overnight
        assert sofr.fixing_lag == 0
        assert sofr.observation_shift == 2
        assert sofr.day_count == DayCountConvention.ACT_360
        assert sofr.compounding == CompoundingMethod.COMPOUNDED
        assert sofr.administrator == "FRBNY"

    def test_estr(self):
        estr = get_rate_index("ESTR")
        assert estr.currency == "EUR"
        assert estr.is_overnight
        assert estr.observation_shift == 2

    def test_sonia_no_shift(self):
        """SONIA has no observation shift (UK convention)."""
        sonia = get_rate_index("SONIA")
        assert sonia.currency == "GBP"
        assert sonia.observation_shift == 0
        assert sonia.day_count == DayCountConvention.ACT_365_FIXED

    def test_tona(self):
        tona = get_rate_index("TONA")
        assert tona.currency == "JPY"
        assert tona.day_count == DayCountConvention.ACT_365_FIXED

    def test_euribor_3m_is_term(self):
        """EURIBOR is a term rate, not overnight."""
        euribor = get_rate_index("EURIBOR_3M")
        assert not euribor.is_overnight
        assert euribor.tenor_months == 3
        assert euribor.fixing_lag == 2
        assert euribor.compounding == CompoundingMethod.FLAT

    def test_euribor_6m(self):
        euribor = get_rate_index("EURIBOR_6M")
        assert euribor.tenor_months == 6

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown rate index"):
            get_rate_index("NONEXISTENT")

    def test_all_indices_count(self):
        """We have 11 registered indices."""
        indices = all_rate_indices()
        assert len(indices) == 11

    def test_overnight_indices(self):
        """8 overnight RFR indices (one per G10 currency except SEK/NOK)."""
        overnight = overnight_indices()
        assert len(overnight) == 8
        assert all(idx.is_overnight for idx in overnight)

    def test_indices_for_currency(self):
        eur = indices_for_currency("EUR")
        names = {idx.name for idx in eur}
        assert "ESTR" in names
        assert "EURIBOR_3M" in names
        assert "EURIBOR_6M" in names

    def test_frozen_immutable(self):
        """RateIndex is frozen — cannot be modified."""
        sofr = get_rate_index("SOFR")
        with pytest.raises(AttributeError):
            sofr.fixing_lag = 5

    def test_all_g10_rfr_covered(self):
        """Major G10 overnight RFRs are all present."""
        expected = {"SOFR", "ESTR", "SONIA", "TONA", "SARON", "CORRA", "AONIA", "NZOCR"}
        actual = {idx.name for idx in overnight_indices()}
        assert expected == actual


# ---- FX6: Calendar-Aware Lag ----

class TestCalendarAwareLag:
    def test_add_business_days_forward(self):
        """add_business_days(+2) skips weekends."""
        cal = USSettlementCalendar()
        # Friday Apr 17 2026 + 2 business days = Tuesday Apr 21
        result = cal.add_business_days(date(2026, 4, 17), 2)
        assert result == date(2026, 4, 21)

    def test_add_business_days_backward(self):
        """add_business_days(-2) goes backward skipping weekends."""
        cal = USSettlementCalendar()
        # Tuesday Apr 21 2026 - 2 business days = Friday Apr 17
        result = cal.add_business_days(date(2026, 4, 21), -2)
        assert result == date(2026, 4, 17)

    def test_add_business_days_skips_holiday(self):
        """Business day calculation skips holidays."""
        cal = USSettlementCalendar()
        # July 3 2026 (Fri), July 4 is Sat → observed Fri July 3
        # Actually let's use a clearer example: MLK Day 2026 is Jan 19 (Mon)
        # Jan 16 (Fri) + 1 business day should skip MLK → Jan 20 (Tue)
        result = cal.add_business_days(date(2026, 1, 16), 1)
        assert result == date(2026, 1, 20)  # Skips weekend + MLK

    def test_get_with_lag_calendar(self):
        """get_with_lag uses calendar-aware business days."""
        cal = USSettlementCalendar()
        store = FixingsStore()
        # Store fixing on Friday Apr 17
        store.set("SOFR", date(2026, 4, 17), 0.043)
        # Look up with lag=2 from Tuesday Apr 21 → Friday Apr 17
        result = store.get_with_lag("SOFR", date(2026, 4, 21), lag=2, calendar=cal)
        assert result == 0.043

    def test_get_with_lag_no_calendar(self):
        """Without calendar, lag uses calendar days."""
        store = FixingsStore()
        store.set("SOFR", date(2026, 4, 19), 0.043)
        # Lag 2 calendar days from Apr 21 → Apr 19
        result = store.get_with_lag("SOFR", date(2026, 4, 21), lag=2, calendar=None)
        assert result == 0.043

    def test_get_with_lag_missing(self):
        """Returns None if no fixing at lagged date."""
        cal = USSettlementCalendar()
        store = FixingsStore()
        result = store.get_with_lag("SOFR", date(2026, 4, 21), lag=2, calendar=cal)
        assert result is None

    def test_zero_lag(self):
        """Zero lag returns fixing at the reference date itself."""
        cal = USSettlementCalendar()
        store = FixingsStore()
        store.set("SOFR", date(2026, 4, 21), 0.043)
        result = store.get_with_lag("SOFR", date(2026, 4, 21), lag=0, calendar=cal)
        assert result == 0.043
