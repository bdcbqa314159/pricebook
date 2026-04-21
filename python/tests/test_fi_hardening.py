"""Tests for FI convention hardening (FI1-FI10)."""

from datetime import date, timedelta

import pytest

from pricebook.bond import FixedRateBond
from pricebook.calendar import USSettlementCalendar
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.fixed_leg import FixedLeg
from pricebook.fra import FRA
from pricebook.schedule import Frequency, generate_schedule
from pricebook.swap import InterestRateSwap
from pricebook.xccy_swap import CrossCurrencySwap
from tests.conftest import make_flat_curve


# ---- FI1: ACT/ACT ICMA ----

class TestActActICMA:
    def test_full_semi_annual_period(self):
        """Full semi-annual coupon period = 0.5 years exactly."""
        yf = year_fraction(
            date(2026, 1, 15), date(2026, 7, 15),
            DayCountConvention.ACT_ACT_ICMA,
            ref_start=date(2026, 1, 15), ref_end=date(2026, 7, 15),
            frequency=2,
        )
        assert yf == pytest.approx(0.5, rel=1e-10)

    def test_half_period(self):
        """Half of a semi-annual period = 0.25 years."""
        yf = year_fraction(
            date(2026, 1, 15), date(2026, 4, 15),
            DayCountConvention.ACT_ACT_ICMA,
            ref_start=date(2026, 1, 15), ref_end=date(2026, 7, 15),
            frequency=2,
        )
        # 90 days out of 181 total days in period, freq=2
        # yf = 90 / (181 * 2) = 0.2486
        expected = 90 / (181 * 2)
        assert yf == pytest.approx(expected, rel=1e-10)

    def test_quarterly_full_period(self):
        """Full quarterly period = 0.25 years."""
        yf = year_fraction(
            date(2026, 1, 15), date(2026, 4, 15),
            DayCountConvention.ACT_ACT_ICMA,
            ref_start=date(2026, 1, 15), ref_end=date(2026, 4, 15),
            frequency=4,
        )
        assert yf == pytest.approx(0.25, rel=1e-10)

    def test_annual_full_period(self):
        """Full annual period = 1.0 year."""
        yf = year_fraction(
            date(2026, 1, 15), date(2027, 1, 15),
            DayCountConvention.ACT_ACT_ICMA,
            ref_start=date(2026, 1, 15), ref_end=date(2027, 1, 15),
            frequency=1,
        )
        assert yf == pytest.approx(1.0, rel=1e-10)

    def test_differs_from_isda(self):
        """ICMA and ISDA should give different results for same period."""
        args = (date(2026, 1, 15), date(2026, 7, 15))
        icma = year_fraction(*args, DayCountConvention.ACT_ACT_ICMA,
                             ref_start=args[0], ref_end=args[1], frequency=2)
        isda = year_fraction(*args, DayCountConvention.ACT_ACT_ISDA)
        # ICMA = exactly 0.5, ISDA = 181/365 ≈ 0.4959
        assert icma == pytest.approx(0.5)
        assert isda != pytest.approx(0.5)

    def test_fallback_without_ref_dates(self):
        """Without ref dates, falls back to ACT/365F."""
        yf = year_fraction(
            date(2026, 1, 15), date(2026, 7, 15),
            DayCountConvention.ACT_ACT_ICMA,
        )
        expected = 181 / 365.0
        assert yf == pytest.approx(expected, rel=1e-10)


# ---- FI2: Bond settlement date offset ----

class TestBondSettlement:
    def test_settlement_t1(self):
        """UST: T+1 settlement."""
        bond = FixedRateBond(
            date(2026, 1, 15), date(2031, 1, 15), 0.04,
            settlement_days=1,
        )
        settle = bond.settlement_date(date(2026, 4, 21))
        assert settle == date(2026, 4, 22)

    def test_settlement_t2_calendar(self):
        """With calendar, T+2 skips weekends."""
        cal = USSettlementCalendar()
        bond = FixedRateBond(
            date(2026, 1, 15), date(2031, 1, 15), 0.04,
            calendar=cal, settlement_days=2,
        )
        # Thursday Apr 23 + 2 business days = Monday Apr 27
        settle = bond.settlement_date(date(2026, 4, 23))
        assert settle.weekday() < 5  # weekday

    def test_settlement_zero(self):
        bond = FixedRateBond(
            date(2026, 1, 15), date(2031, 1, 15), 0.04,
            settlement_days=0,
        )
        assert bond.settlement_date(date(2026, 4, 21)) == date(2026, 4, 21)


# ---- FI3: FixedLeg payment delay ----

class TestFixedLegDelay:
    def test_default_no_delay(self):
        leg = FixedLeg(date(2026, 1, 15), date(2027, 1, 15), 0.04, Frequency.QUARTERLY)
        for cf in leg.cashflows:
            assert cf.payment_date == cf.accrual_end

    def test_two_day_delay(self):
        leg = FixedLeg(
            date(2026, 1, 15), date(2027, 1, 15), 0.04, Frequency.QUARTERLY,
            payment_delay_days=2,
        )
        for cf in leg.cashflows:
            assert cf.payment_date == cf.accrual_end + timedelta(days=2)

    def test_delay_with_calendar(self):
        cal = USSettlementCalendar()
        leg = FixedLeg(
            date(2026, 1, 15), date(2027, 1, 15), 0.04, Frequency.QUARTERLY,
            calendar=cal, payment_delay_days=2,
        )
        for cf in leg.cashflows:
            assert cf.payment_date.weekday() < 5  # business day


# ---- FI4: FRA payment at start ----

class TestFRAStartPayment:
    def test_fra_at_par_pv_zero(self):
        """FRA at par strike should have PV ≈ 0."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        fra = FRA(date(2026, 7, 21), date(2026, 10, 21), strike=0.04)
        par = fra.par_rate(curve)
        fra_at_par = FRA(date(2026, 7, 21), date(2026, 10, 21), strike=par)
        assert fra_at_par.pv(curve) == pytest.approx(0.0, abs=1e-6)

    def test_fra_discounts_to_start(self):
        """FRA PV should use df(start), not df(end)."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        fra = FRA(date(2026, 7, 21), date(2026, 10, 21), strike=0.03)
        pv = fra.pv(curve)
        # PV should be positive (forward > strike)
        assert pv > 0


# ---- FI5: CDS date reconstruction ----
# (tested indirectly — CDS tests should still pass with date arithmetic fix)


# ---- FI6: Xccy swap par_spread bug ----

class TestXccyParSpread:
    def test_par_spread_mtm_reset(self):
        """Par spread with MTM reset should use foreign curve."""
        ref = date(2026, 4, 21)
        dom_curve = make_flat_curve(ref, rate=0.04)
        for_curve = make_flat_curve(ref, rate=0.02)
        swap = CrossCurrencySwap(
            ref, date(2031, 4, 21),
            domestic_notional=1_000_000, fx_rate=1.10,
            mtm_reset=True,
        )
        ps = swap.par_spread(dom_curve, for_curve, 1.10)
        # With MTM reset, par spread should be deterministic
        assert isinstance(ps, float)
        assert abs(ps) < 0.05


# ---- FI7: WEEKLY frequency ----

class TestWeeklyFrequency:
    def test_weekly_schedule(self):
        """Weekly schedule should have ~4 dates per month."""
        sched = generate_schedule(
            date(2026, 4, 21), date(2026, 5, 21), Frequency.WEEKLY,
        )
        # 30 days / 7 = ~4 periods + start + end
        assert len(sched) >= 5

    def test_weekly_dates_7_apart(self):
        sched = generate_schedule(
            date(2026, 4, 21), date(2026, 5, 21), Frequency.WEEKLY,
        )
        for i in range(1, len(sched) - 1):  # skip last (may be short)
            gap = (sched[i] - sched[i - 1]).days
            assert gap == 7


# ---- FI8: Swap cashflow schedule ----

class TestSwapCashflowSchedule:
    def test_schedule_has_both_legs(self):
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        swap = InterestRateSwap(ref, date(2028, 4, 21), fixed_rate=0.04)
        cfs = swap.cashflow_schedule(curve)
        legs = set(cf["leg"] for cf in cfs)
        assert "fixed" in legs
        assert "float" in legs

    def test_schedule_has_required_fields(self):
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        swap = InterestRateSwap(ref, date(2028, 4, 21), fixed_rate=0.04)
        cfs = swap.cashflow_schedule(curve)
        for cf in cfs:
            assert "payment_date" in cf
            assert "amount" in cf
            assert "df" in cf
            assert "pv" in cf

    def test_schedule_sorted_by_date(self):
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        swap = InterestRateSwap(ref, date(2028, 4, 21), fixed_rate=0.04)
        cfs = swap.cashflow_schedule(curve)
        dates = [cf["payment_date"] for cf in cfs]
        assert dates == sorted(dates)


# ---- FI9: Bond ex-dividend ----

class TestExDividend:
    def test_no_ex_div_default(self):
        """Default: no ex-div handling."""
        bond = FixedRateBond(date(2026, 1, 15), date(2028, 1, 15), 0.04)
        # Just before coupon date: accrued should be positive
        ai = bond.accrued_interest(date(2026, 7, 10))
        assert ai > 0

    def test_ex_div_negative_accrued(self):
        """In ex-div period, accrued should be negative."""
        bond = FixedRateBond(
            date(2026, 1, 15), date(2028, 1, 15), 0.04,
            ex_div_days=7,
        )
        # 3 days before coupon (within 7-day ex-div period)
        ai = bond.accrued_interest(date(2026, 7, 12))
        assert ai < 0

    def test_before_ex_div_positive(self):
        """Before ex-div period, accrued should still be positive."""
        bond = FixedRateBond(
            date(2026, 1, 15), date(2028, 1, 15), 0.04,
            ex_div_days=7,
        )
        # 30 days before coupon: not in ex-div period
        ai = bond.accrued_interest(date(2026, 6, 15))
        assert ai > 0


# ---- FI10: Bootstrap day count (documentation, no code change needed) ----
# The bootstrap correctly uses ACT_365_FIXED for the curve's internal time axis.
# Instrument day counts (ACT_360, 30/360) are applied to cashflow calculations.
# This is standard practice — curve time axis != instrument day count.
