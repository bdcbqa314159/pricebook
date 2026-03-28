"""Tests for fixed leg."""

import pytest
from datetime import date

from pricebook.fixed_leg import FixedLeg, Cashflow
from pricebook.schedule import Frequency
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


def _flat_curve(ref: date, rate: float = 0.05) -> DiscountCurve:
    """Build a flat discount curve at the given continuously compounded rate."""
    import math

    tenors_years = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    dates = [date(ref.year + int(t), ref.month, ref.day) if t >= 1
             else date.fromordinal(ref.toordinal() + int(t * 365))
             for t in tenors_years]
    dfs = [math.exp(-rate * t) for t in tenors_years]
    return DiscountCurve(ref, dates, dfs)


class TestCashflows:
    """Cashflow generation."""

    def test_quarterly_1y_generates_4_cashflows(self):
        leg = FixedLeg(
            date(2024, 1, 15), date(2025, 1, 15),
            rate=0.05, frequency=Frequency.QUARTERLY,
        )
        assert len(leg.cashflows) == 4

    def test_semi_annual_2y_generates_4_cashflows(self):
        leg = FixedLeg(
            date(2024, 1, 15), date(2026, 1, 15),
            rate=0.05, frequency=Frequency.SEMI_ANNUAL,
        )
        assert len(leg.cashflows) == 4

    def test_annual_3y_generates_3_cashflows(self):
        leg = FixedLeg(
            date(2024, 3, 1), date(2027, 3, 1),
            rate=0.04, frequency=Frequency.ANNUAL,
        )
        assert len(leg.cashflows) == 3

    def test_cashflow_amount(self):
        leg = FixedLeg(
            date(2024, 1, 15), date(2025, 1, 15),
            rate=0.05, frequency=Frequency.ANNUAL,
            notional=1_000_000.0,
            day_count=DayCountConvention.THIRTY_360,
        )
        cf = leg.cashflows[0]
        # 30/360: 1 year exactly = 1.0 year fraction
        assert cf.year_frac == pytest.approx(1.0)
        assert cf.amount == pytest.approx(50_000.0)

    def test_cashflow_dates_match_schedule(self):
        leg = FixedLeg(
            date(2024, 1, 15), date(2025, 1, 15),
            rate=0.05, frequency=Frequency.QUARTERLY,
        )
        assert leg.cashflows[0].accrual_start == date(2024, 1, 15)
        assert leg.cashflows[0].accrual_end == date(2024, 4, 15)
        assert leg.cashflows[-1].accrual_end == date(2025, 1, 15)

    def test_payment_date_equals_accrual_end(self):
        leg = FixedLeg(
            date(2024, 1, 15), date(2025, 1, 15),
            rate=0.05, frequency=Frequency.QUARTERLY,
        )
        for cf in leg.cashflows:
            assert cf.payment_date == cf.accrual_end

    def test_sum_of_year_fracs_approximately_one_year(self):
        leg = FixedLeg(
            date(2024, 1, 15), date(2025, 1, 15),
            rate=0.05, frequency=Frequency.QUARTERLY,
            day_count=DayCountConvention.THIRTY_360,
        )
        total_yf = sum(cf.year_frac for cf in leg.cashflows)
        assert total_yf == pytest.approx(1.0, abs=0.01)


class TestPresentValue:
    """Present value calculations."""

    def test_pv_at_zero_rates(self):
        """With flat rates at 0, PV = sum of undiscounted cashflows."""
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.0)
        leg = FixedLeg(
            ref, date(2025, 1, 15),
            rate=0.05, frequency=Frequency.QUARTERLY,
            notional=1_000_000.0,
            day_count=DayCountConvention.THIRTY_360,
        )
        pv = leg.pv(curve)
        total_cashflows = sum(cf.amount for cf in leg.cashflows)
        assert pv == pytest.approx(total_cashflows, rel=1e-4)

    def test_pv_positive_for_positive_rate(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        leg = FixedLeg(
            ref, date(2025, 1, 15),
            rate=0.05, frequency=Frequency.QUARTERLY,
        )
        assert leg.pv(curve) > 0

    def test_pv_increases_with_rate(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.03)
        leg_low = FixedLeg(ref, date(2025, 1, 15), rate=0.03, frequency=Frequency.QUARTERLY)
        leg_high = FixedLeg(ref, date(2025, 1, 15), rate=0.06, frequency=Frequency.QUARTERLY)
        assert leg_high.pv(curve) > leg_low.pv(curve)

    def test_pv_decreases_with_higher_discount_rate(self):
        ref = date(2024, 1, 15)
        leg = FixedLeg(ref, date(2025, 1, 15), rate=0.05, frequency=Frequency.QUARTERLY)
        pv_low = leg.pv(_flat_curve(ref, rate=0.02))
        pv_high = leg.pv(_flat_curve(ref, rate=0.10))
        assert pv_low > pv_high


class TestAnnuity:
    """Annuity factor."""

    def test_annuity_times_rate_times_notional_equals_pv(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        leg = FixedLeg(
            ref, date(2025, 1, 15),
            rate=0.04, frequency=Frequency.QUARTERLY,
            notional=1_000_000.0,
        )
        assert leg.pv(curve) == pytest.approx(
            leg.rate * leg.notional * leg.annuity(curve), rel=1e-10
        )

    def test_annuity_positive(self):
        ref = date(2024, 1, 15)
        curve = _flat_curve(ref, rate=0.05)
        leg = FixedLeg(ref, date(2027, 1, 15), rate=0.04, frequency=Frequency.SEMI_ANNUAL)
        assert leg.annuity(curve) > 0


class TestValidation:
    """Input validation."""

    def test_zero_notional_raises(self):
        with pytest.raises(ValueError):
            FixedLeg(date(2024, 1, 1), date(2025, 1, 1), rate=0.05,
                     frequency=Frequency.QUARTERLY, notional=0.0)

    def test_negative_notional_raises(self):
        with pytest.raises(ValueError):
            FixedLeg(date(2024, 1, 1), date(2025, 1, 1), rate=0.05,
                     frequency=Frequency.QUARTERLY, notional=-100.0)
