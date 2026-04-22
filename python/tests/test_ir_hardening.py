"""Tests for IR hardening fixes."""

from datetime import date, timedelta

import pytest

from pricebook.amortising_swap import AmortisingSwap
from pricebook.day_count import DayCountConvention
from pricebook.schedule import Frequency
from pricebook.swap import InterestRateSwap, SwapDirection
from pricebook.zc_swap import ZeroCouponSwap
from tests.conftest import make_flat_curve


# ---- #3: Swap passes payment_delay to FixedLeg ----

class TestSwapFixedLegDelay:
    def test_fixed_leg_gets_delay(self):
        """FixedLeg should receive payment_delay_days from swap."""
        swap = InterestRateSwap(
            date(2026, 4, 21), date(2031, 4, 21), fixed_rate=0.04,
            payment_delay_days=2,
        )
        for cf in swap.fixed_leg.cashflows:
            assert cf.payment_date == cf.accrual_end + timedelta(days=2)

    def test_both_legs_delayed(self):
        """Both fixed and floating legs should have the same delay."""
        swap = InterestRateSwap(
            date(2026, 4, 21), date(2031, 4, 21), fixed_rate=0.04,
            payment_delay_days=2,
        )
        for fcf, flcf in zip(swap.fixed_leg.cashflows, swap.floating_leg.cashflows):
            assert fcf.payment_date != fcf.accrual_end
            assert flcf.payment_date != flcf.accrual_end

    def test_no_delay_backward_compat(self):
        swap = InterestRateSwap(
            date(2026, 4, 21), date(2031, 4, 21), fixed_rate=0.04,
        )
        for cf in swap.fixed_leg.cashflows:
            assert cf.payment_date == cf.accrual_end


# ---- #5: AmortisingSwap typo fixed ----

class TestAmortisingSwapName:
    def test_class_importable(self):
        """AmortisingSwap (single 's') should be importable."""
        assert AmortisingSwap is not None

    def test_amortising_factory(self):
        swap = AmortisingSwap.amortising(
            date(2026, 4, 21), date(2031, 4, 21), 0.04, 1_000_000,
        )
        assert swap.notionals[0] > swap.notionals[-1]


# ---- #8: ZC swap day_count parameter ----

class TestZCSwapDayCount:
    def test_default_act365(self):
        zc = ZeroCouponSwap(date(2026, 4, 21), date(2031, 4, 21), 0.04)
        assert zc.day_count == DayCountConvention.ACT_365_FIXED

    def test_act360(self):
        """EUR ZC swap uses ACT/360."""
        zc = ZeroCouponSwap(
            date(2026, 4, 21), date(2031, 4, 21), 0.04,
            day_count=DayCountConvention.ACT_360,
        )
        assert zc.day_count == DayCountConvention.ACT_360
        # Tenor should differ from ACT/365
        zc_365 = ZeroCouponSwap(date(2026, 4, 21), date(2031, 4, 21), 0.04)
        assert zc.fixed_amount() != zc_365.fixed_amount()

    def test_par_rate_with_day_count(self):
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        zc = ZeroCouponSwap(ref, date(2031, 4, 21), 0.04,
                            day_count=DayCountConvention.ACT_360)
        pr = zc.par_rate(curve)
        assert pr > 0


# ---- #9: Swap DV01 ----

class TestSwapDV01:
    def test_payer_dv01_positive(self):
        """Payer swap (receive float) gains when rates rise → DV01 > 0."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        swap = InterestRateSwap(ref, date(2031, 4, 21), fixed_rate=0.04,
                                direction=SwapDirection.PAYER)
        dv01 = swap.dv01(curve)
        assert dv01 > 0

    def test_receiver_dv01_negative(self):
        """Receiver swap (pay float) loses when rates rise → DV01 < 0."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        swap = InterestRateSwap(ref, date(2031, 4, 21), fixed_rate=0.04,
                                direction=SwapDirection.RECEIVER)
        dv01 = swap.dv01(curve)
        assert dv01 < 0

    def test_dv01_scales_with_notional(self):
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        swap1 = InterestRateSwap(ref, date(2031, 4, 21), 0.04, notional=1_000_000)
        swap2 = InterestRateSwap(ref, date(2031, 4, 21), 0.04, notional=2_000_000)
        assert swap2.dv01(curve) == pytest.approx(2 * swap1.dv01(curve), rel=1e-6)

    def test_longer_swap_larger_dv01(self):
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        swap_5y = InterestRateSwap(ref, date(2031, 4, 21), 0.04)
        swap_10y = InterestRateSwap(ref, date(2036, 4, 21), 0.04)
        assert abs(swap_10y.dv01(curve)) > abs(swap_5y.dv01(curve))


# ---- #10: Swap annuity ----

class TestSwapAnnuity:
    def test_annuity_positive(self):
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        swap = InterestRateSwap(ref, date(2031, 4, 21), 0.04)
        assert swap.annuity(curve) > 0

    def test_annuity_matches_fixed_leg(self):
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        swap = InterestRateSwap(ref, date(2031, 4, 21), 0.04)
        assert swap.annuity(curve) == pytest.approx(swap.fixed_leg.annuity(curve))

    def test_par_rate_uses_annuity(self):
        """par_rate = PV_float / (notional × annuity)."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        swap = InterestRateSwap(ref, date(2031, 4, 21), 0.04)
        pv_float = swap.floating_leg.pv(curve)
        expected_par = pv_float / (swap.notional * swap.annuity(curve))
        assert swap.par_rate(curve) == pytest.approx(expected_par, rel=1e-10)
