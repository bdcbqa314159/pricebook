"""Tests for fixed-rate bond."""

import pytest
from datetime import date

from pricebook.bond import FixedRateBond
from pricebook.schedule import Frequency
from pricebook.day_count import DayCountConvention
from tests.conftest import make_flat_curve


class TestDirtyPrice:

    def test_par_bond_prices_near_100(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        dp = bond.dirty_price(curve)
        assert 95.0 < dp < 105.0

    def test_zero_coupon_dirty_price(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.0)
        dp = bond.dirty_price(curve)
        expected = curve.df(date(2029, 1, 15)) * 100.0
        assert dp == pytest.approx(expected, rel=1e-6)

    def test_higher_coupon_higher_price(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        bond_low = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.03)
        bond_high = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.07)
        assert bond_high.dirty_price(curve) > bond_low.dirty_price(curve)

    def test_higher_yield_lower_price(self):
        ref = date(2024, 1, 15)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        dp_low = bond.dirty_price(make_flat_curve(ref, rate=0.03))
        dp_high = bond.dirty_price(make_flat_curve(ref, rate=0.08))
        assert dp_low > dp_high


class TestAccruedInterest:

    def test_accrued_at_coupon_date_is_zero(self):
        bond = FixedRateBond(date(2024, 1, 15), date(2026, 1, 15), coupon_rate=0.06)
        ai = bond.accrued_interest(date(2024, 1, 15))
        assert ai == pytest.approx(0.0)

    def test_accrued_mid_period(self):
        bond = FixedRateBond(
            date(2024, 1, 15), date(2025, 1, 15),
            coupon_rate=0.06, frequency=Frequency.SEMI_ANNUAL,
            day_count=DayCountConvention.THIRTY_360,
        )
        ai = bond.accrued_interest(date(2024, 4, 15))
        assert ai == pytest.approx(1.5, rel=1e-2)

    def test_accrued_after_maturity_is_zero(self):
        bond = FixedRateBond(date(2024, 1, 15), date(2025, 1, 15), coupon_rate=0.06)
        ai = bond.accrued_interest(date(2025, 6, 15))
        assert ai == pytest.approx(0.0)


class TestCleanPrice:

    def test_clean_equals_dirty_at_coupon_date(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        assert bond.clean_price(curve) == pytest.approx(bond.dirty_price(curve), rel=1e-6)

    def test_clean_less_than_dirty_mid_period(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        settlement = date(2024, 4, 15)
        clean = bond.clean_price(curve, settlement)
        dirty = bond.dirty_price(curve)
        assert clean < dirty

    def test_clean_price_default_settlement(self):
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        assert bond.clean_price(curve) == bond.clean_price(curve, ref)


class TestYieldToMaturity:

    def test_ytm_round_trip(self):
        """Compute price from curve, then recover YTM, then price from YTM."""
        ref = date(2024, 1, 15)
        curve = make_flat_curve(ref, rate=0.05)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.04)
        dp = bond.dirty_price(curve)
        ytm = bond.yield_to_maturity(dp)
        dp_recovered = bond._price_from_ytm(ytm)
        assert dp_recovered == pytest.approx(dp, abs=0.01)

    def test_par_bond_ytm_equals_coupon(self):
        """A bond priced at exactly 100 has YTM = coupon rate."""
        ref = date(2024, 1, 15)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.06)
        ytm = bond.yield_to_maturity(100.0)
        assert ytm == pytest.approx(0.06, abs=1e-4)

    def test_discount_bond_ytm_above_coupon(self):
        """Bond below par has YTM > coupon rate."""
        ref = date(2024, 1, 15)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.04)
        ytm = bond.yield_to_maturity(95.0)
        assert ytm > 0.04

    def test_premium_bond_ytm_below_coupon(self):
        """Bond above par has YTM < coupon rate."""
        ref = date(2024, 1, 15)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.06)
        ytm = bond.yield_to_maturity(105.0)
        assert ytm < 0.06


class TestDuration:

    def test_macaulay_duration_positive(self):
        ref = date(2024, 1, 15)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        mac = bond.macaulay_duration(0.05)
        assert mac > 0

    def test_macaulay_duration_less_than_maturity(self):
        """Macaulay duration of a coupon bond < maturity."""
        ref = date(2024, 1, 15)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        mat_years = 5.0
        mac = bond.macaulay_duration(0.05)
        assert mac < mat_years

    def test_zero_coupon_duration_equals_maturity(self):
        """Zero-coupon bond: Macaulay duration = maturity."""
        ref = date(2024, 1, 15)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.0)
        mac = bond.macaulay_duration(0.05)
        assert mac == pytest.approx(5.0, abs=0.05)

    def test_modified_less_than_macaulay(self):
        ref = date(2024, 1, 15)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        ytm = 0.05
        assert bond.modified_duration(ytm) < bond.macaulay_duration(ytm)

    def test_longer_bond_higher_duration(self):
        ref = date(2024, 1, 15)
        bond_2y = FixedRateBond(ref, date(2026, 1, 15), coupon_rate=0.05)
        bond_10y = FixedRateBond(ref, date(2034, 1, 15), coupon_rate=0.05)
        assert bond_10y.macaulay_duration(0.05) > bond_2y.macaulay_duration(0.05)

    def test_duration_approximates_bump(self):
        """Modified duration * dp ≈ actual price change for small yield shifts."""
        ref = date(2024, 1, 15)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        ytm = 0.05
        price = bond._price_from_ytm(ytm)
        mod_dur = bond.modified_duration(ytm)

        bp = 0.0001  # 1 basis point
        price_up = bond._price_from_ytm(ytm + bp)
        actual_change = price_up - price
        approx_change = -mod_dur * price * bp

        assert actual_change == pytest.approx(approx_change, rel=0.01)


class TestConvexity:

    def test_convexity_positive(self):
        ref = date(2024, 1, 15)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        assert bond.convexity(0.05) > 0

    def test_convexity_improves_duration_approx(self):
        """Duration + convexity correction should be closer than duration alone."""
        ref = date(2024, 1, 15)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        ytm = 0.05
        price = bond._price_from_ytm(ytm)
        mod_dur = bond.modified_duration(ytm)
        conv = bond.convexity(ytm)

        dy = 0.01  # 100bp shift
        price_up = bond._price_from_ytm(ytm + dy)
        actual_change = price_up - price

        dur_approx = -mod_dur * price * dy
        dur_conv_approx = dur_approx + 0.5 * conv * price * dy ** 2

        err_dur = abs(actual_change - dur_approx)
        err_dur_conv = abs(actual_change - dur_conv_approx)
        assert err_dur_conv < err_dur


class TestDV01:

    def test_dv01_positive(self):
        ref = date(2024, 1, 15)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        assert bond.dv01_yield(0.05) > 0

    def test_dv01_matches_bump(self):
        """DV01 ≈ |price(y) - price(y + 1bp)|."""
        ref = date(2024, 1, 15)
        bond = FixedRateBond(ref, date(2029, 1, 15), coupon_rate=0.05)
        ytm = 0.05
        dv01 = bond.dv01_yield(ytm)
        bump_change = abs(bond._price_from_ytm(ytm + 0.0001) - bond._price_from_ytm(ytm))
        assert dv01 == pytest.approx(bump_change, rel=0.01)


class TestValidation:

    def test_negative_face_value_raises(self):
        with pytest.raises(ValueError):
            FixedRateBond(date(2024, 1, 1), date(2025, 1, 1), coupon_rate=0.05, face_value=-100.0)

    def test_zero_face_value_raises(self):
        with pytest.raises(ValueError):
            FixedRateBond(date(2024, 1, 1), date(2025, 1, 1), coupon_rate=0.05, face_value=0.0)

    def test_issue_after_maturity_raises(self):
        with pytest.raises(ValueError):
            FixedRateBond(date(2026, 1, 1), date(2025, 1, 1), coupon_rate=0.05)
