"""Tests for bond hardening fixes (BH1-BH7)."""

import math
from datetime import date

import pytest

from pricebook.bond import FixedRateBond
from pricebook.schedule import Frequency
from pricebook.day_count import DayCountConvention
from tests.conftest import make_flat_curve


# ---- BH1: YTM/duration from settlement ----

class TestSettlementBasedAnalytics:
    def test_ytm_seasoned_bond(self):
        """Seasoned bond YTM should be reasonable, not distorted by issue_date."""
        bond = FixedRateBond(
            date(2020, 1, 15), date(2030, 1, 15), coupon_rate=0.04,
            frequency=Frequency.SEMI_ANNUAL,
        )
        settlement = date(2026, 1, 15)  # 6 years after issue
        # At par, YTM should equal coupon
        ytm = bond.yield_to_maturity(100.0, settlement=settlement)
        assert ytm == pytest.approx(0.04, abs=0.002)

    def test_ytm_at_issue_vs_settlement(self):
        """YTM at issue should differ from YTM at later settlement for same price."""
        bond = FixedRateBond(
            date(2020, 1, 15), date(2030, 1, 15), coupon_rate=0.05,
        )
        # At 95 (discount), YTM from issue vs from 5Y later
        ytm_issue = bond.yield_to_maturity(95.0, settlement=date(2020, 1, 15))
        ytm_5y = bond.yield_to_maturity(95.0, settlement=date(2025, 1, 15))
        # Shorter remaining maturity at same discount → higher YTM
        assert ytm_5y > ytm_issue

    def test_duration_seasoned_less_than_original(self):
        """Seasoned bond has shorter duration than at issue."""
        bond = FixedRateBond(
            date(2020, 1, 15), date(2030, 1, 15), coupon_rate=0.04,
        )
        dur_issue = bond.macaulay_duration(0.04, settlement=date(2020, 1, 15))
        dur_5y = bond.macaulay_duration(0.04, settlement=date(2025, 1, 15))
        assert dur_5y < dur_issue

    def test_duration_near_maturity_near_zero(self):
        """Duration very near maturity should be very small."""
        bond = FixedRateBond(
            date(2020, 1, 15), date(2026, 7, 15), coupon_rate=0.04,
        )
        dur = bond.macaulay_duration(0.04, settlement=date(2026, 1, 15))
        assert dur < 0.6  # Less than 6 months to maturity

    def test_convexity_seasoned(self):
        """Convexity should be lower for seasoned bond (shorter remaining life)."""
        bond = FixedRateBond(
            date(2020, 1, 15), date(2030, 1, 15), coupon_rate=0.04,
        )
        conv_issue = bond.convexity(0.04, settlement=date(2020, 1, 15))
        conv_5y = bond.convexity(0.04, settlement=date(2025, 1, 15))
        assert conv_5y < conv_issue

    def test_dv01_seasoned(self):
        """DV01 should decrease as bond ages."""
        bond = FixedRateBond(
            date(2020, 1, 15), date(2030, 1, 15), coupon_rate=0.04,
        )
        dv01_issue = bond.dv01_yield(0.04, settlement=date(2020, 1, 15))
        dv01_5y = bond.dv01_yield(0.04, settlement=date(2025, 1, 15))
        assert dv01_5y < dv01_issue


# ---- BH2: dirty_price filters past cashflows ----

class TestPastCashflowFiltering:
    def test_dirty_price_seasoned_reasonable(self):
        """Dirty price of seasoned par bond should be near 100, not inflated."""
        bond = FixedRateBond(
            date(2020, 1, 15), date(2030, 1, 15), coupon_rate=0.04,
        )
        # Curve at 4% as of 2025
        curve = make_flat_curve(date(2025, 1, 15), rate=0.04)
        dirty = bond.dirty_price(curve)
        # Par bond at par rate → dirty price near 100
        assert 99.0 < dirty < 101.0

    def test_future_cashflows_only(self):
        """_future_cashflows should exclude past payments."""
        bond = FixedRateBond(
            date(2020, 1, 15), date(2030, 1, 15), coupon_rate=0.04,
            frequency=Frequency.SEMI_ANNUAL,
        )
        all_cfs = bond.coupon_leg.cashflows
        future_cfs = bond._future_cashflows(date(2025, 1, 15))
        # Should have fewer cashflows than total
        assert len(future_cfs) < len(all_cfs)
        # All future cashflows should be after settlement
        for cf in future_cfs:
            assert cf.payment_date > date(2025, 1, 15)
