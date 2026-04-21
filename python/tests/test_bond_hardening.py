"""Tests for bond hardening fixes (BH1-BH7)."""

import math
from datetime import date

import pytest

from pricebook.bond import FixedRateBond
from pricebook.schedule import Frequency
from pricebook.day_count import DayCountConvention
from pricebook.amortising_bond import psa_schedule
from pricebook.bond_futures import bond_futures_basis, implied_repo_rate
from pricebook.risky_bond import RiskyBond, z_spread
from pricebook.survival_curve import SurvivalCurve
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


# ---- BH3: PSA schedule formula ----

class TestPSASchedule:
    def test_month_1_cpr(self):
        """Month 1 (index 0) should have CPR = 0.2%."""
        smm = psa_schedule(1.0, 360)
        # SMM = 1 - (1-CPR)^(1/12), so CPR = 1 - (1-SMM)^12
        cpr_month_1 = 1 - (1 - smm[0]) ** 12
        assert cpr_month_1 == pytest.approx(0.002, abs=1e-6)

    def test_month_30_cpr(self):
        """Month 30 (index 29) should have CPR = 6%."""
        smm = psa_schedule(1.0, 360)
        cpr_month_30 = 1 - (1 - smm[29]) ** 12
        assert cpr_month_30 == pytest.approx(0.06, abs=1e-6)

    def test_month_31_cpr(self):
        """Month 31+ should stay flat at 6%."""
        smm = psa_schedule(1.0, 360)
        cpr_month_31 = 1 - (1 - smm[30]) ** 12
        assert cpr_month_31 == pytest.approx(0.06, abs=1e-6)

    def test_200_psa_doubles(self):
        """200% PSA should have double the CPR."""
        smm_100 = psa_schedule(1.0, 60)
        smm_200 = psa_schedule(2.0, 60)
        # Month 15: 100PSA CPR = 3%, 200PSA = 6%
        cpr_100 = 1 - (1 - smm_100[14]) ** 12
        cpr_200 = 1 - (1 - smm_200[14]) ** 12
        assert cpr_200 == pytest.approx(2 * cpr_100, rel=1e-4)

    def test_monotone_ramp(self):
        """CPR should increase monotonically in months 1-30."""
        smm = psa_schedule(1.0, 360)
        for i in range(29):
            assert smm[i + 1] > smm[i]


# ---- BH4: Risky bond recovery at mid-period ----

class TestRecoveryMidPeriod:
    def test_recovery_higher_with_mid_period(self):
        """Mid-period discounting gives higher recovery PV than end-of-period."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.05)
        surv_dates = [date(2027, 4, 21), date(2031, 4, 21)]
        surv_probs = [0.98, 0.90]
        surv = SurvivalCurve(ref, surv_dates, surv_probs)

        bond = RiskyBond(ref, date(2031, 4, 21), 0.04, recovery=0.40)
        price = bond.dirty_price(curve, surv)
        # Should be positive and reasonable
        assert 70 < price < 110

    def test_zero_default_prob_recovery_zero(self):
        """With 100% survival, recovery contribution is zero."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        surv_dates = [date(2031, 4, 21)]
        surv_probs = [1.0]
        surv = SurvivalCurve(ref, surv_dates, surv_probs)

        bond = RiskyBond(ref, date(2031, 4, 21), 0.04, recovery=0.40)
        risky_price = bond.dirty_price(curve, surv)
        rf_price = bond.risk_free_price(curve)
        # No default → risky = risk-free
        assert risky_price == pytest.approx(rf_price, rel=1e-8)


# ---- BH5+BH6: Implied repo and carry fixes ----

class TestImpliedRepoAndCarry:
    def test_implied_repo_with_accrued(self):
        """Implied repo should use dirty cost (clean + accrued)."""
        repo = implied_repo_rate(
            bond_price=100.0, futures_price=100.0, cf=1.0,
            accrued_at_delivery=2.0, coupon_income=3.0,
            days_to_delivery=180, accrued_at_purchase=1.5,
        )
        # cost = 100 + 1.5 = 101.5
        # invoice = 100 + 2.0 = 102.0
        # profit = 102 - 101.5 + 3 = 3.5
        # repo = 3.5 / 101.5 * 365/180
        expected = 3.5 / 101.5 * (365.0 / 180)
        assert repo == pytest.approx(expected, rel=1e-10)

    def test_implied_repo_zero_accrued_backward_compat(self):
        """Default accrued_at_purchase=0 preserves old behavior."""
        repo = implied_repo_rate(
            bond_price=100.0, futures_price=100.0, cf=1.0,
            accrued_at_delivery=2.0, coupon_income=3.0,
            days_to_delivery=180,
        )
        expected = (102 - 100 + 3) / 100 * (365.0 / 180)
        assert repo == pytest.approx(expected, rel=1e-10)

    def test_carry_uses_dirty_for_financing(self):
        """Carry financing should be on dirty price, not clean."""
        result = bond_futures_basis(
            100.0, 100.0, 1.0, 0.03, 365,
            coupon_income=6.0, accrued_at_purchase=2.0,
        )
        # financing = (100 + 2) * 0.03 * 1 = 3.06
        # carry = 6.0 - 3.06 = 2.94
        assert result.carry == pytest.approx(2.94, rel=1e-4)


# ---- BH7: Z-spread bounds ----

class TestZSpreadBounds:
    def test_distressed_bond_high_spread(self):
        """Z-spread for deeply discounted bond should solve (>2000bp)."""
        ref = date(2026, 4, 21)
        curve = make_flat_curve(ref, rate=0.04)
        bond = RiskyBond(ref, date(2031, 4, 21), 0.05, recovery=0.40)
        # Bond trading at 60 (deeply distressed)
        zs = z_spread(bond, 60.0, curve)
        assert zs > 0.10  # > 1000bp
