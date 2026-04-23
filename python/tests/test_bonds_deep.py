"""Deep tests for bonds — DD3 hardening.

Covers: YTM round-trip, duration/convexity formulas, callable bounds,
accrued interest, conversion factor, bond forward carry.
"""

import math
import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.bond import FixedRateBond
from pricebook.bond_futures import conversion_factor, implied_repo_rate, bond_futures_basis
from pricebook.callable_bond import callable_bond_price, puttable_bond_price
from pricebook.risky_bond import RiskyBond
from pricebook.zc_swap import ZeroCouponSwap
from pricebook.day_count import DayCountConvention
from pricebook.schedule import Frequency
from pricebook.swap import SwapDirection
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


class TestYTMRoundTrip:

    def test_par_bond_ytm_equals_coupon(self):
        """A par bond's YTM equals its coupon rate."""
        curve = make_flat_curve(REF, 0.05)
        bond = FixedRateBond(REF, REF + relativedelta(years=5), 0.05)
        dirty = bond.dirty_price(curve)
        ytm = bond.yield_to_maturity(dirty, REF)
        assert ytm == pytest.approx(0.05, abs=0.003)

    def test_ytm_price_round_trip(self):
        """price_from_ytm(ytm(price)) = price."""
        bond = FixedRateBond(REF, REF + relativedelta(years=10), 0.04)
        test_price = 95.0
        ytm = bond.yield_to_maturity(test_price, REF)
        recovered = bond._price_from_ytm(ytm, REF)
        assert recovered == pytest.approx(test_price, abs=0.01)

    def test_negative_yield(self):
        """Bond priced above par with very low coupon can have negative yield."""
        bond = FixedRateBond(REF, REF + relativedelta(years=2), 0.0)
        ytm = bond.yield_to_maturity(101.0, REF)
        assert ytm < 0

    def test_high_yield_distressed(self):
        """Deeply discounted bond (distressed) has high YTM."""
        bond = FixedRateBond(REF, REF + relativedelta(years=5), 0.05)
        ytm = bond.yield_to_maturity(50.0, REF)
        assert ytm > 0.15


class TestDurationConvexity:

    def test_modified_lt_macaulay(self):
        """Modified duration < Macaulay duration (for positive yields)."""
        bond = FixedRateBond(REF, REF + relativedelta(years=10), 0.05)
        ytm = 0.05
        mac = bond.macaulay_duration(ytm, REF)
        mod = bond.modified_duration(ytm, REF)
        assert mod < mac

    def test_macaulay_duration_zero_coupon(self):
        """Zero coupon bond: Macaulay duration = maturity."""
        bond = FixedRateBond(REF, REF + relativedelta(years=5), 0.0)
        mac = bond.macaulay_duration(0.05, REF)
        expected = 5.0  # approximately
        assert mac == pytest.approx(expected, abs=0.1)

    def test_convexity_positive(self):
        """Convexity is always positive for bullet bonds."""
        bond = FixedRateBond(REF, REF + relativedelta(years=10), 0.05)
        c = bond.convexity(0.05, REF)
        assert c > 0

    def test_dv01_positive(self):
        """DV01 is positive (price falls when yield rises)."""
        bond = FixedRateBond(REF, REF + relativedelta(years=10), 0.05)
        dv01 = bond.dv01_yield(0.05, REF)
        assert dv01 > 0

    def test_longer_bond_higher_duration(self):
        """Longer maturity → higher duration."""
        bond5 = FixedRateBond(REF, REF + relativedelta(years=5), 0.05)
        bond30 = FixedRateBond(REF, REF + relativedelta(years=30), 0.05)
        assert bond30.modified_duration(0.05, REF) > bond5.modified_duration(0.05, REF)


class TestAccruedInterest:

    def test_accrued_at_issue_is_zero(self):
        bond = FixedRateBond(REF, REF + relativedelta(years=5), 0.05)
        assert bond.accrued_interest(REF) == pytest.approx(0.0, abs=0.01)

    def test_accrued_mid_period(self):
        """Accrued at mid-period should be ~half the coupon."""
        bond = FixedRateBond(REF, REF + relativedelta(years=5), 0.06,
                             frequency=Frequency.SEMI_ANNUAL)
        mid = REF + relativedelta(months=3)  # mid of first 6M period
        accrued = bond.accrued_interest(mid)
        # 6% annual, semi-annual → 3% per period. Mid-period ≈ 1.5%
        assert 1.0 < accrued < 2.0

    def test_clean_plus_accrued_equals_dirty(self):
        curve = make_flat_curve(REF, 0.05)
        bond = FixedRateBond(REF, REF + relativedelta(years=5), 0.05)
        settle = REF + relativedelta(months=2)
        dirty = bond.dirty_price(curve)
        clean = bond.clean_price(curve, settle)
        accrued = bond.accrued_interest(settle)
        assert dirty == pytest.approx(clean + accrued, abs=0.01)


class TestCallableBondBounds:

    def _hw(self):
        from pricebook.hull_white import HullWhite
        curve = make_flat_curve(REF, 0.04)
        return HullWhite(a=0.03, sigma=0.01, curve=curve)

    def test_callable_leq_straight(self):
        """Callable bond ≤ straight bond (call option benefits issuer)."""
        hw = self._hw()
        straight = callable_bond_price(hw, 0.05, 10, call_dates_years=[], n_steps=50)
        callable_ = callable_bond_price(hw, 0.05, 10, n_steps=50)
        assert callable_ <= straight + 0.01

    def test_puttable_geq_straight(self):
        """Puttable bond ≥ straight bond (put option benefits holder)."""
        hw = self._hw()
        straight = puttable_bond_price(hw, 0.05, 10, put_dates_years=[], n_steps=50)
        puttable = puttable_bond_price(hw, 0.05, 10, n_steps=50)
        assert puttable >= straight - 0.01

    def test_callable_positive(self):
        hw = self._hw()
        price = callable_bond_price(hw, 0.05, 5, n_steps=50)
        assert price > 0


class TestConversionFactor:

    def test_par_bond_cf_one(self):
        """6% coupon at 6% standard yield → CF ≈ 1.0."""
        cf = conversion_factor(0.06, 10.0, yield_standard=0.06)
        assert cf == pytest.approx(1.0, abs=0.01)

    def test_higher_coupon_higher_cf(self):
        """Higher coupon → higher CF."""
        cf_low = conversion_factor(0.04, 10.0)
        cf_high = conversion_factor(0.08, 10.0)
        assert cf_high > cf_low

    def test_longer_maturity_effect(self):
        """For coupon < standard yield, longer maturity → lower CF."""
        cf_5 = conversion_factor(0.04, 5.0)
        cf_30 = conversion_factor(0.04, 30.0)
        assert cf_30 < cf_5  # discount bond: longer = cheaper


class TestRiskyBond:

    def test_risky_leq_riskfree(self):
        """Risky bond ≤ risk-free bond (default risk reduces value)."""
        curve = make_flat_curve(REF, 0.04)
        from pricebook.survival_curve import SurvivalCurve
        surv = SurvivalCurve.flat(REF, 0.02)  # 2% annual hazard
        risky = RiskyBond(REF, REF + relativedelta(years=5), 0.05, recovery=0.4)
        rf_bond = FixedRateBond(REF, REF + relativedelta(years=5), 0.05)
        risky_price = risky.dirty_price(curve, surv)
        rf_price = rf_bond.dirty_price(curve)
        assert risky_price < rf_price
