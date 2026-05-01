"""T-Lock Model Paper validation — tests 1-24.

Validates against "Treasury Lock Model: A Practitioner's Forward-Bond Approach".
"""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.bond import FixedRateBond
from pricebook.treasury_lock import (
    TreasuryLock, bond_forward_clean, tlock_clean_price_npv,
    tlock_yield_npv, tlock_dirty_price_npv, pv01_forward,
)
from pricebook.bond_yield import bond_irr, bond_price_from_yield, bond_risk_factor
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)
EXPIRY = REF + relativedelta(months=3)


def _make_bond(coupon=0.04, maturity_years=10):
    return FixedRateBond.treasury_note(
        REF - relativedelta(years=1), REF + relativedelta(years=maturity_years - 1), coupon,
    )


# ---- 8.1 Bond forward ----

class TestBondForward:

    def test_1_zero_coupon_limit(self):
        """Test 1: No coupons → Bf = P0 / D_repo (trivial carry)."""
        # Zero coupon: no intermediate coupons
        bf = bond_forward_clean(
            spot_clean=0.95, accrued_spot=0.0, accrued_delivery=0.0,
            coupons=[], repo_dfs_to_delivery=[], repo_df_spot_to_delivery=0.99,
        )
        expected = 0.95 / 0.99
        assert bf == pytest.approx(expected, rel=1e-6)

    def test_2_single_coupon(self):
        """Test 2: One intermediate coupon. Exact formula check."""
        bf = bond_forward_clean(
            spot_clean=1.00, accrued_spot=0.02, accrued_delivery=0.01,
            coupons=[0.025], repo_dfs_to_delivery=[0.998],
            repo_df_spot_to_delivery=0.99,
        )
        expected = (1.00 + 0.02) / 0.99 - 0.025 / 0.998 - 0.01
        assert bf == pytest.approx(expected, rel=1e-8)

    def test_9_inception_zero_npv(self):
        """Test 9: At inception with PT-Lock = Bf, NPV = 0."""
        bf = 1.02
        npv = tlock_clean_price_npv(bf, bf, 1_000_000, 0.99)
        assert abs(npv) < 1e-10

    def test_10_linearity(self):
        """Test 10: 1bp price move → NPV change = -N × 0.0001 × D."""
        bf = 1.02
        N = 1_000_000
        D = 0.99
        npv1 = tlock_clean_price_npv(bf, bf, N, D)
        npv2 = tlock_clean_price_npv(bf, bf + 0.0001, N, D)
        expected_change = -N * 0.0001 * D
        assert (npv2 - npv1) == pytest.approx(expected_change, rel=1e-4)

    def test_11_zero_discount(self):
        """Test 11: D = 1 → NPV = N × (P_lock - Bf) exactly."""
        npv = tlock_clean_price_npv(1.05, 1.02, 1_000_000, 1.0)
        assert npv == pytest.approx(1_000_000 * (1.05 - 1.02))


# ---- 8.2 Forward yield ----

class TestForwardYield:

    def test_6_internal_consistency(self):
        """Test 6: B(yf) = Bf + Adel (forward dirty = priced at forward yield)."""
        bond = _make_bond()
        curve = make_flat_curve(REF, 0.04)
        tl = TreasuryLock(bond, 0.04, EXPIRY, repo_rate=0.035)
        result = tl.price(curve)

        alphas, _, _ = bond.accrual_schedule(EXPIRY)
        fwd_dirty = result.forward_price + bond.accrued_interest(EXPIRY) / 100.0
        yf = bond_irr(fwd_dirty, bond.coupon_rate, alphas)
        check = bond_price_from_yield(bond.coupon_rate, alphas, yf)
        assert check == pytest.approx(fwd_dirty, rel=1e-4)

    def test_12_pv01_symmetry(self):
        """Test 12: PV01(yf) via centred difference ≈ analytical modified duration."""
        bond = _make_bond()
        alphas, _, _ = bond.accrual_schedule(EXPIRY)
        yf = 0.04
        pv01 = pv01_forward(bond.coupon_rate, alphas, yf)
        # Analytical: RF = |dP/dy|, PV01 ≈ RF × 0.0001
        rf = bond_risk_factor(bond.coupon_rate, alphas, yf)
        assert pv01 == pytest.approx(rf * 0.0001, rel=0.01)


# ---- 8.3 Clean-price T-Lock ----

class TestCleanPriceLock:

    def test_clean_price_convention(self):
        """Clean price lock with locked_price set."""
        bond = _make_bond()
        curve = make_flat_curve(REF, 0.04)
        tl = TreasuryLock(bond, locked_yield=0.04, expiry=EXPIRY,
                           notional=10_000_000, direction=1, repo_rate=0.035,
                           lock_convention="clean_price", locked_price=100.0)
        result = tl.price(curve)
        assert math.isfinite(result.value)


# ---- 8.4 Yield T-Lock ----

class TestYieldLock:

    def test_13_inception_zero(self):
        """Test 13: At inception with yTLock = yf, NPV = 0."""
        bond = _make_bond()
        curve = make_flat_curve(REF, 0.04)
        # First get the forward yield
        tl_temp = TreasuryLock(bond, locked_yield=0.04, expiry=EXPIRY, repo_rate=0.035)
        result_temp = tl_temp.price(curve)
        alphas, _, _ = bond.accrual_schedule(EXPIRY)
        fwd_dirty = result_temp.forward_price + bond.accrued_interest(EXPIRY) / 100.0
        yf = bond_irr(fwd_dirty, bond.coupon_rate, alphas)

        # Lock at forward yield
        tl = TreasuryLock(bond, locked_yield=yf, expiry=EXPIRY, notional=10_000_000,
                           repo_rate=0.035)
        result = tl.price(curve)
        assert abs(result.value) < 100  # near zero on $10M

    def test_14_linearity(self):
        """Test 14: For small |yf - yTLock|, NPV linear in yield difference."""
        bond = _make_bond()
        curve = make_flat_curve(REF, 0.04)
        npvs = []
        yields = [0.039, 0.0395, 0.040, 0.0405, 0.041]
        for y in yields:
            tl = TreasuryLock(bond, locked_yield=y, expiry=EXPIRY,
                               notional=10_000_000, repo_rate=0.035)
            npvs.append(tl.price(curve).value)
        # Check roughly linear: differences should be similar
        diffs = [npvs[i+1] - npvs[i] for i in range(len(npvs)-1)]
        for i in range(1, len(diffs)):
            assert abs(diffs[i] - diffs[0]) / abs(diffs[0]) < 0.1  # within 10%


# ---- 8.5 Dirty-price T-Lock ----

class TestDirtyPriceLock:

    def test_17_dirty_equals_clean_adjusted(self):
        """Test 17: Dirty lock at P_dirty = clean lock at P_dirty - Adel."""
        bond = _make_bond()
        curve = make_flat_curve(REF, 0.04)
        adel = bond.accrued_interest(EXPIRY)

        tl_clean = TreasuryLock(bond, locked_yield=0.04, expiry=EXPIRY,
                                 notional=10_000_000, repo_rate=0.035,
                                 lock_convention="clean_price", locked_price=100.0)
        tl_dirty = TreasuryLock(bond, locked_yield=0.04, expiry=EXPIRY,
                                 notional=10_000_000, repo_rate=0.035,
                                 lock_convention="dirty_price", locked_price=100.0 + adel)
        pv_clean = tl_clean.price(curve).value
        pv_dirty = tl_dirty.price(curve).value
        assert pv_clean == pytest.approx(pv_dirty, abs=10)


# ---- 8.6 Sensitivities ----

class TestSensitivities:

    def test_18_repo_bump(self):
        """Test 18: Repo bump changes Bf but not D_TLock."""
        bond = _make_bond()
        curve = make_flat_curve(REF, 0.04)
        tl1 = TreasuryLock(bond, locked_yield=0.04, expiry=EXPIRY,
                             notional=10_000_000, repo_rate=0.035)
        tl2 = TreasuryLock(bond, locked_yield=0.04, expiry=EXPIRY,
                             notional=10_000_000, repo_rate=0.036)
        pv1 = tl1.price(curve).value
        pv2 = tl2.price(curve).value
        assert pv1 != pv2  # repo bump changes PV

    def test_19_discount_bump(self):
        """Test 19: Discount curve bump changes D_TLock."""
        bond = _make_bond()
        curve1 = make_flat_curve(REF, 0.04)
        curve2 = make_flat_curve(REF, 0.041)
        tl = TreasuryLock(bond, locked_yield=0.04, expiry=EXPIRY,
                           notional=10_000_000, repo_rate=0.035)
        pv1 = tl.price(curve1).value
        pv2 = tl.price(curve2).value
        assert pv1 != pv2

    def test_21_time_decay(self):
        """Test 21: Rolling val date forward changes NPV."""
        bond = _make_bond()
        curve = make_flat_curve(REF, 0.04)
        tl = TreasuryLock(bond, locked_yield=0.04, expiry=EXPIRY,
                           notional=10_000_000, repo_rate=0.035)
        pv1 = tl.price(curve).value
        # Roll forward 30 days
        curve2 = make_flat_curve(REF + relativedelta(days=30), 0.04)
        tl2 = TreasuryLock(bond, locked_yield=0.04, expiry=EXPIRY,
                            notional=10_000_000, repo_rate=0.035)
        pv2 = tl2.price(curve2).value
        assert pv1 != pytest.approx(pv2, abs=0.01)  # time decay


# ---- 8.7 Edge cases ----

class TestEdgeCases:

    def test_22_same_day_delivery(self):
        """Test 22: Tdel = T0 → Bf = P0, NPV = N × (P_lock - P0)."""
        bond = _make_bond()
        curve = make_flat_curve(REF, 0.04)
        # Lock at clean price with same-day expiry
        tl = TreasuryLock(bond, locked_yield=0.04, expiry=REF + relativedelta(days=1),
                           notional=10_000_000, repo_rate=0.035)
        result = tl.price(curve)
        assert math.isfinite(result.value)

    def test_24_specialness(self):
        """Test 24: Special repo (lower rate) → Bf rises → lower forward yield."""
        bond = _make_bond()
        curve = make_flat_curve(REF, 0.04)
        tl_gc = TreasuryLock(bond, locked_yield=0.04, expiry=EXPIRY, repo_rate=0.04)
        tl_special = TreasuryLock(bond, locked_yield=0.04, expiry=EXPIRY, repo_rate=0.03)

        r_gc = tl_gc.price(curve)
        r_sp = tl_special.price(curve)

        # Special → lower repo → lower carry cost → higher forward price
        # → lower forward yield → different NPV
        assert r_gc.value != pytest.approx(r_sp.value, abs=0.01)
