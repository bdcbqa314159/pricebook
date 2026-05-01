"""Tests for asset swap depth: proceeds convention, z-spread/ASW bridge."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.bond import FixedRateBond
from pricebook.par_asset_swap import (
    ParAssetSwap, ProceedsAssetSwap,
    asw_vs_zspread, SpreadComparison,
)
from pricebook.schedule import Frequency
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


def _make_bond(coupon=0.04):
    return FixedRateBond(
        issue_date=REF - relativedelta(years=1),
        maturity=REF + relativedelta(years=9),
        coupon_rate=coupon,
        frequency=Frequency.SEMI_ANNUAL,
        face_value=100.0,
    )


# ---- Par vs Proceeds ----

class TestParVsProceeds:

    def test_at_par_equal(self):
        """At par (market=100), par ASW = proceeds ASW."""
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        par = ParAssetSwap(bond, REF, 100.0).price(dc)
        proc = ProceedsAssetSwap(bond, REF, 100.0).price(dc)
        assert par.asw_spread == pytest.approx(proc.asw_spread, abs=0.001)

    def test_upfront_differs(self):
        """For discount bond, par has upfront but proceeds doesn't."""
        bond = _make_bond(coupon=0.02)
        dc = make_flat_curve(REF, 0.04)
        par = ParAssetSwap(bond, REF, 90.0).price(dc)
        proc = ProceedsAssetSwap(bond, REF, 90.0).price(dc)
        assert par.upfront == pytest.approx(10.0)
        assert proc.upfront == 0.0
        # Spreads are the same when clean = dirty (both use same formula)
        # but par convention has material upfront

    def test_proceeds_no_upfront(self):
        """Proceeds ASW has zero upfront."""
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        proc = ProceedsAssetSwap(bond, REF, 95.0).price(dc)
        assert proc.upfront == 0.0

    def test_par_upfront(self):
        """Par ASW upfront = 100 - market."""
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        par = ParAssetSwap(bond, REF, 95.0).price(dc)
        assert par.upfront == pytest.approx(5.0)

    def test_premium_bond(self):
        """Premium bond (price > 100): par ASW has negative upfront."""
        bond = _make_bond(coupon=0.06)
        dc = make_flat_curve(REF, 0.04)
        par = ParAssetSwap(bond, REF, 105.0).price(dc)
        assert par.upfront < 0

    def test_both_positive_spread_for_credit(self):
        """Both conventions should give positive spread for sub-par credit bond."""
        bond = _make_bond(coupon=0.04)
        dc = make_flat_curve(REF, 0.04)
        par = ParAssetSwap(bond, REF, 95.0).price(dc)
        proc = ProceedsAssetSwap(bond, REF, 95.0).price(dc)
        assert par.asw_spread > 0
        assert proc.asw_spread > 0


# ---- Z-spread / ASW bridge ----

class TestZSpreadBridge:

    def test_at_par_all_equal(self):
        """At par: z-spread ≈ par ASW ≈ proceeds ASW."""
        bond = _make_bond(coupon=0.04)
        dc = make_flat_curve(REF, 0.04)
        # Price the bond at risk-free to get par price
        rf_price = bond.dirty_price(dc)
        comp = asw_vs_zspread(bond, rf_price, dc, REF)
        # At risk-free price, z-spread ≈ 0 and all ASW ≈ 0
        assert abs(comp.z_spread) < 0.005
        assert abs(comp.par_asw_spread) < 0.005

    def test_credit_bond_positive(self):
        """Credit bond below par: all spreads positive."""
        bond = _make_bond(coupon=0.04)
        dc = make_flat_curve(REF, 0.04)
        comp = asw_vs_zspread(bond, 92.0, dc, REF)
        assert comp.z_spread > 0
        assert comp.par_asw_spread > 0
        assert comp.proceeds_asw_spread > 0

    def test_basis_small_near_par(self):
        """Near par: ASW-zspread basis should be small."""
        bond = _make_bond(coupon=0.04)
        dc = make_flat_curve(REF, 0.04)
        comp = asw_vs_zspread(bond, 99.0, dc, REF)
        assert abs(comp.par_asw_basis) < 50  # < 50bp basis

    def test_basis_larger_off_par(self):
        """Deep discount: par ASW basis vs z-spread widens."""
        bond = _make_bond(coupon=0.02)
        dc = make_flat_curve(REF, 0.04)
        near = asw_vs_zspread(bond, 98.0, dc, REF)
        far = asw_vs_zspread(bond, 80.0, dc, REF)
        assert abs(far.par_asw_basis) > abs(near.par_asw_basis)

    def test_to_dict(self):
        bond = _make_bond()
        dc = make_flat_curve(REF, 0.04)
        comp = asw_vs_zspread(bond, 95.0, dc, REF)
        d = comp.to_dict()
        assert "z_spread" in d
        assert "par_asw_spread" in d
        assert "proceeds_asw_basis_bp" in d
