"""Tests for basis trading (bond vs CDS)."""

import pytest
from datetime import date

from pricebook.basis_trade import (
    cds_bond_basis, negative_basis_trade, positive_basis_trade,
    basis_monitor,
)
from pricebook.cds import CDS
from pricebook.risky_bond import RiskyBond
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


REF = date(2024, 1, 15)
END = date(2029, 1, 15)


def _dc(rate=0.05):
    return DiscountCurve.flat(REF, rate)


def _sc(hazard=0.02):
    return SurvivalCurve.flat(REF, hazard)


def _cds(spread=0.01):
    return CDS(REF, END, spread, notional=1_000_000, recovery=0.4)


def _bond(coupon=0.06):
    return RiskyBond(REF, END, coupon, notional=100, recovery=0.4)


# ---- CDS-bond basis ----

class TestBasis:
    def test_matched_near_zero(self):
        """Matched credit curve → basis near zero."""
        dc = _dc()
        sc = _sc()
        cds = _cds()
        bond = _bond()
        # Price the bond on the same credit curve
        bond_price = bond.dirty_price(dc, sc)
        result = cds_bond_basis(cds, bond, bond_price, dc, sc)
        # Basis should be small (not exactly zero due to convention differences)
        assert abs(result.basis_z) < 0.01

    def test_negative_basis(self):
        """Bond priced cheap (high z-spread) → negative basis."""
        dc = _dc()
        sc = _sc(0.01)  # low hazard → low CDS spread
        cds = _cds()
        bond = _bond()
        # Force bond to trade at a discount (high z-spread)
        cheap_price = bond.dirty_price(dc, sc) * 0.95
        result = cds_bond_basis(cds, bond, cheap_price, dc, sc)
        assert result.basis_z < 0

    def test_both_basis_measures(self):
        dc = _dc()
        sc = _sc()
        bond = _bond()
        bond_price = bond.dirty_price(dc, sc)
        result = cds_bond_basis(_cds(), bond, bond_price, dc, sc)
        assert result.cds_spread > 0
        assert result.z_spread_val != 0
        assert result.asw_spread != 0


# ---- Basis trade construction ----

class TestNegativeBasis:
    def test_carry_positive_when_coupon_above_spread(self):
        """Bond coupon > CDS spread → positive carry."""
        dc = _dc()
        sc = _sc()
        cds = _cds(spread=0.01)  # 100bp
        bond = _bond(coupon=0.06)  # 6%
        bond_price = bond.dirty_price(dc, sc)
        result = negative_basis_trade(cds, bond, bond_price, dc, sc)
        assert result.carry > 0

    def test_trade_type(self):
        dc = _dc()
        sc = _sc()
        result = negative_basis_trade(
            _cds(), _bond(), _bond().dirty_price(dc, sc), dc, sc,
        )
        assert result.trade_type == "negative_basis"


class TestPositiveBasis:
    def test_opposite_carry(self):
        """Positive basis trade has opposite carry to negative."""
        dc = _dc()
        sc = _sc()
        cds = _cds(spread=0.01)
        bond = _bond(coupon=0.06)
        bond_price = bond.dirty_price(dc, sc)
        neg = negative_basis_trade(cds, bond, bond_price, dc, sc)
        pos = positive_basis_trade(cds, bond, bond_price, dc, sc)
        assert neg.carry == pytest.approx(-pos.carry)

    def test_trade_type(self):
        dc = _dc()
        sc = _sc()
        result = positive_basis_trade(
            _cds(), _bond(), _bond().dirty_price(dc, sc), dc, sc,
        )
        assert result.trade_type == "positive_basis"


# ---- Basis monitor ----

class TestBasisMonitor:
    def test_no_history(self):
        result = basis_monitor("ACME", -0.002)
        assert result.signal == "fair"
        assert result.z_score is None

    def test_extreme_negative_signal(self):
        history = [0.001, 0.002, -0.001, 0.000, 0.001]
        result = basis_monitor("ACME", -0.05, history=history, threshold=2.0)
        assert result.signal == "negative"

    def test_extreme_positive_signal(self):
        history = [0.001, 0.002, -0.001, 0.000, 0.001]
        result = basis_monitor("ACME", 0.05, history=history, threshold=2.0)
        assert result.signal == "positive"

    def test_fair_within_threshold(self):
        history = [0.001, 0.002, -0.001, 0.000, 0.001]
        result = basis_monitor("ACME", 0.001, history=history, threshold=2.0)
        assert result.signal == "fair"

    def test_percentile(self):
        history = [0.001, 0.002, 0.003, 0.004, 0.005]
        result = basis_monitor("ACME", 0.003, history=history)
        assert result.percentile == pytest.approx(60.0)

    def test_name(self):
        result = basis_monitor("ACME", 0.001)
        assert result.name == "ACME"
