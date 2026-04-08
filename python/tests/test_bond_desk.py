"""Tests for bond desk tools."""

import pytest
from datetime import date

from pricebook.bond_desk import (
    fit_curve_from_bonds, bond_rich_cheap,
    repo_carry, securities_lending_fee,
)
from pricebook.bond import FixedRateBond
from pricebook.discount_curve import DiscountCurve


REF = date(2024, 1, 15)


def _curve(rate=0.05):
    return DiscountCurve.flat(REF, rate)


def _bond(coupon=0.05, maturity_year=2034):
    return FixedRateBond(REF, date(maturity_year, 1, 15), coupon)


# ---- Curve fitting ----

class TestCurveFitting:
    def test_reprices_input_bonds(self):
        """Fitted curve should reprice input bonds."""
        curve = _curve()
        b1 = _bond(0.04, 2027)
        b2 = _bond(0.05, 2029)
        b3 = _bond(0.06, 2034)
        bonds = [
            (b1, b1.dirty_price(curve) - b1.accrued_interest(REF) / b1.face_value * 100),
            (b2, b2.dirty_price(curve) - b2.accrued_interest(REF) / b2.face_value * 100),
            (b3, b3.dirty_price(curve) - b3.accrued_interest(REF) / b3.face_value * 100),
        ]
        fitted, results = fit_curve_from_bonds(REF, bonds)
        for r in results:
            assert abs(r.residual) < 0.5

    def test_returns_curve_and_results(self):
        curve = _curve()
        b1 = _bond(0.05, 2029)
        price = b1.dirty_price(curve) - b1.accrued_interest(REF) / b1.face_value * 100
        fitted, results = fit_curve_from_bonds(REF, [(b1, price)])
        assert isinstance(fitted, DiscountCurve)
        assert len(results) == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            fit_curve_from_bonds(REF, [])


# ---- Rich/cheap ----

class TestBondRichCheap:
    def test_fair_at_model(self):
        curve = _curve()
        b = _bond()
        model_clean = b.dirty_price(curve) - b.accrued_interest(REF) / b.face_value * 100
        results = bond_rich_cheap([(b, model_clean)], curve)
        assert results[0].signal == "fair"

    def test_rich_above_model(self):
        curve = _curve()
        b = _bond()
        model_clean = b.dirty_price(curve) - b.accrued_interest(REF) / b.face_value * 100
        results = bond_rich_cheap([(b, model_clean + 1.0)], curve, threshold=0.5)
        assert results[0].signal == "rich"
        assert results[0].spread > 0

    def test_cheap_below_model(self):
        curve = _curve()
        b = _bond()
        model_clean = b.dirty_price(curve) - b.accrued_interest(REF) / b.face_value * 100
        results = bond_rich_cheap([(b, model_clean - 1.0)], curve, threshold=0.5)
        assert results[0].signal == "cheap"
        assert results[0].spread < 0


# ---- Repo ----

class TestRepoCarry:
    def test_positive_carry(self):
        """Coupon > repo cost → positive carry."""
        result = repo_carry(100.0, 0.06, 0.03, 365)
        assert result["carry"] == pytest.approx(3.0)
        assert result["coupon_income"] == pytest.approx(6.0)
        assert result["financing_cost"] == pytest.approx(3.0)

    def test_negative_carry(self):
        """Repo cost > coupon → negative carry."""
        result = repo_carry(100.0, 0.02, 0.05, 365)
        assert result["carry"] < 0

    def test_breakeven_repo(self):
        """Breakeven repo = coupon × face / price."""
        result = repo_carry(100.0, 0.05, 0.03, 365)
        assert result["breakeven_repo"] == pytest.approx(0.05)

    def test_scales_with_term(self):
        r90 = repo_carry(100.0, 0.06, 0.03, 90)
        r365 = repo_carry(100.0, 0.06, 0.03, 365)
        assert r365["carry"] == pytest.approx(r90["carry"] * 365 / 90, rel=0.01)


class TestSecuritiesLending:
    def test_fee_calculation(self):
        fee = securities_lending_fee(100.0, 50.0, 365)
        # 50bp on 100 for 1 year = 0.50
        assert fee == pytest.approx(0.50)

    def test_shorter_term_lower_fee(self):
        f90 = securities_lending_fee(100.0, 50.0, 90)
        f365 = securities_lending_fee(100.0, 50.0, 365)
        assert f90 < f365
