"""Tests for Treasury bill: discount yield, BEY, MMY, PV, risk, serialisation."""

from __future__ import annotations

import math
from datetime import date

import pytest
from dateutil.relativedelta import relativedelta

from pricebook.tbill import TreasuryBill
from tests.conftest import make_flat_curve


REF = date(2024, 7, 15)


# ---- Construction and pricing ----

class TestConstruction:

    def test_from_discount_yield_90day(self):
        """90-day T-bill at 5% discount → price = 98.75."""
        mat = REF + relativedelta(days=90)
        bill = TreasuryBill.from_discount_yield(REF, mat, 0.05)
        assert bill.price == pytest.approx(100 * (1 - 0.05 * 90 / 360))
        assert bill.price == pytest.approx(98.75)

    def test_from_discount_yield_180day(self):
        mat = REF + relativedelta(days=180)
        bill = TreasuryBill.from_discount_yield(REF, mat, 0.05)
        assert bill.price == pytest.approx(100 * (1 - 0.05 * 180 / 360))
        assert bill.price == pytest.approx(97.50)

    def test_from_price(self):
        mat = REF + relativedelta(days=90)
        bill = TreasuryBill(REF, mat, price=98.75)
        assert bill.discount_yield == pytest.approx(0.05, rel=1e-6)

    def test_from_bey_short(self):
        """BEY constructor for ≤ 182 days."""
        mat = REF + relativedelta(days=90)
        bey = (100 - 98.75) / 98.75 * (365 / 90)
        bill = TreasuryBill.from_bond_equivalent_yield(REF, mat, bey)
        assert bill.price == pytest.approx(98.75, rel=1e-4)

    def test_from_bey_long(self):
        """BEY constructor for > 182 days."""
        mat = REF + relativedelta(days=270)
        bill = TreasuryBill.from_bond_equivalent_yield(REF, mat, 0.055)
        assert bill.price > 0
        assert bill.price < 100

    def test_validation(self):
        mat = REF + relativedelta(days=90)
        with pytest.raises(ValueError):
            TreasuryBill(mat, REF, 99)  # settlement after maturity
        with pytest.raises(ValueError):
            TreasuryBill(REF, mat, -5)  # negative price


# ---- Yield conventions ----

class TestYields:

    def test_discount_yield(self):
        """d = (F - P) / F × 360/days."""
        mat = REF + relativedelta(days=90)
        bill = TreasuryBill(REF, mat, price=98.75)
        assert bill.discount_yield == pytest.approx(0.05, rel=1e-6)

    def test_bey_short(self):
        """BEY = (F - P) / P × 365/days for ≤ 182 days."""
        mat = REF + relativedelta(days=90)
        bill = TreasuryBill(REF, mat, price=98.75)
        expected = (100 - 98.75) / 98.75 * (365 / 90)
        assert bill.bey == pytest.approx(expected, rel=1e-6)

    def test_bey_greater_than_discount(self):
        """BEY > discount yield (always, for T-bills)."""
        mat = REF + relativedelta(days=90)
        bill = TreasuryBill(REF, mat, price=98.75)
        assert bill.bey > bill.discount_yield

    def test_money_market_yield(self):
        """MMY = (F - P) / P × 360/days."""
        mat = REF + relativedelta(days=90)
        bill = TreasuryBill(REF, mat, price=98.75)
        expected = (100 - 98.75) / 98.75 * (360 / 90)
        assert bill.money_market_yield == pytest.approx(expected, rel=1e-6)

    def test_mmyield_between_discount_and_bey(self):
        """discount < MMY < BEY for standard T-bills."""
        mat = REF + relativedelta(days=90)
        bill = TreasuryBill(REF, mat, price=98.75)
        assert bill.discount_yield < bill.money_market_yield < bill.bey

    def test_bey_long_maturity(self):
        """BEY for > 182 days uses semi-annual formula."""
        mat = REF + relativedelta(days=270)
        bill = TreasuryBill(REF, mat, price=96.0)
        bey = bill.bey
        assert bey > 0
        assert bey > bill.discount_yield


# ---- Cross-conversions ----

class TestConversions:

    def test_discount_to_bey_roundtrip(self):
        """discount → BEY → discount round-trip."""
        d = 0.05
        days = 90
        bey = TreasuryBill.discount_to_bey(d, days)
        d_back = TreasuryBill.bey_to_discount(bey, days)
        assert d_back == pytest.approx(d, rel=1e-6)

    def test_discount_to_mmyield(self):
        d = 0.05
        days = 90
        mmy = TreasuryBill.discount_to_mmyield(d, days)
        # MMY = d / (1 - d × days/360)
        expected = d / (1 - d * days / 360)
        assert mmy == pytest.approx(expected, rel=1e-6)

    def test_roundtrip_long(self):
        """Round-trip for > 182 day bill."""
        d = 0.04
        days = 270
        bey = TreasuryBill.discount_to_bey(d, days)
        d_back = TreasuryBill.bey_to_discount(bey, days)
        assert d_back == pytest.approx(d, rel=1e-4)


# ---- PV and risk ----

class TestPVRisk:

    def test_pv_matches_price(self):
        """PV on flat curve at implied yield should ≈ price."""
        mat = REF + relativedelta(days=90)
        bill = TreasuryBill(REF, mat, price=98.75)
        # Implied continuous rate from price
        r = -math.log(98.75 / 100) / (90 / 365)
        dc = make_flat_curve(REF, r)
        pv = bill.pv(dc)
        assert pv == pytest.approx(98.75, rel=0.01)

    def test_dv01_negative(self):
        """Higher rates → lower PV → DV01 < 0."""
        mat = REF + relativedelta(days=180)
        bill = TreasuryBill(REF, mat, price=97.5)
        dc = make_flat_curve(REF, 0.05)
        assert bill.dv01(dc) < 0

    def test_duration_equals_time(self):
        """Zero-coupon: Macaulay duration = time to maturity."""
        mat = REF + relativedelta(days=90)
        bill = TreasuryBill(REF, mat, price=98.75)
        assert bill.duration == pytest.approx(90 / 365, rel=1e-6)

    def test_modified_duration(self):
        mat = REF + relativedelta(days=180)
        bill = TreasuryBill(REF, mat, price=97.5)
        assert bill.modified_duration < bill.duration
        assert bill.modified_duration > 0

    def test_implied_repo(self):
        """Implied repo from spot/forward prices."""
        r = TreasuryBill.implied_repo_rate(98.75, 99.50, 90)
        expected = (99.50 / 98.75 - 1) * (360 / 90)
        assert r == pytest.approx(expected)


# ---- Serialisation ----

class TestSerialisation:

    def test_round_trip(self):
        from pricebook.serialisable import from_dict
        mat = REF + relativedelta(days=90)
        bill = TreasuryBill(REF, mat, price=98.75)
        d = bill.to_dict()
        bill2 = from_dict(d)
        assert bill2.price == bill.price
        assert bill2.settlement == bill.settlement
        assert bill2.maturity == bill.maturity
        assert bill2.discount_yield == pytest.approx(bill.discount_yield)
