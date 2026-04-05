"""Tests for deal structuring."""

import json
import pytest
from datetime import date

from pricebook.deal import Deal, DealRole
from pricebook.trade import Trade
from pricebook.swaption import Swaption
from pricebook.pricing_context import PricingContext
from pricebook.discount_curve import DiscountCurve
from pricebook.vol_surface import FlatVol


REF = date(2024, 1, 15)


def _ctx():
    return PricingContext.simple(REF, rate=0.05, vol=0.20)


def _swn(strike=0.05):
    return Swaption(date(2025, 1, 15), date(2030, 1, 15), strike=strike)


class TestDealContainer:
    def test_create_deal(self):
        deal = Deal("d1", counterparty="ACME")
        assert deal.deal_id == "d1"
        assert deal.size == 0

    def test_add_components(self):
        deal = Deal("d1")
        deal.add("swn1", Trade(_swn(), trade_id="t1"), DealRole.PRINCIPAL)
        deal.add("swn2", Trade(_swn(0.04), trade_id="t2"), DealRole.HEDGE)
        assert deal.size == 2

    def test_get_component(self):
        deal = Deal("d1")
        deal.add("swn", Trade(_swn(), trade_id="t1"))
        comp = deal.get("swn")
        assert comp.name == "swn"
        assert comp.role == DealRole.PRINCIPAL

    def test_get_missing_raises(self):
        deal = Deal("d1")
        with pytest.raises(KeyError):
            deal.get("missing")

    def test_duplicate_name_raises(self):
        deal = Deal("d1")
        deal.add("swn", Trade(_swn()))
        with pytest.raises(ValueError, match="already exists"):
            deal.add("swn", Trade(_swn()))

    def test_by_role(self):
        deal = Deal("d1")
        deal.add("p1", Trade(_swn()), DealRole.PRINCIPAL)
        deal.add("h1", Trade(_swn(0.04)), DealRole.HEDGE)
        deal.add("h2", Trade(_swn(0.06)), DealRole.HEDGE)
        assert len(deal.by_role(DealRole.HEDGE)) == 2
        assert len(deal.by_role(DealRole.PRINCIPAL)) == 1

    def test_linked_to(self):
        deal = Deal("d1")
        deal.add("bond", Trade(_swn()), DealRole.PRINCIPAL)
        deal.add("hedge", Trade(_swn(0.04)), DealRole.HEDGE, linked_to="bond")
        assert deal.get("hedge").linked_to == "bond"


class TestDealPricing:
    def test_pv(self):
        ctx = _ctx()
        deal = Deal("d1")
        deal.add("swn", Trade(_swn(), trade_id="t1"))
        pv = deal.pv(ctx)
        assert pv > 0

    def test_pv_sum_of_components(self):
        ctx = _ctx()
        t1 = Trade(_swn(0.05), trade_id="t1")
        t2 = Trade(_swn(0.04), trade_id="t2")
        deal = Deal("d1")
        deal.add("a", t1)
        deal.add("b", t2)
        assert deal.pv(ctx) == pytest.approx(t1.pv(ctx) + t2.pv(ctx))

    def test_pv_by_component(self):
        ctx = _ctx()
        deal = Deal("d1")
        deal.add("swn", Trade(_swn(), trade_id="t1"))
        pvs = deal.pv_by_component(ctx)
        assert "swn" in pvs
        assert pvs["swn"] > 0

    def test_pv_by_role(self):
        ctx = _ctx()
        deal = Deal("d1")
        deal.add("p1", Trade(_swn(0.05)), DealRole.PRINCIPAL)
        deal.add("h1", Trade(_swn(0.04), direction=-1), DealRole.HEDGE)
        pvs = deal.pv_by_role(ctx)
        assert "principal" in pvs
        assert "hedge" in pvs


class TestDealRisk:
    def test_dv01(self):
        ctx = _ctx()
        deal = Deal("d1")
        deal.add("swn", Trade(_swn()))
        dv01 = deal.dv01(ctx)
        assert dv01 != 0.0

    def test_risk_report(self):
        ctx = _ctx()
        deal = Deal("d1", counterparty="ACME")
        deal.add("swn", Trade(_swn()))
        report = deal.risk_report(ctx)
        assert report["deal_id"] == "d1"
        assert "total_pv" in report
        assert "dv01" in report
        assert "pv_by_component" in report

    def test_risk_report_json(self):
        ctx = _ctx()
        deal = Deal("d1")
        deal.add("swn", Trade(_swn()))
        report = deal.risk_report(ctx)
        j = json.dumps(report)
        assert isinstance(j, str)


class TestDealSerialization:
    def test_roundtrip(self):
        deal = Deal("d1", counterparty="ACME", book="swaps", desk="rates")
        deal.add("swn", Trade(_swn(), trade_id="t1"), DealRole.PRINCIPAL)
        deal.add("hedge", Trade(_swn(0.04), trade_id="t2"), DealRole.HEDGE, linked_to="swn")

        d = deal.to_dict()
        deal2 = Deal.from_dict(d)

        assert deal2.deal_id == "d1"
        assert deal2.counterparty == "ACME"
        assert deal2.size == 2
        assert deal2.get("hedge").linked_to == "swn"
        assert deal2.get("hedge").role == DealRole.HEDGE

    def test_json_roundtrip(self):
        deal = Deal("d1")
        deal.add("swn", Trade(_swn(), trade_id="t1"))
        j = deal.to_json()
        deal2 = Deal.from_json(j)
        assert deal2.size == 1
