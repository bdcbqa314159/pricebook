"""Tests for serialization: curves, instruments, trades, contexts."""

import json
import math
import pytest
from datetime import date

from pricebook.serialization import (
    discount_curve_to_dict,
    discount_curve_from_dict,
    survival_curve_to_dict,
    survival_curve_from_dict,
    pricing_context_to_dict,
    pricing_context_from_dict,
    instrument_to_dict,
    instrument_from_dict,
    trade_to_dict,
    trade_from_dict,
    portfolio_to_dict,
    portfolio_from_dict,
    to_json,
)
from pricebook.pricing_context import PricingContext
from pricebook.vol_surface import FlatVol
from pricebook.trade import Trade, Portfolio
from tests.conftest import make_flat_curve, make_flat_survival


REF = date(2024, 1, 15)


# ---------------------------------------------------------------------------
# Slice 68: Curve + Context serialization
# ---------------------------------------------------------------------------


class TestDiscountCurveSerialization:
    def test_roundtrip(self):
        curve = make_flat_curve(REF, 0.05)
        d = discount_curve_to_dict(curve)
        curve2 = discount_curve_from_dict(d)
        for t in [0.5, 1.0, 2.0, 5.0, 10.0]:
            dt = date.fromordinal(REF.toordinal() + int(t * 365))
            assert curve2.df(dt) == pytest.approx(curve.df(dt), rel=1e-10)

    def test_json_roundtrip(self):
        curve = make_flat_curve(REF, 0.05)
        j = to_json(curve)
        d = json.loads(j)
        curve2 = discount_curve_from_dict(d)
        dt = date.fromordinal(REF.toordinal() + 365)
        assert curve2.df(dt) == pytest.approx(curve.df(dt), rel=1e-10)

    def test_dict_has_type(self):
        curve = make_flat_curve(REF, 0.05)
        d = discount_curve_to_dict(curve)
        assert d["type"] == "discount_curve"

    def test_reference_date_preserved(self):
        curve = make_flat_curve(REF, 0.05)
        d = discount_curve_to_dict(curve)
        curve2 = discount_curve_from_dict(d)
        assert curve2.reference_date == REF


class TestSurvivalCurveSerialization:
    def test_roundtrip(self):
        curve = make_flat_survival(REF, 0.02)
        d = survival_curve_to_dict(curve)
        curve2 = survival_curve_from_dict(d)
        for t in [1.0, 3.0, 5.0]:
            dt = date.fromordinal(REF.toordinal() + int(t * 365))
            assert curve2.survival(dt) == pytest.approx(curve.survival(dt), rel=1e-6)

    def test_dict_has_type(self):
        curve = make_flat_survival(REF, 0.02)
        d = survival_curve_to_dict(curve)
        assert d["type"] == "survival_curve"


class TestPricingContextSerialization:
    def test_roundtrip(self):
        curve = make_flat_curve(REF, 0.05)
        surv = make_flat_survival(REF, 0.02)
        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=curve,
            vol_surfaces={"ir": FlatVol(0.20)},
            credit_curves={"ACME": surv},
            fx_spots={("EUR", "USD"): 1.085},
        )
        d = pricing_context_to_dict(ctx)
        ctx2 = pricing_context_from_dict(d)

        assert ctx2.valuation_date == REF
        dt = date.fromordinal(REF.toordinal() + 365)
        assert ctx2.discount_curve.df(dt) == pytest.approx(ctx.discount_curve.df(dt), rel=1e-10)
        assert ctx2.vol_surfaces["ir"].vol(1.0, 0.05) == pytest.approx(0.20)
        assert ctx2.fx_spots[("EUR", "USD")] == pytest.approx(1.085)

    def test_json_roundtrip(self):
        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=make_flat_curve(REF, 0.05),
        )
        j = to_json(ctx)
        ctx2 = pricing_context_from_dict(json.loads(j))
        assert ctx2.valuation_date == REF

    def test_minimal_context(self):
        ctx = PricingContext(valuation_date=REF)
        d = pricing_context_to_dict(ctx)
        ctx2 = pricing_context_from_dict(d)
        assert ctx2.valuation_date == REF
        assert ctx2.discount_curve is None


# ---------------------------------------------------------------------------
# Slice 69: Instrument serialization
# ---------------------------------------------------------------------------


class TestSwapSerialization:
    def test_roundtrip(self):
        from pricebook.swap import InterestRateSwap
        swap = InterestRateSwap(date(2024, 1, 15), date(2029, 1, 15), 0.05)
        d = instrument_to_dict(swap)
        swap2 = instrument_from_dict(d)
        curve = make_flat_curve(REF, 0.05)
        assert swap2.pv(curve) == pytest.approx(swap.pv(curve), rel=1e-10)

    def test_type_field(self):
        from pricebook.swap import InterestRateSwap
        swap = InterestRateSwap(date(2024, 1, 15), date(2029, 1, 15), 0.05)
        d = instrument_to_dict(swap)
        assert d["type"] == "irs"


class TestBondSerialization:
    def test_roundtrip(self):
        from pricebook.bond import FixedRateBond
        bond = FixedRateBond(date(2020, 1, 15), date(2030, 1, 15), 0.04)
        d = instrument_to_dict(bond)
        bond2 = instrument_from_dict(d)
        assert bond2.coupon_rate == bond.coupon_rate
        assert bond2.maturity == bond.maturity
        assert bond2.face_value == bond.face_value


class TestCDSSerialization:
    def test_roundtrip(self):
        from pricebook.cds import CDS
        cds = CDS(date(2024, 1, 15), date(2029, 1, 15), 0.01)
        d = instrument_to_dict(cds)
        cds2 = instrument_from_dict(d)
        curve = make_flat_curve(REF, 0.05)
        surv = make_flat_survival(REF, 0.02)
        assert cds2.pv(curve, surv) == pytest.approx(cds.pv(curve, surv), rel=1e-10)


class TestFRASerialization:
    def test_roundtrip(self):
        from pricebook.fra import FRA
        fra = FRA(date(2024, 7, 15), date(2024, 10, 15), 0.05)
        d = instrument_to_dict(fra)
        fra2 = instrument_from_dict(d)
        curve = make_flat_curve(REF, 0.05)
        assert fra2.pv(curve) == pytest.approx(fra.pv(curve), rel=1e-10)


class TestSwaptionSerialization:
    def test_roundtrip(self):
        from pricebook.swaption import Swaption
        swn = Swaption(date(2025, 1, 15), date(2030, 1, 15), 0.05)
        d = instrument_to_dict(swn)
        swn2 = instrument_from_dict(d)
        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=make_flat_curve(REF, 0.05),
            vol_surfaces={"ir": FlatVol(0.20)},
        )
        assert swn2.pv_ctx(ctx) == pytest.approx(swn.pv_ctx(ctx), rel=1e-10)


class TestTradeSerialization:
    def test_roundtrip(self):
        from pricebook.swap import InterestRateSwap
        swap = InterestRateSwap(date(2024, 1, 15), date(2029, 1, 15), 0.05)
        trade = Trade(swap, direction=1, notional_scale=2.0,
                      trade_id="t1", counterparty="ACME",
                      trade_date=date(2024, 1, 10))
        d = trade_to_dict(trade)
        trade2 = trade_from_dict(d)
        assert trade2.trade_id == "t1"
        assert trade2.counterparty == "ACME"
        assert trade2.direction == 1
        assert trade2.notional_scale == 2.0


class TestPortfolioSerialization:
    def test_roundtrip(self):
        from pricebook.swap import InterestRateSwap
        from pricebook.cds import CDS
        swap = InterestRateSwap(date(2024, 1, 15), date(2029, 1, 15), 0.05)
        cds = CDS(date(2024, 1, 15), date(2029, 1, 15), 0.01)
        port = Portfolio([
            Trade(swap, trade_id="swap1"),
            Trade(cds, trade_id="cds1"),
        ], name="test_book")
        d = portfolio_to_dict(port)
        port2 = portfolio_from_dict(d)
        assert port2.name == "test_book"
        assert len(port2.trades) == 2

    def test_json_roundtrip(self):
        from pricebook.swap import InterestRateSwap
        swap = InterestRateSwap(date(2024, 1, 15), date(2029, 1, 15), 0.05)
        port = Portfolio([Trade(swap, trade_id="s1")], name="book")
        j = to_json(port)
        port2 = portfolio_from_dict(json.loads(j))
        assert len(port2.trades) == 1


class TestUnknownInstrument:
    def test_raises(self):
        with pytest.raises(ValueError, match="has no to_dict"):
            instrument_to_dict(object())

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown type"):
            instrument_from_dict({"type": "unknown", "params": {}})
