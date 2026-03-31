"""Tests for Trade and Portfolio."""

import pytest
from datetime import date

from pricebook.trade import Trade, Portfolio
from pricebook.pricing_context import PricingContext
from pricebook.swaption import Swaption, SwaptionType
from pricebook.vol_surface import FlatVol
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


@pytest.fixture
def ctx():
    return PricingContext(
        valuation_date=REF,
        discount_curve=make_flat_curve(REF, 0.03),
        vol_surfaces={"ir": FlatVol(0.20)},
    )


@pytest.fixture
def swaption():
    return Swaption(
        expiry=date(2025, 1, 15),
        swap_end=date(2030, 1, 15),
        strike=0.03,
    )


class TestTrade:
    def test_long_trade(self, ctx, swaption):
        trade = Trade(instrument=swaption, direction=1)
        pv = trade.pv(ctx)
        assert pv > 0

    def test_short_trade(self, ctx, swaption):
        trade = Trade(instrument=swaption, direction=-1)
        pv = trade.pv(ctx)
        assert pv < 0

    def test_long_short_cancel(self, ctx, swaption):
        long = Trade(instrument=swaption, direction=1)
        short = Trade(instrument=swaption, direction=-1)
        assert long.pv(ctx) + short.pv(ctx) == pytest.approx(0.0)

    def test_notional_scale(self, ctx, swaption):
        t1 = Trade(instrument=swaption, direction=1, notional_scale=1.0)
        t2 = Trade(instrument=swaption, direction=1, notional_scale=2.0)
        assert t2.pv(ctx) == pytest.approx(2.0 * t1.pv(ctx))

    def test_metadata(self, swaption):
        trade = Trade(
            instrument=swaption,
            trade_date=REF,
            counterparty="ACME",
            trade_id="T001",
        )
        assert trade.trade_date == REF
        assert trade.counterparty == "ACME"
        assert trade.trade_id == "T001"

    def test_no_pv_ctx_raises(self, ctx):
        trade = Trade(instrument="not_an_instrument")
        with pytest.raises(ValueError, match="pv_ctx"):
            trade.pv(ctx)


class TestPortfolio:
    def test_aggregate_pv(self, ctx, swaption):
        t1 = Trade(instrument=swaption, direction=1)
        t2 = Trade(instrument=swaption, direction=1)
        port = Portfolio([t1, t2])
        assert port.pv(ctx) == pytest.approx(2 * t1.pv(ctx))

    def test_empty_portfolio(self, ctx):
        port = Portfolio()
        assert port.pv(ctx) == 0.0

    def test_add_trade(self, ctx, swaption):
        port = Portfolio()
        port.add(Trade(instrument=swaption))
        assert len(port) == 1
        assert port.pv(ctx) > 0

    def test_pv_by_trade(self, ctx, swaption):
        t1 = Trade(instrument=swaption, direction=1, trade_id="A")
        t2 = Trade(instrument=swaption, direction=-1, trade_id="B")
        port = Portfolio([t1, t2])
        breakdown = port.pv_by_trade(ctx)
        assert len(breakdown) == 2
        assert breakdown[0][0] == "A"
        assert breakdown[1][0] == "B"
        assert breakdown[0][1] + breakdown[1][1] == pytest.approx(0.0)

    def test_portfolio_pv_equals_sum(self, ctx, swaption):
        trades = [
            Trade(instrument=swaption, direction=1, trade_id="T1"),
            Trade(instrument=swaption, direction=-1, notional_scale=0.5, trade_id="T2"),
        ]
        port = Portfolio(trades)
        manual_sum = sum(t.pv(ctx) for t in trades)
        assert port.pv(ctx) == pytest.approx(manual_sum)
