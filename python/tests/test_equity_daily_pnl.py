"""Tests for equity daily P&L: official P&L and Greek attribution."""

import pytest
from datetime import date

from pricebook.discount_curve import DiscountCurve
from pricebook.equity_book import EquityBook
from pricebook.equity_daily_pnl import (
    EquityDailyPnL,
    GreekAttribution,
    EquityBookAttribution,
    TradeGreeks,
    attribute_equity_pnl,
    compute_equity_daily_pnl,
)
from pricebook.pricing_context import PricingContext
from pricebook.swaption import Swaption
from pricebook.trade import Trade
from pricebook.vol_surface import FlatVol


PRIOR = date(2024, 1, 15)
CURRENT = date(2024, 1, 16)


def _ctx(rate: float = 0.05, vol: float = 0.20, val_date: date = PRIOR):
    return PricingContext(
        valuation_date=val_date,
        discount_curve=DiscountCurve.flat(val_date, rate),
        vol_surfaces={"ir": FlatVol(vol)},
    )


def _swaption(strike=0.05, notional=1_000_000):
    return Swaption(date(2025, 1, 15), date(2030, 1, 15),
                    strike=strike, notional=notional)


def _trade(notional=1_000_000, direction=1, trade_id="t1"):
    return Trade(_swaption(notional=notional), direction=direction,
                 trade_id=trade_id)


# ---- Step 1: Official equity P&L ----

class TestComputeEquityDailyPnL:
    def test_empty_book(self):
        book = EquityBook("test")
        pnl = compute_equity_daily_pnl(book, _ctx(), _ctx(val_date=CURRENT))
        assert pnl.book_name == "test"
        assert pnl.prior_pv == 0.0
        assert pnl.current_pv == 0.0
        assert pnl.market_move_pnl == 0.0
        assert pnl.total_pnl == 0.0

    def test_market_move_only(self):
        book = EquityBook("test")
        book.add(_trade(), ticker="AAPL", sector="tech")
        prior = _ctx(rate=0.05, vol=0.20)
        current = _ctx(rate=0.06, vol=0.22, val_date=CURRENT)
        pnl = compute_equity_daily_pnl(book, prior, current)
        # Both PVs are non-zero and different (rate + vol moved).
        assert pnl.prior_pv != 0.0
        assert pnl.current_pv != 0.0
        assert pnl.market_move_pnl == pytest.approx(pnl.current_pv - pnl.prior_pv)
        assert pnl.new_trade_pnl == 0.0
        assert pnl.amendment_pnl == 0.0
        assert pnl.total_pnl == pytest.approx(pnl.market_move_pnl)

    def test_new_trades(self):
        book = EquityBook("test")
        book.add(_trade(trade_id="t_old"), ticker="AAPL")
        prior = _ctx()
        current = _ctx(val_date=CURRENT)
        new = [Trade(_swaption(), trade_id="t_new")]
        pnl = compute_equity_daily_pnl(book, prior, current, new_trades=new)
        assert pnl.new_trade_pnl == pytest.approx(new[0].pv(current))
        assert pnl.total_pnl == pytest.approx(
            pnl.market_move_pnl + pnl.new_trade_pnl
        )

    def test_amendments(self):
        book = EquityBook("test")
        book.add(_trade(), ticker="AAPL")
        pnl = compute_equity_daily_pnl(
            book, _ctx(), _ctx(val_date=CURRENT),
            amendments={"t1": 12_345.0},
        )
        assert pnl.amendment_pnl == 12_345.0
        assert pnl.total_pnl == pytest.approx(
            pnl.market_move_pnl + pnl.amendment_pnl
        )

    def test_full_decomposition(self):
        book = EquityBook("test")
        book.add(_trade(), ticker="AAPL")
        prior = _ctx(rate=0.05, vol=0.20)
        current = _ctx(rate=0.06, vol=0.22, val_date=CURRENT)
        new = [Trade(_swaption(), trade_id="t_new")]
        pnl = compute_equity_daily_pnl(
            book, prior, current,
            new_trades=new, amendments={"t1": 1_000.0},
        )
        assert pnl.total_pnl == pytest.approx(
            pnl.market_move_pnl + pnl.new_trade_pnl + pnl.amendment_pnl
        )

    def test_dates_recorded(self):
        book = EquityBook("test")
        prior = _ctx()
        current = _ctx(val_date=CURRENT)
        pnl = compute_equity_daily_pnl(book, prior, current)
        assert pnl.prior_date == PRIOR
        assert pnl.current_date == CURRENT


# ---- Step 2: Greek-based attribution ----

class TestAttributeEquityPnL:
    def test_empty_book(self):
        book = EquityBook("test")
        attrib = attribute_equity_pnl(
            book, _ctx(), _ctx(val_date=CURRENT), spot_changes={},
        )
        assert attrib.total_pnl == 0.0
        assert attrib.delta_pnl == 0.0
        assert attrib.unexplained == 0.0
        assert attrib.by_trade == []

    def test_delta_only(self):
        book = EquityBook("test")
        book.add(_trade(trade_id="t1"), ticker="AAPL")
        attrib = attribute_equity_pnl(
            book, _ctx(), _ctx(val_date=CURRENT),
            spot_changes={"AAPL": 2.0},
            greeks={"t1": TradeGreeks(delta=100.0)},
        )
        assert attrib.delta_pnl == pytest.approx(200.0)
        assert attrib.gamma_pnl == 0.0
        assert attrib.vega_pnl == 0.0

    def test_gamma_term(self):
        book = EquityBook("test")
        book.add(_trade(trade_id="t1"), ticker="AAPL")
        attrib = attribute_equity_pnl(
            book, _ctx(), _ctx(val_date=CURRENT),
            spot_changes={"AAPL": 4.0},
            greeks={"t1": TradeGreeks(gamma=10.0)},
        )
        # 0.5 × 10 × 16 = 80
        assert attrib.gamma_pnl == pytest.approx(80.0)

    def test_vega_term(self):
        book = EquityBook("test")
        book.add(_trade(trade_id="t1"), ticker="AAPL")
        attrib = attribute_equity_pnl(
            book, _ctx(), _ctx(val_date=CURRENT),
            spot_changes={"AAPL": 0.0},
            vol_changes={"AAPL": 0.01},
            greeks={"t1": TradeGreeks(vega=5_000.0)},
        )
        assert attrib.vega_pnl == pytest.approx(50.0)

    def test_theta_term(self):
        book = EquityBook("test")
        book.add(_trade(trade_id="t1"), ticker="AAPL")
        # 1 calendar day between PRIOR and CURRENT
        attrib = attribute_equity_pnl(
            book, _ctx(), _ctx(val_date=CURRENT),
            spot_changes={},
            greeks={"t1": TradeGreeks(theta=-25.0)},
        )
        assert attrib.theta_pnl == pytest.approx(-25.0)

    def test_rho_term(self):
        book = EquityBook("test")
        book.add(_trade(trade_id="t1"), ticker="AAPL")
        attrib = attribute_equity_pnl(
            book, _ctx(), _ctx(val_date=CURRENT),
            spot_changes={},
            rate_change=0.001,
            greeks={"t1": TradeGreeks(rho=10_000.0)},
        )
        assert attrib.rho_pnl == pytest.approx(10.0)

    def test_short_position_flips_sign(self):
        book = EquityBook("test")
        book.add(_trade(trade_id="t_short", direction=-1), ticker="AAPL")
        attrib = attribute_equity_pnl(
            book, _ctx(), _ctx(val_date=CURRENT),
            spot_changes={"AAPL": 2.0},
            greeks={"t_short": TradeGreeks(delta=100.0)},
        )
        assert attrib.delta_pnl == pytest.approx(-200.0)

    def test_explained_plus_unexplained_equals_total(self):
        book = EquityBook("test")
        book.add(_trade(trade_id="t1"), ticker="AAPL")
        prior = _ctx(rate=0.05, vol=0.20)
        current = _ctx(rate=0.06, vol=0.22, val_date=CURRENT)
        attrib = attribute_equity_pnl(
            book, prior, current,
            spot_changes={"AAPL": 1.0}, vol_changes={"AAPL": 0.02},
            rate_change=0.01,
            greeks={"t1": TradeGreeks(
                delta=50.0, gamma=2.0, vega=1_000.0, theta=-10.0, rho=500.0,
            )},
        )
        explained = attrib.explained
        assert explained + attrib.unexplained == pytest.approx(attrib.total_pnl)

    def test_per_trade_sums_equal_book(self):
        book = EquityBook("test")
        book.add(_trade(trade_id="t1"), ticker="AAPL")
        book.add(_trade(trade_id="t2"), ticker="MSFT")
        attrib = attribute_equity_pnl(
            book, _ctx(), _ctx(val_date=CURRENT),
            spot_changes={"AAPL": 1.0, "MSFT": 2.0},
            greeks={
                "t1": TradeGreeks(delta=100.0),
                "t2": TradeGreeks(delta=200.0),
            },
        )
        assert len(attrib.by_trade) == 2
        assert attrib.delta_pnl == pytest.approx(
            sum(a.delta_pnl for a in attrib.by_trade)
        )
        assert attrib.delta_pnl == pytest.approx(100.0 + 400.0)

    def test_per_ticker_aggregation(self):
        book = EquityBook("test")
        book.add(_trade(trade_id="t1"), ticker="AAPL")
        book.add(_trade(trade_id="t2"), ticker="AAPL")
        book.add(_trade(trade_id="t3"), ticker="MSFT")
        attrib = attribute_equity_pnl(
            book, _ctx(), _ctx(val_date=CURRENT),
            spot_changes={"AAPL": 1.0, "MSFT": 2.0},
            greeks={
                "t1": TradeGreeks(delta=100.0),
                "t2": TradeGreeks(delta=50.0),
                "t3": TradeGreeks(delta=30.0),
            },
        )
        assert "AAPL" in attrib.by_ticker
        assert "MSFT" in attrib.by_ticker
        assert attrib.by_ticker["AAPL"]["delta_pnl"] == pytest.approx(150.0)
        assert attrib.by_ticker["MSFT"]["delta_pnl"] == pytest.approx(60.0)

    def test_missing_greeks_default_zero(self):
        book = EquityBook("test")
        book.add(_trade(trade_id="t1"), ticker="AAPL")
        # No greeks supplied → all sensitivities zero, but total_pnl is still
        # the actual re-priced move (so unexplained == total).
        attrib = attribute_equity_pnl(
            book, _ctx(), _ctx(val_date=CURRENT),
            spot_changes={"AAPL": 1.0},
        )
        assert attrib.delta_pnl == 0.0
        assert attrib.gamma_pnl == 0.0
        assert attrib.unexplained == pytest.approx(attrib.total_pnl)

    def test_notional_scale_applied(self):
        book = EquityBook("test")
        scaled = Trade(_swaption(), trade_id="t1", notional_scale=2.5)
        book.add(scaled, ticker="AAPL")
        attrib = attribute_equity_pnl(
            book, _ctx(), _ctx(val_date=CURRENT),
            spot_changes={"AAPL": 1.0},
            greeks={"t1": TradeGreeks(delta=100.0)},
        )
        assert attrib.delta_pnl == pytest.approx(250.0)
