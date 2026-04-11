"""Tests for FX book."""

import pytest
from datetime import date

from pricebook.fx_book import (
    CurrencyExposure,
    FXBook,
    FXLimits,
    FXPairPosition,
)
from pricebook.swaption import Swaption
from pricebook.trade import Trade


def _instr():
    return Swaption(date(2025, 1, 15), date(2030, 1, 15),
                    strike=0.05, notional=1_000_000)


def _trade(direction=1, trade_id="t"):
    return Trade(_instr(), direction=direction, trade_id=trade_id)


# ---- Step 1: FX book + aggregation ----

class TestFXBook:
    def test_create_empty(self):
        book = FXBook("test")
        assert len(book) == 0
        assert book.n_pairs == 0

    def test_add_trade(self):
        book = FXBook("test")
        book.add(_trade(), pair="EUR/USD", notional=10_000_000,
                 spot_rate=1.085)
        assert len(book) == 1
        assert book.n_pairs == 1


class TestPositionsByPair:
    def test_single_long(self):
        book = FXBook("test")
        book.add(_trade(direction=1), pair="EUR/USD",
                 notional=10_000_000, spot_rate=1.085, reporting_rate=1.085)
        positions = book.positions_by_pair()
        assert len(positions) == 1
        p = positions[0]
        assert p.pair == "EUR/USD"
        assert p.base_ccy == "EUR"
        assert p.quote_ccy == "USD"
        assert p.net_notional == pytest.approx(10_000_000)
        assert p.long_notional == pytest.approx(10_000_000)
        assert p.short_notional == pytest.approx(0)
        assert p.net_pv_reporting == pytest.approx(10_850_000)

    def test_long_short_netting(self):
        book = FXBook("test")
        book.add(_trade(direction=1), pair="EUR/USD", notional=10_000_000)
        book.add(_trade(direction=-1), pair="EUR/USD", notional=4_000_000)
        positions = book.positions_by_pair()
        assert positions[0].net_notional == pytest.approx(6_000_000)
        assert positions[0].long_notional == pytest.approx(10_000_000)
        assert positions[0].short_notional == pytest.approx(4_000_000)
        assert positions[0].trade_count == 2

    def test_multiple_pairs(self):
        book = FXBook("test")
        book.add(_trade(), pair="EUR/USD", notional=10_000_000)
        book.add(_trade(), pair="GBP/USD", notional=5_000_000)
        positions = book.positions_by_pair()
        assert len(positions) == 2
        assert {p.pair for p in positions} == {"EUR/USD", "GBP/USD"}


class TestNetCurrencyExposure:
    def test_single_trade_creates_two_exposures(self):
        """Step 1 test: net ccy exposure aggregates correctly."""
        book = FXBook("test")
        book.add(_trade(direction=1), pair="EUR/USD",
                 notional=10_000_000, spot_rate=1.085)
        exposures = book.net_currency_exposure()
        exp_map = {e.currency: e.net_exposure for e in exposures}
        # Long EUR/USD: +10M EUR, −10M × 1.085 = −10.85M USD
        assert exp_map["EUR"] == pytest.approx(10_000_000)
        assert exp_map["USD"] == pytest.approx(-10_850_000)

    def test_offsetting_trades(self):
        book = FXBook("test")
        book.add(_trade(direction=1), pair="EUR/USD",
                 notional=10_000_000, spot_rate=1.085)
        book.add(_trade(direction=-1), pair="EUR/USD",
                 notional=10_000_000, spot_rate=1.085)
        exposures = book.net_currency_exposure()
        exp_map = {e.currency: e.net_exposure for e in exposures}
        assert exp_map["EUR"] == pytest.approx(0.0)
        assert exp_map["USD"] == pytest.approx(0.0)

    def test_multiple_pairs_aggregate_per_ccy(self):
        book = FXBook("test")
        # Long EUR/USD: +EUR, −USD
        book.add(_trade(direction=1), pair="EUR/USD",
                 notional=10_000_000, spot_rate=1.085)
        # Long GBP/USD: +GBP, −USD
        book.add(_trade(direction=1), pair="GBP/USD",
                 notional=5_000_000, spot_rate=1.27)
        exposures = book.net_currency_exposure()
        exp_map = {e.currency: e.net_exposure for e in exposures}
        assert exp_map["EUR"] == pytest.approx(10_000_000)
        assert exp_map["GBP"] == pytest.approx(5_000_000)
        # USD: −10M × 1.085 − 5M × 1.27 = −10.85M − 6.35M = −17.2M
        assert exp_map["USD"] == pytest.approx(-17_200_000)

    def test_short_position(self):
        book = FXBook("test")
        book.add(_trade(direction=-1), pair="EUR/USD",
                 notional=10_000_000, spot_rate=1.085)
        exposures = book.net_currency_exposure()
        exp_map = {e.currency: e.net_exposure for e in exposures}
        assert exp_map["EUR"] == pytest.approx(-10_000_000)
        assert exp_map["USD"] == pytest.approx(10_850_000)


class TestGrossNotional:
    def test_gross(self):
        book = FXBook("test")
        book.add(_trade(direction=1), pair="EUR/USD", notional=10_000_000)
        book.add(_trade(direction=-1), pair="GBP/USD", notional=5_000_000)
        assert book.gross_notional() == pytest.approx(15_000_000)


# ---- Step 2: limits ----

class TestFXLimits:
    def test_per_pair_breach(self):
        limits = FXLimits(max_notional_per_pair={"EUR/USD": 5_000_000})
        book = FXBook("test", limits=limits)
        book.add(_trade(), pair="EUR/USD", notional=10_000_000)
        breaches = book.check_limits()
        assert len(breaches) == 1
        assert breaches[0].limit_type == "per_pair"

    def test_per_pair_ok(self):
        limits = FXLimits(max_notional_per_pair={"EUR/USD": 20_000_000})
        book = FXBook("test", limits=limits)
        book.add(_trade(), pair="EUR/USD", notional=10_000_000)
        assert book.check_limits() == []

    def test_per_ccy_breach(self):
        limits = FXLimits(max_exposure_per_ccy={"EUR": 5_000_000})
        book = FXBook("test", limits=limits)
        book.add(_trade(), pair="EUR/USD", notional=10_000_000,
                 spot_rate=1.085)
        breaches = book.check_limits()
        assert any(b.limit_type == "per_ccy" for b in breaches)

    def test_gross_notional_breach(self):
        limits = FXLimits(max_gross_notional=10_000_000)
        book = FXBook("test", limits=limits)
        book.add(_trade(direction=1), pair="EUR/USD", notional=8_000_000)
        book.add(_trade(direction=-1), pair="GBP/USD", notional=8_000_000)
        # gross = 16M > 10M
        breaches = book.check_limits()
        assert any(b.limit_type == "gross_notional" for b in breaches)

    def test_no_breaches(self):
        limits = FXLimits(
            max_notional_per_pair={"EUR/USD": 20_000_000},
            max_gross_notional=50_000_000,
        )
        book = FXBook("test", limits=limits)
        book.add(_trade(), pair="EUR/USD", notional=10_000_000)
        assert book.check_limits() == []
