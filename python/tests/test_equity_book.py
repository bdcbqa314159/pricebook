"""Tests for equity book."""

import pytest
from datetime import date

from pricebook.equity_book import (
    EquityBook, EquityLimits, EquityPosition, SectorExposure,
)
from pricebook.trade import Trade
from pricebook.swaption import Swaption


REF = date(2024, 1, 15)


def _option(strike=100.0, notional=1_000_000):
    """Use Swaption as a notional-bearing instrument for testing."""
    return Swaption(date(2025, 1, 15), date(2030, 1, 15), strike=strike, notional=notional)


def _trade(notional=1_000_000, direction=1, trade_id="t1"):
    return Trade(_option(notional=notional), direction=direction, trade_id=trade_id)


# ---- Basic book ----

class TestEquityBook:
    def test_create_empty(self):
        book = EquityBook("US_Vol")
        assert book.name == "US_Vol"
        assert len(book) == 0
        assert book.n_names == 0
        assert book.n_sectors == 0

    def test_add_trade(self):
        book = EquityBook("US_Vol")
        book.add(_trade(), ticker="AAPL", sector="tech", spot=180.0)
        assert len(book) == 1
        assert book.n_names == 1
        assert book.n_sectors == 1

    def test_currency_default(self):
        book = EquityBook("EU_Vol", currency="EUR")
        book.add(_trade(), ticker="SAP")
        entries = book.entries
        assert entries[0].currency == "EUR"


# ---- Position aggregation ----

class TestPositionsByTicker:
    def test_single_position(self):
        book = EquityBook("test")
        book.add(_trade(notional=10_000_000), ticker="AAPL", sector="tech")
        positions = book.positions_by_ticker()
        assert len(positions) == 1
        assert positions[0].ticker == "AAPL"
        assert positions[0].net_notional == 10_000_000
        assert positions[0].long_notional == 10_000_000
        assert positions[0].short_notional == 0

    def test_long_short_net(self):
        book = EquityBook("test")
        book.add(_trade(notional=10_000_000, direction=1), ticker="AAPL", sector="tech")
        book.add(_trade(notional=4_000_000, direction=-1), ticker="AAPL", sector="tech")
        positions = book.positions_by_ticker()
        assert len(positions) == 1
        assert positions[0].net_notional == pytest.approx(6_000_000)
        assert positions[0].long_notional == 10_000_000
        assert positions[0].short_notional == 4_000_000
        assert positions[0].trade_count == 2

    def test_multiple_tickers(self):
        book = EquityBook("test")
        book.add(_trade(notional=10_000_000), ticker="AAPL", sector="tech")
        book.add(_trade(notional=5_000_000), ticker="MSFT", sector="tech")
        positions = book.positions_by_ticker()
        assert len(positions) == 2
        tickers = {p.ticker for p in positions}
        assert tickers == {"AAPL", "MSFT"}

    def test_delta_exposure(self):
        book = EquityBook("test")
        book.add(_trade(notional=10_000_000), ticker="AAPL", sector="tech",
                 delta_per_unit=0.5)
        positions = book.positions_by_ticker()
        assert positions[0].delta_exposure == pytest.approx(5_000_000)


# ---- Sector exposures ----

class TestSectorExposures:
    def test_single_sector(self):
        book = EquityBook("test")
        book.add(_trade(notional=10_000_000), ticker="AAPL", sector="tech")
        book.add(_trade(notional=5_000_000), ticker="MSFT", sector="tech")
        exposures = book.exposures_by_sector()
        assert len(exposures) == 1
        assert exposures[0].sector == "tech"
        assert exposures[0].net_notional == 15_000_000
        assert exposures[0].n_names == 2

    def test_multiple_sectors(self):
        book = EquityBook("test")
        book.add(_trade(notional=10_000_000), ticker="AAPL", sector="tech")
        book.add(_trade(notional=8_000_000), ticker="JPM", sector="financial")
        book.add(_trade(notional=5_000_000), ticker="XOM", sector="energy")
        exposures = book.exposures_by_sector()
        assert len(exposures) == 3
        sectors = {e.sector for e in exposures}
        assert sectors == {"tech", "financial", "energy"}

    def test_sector_long_short(self):
        book = EquityBook("test")
        book.add(_trade(notional=10_000_000, direction=1), ticker="AAPL", sector="tech")
        book.add(_trade(notional=4_000_000, direction=-1), ticker="MSFT", sector="tech")
        exposures = book.exposures_by_sector()
        assert exposures[0].net_notional == 6_000_000
        assert exposures[0].long_notional == 10_000_000
        assert exposures[0].short_notional == 4_000_000


# ---- Total exposures ----

class TestExposures:
    def test_net_exposure(self):
        book = EquityBook("test")
        book.add(_trade(notional=10_000_000, direction=1), ticker="AAPL")
        book.add(_trade(notional=4_000_000, direction=-1), ticker="MSFT")
        assert book.net_exposure() == pytest.approx(6_000_000)

    def test_gross_exposure(self):
        book = EquityBook("test")
        book.add(_trade(notional=10_000_000, direction=1), ticker="AAPL")
        book.add(_trade(notional=4_000_000, direction=-1), ticker="MSFT")
        assert book.gross_exposure() == pytest.approx(14_000_000)

    def test_beta_weighted(self):
        book = EquityBook("test")
        book.add(_trade(notional=10_000_000), ticker="AAPL", beta=1.2)
        book.add(_trade(notional=10_000_000), ticker="JPM", beta=0.8)
        # 10M × 1.2 + 10M × 0.8 = 20M
        assert book.beta_weighted_exposure() == pytest.approx(20_000_000)


# ---- Limits ----

class TestEquityLimits:
    def test_per_name_breach(self):
        limits = EquityLimits(max_notional_per_name={"AAPL": 5_000_000})
        book = EquityBook("test", limits=limits)
        book.add(_trade(notional=10_000_000), ticker="AAPL", sector="tech")
        breaches = book.check_limits()
        assert len(breaches) == 1
        assert breaches[0].limit_type == "per_name"

    def test_per_name_ok(self):
        limits = EquityLimits(max_notional_per_name={"AAPL": 20_000_000})
        book = EquityBook("test", limits=limits)
        book.add(_trade(notional=10_000_000), ticker="AAPL", sector="tech")
        breaches = book.check_limits()
        assert len(breaches) == 0

    def test_per_sector_breach(self):
        limits = EquityLimits(max_notional_per_sector={"tech": 12_000_000})
        book = EquityBook("test", limits=limits)
        book.add(_trade(notional=10_000_000), ticker="AAPL", sector="tech")
        book.add(_trade(notional=5_000_000), ticker="MSFT", sector="tech")
        breaches = book.check_limits()
        assert any(b.limit_type == "per_sector" for b in breaches)

    def test_net_exposure_breach(self):
        limits = EquityLimits(max_net_exposure=5_000_000)
        book = EquityBook("test", limits=limits)
        book.add(_trade(notional=10_000_000), ticker="AAPL", sector="tech")
        breaches = book.check_limits()
        assert any(b.limit_type == "net_exposure" for b in breaches)

    def test_gross_exposure_breach(self):
        limits = EquityLimits(max_gross_exposure=10_000_000)
        book = EquityBook("test", limits=limits)
        book.add(_trade(notional=8_000_000, direction=1), ticker="AAPL")
        book.add(_trade(notional=8_000_000, direction=-1), ticker="MSFT")
        # net = 0 but gross = 16M
        breaches = book.check_limits()
        assert any(b.limit_type == "gross_exposure" for b in breaches)

    def test_beta_exposure_breach(self):
        limits = EquityLimits(max_beta_exposure=10_000_000)
        book = EquityBook("test", limits=limits)
        book.add(_trade(notional=10_000_000), ticker="AAPL", beta=1.5)
        # 10M × 1.5 = 15M > 10M
        breaches = book.check_limits()
        assert any(b.limit_type == "beta_exposure" for b in breaches)

    def test_no_breaches(self):
        limits = EquityLimits(
            max_net_exposure=20_000_000,
            max_gross_exposure=50_000_000,
            max_notional_per_name={"AAPL": 20_000_000},
        )
        book = EquityBook("test", limits=limits)
        book.add(_trade(notional=10_000_000), ticker="AAPL", sector="tech")
        breaches = book.check_limits()
        assert len(breaches) == 0
