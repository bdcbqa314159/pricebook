"""Tests for exotic options book."""

import pytest
from datetime import date

from pricebook.exotic_book import (
    ExoticBook,
    ExoticEntry,
    ExoticHedgeResult,
    ModelComparison,
    hedge_exotic_book,
    model_risk_comparison,
)


def _barrier():
    return ExoticEntry("b1", "barrier", "equity", "AAPL", date(2024, 6, 15),
                       1e6, delta=400, gamma=30, vega=800, theta=-15)

def _digital():
    return ExoticEntry("d1", "digital", "fx", "EUR/USD", date(2024, 6, 15),
                       5e5, delta=200, gamma=50, vega=300, theta=-10)

def _asian():
    return ExoticEntry("a1", "asian", "commodity", "WTI", date(2024, 9, 15),
                       2e6, delta=600, gamma=10, vega=500, theta=-20)


# ---- Step 1: position management ----

class TestExoticBook:
    def test_empty(self):
        book = ExoticBook("test")
        assert len(book) == 0

    def test_aggregate_by_type(self):
        """Step 1 test: aggregate exotic positions by type and underlying."""
        book = ExoticBook("test")
        book.add(_barrier())
        book.add(_digital())
        book.add(_asian())
        by_type = book.by_type()
        assert len(by_type) == 3
        assert {t.exotic_type for t in by_type} == {"barrier", "digital", "asian"}

    def test_aggregate_by_underlying(self):
        book = ExoticBook("test")
        book.add(_barrier())
        book.add(ExoticEntry("b2", "barrier", "equity", "AAPL",
                             delta=100, vega=200, gamma=10))
        by_und = book.by_underlying()
        aapl = next(u for u in by_und if u.underlying == "AAPL")
        assert aapl.net_delta == pytest.approx(500)
        assert aapl.net_vega == pytest.approx(1000)
        assert aapl.n_positions == 2

    def test_total_delta_vega(self):
        book = ExoticBook("test")
        book.add(_barrier())
        book.add(_digital())
        assert book.total_delta() == pytest.approx(600)
        assert book.total_vega() == pytest.approx(1100)

    def test_multiple_types_same_underlying(self):
        book = ExoticBook("test")
        book.add(_barrier())
        book.add(ExoticEntry("d2", "digital", "equity", "AAPL",
                             delta=100, vega=50))
        by_und = book.by_underlying()
        aapl = next(u for u in by_und if u.underlying == "AAPL")
        assert aapl.n_positions == 2


# ---- Step 2: Greeks + hedging ----

class TestModelRiskComparison:
    def test_comparison(self):
        greeks = {
            "black_scholes": {"delta": 0.55, "gamma": 0.02, "vega": 120, "theta": -5},
            "local_vol": {"delta": 0.52, "gamma": 0.025, "vega": 115, "theta": -4.5},
            "slv": {"delta": 0.53, "gamma": 0.022, "vega": 118, "theta": -4.8},
        }
        result = model_risk_comparison(greeks, "barrier_1")
        assert result.max_delta_diff == pytest.approx(0.03)
        assert result.max_vega_diff == pytest.approx(5.0)
        assert result.trade_id == "barrier_1"

    def test_single_model(self):
        greeks = {"bs": {"delta": 0.5, "vega": 100}}
        result = model_risk_comparison(greeks)
        assert result.max_delta_diff == 0.0
        assert result.max_vega_diff == 0.0


class TestHedgeExoticBook:
    def test_hedge_flattens(self):
        """Step 2 test: hedge flattens delta and vega exposure."""
        book = ExoticBook("test")
        book.add(_barrier())
        book.add(_digital())
        result = hedge_exotic_book(book, vanilla_delta_per_unit=1.0,
                                    vanilla_vega_per_unit=100.0)
        assert result.residual_delta == pytest.approx(0.0)
        assert result.residual_vega == pytest.approx(0.0)

    def test_hedge_quantities(self):
        book = ExoticBook("test")
        book.add(ExoticEntry("t1", "barrier", "equity", "AAPL",
                             delta=1000, vega=500))
        result = hedge_exotic_book(book, 1.0, 100.0)
        assert result.hedge_delta_qty == pytest.approx(-1000)
        assert result.hedge_vega_qty == pytest.approx(-5)

    def test_empty_book(self):
        result = hedge_exotic_book(ExoticBook("test"))
        assert result.hedge_delta_qty == 0.0
        assert result.hedge_vega_qty == 0.0
