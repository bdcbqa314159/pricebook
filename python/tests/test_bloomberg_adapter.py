"""Tests for Bloomberg adapter: ticker parsing, field mapping, request building."""

from __future__ import annotations

import math

import pytest

from pricebook.bloomberg_adapter import (
    parse_bbg_ticker,
    BloombergQuoteAdapter,
    BloombergResultFormatter,
    irs_request_from_bbg,
    bond_request_from_bbg,
)
from pricebook.pricing_schema import (
    PricingRequest, PricingResponse, TradeResult, QuoteMsg,
    MarketDataEnvelope,
)
from pricebook.pricing_server import _handle_request


# ---- Ticker parsing ----

class TestTickerParsing:

    def test_usd_swap(self):
        q = parse_bbg_ticker("USSW5 Curncy")
        assert q is not None
        assert q.type == "swap_rate"
        assert q.tenor == "5Y"
        assert q.currency == "USD"

    def test_eur_swap(self):
        q = parse_bbg_ticker("EUSA10 Curncy")
        assert q is not None
        assert q.tenor == "10Y"
        assert q.currency == "EUR"

    def test_gbp_swap(self):
        q = parse_bbg_ticker("BPSW3 Curncy")
        assert q is not None
        assert q.currency == "GBP"
        assert q.tenor == "3Y"

    def test_usd_deposit(self):
        q = parse_bbg_ticker("US0003M Index")
        assert q is not None
        assert q.type == "deposit_rate"
        assert q.tenor == "3M"
        assert q.currency == "USD"

    def test_eur_deposit(self):
        q = parse_bbg_ticker("EU0006M Index")
        assert q is not None
        assert q.tenor == "6M"
        assert q.currency == "EUR"

    def test_unknown_ticker(self):
        q = parse_bbg_ticker("AAPL US Equity")
        assert q is None

    def test_case_insensitive(self):
        q = parse_bbg_ticker("ussw5 curncy")
        assert q is not None
        assert q.currency == "USD"

    def test_no_suffix(self):
        q = parse_bbg_ticker("USSW5")
        assert q is not None
        assert q.tenor == "5Y"


# ---- BloombergQuoteAdapter ----

class TestBloombergQuoteAdapter:

    def test_from_bbg_snapshot(self):
        adapter = BloombergQuoteAdapter()
        md = adapter.from_bbg_snapshot({
            "US0003M Index": 0.030,
            "USSW1 Curncy": 0.032,
            "USSW5 Curncy": 0.035,
            "USSW10 Curncy": 0.037,
        })
        assert md.mode == "quotes"
        assert len(md.quotes) == 4

    def test_skips_unknown(self):
        adapter = BloombergQuoteAdapter()
        md = adapter.from_bbg_snapshot({
            "USSW5 Curncy": 0.035,
            "AAPL US Equity": 150.0,
        })
        assert len(md.quotes) == 1

    def test_to_pricing_request(self):
        adapter = BloombergQuoteAdapter()
        from pricebook.pricing_schema import irs_trade
        req = adapter.to_pricing_request(
            valuation_date="2026-04-28",
            market_data={"USSW5 Curncy": 0.035, "US0003M Index": 0.030},
            trades=[irs_trade("T1", "USD", 0.035, "2031-04-28")],
        )
        assert req.valuation_date == "2026-04-28"
        assert len(req.trades) == 1


# ---- BloombergResultFormatter ----

class TestBloombergResultFormatter:

    def test_basic_formatting(self):
        fmt = BloombergResultFormatter()
        result = TradeResult(trade_id="T1", pv=12345.67, currency="USD",
                             greeks={"delta": 5000, "dv01": 450})
        fields = fmt.to_bbg_fields(result)
        assert fields["PX_LAST"] == 12345.67
        assert fields["CRNCY"] == "USD"
        assert fields["DELTA"] == 5000
        assert fields["DUR_ADJ_MID"] == 450

    def test_format_multiple(self):
        fmt = BloombergResultFormatter()
        results = [
            TradeResult(trade_id="T1", pv=100),
            TradeResult(trade_id="T2", pv=200),
        ]
        formatted = fmt.format_results(results)
        assert "T1" in formatted
        assert "T2" in formatted
        assert formatted["T1"]["PX_LAST"] == 100


# ---- Request templates ----

class TestRequestTemplates:

    def test_irs_request(self):
        req = irs_request_from_bbg(
            valuation_date="2026-04-28",
            currency="USD",
            fixed_rate=0.035,
            maturity_tenor="5Y",
            market_data={"USSW5 Curncy": 0.035, "US0003M Index": 0.030},
        )
        assert req.valuation_date == "2026-04-28"
        assert len(req.trades) == 1
        md = req.get_market_data()
        assert len(md.quotes) == 2

    def test_bond_request(self):
        req = bond_request_from_bbg(
            valuation_date="2026-04-28",
            coupon_rate=0.05,
            maturity="2036-04-28",
        )
        assert len(req.trades) == 1


# ---- End-to-end: BBG → Request → Price → BBG ----

class TestEndToEnd:

    def test_bbg_to_pv(self):
        """Full pipeline: BBG data → PricingRequest → price → BBG fields."""
        req = irs_request_from_bbg(
            valuation_date="2026-04-28",
            currency="USD",
            fixed_rate=0.035,
            maturity_tenor="5Y",
            notional=10_000_000,
            market_data={
                "US0003M Index": 0.030,
                "USSW1 Curncy": 0.032,
                "USSW5 Curncy": 0.035,
                "USSW10 Curncy": 0.037,
            },
        )
        resp = _handle_request(req)
        assert resp.status == "ok"

        results = resp.get_results()
        assert len(results) == 1
        assert math.isfinite(results[0].pv)

        # Format for Bloomberg
        fmt = BloombergResultFormatter()
        fields = fmt.to_bbg_fields(results[0])
        assert "PX_LAST" in fields
        assert math.isfinite(fields["PX_LAST"])
