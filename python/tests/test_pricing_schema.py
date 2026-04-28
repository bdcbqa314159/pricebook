"""Tests for pricing service message schema."""

from __future__ import annotations

import json

import pytest

from pricebook.pricing_schema import (
    PricingRequest, PricingResponse,
    MarketDataEnvelope, TradeEnvelope, TradeResult,
    PricingConfig, QuoteMsg, CurveMsg, CSAMsg,
    irs_trade, bond_trade, quotes_market_data, curves_market_data,
)


# ---- QuoteMsg ----

class TestQuoteMsg:
    def test_round_trip(self):
        q = QuoteMsg(type="swap_rate", tenor="5Y", value=0.035, currency="USD")
        d = q.to_dict()
        q2 = QuoteMsg.from_dict(d)
        assert q2.type == q.type
        assert q2.value == q.value
        assert q2.to_dict() == d

    def test_defaults(self):
        q = QuoteMsg.from_dict({"type": "deposit_rate", "tenor": "3M", "value": 0.03})
        assert q.currency == "USD"
        assert q.name == ""


# ---- CurveMsg ----

class TestCurveMsg:
    def test_round_trip(self):
        c = CurveMsg(name="USD_OIS", dates=["2027-04-27"], values=[0.97])
        d = c.to_dict()
        c2 = CurveMsg.from_dict(d)
        assert c2.name == "USD_OIS"
        assert c2.values == [0.97]
        assert c2.to_dict() == d


# ---- MarketDataEnvelope ----

class TestMarketDataEnvelope:
    def test_quotes_mode(self):
        md = MarketDataEnvelope(mode="quotes", quotes=[
            QuoteMsg("swap_rate", "5Y", 0.035).to_dict(),
        ])
        d = md.to_dict()
        assert d["mode"] == "quotes"
        md2 = MarketDataEnvelope.from_dict(d)
        assert len(md2.quotes) == 1

    def test_curves_mode(self):
        md = MarketDataEnvelope(mode="curves", curves=[
            CurveMsg("USD_OIS", ["2027-04-27"], [0.97]).to_dict(),
        ])
        d = md.to_dict()
        md2 = MarketDataEnvelope.from_dict(d)
        assert md2.mode == "curves"

    def test_with_fixings(self):
        md = MarketDataEnvelope(
            mode="quotes", quotes=[],
            fixings={"SOFR": {"2026-04-27": 0.043}},
        )
        d = md.to_dict()
        md2 = MarketDataEnvelope.from_dict(d)
        assert md2.fixings["SOFR"]["2026-04-27"] == 0.043

    def test_empty_fields_omitted(self):
        md = MarketDataEnvelope(mode="quotes")
        d = md.to_dict()
        assert "curves" not in d
        assert "fixings" not in d


# ---- TradeEnvelope ----

class TestTradeEnvelope:
    def test_round_trip(self):
        t = TradeEnvelope(trade_id="T1", instrument_type="irs",
                          params={"fixed_rate": 0.035, "maturity": "2031-04-27"})
        d = t.to_dict()
        t2 = TradeEnvelope.from_dict(d)
        assert t2.trade_id == "T1"
        assert t2.params["fixed_rate"] == 0.035

    def test_with_csa(self):
        csa = CSAMsg(currency="EUR", threshold=1_000_000).to_dict()
        t = TradeEnvelope(trade_id="T1", instrument_type="irs", csa=csa)
        d = t.to_dict()
        assert "csa" in d
        t2 = TradeEnvelope.from_dict(d)
        assert t2.csa["currency"] == "EUR"

    def test_no_csa_omitted(self):
        t = TradeEnvelope(trade_id="T1", instrument_type="irs")
        d = t.to_dict()
        assert "csa" not in d


# ---- PricingConfig ----

class TestPricingConfig:
    def test_defaults(self):
        c = PricingConfig()
        assert c.model == "black"
        assert c.mc_paths == 10_000
        assert c.measures == ["pv"]

    def test_round_trip(self):
        c = PricingConfig(model="sabr", mc_paths=50_000,
                          compute_greeks=True, measures=["pv", "delta", "vega"])
        d = c.to_dict()
        c2 = PricingConfig.from_dict(d)
        assert c2.model == "sabr"
        assert c2.compute_greeks is True
        assert c2.measures == ["pv", "delta", "vega"]


# ---- PricingRequest ----

class TestPricingRequest:
    def test_round_trip(self):
        req = PricingRequest(
            valuation_date="2026-04-28",
            trades=[irs_trade("T1", "USD", 0.035, "2031-04-28")],
            market_data=quotes_market_data([
                QuoteMsg("swap_rate", "5Y", 0.035).to_dict(),
            ]),
            config=PricingConfig(compute_greeks=True).to_dict(),
            metadata={"user": "desk_a"},
        )
        d = req.to_dict()
        req2 = PricingRequest.from_dict(d)
        assert req2.request_id == req.request_id
        assert req2.valuation_date == "2026-04-28"
        assert len(req2.trades) == 1
        assert req2.metadata["user"] == "desk_a"

    def test_json_serializable(self):
        req = PricingRequest(
            valuation_date="2026-04-28",
            trades=[irs_trade("T1", "USD", 0.035, "2031-04-28")],
        )
        s = json.dumps(req.to_dict())
        d = json.loads(s)
        req2 = PricingRequest.from_dict(d)
        assert req2.valuation_date == req.valuation_date

    def test_auto_uuid(self):
        req = PricingRequest(valuation_date="2026-04-28")
        assert len(req.request_id) == 36  # UUID format

    def test_get_helpers(self):
        req = PricingRequest(
            valuation_date="2026-04-28",
            trades=[irs_trade("T1", "USD", 0.035, "2031-04-28")],
            market_data=quotes_market_data([]),
            config=PricingConfig(model="sabr").to_dict(),
        )
        assert req.get_config().model == "sabr"
        assert req.get_market_data().mode == "quotes"
        assert req.get_trades()[0].trade_id == "T1"

    def test_missing_valuation_date_raises(self):
        with pytest.raises(KeyError):
            PricingRequest.from_dict({"trades": []})


# ---- PricingResponse ----

class TestPricingResponse:
    def test_round_trip(self):
        resp = PricingResponse(
            request_id="abc-123",
            status="ok",
            results=[TradeResult(trade_id="T1", pv=12345.67).to_dict()],
            compute_time_ms=42.5,
            server_version="0.389.0",
        )
        d = resp.to_dict()
        resp2 = PricingResponse.from_dict(d)
        assert resp2.request_id == "abc-123"
        assert resp2.compute_time_ms == 42.5
        r = resp2.get_results()
        assert r[0].pv == 12345.67

    def test_error_response(self):
        resp = PricingResponse(
            request_id="abc-123",
            status="error",
            errors=[{"trade_id": "T1", "code": "UNKNOWN_INSTRUMENT", "message": "bad type"}],
        )
        d = resp.to_dict()
        resp2 = PricingResponse.from_dict(d)
        assert resp2.status == "error"
        assert resp2.errors[0]["code"] == "UNKNOWN_INSTRUMENT"


# ---- TradeResult ----

class TestTradeResult:
    def test_with_greeks(self):
        r = TradeResult(trade_id="T1", pv=100_000, currency="EUR",
                        greeks={"delta": 5000, "gamma": 50, "vega": 3000})
        d = r.to_dict()
        assert d["greeks"]["delta"] == 5000

    def test_empty_fields_omitted(self):
        r = TradeResult(trade_id="T1", pv=100)
        d = r.to_dict()
        assert "greeks" not in d
        assert "error_message" not in d


# ---- Convenience builders ----

class TestBuilders:
    def test_irs_trade(self):
        d = irs_trade("T1", "USD", 0.035, "2031-04-28", notional=10_000_000)
        t = TradeEnvelope.from_dict(d)
        assert t.instrument_type == "irs"
        assert t.params["fixed_rate"] == 0.035

    def test_bond_trade(self):
        d = bond_trade("B1", 0.05, "2036-04-28", face_value=1000)
        t = TradeEnvelope.from_dict(d)
        assert t.instrument_type == "bond"
        assert t.params["coupon_rate"] == 0.05

    def test_quotes_market_data(self):
        d = quotes_market_data([
            QuoteMsg("deposit_rate", "3M", 0.03).to_dict(),
            QuoteMsg("swap_rate", "5Y", 0.035).to_dict(),
        ])
        md = MarketDataEnvelope.from_dict(d)
        assert md.mode == "quotes"
        assert len(md.quotes) == 2

    def test_full_request(self):
        """Build a complete request using convenience functions."""
        req = PricingRequest(
            valuation_date="2026-04-28",
            trades=[
                irs_trade("IRS_5Y", "USD", 0.035, "2031-04-28"),
                bond_trade("BOND_10Y", 0.05, "2036-04-28"),
            ],
            market_data=quotes_market_data([
                QuoteMsg("deposit_rate", "3M", 0.030).to_dict(),
                QuoteMsg("swap_rate", "1Y", 0.032).to_dict(),
                QuoteMsg("swap_rate", "5Y", 0.035).to_dict(),
                QuoteMsg("swap_rate", "10Y", 0.037).to_dict(),
            ]),
            config=PricingConfig(compute_greeks=True, measures=["pv", "dv01"]).to_dict(),
        )
        s = json.dumps(req.to_dict())
        assert len(s) < 2000  # compact
        req2 = PricingRequest.from_dict(json.loads(s))
        assert len(req2.trades) == 2
