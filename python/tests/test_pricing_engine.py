"""Tests for pricing engine: JSON in → price/risk out."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta

import pytest

from pricebook.discount_curve import DiscountCurve
from pricebook.pricing_engine import price_from_json, price_from_dict
from pricebook.serialization import discount_curve_to_dict, survival_curve_to_dict


REF = date(2026, 4, 28)


def _flat_curve_dict(rate=0.03):
    curve = DiscountCurve.flat(REF, rate)
    return discount_curve_to_dict(curve)


# ---- Basic pricing ----

class TestBasicPricing:

    def test_irs_from_flat_curve(self):
        """Price an IRS from a flat discount curve."""
        request = {
            "valuation_date": "2026-04-28",
            "market_data": {"discount_curve": _flat_curve_dict()},
            "trades": [{"type": "irs", "params": {
                "start": "2026-04-28", "end": "2031-04-28",
                "fixed_rate": 0.035, "notional": 10_000_000,
            }}],
        }
        result = price_from_dict(request)
        assert result["status"] == "ok"
        assert len(result["results"]) == 1
        r = result["results"][0]
        assert r["instrument_type"] == "irs"
        assert math.isfinite(r["pv"])

    def test_irs_json_round_trip(self):
        """price_from_json returns valid JSON."""
        req = json.dumps({
            "valuation_date": "2026-04-28",
            "market_data": {"discount_curve": _flat_curve_dict()},
            "trades": [{"type": "irs", "params": {
                "start": "2026-04-28", "end": "2031-04-28", "fixed_rate": 0.035,
            }}],
        })
        resp = price_from_json(req)
        result = json.loads(resp)
        assert result["status"] == "ok"
        assert math.isfinite(result["results"][0]["pv"])

    def test_bond(self):
        request = {
            "valuation_date": "2026-04-28",
            "market_data": {"discount_curve": _flat_curve_dict()},
            "trades": [{"type": "bond", "params": {
                "issue_date": "2026-04-28", "maturity": "2036-04-28",
                "coupon_rate": 0.05, "face_value": 100,
            }}],
        }
        result = price_from_dict(request)
        assert result["status"] == "ok"
        assert result["results"][0]["pv"] > 0

    def test_cds(self):
        request = {
            "valuation_date": "2026-04-28",
            "market_data": {"discount_curve": _flat_curve_dict()},
            "trades": [{"type": "cds", "params": {
                "start": "2026-04-28", "end": "2031-04-28",
                "spread": 0.01, "notional": 10_000_000,
            }}],
        }
        result = price_from_dict(request)
        assert result["status"] == "ok"

    def test_deposit(self):
        request = {
            "valuation_date": "2026-04-28",
            "market_data": {"discount_curve": _flat_curve_dict()},
            "trades": [{"type": "deposit", "params": {
                "start": "2026-04-28", "end": "2026-07-28", "rate": 0.03,
            }}],
        }
        result = price_from_dict(request)
        assert result["status"] == "ok"

    def test_term_loan(self):
        request = {
            "valuation_date": "2026-04-28",
            "market_data": {"discount_curve": _flat_curve_dict()},
            "trades": [{"type": "term_loan", "params": {
                "start": "2026-04-28", "end": "2031-04-28",
                "spread": 0.03, "notional": 10_000_000,
            }}],
        }
        result = price_from_dict(request)
        assert result["status"] == "ok"
        assert result["results"][0]["pv"] > 0

    def test_cln(self):
        request = {
            "valuation_date": "2026-04-28",
            "market_data": {"discount_curve": _flat_curve_dict()},
            "trades": [{"type": "cln", "params": {
                "start": "2026-04-28", "end": "2031-04-28",
                "coupon_rate": 0.06, "notional": 1_000_000,
            }}],
        }
        result = price_from_dict(request)
        assert result["status"] == "ok"


# ---- Multiple trades ----

class TestMultipleTrades:

    def test_mixed_portfolio(self):
        """Price IRS + bond + CDS in one request."""
        request = {
            "valuation_date": "2026-04-28",
            "market_data": {"discount_curve": _flat_curve_dict()},
            "trades": [
                {"type": "irs", "params": {"start": "2026-04-28", "end": "2031-04-28", "fixed_rate": 0.035}},
                {"type": "bond", "params": {"issue_date": "2026-04-28", "maturity": "2036-04-28", "coupon_rate": 0.05}},
                {"type": "cds", "params": {"start": "2026-04-28", "end": "2031-04-28", "spread": 0.01}},
            ],
        }
        result = price_from_dict(request)
        assert result["status"] == "ok"
        assert result["n_trades"] == 3
        assert len(result["results"]) == 3
        for r in result["results"]:
            assert math.isfinite(r["pv"])


# ---- TRS ----

class TestTRSPricing:

    def test_equity_trs(self):
        request = {
            "valuation_date": "2026-04-28",
            "market_data": {"discount_curve": _flat_curve_dict()},
            "trades": [{"type": "trs", "params": {
                "underlying": 100.0, "notional": 10_000_000,
                "start": "2026-04-28", "end": "2027-04-28",
            }}],
        }
        result = price_from_dict(request)
        assert result["status"] == "ok"
        assert math.isfinite(result["results"][0]["pv"])


# ---- Greeks ----

class TestGreeks:

    def test_dv01(self):
        request = {
            "valuation_date": "2026-04-28",
            "market_data": {"discount_curve": _flat_curve_dict()},
            "trades": [{"type": "irs", "params": {
                "start": "2026-04-28", "end": "2031-04-28", "fixed_rate": 0.035,
            }}],
            "config": {"compute_greeks": True, "measures": ["pv", "dv01"]},
        }
        result = price_from_dict(request)
        assert result["status"] == "ok"
        r = result["results"][0]
        assert "greeks" in r
        assert "dv01" in r["greeks"]
        assert math.isfinite(r["greeks"]["dv01"])

    def test_no_greeks_by_default(self):
        request = {
            "valuation_date": "2026-04-28",
            "market_data": {"discount_curve": _flat_curve_dict()},
            "trades": [{"type": "irs", "params": {
                "start": "2026-04-28", "end": "2031-04-28", "fixed_rate": 0.035,
            }}],
        }
        result = price_from_dict(request)
        r = result["results"][0]
        assert "greeks" not in r


# ---- Market data modes ----

class TestMarketDataModes:

    def test_flat_rate_shortcut(self):
        """Use flat_rate instead of full curve."""
        request = {
            "valuation_date": "2026-04-28",
            "market_data": {"flat_rate": 0.04},
            "trades": [{"type": "irs", "params": {
                "start": "2026-04-28", "end": "2031-04-28", "fixed_rate": 0.04,
            }}],
        }
        result = price_from_dict(request)
        assert result["status"] == "ok"
        # At-par swap: PV should be near zero
        assert abs(result["results"][0]["pv"]) < 50_000

    def test_empty_market_data(self):
        """Missing market_data → fallback flat curve."""
        request = {
            "valuation_date": "2026-04-28",
            "trades": [{"type": "irs", "params": {
                "start": "2026-04-28", "end": "2031-04-28", "fixed_rate": 0.035,
            }}],
        }
        result = price_from_dict(request)
        assert result["status"] == "ok"

    def test_with_vol_surface(self):
        """FlatVol in market data."""
        request = {
            "valuation_date": "2026-04-28",
            "market_data": {
                "discount_curve": _flat_curve_dict(),
                "vol_surfaces": {"equity": {"type": "FlatVol", "vol": 0.20}},
            },
            "trades": [{"type": "irs", "params": {
                "start": "2026-04-28", "end": "2031-04-28", "fixed_rate": 0.035,
            }}],
        }
        result = price_from_dict(request)
        assert result["status"] == "ok"

    def test_vol_as_scalar(self):
        """Vol surface as plain float → auto-wrap in FlatVol."""
        request = {
            "valuation_date": "2026-04-28",
            "market_data": {
                "discount_curve": _flat_curve_dict(),
                "vol_surfaces": {"equity": 0.25},
            },
            "trades": [{"type": "irs", "params": {
                "start": "2026-04-28", "end": "2031-04-28", "fixed_rate": 0.035,
            }}],
        }
        result = price_from_dict(request)
        assert result["status"] == "ok"


# ---- Tenor shorthand ----

class TestTenorShorthand:

    def test_maturity_as_tenor(self):
        """Use 'maturity': '5Y' instead of explicit end date."""
        request = {
            "valuation_date": "2026-04-28",
            "market_data": {"flat_rate": 0.03},
            "trades": [{"type": "irs", "params": {
                "fixed_rate": 0.035, "maturity": "5Y",
            }}],
        }
        result = price_from_dict(request)
        assert result["status"] == "ok"
        assert math.isfinite(result["results"][0]["pv"])


# ---- Error handling ----

class TestErrorHandling:

    def test_bad_instrument_type(self):
        request = {
            "valuation_date": "2026-04-28",
            "trades": [{"type": "nonexistent", "params": {}}],
        }
        result = price_from_dict(request)
        assert result["status"] in ("error", "partial")
        assert result["results"][0]["status"] == "error"

    def test_missing_valuation_date(self):
        request = {"trades": []}
        result = price_from_dict(request)
        assert result["status"] == "error"

    def test_partial_failure(self):
        """One good trade + one bad → partial status."""
        request = {
            "valuation_date": "2026-04-28",
            "market_data": {"flat_rate": 0.03},
            "trades": [
                {"type": "irs", "params": {"start": "2026-04-28", "end": "2031-04-28", "fixed_rate": 0.035}},
                {"type": "nonexistent", "params": {}},
            ],
        }
        result = price_from_dict(request)
        assert result["status"] == "partial"
        assert result["results"][0]["status"] == "ok"
        assert result["results"][1]["status"] == "error"

    def test_compute_time(self):
        request = {
            "valuation_date": "2026-04-28",
            "market_data": {"flat_rate": 0.03},
            "trades": [{"type": "irs", "params": {"start": "2026-04-28", "end": "2031-04-28", "fixed_rate": 0.035}}],
        }
        result = price_from_dict(request)
        assert result["compute_time_ms"] > 0


# ---- PV matches direct pricing ----

class TestPVAccuracy:

    def test_irs_matches_direct(self):
        """PV from engine matches direct InterestRateSwap.pv()."""
        from pricebook.swap import InterestRateSwap

        end = REF + timedelta(days=1825)  # 2031-04-27
        curve = DiscountCurve.flat(REF, 0.03)
        irs = InterestRateSwap(REF, end, fixed_rate=0.035, notional=10_000_000)
        direct_pv = irs.pv(curve)

        request = {
            "valuation_date": REF.isoformat(),
            "market_data": {"flat_rate": 0.03},
            "trades": [{"type": "irs", "params": {
                "start": REF.isoformat(), "end": end.isoformat(),
                "fixed_rate": 0.035, "notional": 10_000_000,
            }}],
        }
        result = price_from_dict(request)
        engine_pv = result["results"][0]["pv"]
        assert engine_pv == pytest.approx(direct_pv, abs=1e-8)
