"""Tests for pricing server + client: end-to-end request/response."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from datetime import date, timedelta

import pytest

from pricebook.pricing_client import PricingClient
from pricebook.pricing_codec import Codec
from pricebook.pricing_schema import (
    PricingRequest, PricingResponse, PricingConfig, QuoteMsg, TradeResult,
    irs_trade, bond_trade, quotes_market_data,
)
from pricebook.pricing_server import PricingServer, _handle_request, _tenor_to_date


REF = date(2026, 4, 28)


# ---- Unit tests (no server) ----

class TestTenorToDate:
    def test_months(self):
        assert _tenor_to_date(REF, "3M") == date(2026, 7, 28)

    def test_years(self):
        assert _tenor_to_date(REF, "5Y") == date(2031, 4, 28)

    def test_days(self):
        assert _tenor_to_date(REF, "30D") == REF + timedelta(days=30)

    def test_weeks(self):
        assert _tenor_to_date(REF, "2W") == REF + timedelta(weeks=2)

    def test_invalid(self):
        with pytest.raises(ValueError, match="Unknown tenor"):
            _tenor_to_date(REF, "5X")


class TestHandleRequest:

    def test_basic_irs(self):
        """Price an IRS from quotes — full pipeline without server."""
        req = PricingRequest(
            valuation_date="2026-04-28",
            trades=[irs_trade("IRS_5Y", "USD", 0.035, "2031-04-28",
                              notional=1_000_000)],
            market_data=quotes_market_data([
                QuoteMsg("deposit_rate", "3M", 0.030).to_dict(),
                QuoteMsg("swap_rate", "1Y", 0.032).to_dict(),
                QuoteMsg("swap_rate", "5Y", 0.035).to_dict(),
                QuoteMsg("swap_rate", "10Y", 0.037).to_dict(),
            ]),
        )
        resp = _handle_request(req)
        assert resp.status == "ok"
        assert len(resp.results) == 1
        r = TradeResult.from_dict(resp.results[0])
        assert math.isfinite(r.pv)
        assert resp.compute_time_ms > 0

    def test_multiple_trades(self):
        req = PricingRequest(
            valuation_date="2026-04-28",
            trades=[
                irs_trade("T1", "USD", 0.035, "2031-04-28"),
                irs_trade("T2", "USD", 0.040, "2036-04-28"),
            ],
            market_data=quotes_market_data([
                QuoteMsg("deposit_rate", "3M", 0.030).to_dict(),
                QuoteMsg("swap_rate", "5Y", 0.035).to_dict(),
                QuoteMsg("swap_rate", "10Y", 0.037).to_dict(),
            ]),
        )
        resp = _handle_request(req)
        assert len(resp.results) == 2
        for r_dict in resp.results:
            r = TradeResult.from_dict(r_dict)
            assert math.isfinite(r.pv)

    def test_bad_instrument_type(self):
        req = PricingRequest(
            valuation_date="2026-04-28",
            trades=[{"trade_id": "BAD", "instrument_type": "nonexistent",
                     "params": {}}],
            market_data=quotes_market_data([
                QuoteMsg("swap_rate", "5Y", 0.035).to_dict(),
            ]),
        )
        resp = _handle_request(req)
        assert resp.status in ("error", "partial")
        assert len(resp.errors) > 0

    def test_empty_market_data(self):
        """Empty quotes → uses flat fallback curve."""
        req = PricingRequest(
            valuation_date="2026-04-28",
            trades=[irs_trade("T1", "USD", 0.035, "2031-04-28")],
            market_data=quotes_market_data([]),
        )
        resp = _handle_request(req)
        assert resp.status == "ok"

    def test_server_version(self):
        req = PricingRequest(valuation_date="2026-04-28")
        resp = _handle_request(req)
        assert resp.server_version != ""


# ---- Integration: server + client ----

def _start_server_thread(server, loop):
    """Run server in a background thread."""
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.start())
    loop.run_until_complete(server._server.serve_forever())


class TestServerClient:

    @pytest.fixture()
    def server_port(self):
        """Start server on a random port, yield port, then stop."""
        import socket as _socket
        # Find free port
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        loop = asyncio.new_event_loop()
        server = PricingServer(host="127.0.0.1", port=port)

        t = threading.Thread(target=_start_server_thread, args=(server, loop), daemon=True)
        t.start()
        time.sleep(0.3)  # let server start

        yield port

        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=2)

    def test_end_to_end(self, server_port):
        """Client → Server → Response for a 5Y IRS."""
        req = PricingRequest(
            valuation_date="2026-04-28",
            trades=[irs_trade("IRS_5Y", "USD", 0.035, "2031-04-28")],
            market_data=quotes_market_data([
                QuoteMsg("deposit_rate", "3M", 0.030).to_dict(),
                QuoteMsg("swap_rate", "1Y", 0.032).to_dict(),
                QuoteMsg("swap_rate", "5Y", 0.035).to_dict(),
            ]),
        )

        with PricingClient("127.0.0.1", server_port) as client:
            resp = client.price(req)

        assert resp.status == "ok"
        assert len(resp.results) == 1
        r = TradeResult.from_dict(resp.results[0])
        assert math.isfinite(r.pv)

    def test_multiple_requests(self, server_port):
        """Send two requests on the same connection."""
        req1 = PricingRequest(
            valuation_date="2026-04-28",
            trades=[irs_trade("T1", "USD", 0.030, "2031-04-28")],
            market_data=quotes_market_data([
                QuoteMsg("swap_rate", "5Y", 0.035).to_dict(),
            ]),
        )
        req2 = PricingRequest(
            valuation_date="2026-04-28",
            trades=[irs_trade("T2", "USD", 0.040, "2031-04-28")],
            market_data=quotes_market_data([
                QuoteMsg("swap_rate", "5Y", 0.035).to_dict(),
            ]),
        )

        with PricingClient("127.0.0.1", server_port) as client:
            resp1 = client.price(req1)
            resp2 = client.price(req2)

        assert resp1.request_id != resp2.request_id
        r1 = TradeResult.from_dict(resp1.results[0])
        r2 = TradeResult.from_dict(resp2.results[0])
        # Different fixed rates → different PVs
        assert r1.pv != r2.pv
