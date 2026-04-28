"""Pricing server: asyncio TCP server accepting framed pricing requests.

    from pricebook.pricing_server import PricingServer

    server = PricingServer(host="0.0.0.0", port=9090)
    asyncio.run(server.run())

The server accepts framed messages (see pricing_codec.py), dispatches to
the pricing engine, and returns framed responses. CPU-bound pricing runs
in a thread pool to avoid blocking the event loop.

For testing, use PricingClient from pricing_client.py.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

from pricebook.pricing_codec import Codec, CodecFormat, Compression, HEADER_SIZE
from pricebook.pricing_schema import (
    PricingRequest, PricingResponse, TradeResult,
    MarketDataEnvelope, TradeEnvelope, PricingConfig, QuoteMsg,
)

logger = logging.getLogger(__name__)

# Version reported to clients
SERVER_VERSION = "0.391.0"


def _handle_request(request: PricingRequest) -> PricingResponse:
    """Process a pricing request (runs in thread pool).

    1. Build curves from market data
    2. Price each trade
    3. Return results
    """
    t0 = time.monotonic()
    results = []
    errors = []

    try:
        md = request.get_market_data()
        config = request.get_config()
        trades = request.get_trades()
        val_date = date.fromisoformat(request.valuation_date)

        # Build pricing context from market data
        ctx = _build_context(val_date, md)

        # Price each trade
        for te in trades:
            try:
                result = _price_trade(te, ctx, config)
                results.append(result.to_dict())
            except Exception as e:
                errors.append({
                    "trade_id": te.trade_id,
                    "code": type(e).__name__,
                    "message": str(e),
                })
                results.append(TradeResult(
                    trade_id=te.trade_id, status="error",
                    error_message=str(e),
                ).to_dict())

    except Exception as e:
        errors.append({"trade_id": "", "code": type(e).__name__, "message": str(e)})

    elapsed = (time.monotonic() - t0) * 1000

    status = "ok" if not errors else ("partial" if results else "error")

    return PricingResponse(
        request_id=request.request_id,
        status=status,
        results=results,
        compute_time_ms=elapsed,
        server_version=SERVER_VERSION,
        errors=errors,
    )


def _build_context(val_date: date, md: MarketDataEnvelope):
    """Build a PricingContext from market data."""
    from pricebook.pricing_context import PricingContext
    from pricebook.discount_curve import DiscountCurve

    if md.mode == "quotes":
        # Bootstrap from quotes
        from pricebook.bootstrap import bootstrap

        deposits = []
        swaps = []
        for q in md.quotes:
            quote = QuoteMsg.from_dict(q) if isinstance(q, dict) else q
            mat_date = _tenor_to_date(val_date, quote.tenor)
            if quote.type == "deposit_rate":
                deposits.append((mat_date, quote.value))
            elif quote.type == "swap_rate":
                swaps.append((mat_date, quote.value))

        if deposits or swaps:
            curve = bootstrap(val_date, deposits, swaps)
        else:
            curve = DiscountCurve.flat(val_date, 0.03)

        return PricingContext(valuation_date=val_date, discount_curve=curve)

    elif md.mode == "curves":
        # Use pre-built curves
        from pricebook.pricing_schema import CurveMsg

        curves = {}
        for cd in md.curves:
            cm = CurveMsg.from_dict(cd) if isinstance(cd, dict) else cd
            pillar_dates = [date.fromisoformat(d) for d in cm.dates]
            curve = DiscountCurve(val_date, pillar_dates, cm.values)
            curves[cm.name] = curve

        disc = next(iter(curves.values())) if curves else DiscountCurve.flat(val_date, 0.03)
        return PricingContext(valuation_date=val_date, discount_curve=disc)

    return PricingContext(valuation_date=val_date,
                          discount_curve=DiscountCurve.flat(val_date, 0.03))


def _price_trade(te: TradeEnvelope, ctx, config: PricingConfig) -> TradeResult:
    """Price a single trade."""
    val_date = ctx.valuation_date
    params = te.params.copy()

    # Resolve date fields: "maturity" → actual date, add start if missing
    if "maturity" in params:
        mat = params.pop("maturity")
        if isinstance(mat, str):
            if len(mat) <= 4:  # tenor like "5Y"
                mat = _tenor_to_date(val_date, mat)
            else:
                mat = date.fromisoformat(mat)
        params["end"] = mat
    if "start" not in params and "end" in params:
        params["start"] = val_date

    # Remove non-constructor fields
    params.pop("currency", None)

    # Build instrument via serialization registry
    from pricebook.serialization import instrument_from_dict
    inst_dict = {"type": te.instrument_type, "params": params}
    instrument = instrument_from_dict(inst_dict)

    # Price
    if hasattr(instrument, "pv_ctx"):
        pv = instrument.pv_ctx(ctx)
    elif hasattr(instrument, "pv"):
        pv = instrument.pv(ctx.discount_curve)
    else:
        raise ValueError(f"Instrument {te.instrument_type} has no pricing method")

    pv *= te.notional_scale

    # Greeks if requested
    greeks = {}
    if config.compute_greeks and hasattr(instrument, "dv01"):
        try:
            greeks["dv01"] = instrument.dv01(ctx.discount_curve)
        except Exception:
            pass

    return TradeResult(
        trade_id=te.trade_id,
        pv=pv,
        greeks=greeks,
    )


def _tenor_to_date(ref: date, tenor: str) -> date:
    """Convert tenor string to maturity date: '3M' → +3 months, '5Y' → +5 years."""
    from dateutil.relativedelta import relativedelta

    tenor = tenor.strip().upper()
    if tenor.endswith("D"):
        from datetime import timedelta
        return ref + timedelta(days=int(tenor[:-1]))
    elif tenor.endswith("W"):
        from datetime import timedelta
        return ref + timedelta(weeks=int(tenor[:-1]))
    elif tenor.endswith("M"):
        return ref + relativedelta(months=int(tenor[:-1]))
    elif tenor.endswith("Y"):
        return ref + relativedelta(years=int(tenor[:-1]))
    raise ValueError(f"Unknown tenor format: {tenor}")


class PricingServer:
    """Asyncio TCP server for pricing requests.

    Accepts framed messages (see Codec), dispatches to _handle_request
    in a thread pool, returns framed responses.

    Args:
        host: bind address (default "127.0.0.1").
        port: listen port (default 9090).
        codec: Codec instance for encode/decode.
        max_workers: thread pool size for CPU-bound pricing.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9090,
        codec: Codec | None = None,
        max_workers: int = 4,
    ):
        self.host = host
        self.port = port
        self.codec = codec or Codec()
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._server: asyncio.Server | None = None

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        """Handle one client connection (may send multiple requests)."""
        addr = writer.get_extra_info("peername")
        logger.info(f"Connection from {addr}")

        try:
            while True:
                # Read frame length (first 4 bytes)
                length_bytes = await reader.readexactly(4)
                total_len = Codec.read_frame_length(length_bytes)

                # Read rest of frame
                remaining = await reader.readexactly(total_len - 4)
                frame = length_bytes + remaining

                # Decode
                msg = self.codec.decode(frame)
                request = PricingRequest.from_dict(msg)

                # Process in thread pool (CPU-bound)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    self._pool, _handle_request, request,
                )

                # Encode and send response
                resp_bytes = self.codec.encode(response.to_dict())
                writer.write(resp_bytes)
                await writer.drain()

        except asyncio.IncompleteReadError:
            pass  # Client disconnected
        except Exception as e:
            logger.error(f"Error handling {addr}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            logger.info(f"Disconnected {addr}")

    async def start(self) -> None:
        """Start the server."""
        self._server = await asyncio.start_server(
            self._handle_connection, self.host, self.port,
        )
        logger.info(f"Pricing server listening on {self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._pool.shutdown(wait=False)

    async def run(self) -> None:
        """Start and run forever."""
        await self.start()
        async with self._server:
            await self._server.serve_forever()
