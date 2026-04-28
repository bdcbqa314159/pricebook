"""Bloomberg adapter: convert between BBG market data and pricebook schema.

Maps Bloomberg ticker conventions to pricebook QuoteType/tenor and
formats pricing results in Bloomberg-compatible field names.

    from pricebook.bloomberg_adapter import (
        BloombergQuoteAdapter, BloombergResultFormatter,
        irs_request_from_bbg, bond_request_from_bbg,
    )

    adapter = BloombergQuoteAdapter()
    envelope = adapter.from_bbg_snapshot(tickers, values)

    formatter = BloombergResultFormatter()
    bbg_fields = formatter.to_bbg_fields(trade_result)

Ticker conventions follow Bloomberg's standard naming:
    USSW5 Curncy  → USD 5Y swap rate
    US0003M Index → USD 3M deposit rate
    EUSA5 Curncy  → EUR 5Y swap rate
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pricebook.pricing_schema import (
    PricingRequest, PricingConfig, QuoteMsg,
    MarketDataEnvelope, TradeEnvelope, TradeResult,
    irs_trade, quotes_market_data,
)


# ---- Ticker parsing ----

# BBG swap ticker patterns: USSW5, EUSA10, BPSW3, etc.
_SWAP_PATTERN = re.compile(
    r"^(US|EU|BP|JY|SF|CA|AD|NZ)(?:SW|SA)(\d+)\s*(?:Curncy)?$", re.IGNORECASE
)

# BBG deposit/money market: US0003M Index, EU0006M Index
_DEPOSIT_PATTERN = re.compile(
    r"^(US|EU|BP|JY|SF|CA|AD|NZ)00(\d{2})(M|D|W)\s*(?:Index)?$", re.IGNORECASE
)

_CCY_MAP = {
    "US": "USD", "EU": "EUR", "BP": "GBP", "JY": "JPY",
    "SF": "CHF", "CA": "CAD", "AD": "AUD", "NZ": "NZD",
}


def parse_bbg_ticker(ticker: str) -> QuoteMsg | None:
    """Parse a Bloomberg ticker into a QuoteMsg.

    Returns None if the ticker format is not recognised.

    Examples:
        "USSW5 Curncy"  → QuoteMsg(type="swap_rate", tenor="5Y", currency="USD")
        "US0003M Index" → QuoteMsg(type="deposit_rate", tenor="3M", currency="USD")
        "EUSA10 Curncy" → QuoteMsg(type="swap_rate", tenor="10Y", currency="EUR")
    """
    ticker = ticker.strip()

    # Try swap pattern
    m = _SWAP_PATTERN.match(ticker)
    if m:
        prefix, years = m.group(1).upper(), m.group(2)
        ccy = _CCY_MAP.get(prefix, "USD")
        return QuoteMsg(type="swap_rate", tenor=f"{years}Y",
                        value=0.0, currency=ccy, name=ticker)

    # Try deposit pattern
    m = _DEPOSIT_PATTERN.match(ticker)
    if m:
        prefix = m.group(1).upper()
        num = m.group(2)
        unit = m.group(3).upper()
        ccy = _CCY_MAP.get(prefix, "USD")
        tenor = f"{int(num)}{unit}"
        return QuoteMsg(type="deposit_rate", tenor=tenor,
                        value=0.0, currency=ccy, name=ticker)

    return None


class BloombergQuoteAdapter:
    """Converts Bloomberg-style market data to pricebook MarketDataEnvelope.

    Usage:
        adapter = BloombergQuoteAdapter()
        envelope = adapter.from_bbg_snapshot(
            {"USSW5 Curncy": 0.035, "US0003M Index": 0.030})
    """

    def from_bbg_snapshot(
        self, data: dict[str, float],
    ) -> MarketDataEnvelope:
        """Convert {bbg_ticker: value} to MarketDataEnvelope.

        Unrecognised tickers are silently skipped.
        """
        quotes = []
        for ticker, value in data.items():
            q = parse_bbg_ticker(ticker)
            if q is not None:
                q.value = value
                quotes.append(q.to_dict())
        return MarketDataEnvelope(mode="quotes", quotes=quotes)

    def to_pricing_request(
        self,
        valuation_date: str,
        market_data: dict[str, float],
        trades: list[dict],
        config: dict | None = None,
    ) -> PricingRequest:
        """Build a full PricingRequest from Bloomberg data."""
        md = self.from_bbg_snapshot(market_data)
        return PricingRequest(
            valuation_date=valuation_date,
            market_data=md.to_dict(),
            trades=trades,
            config=config or PricingConfig().to_dict(),
        )


# ---- Result formatting ----

# Bloomberg field name mapping
_BBG_FIELD_MAP = {
    "pv": "PX_LAST",
    "dv01": "DUR_ADJ_MID",
    "delta": "DELTA",
    "gamma": "GAMMA",
    "vega": "VEGA",
    "theta": "THETA",
}


class BloombergResultFormatter:
    """Converts pricebook TradeResult to Bloomberg-compatible field names."""

    def to_bbg_fields(self, result: TradeResult) -> dict[str, Any]:
        """Convert TradeResult to Bloomberg field names.

        Returns:
            {"PX_LAST": pv, "DUR_ADJ_MID": dv01, ...}
        """
        fields: dict[str, Any] = {
            "PX_LAST": result.pv,
            "CRNCY": result.currency,
        }
        for key, value in result.greeks.items():
            bbg_name = _BBG_FIELD_MAP.get(key, key.upper())
            fields[bbg_name] = value
        for key, value in result.risk.items():
            fields[key.upper()] = value
        return fields

    def format_results(
        self, results: list[TradeResult],
    ) -> dict[str, dict[str, Any]]:
        """Format multiple results keyed by trade_id.

        Returns:
            {"T1": {"PX_LAST": 12345, ...}, "T2": {...}}
        """
        return {r.trade_id: self.to_bbg_fields(r) for r in results}


# ---- Request templates ----

def irs_request_from_bbg(
    valuation_date: str,
    currency: str,
    fixed_rate: float,
    maturity_tenor: str,
    notional: float = 1_000_000,
    market_data: dict[str, float] | None = None,
) -> PricingRequest:
    """Build IRS pricing request from Bloomberg-style inputs."""
    adapter = BloombergQuoteAdapter()
    md = adapter.from_bbg_snapshot(market_data or {})
    return PricingRequest(
        valuation_date=valuation_date,
        trades=[irs_trade("IRS_1", currency, fixed_rate, maturity_tenor, notional)],
        market_data=md.to_dict(),
    )


def bond_request_from_bbg(
    valuation_date: str,
    coupon_rate: float,
    maturity: str,
    face_value: float = 100,
    market_data: dict[str, float] | None = None,
) -> PricingRequest:
    """Build bond pricing request from Bloomberg-style inputs."""
    from pricebook.pricing_schema import bond_trade
    adapter = BloombergQuoteAdapter()
    md = adapter.from_bbg_snapshot(market_data or {})
    return PricingRequest(
        valuation_date=valuation_date,
        trades=[bond_trade("BOND_1", coupon_rate, maturity, face_value)],
        market_data=md.to_dict(),
    )
