"""Pricing service message schema — language-agnostic request/response protocol.

All messages are dataclasses with to_dict()/from_dict(). All field types are
JSON-native (str, float, int, bool, list, dict). Dates are ISO strings.
Enums are strings. This maps 1:1 to protobuf for future migration.

    from pricebook.pricing_schema import PricingRequest, PricingResponse

    req = PricingRequest(
        valuation_date="2026-04-28",
        trades=[TradeEnvelope(trade_id="T1", instrument_type="irs", params={...})],
        market_data=MarketDataEnvelope(mode="quotes", quotes=[...]),
    )
    wire = req.to_dict()
    req2 = PricingRequest.from_dict(wire)

Wire format designed for: JSON (now), MessagePack (opt-in), Protobuf (future).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


# ---- Market data ----

@dataclass
class QuoteMsg:
    """A single market data quote (Bloomberg-compatible).

    Maps to protobuf: message QuoteMsg { string type = 1; ... }
    """
    type: str          # "deposit_rate", "swap_rate", "cds_spread", "vol_point", "fx_spot"
    tenor: str         # "3M", "5Y", "10Y"
    value: float
    currency: str = "USD"
    name: str = ""     # issuer, index, etc.

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "tenor": self.tenor, "value": self.value,
                "currency": self.currency, "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> QuoteMsg:
        return cls(type=d["type"], tenor=d["tenor"], value=d["value"],
                   currency=d.get("currency", "USD"), name=d.get("name", ""))


@dataclass
class CurveMsg:
    """Pre-built curve pillars.

    Maps to protobuf: message CurveMsg { string name = 1; repeated string dates = 2; ... }
    """
    name: str                    # "USD_OIS", "EURIBOR_3M"
    dates: list[str]             # ISO dates
    values: list[float]          # discount factors or zero rates
    value_type: str = "df"       # "df" or "zero_rate"
    day_count: str = "ACT_365_FIXED"
    interpolation: str = "LOG_LINEAR"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dates": self.dates, "values": self.values,
                "value_type": self.value_type, "day_count": self.day_count,
                "interpolation": self.interpolation}

    @classmethod
    def from_dict(cls, d: dict) -> CurveMsg:
        return cls(name=d["name"], dates=d["dates"], values=d["values"],
                   value_type=d.get("value_type", "df"),
                   day_count=d.get("day_count", "ACT_365_FIXED"),
                   interpolation=d.get("interpolation", "LOG_LINEAR"))


@dataclass
class MarketDataEnvelope:
    """Market data: either raw quotes (bootstrap on server) or pre-built curves.

    mode="quotes": server bootstraps curves from quotes.
    mode="curves": server uses pre-built curves directly.
    """
    mode: str = "quotes"                            # "quotes" | "curves"
    quotes: list[dict] = field(default_factory=list)
    curves: list[dict] = field(default_factory=list)
    fixings: dict[str, dict[str, float]] = field(default_factory=dict)
    # fixings: {"SOFR": {"2026-04-27": 0.043, ...}, "EURIBOR_3M": {...}}

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"mode": self.mode}
        if self.quotes:
            d["quotes"] = self.quotes
        if self.curves:
            d["curves"] = self.curves
        if self.fixings:
            d["fixings"] = self.fixings
        return d

    @classmethod
    def from_dict(cls, d: dict) -> MarketDataEnvelope:
        return cls(mode=d.get("mode", "quotes"),
                   quotes=d.get("quotes", []),
                   curves=d.get("curves", []),
                   fixings=d.get("fixings", {}))


# ---- Trades ----

@dataclass
class CSAMsg:
    """CSA terms for a trade."""
    currency: str = "USD"
    threshold: float = 0.0
    mta: float = 0.0
    margin_frequency: str = "daily"
    collateral_type: str = "cash"  # "cash", "government_bond", "corporate_bond"

    def to_dict(self) -> dict[str, Any]:
        return {"currency": self.currency, "threshold": self.threshold,
                "mta": self.mta, "margin_frequency": self.margin_frequency,
                "collateral_type": self.collateral_type}

    @classmethod
    def from_dict(cls, d: dict) -> CSAMsg:
        return cls(currency=d.get("currency", "USD"),
                   threshold=d.get("threshold", 0.0),
                   mta=d.get("mta", 0.0),
                   margin_frequency=d.get("margin_frequency", "daily"),
                   collateral_type=d.get("collateral_type", "cash"))


@dataclass
class TradeEnvelope:
    """A trade to price.

    instrument_type maps to the serialization.py registry:
    "irs", "bond", "fra", "cds", "swaption", "capfloor", "trs", "cln", etc.
    params are instrument-specific constructor arguments.
    """
    trade_id: str
    instrument_type: str                      # registry key
    params: dict[str, Any] = field(default_factory=dict)
    direction: str = "payer"                  # "payer" | "receiver"
    notional_scale: float = 1.0
    csa: dict | None = None                   # CSAMsg.to_dict() or None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"trade_id": self.trade_id,
                "instrument_type": self.instrument_type,
                "params": self.params,
                "direction": self.direction,
                "notional_scale": self.notional_scale}
        if self.csa is not None:
            d["csa"] = self.csa
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TradeEnvelope:
        return cls(trade_id=d["trade_id"],
                   instrument_type=d["instrument_type"],
                   params=d.get("params", {}),
                   direction=d.get("direction", "payer"),
                   notional_scale=d.get("notional_scale", 1.0),
                   csa=d.get("csa"))


# ---- Config ----

@dataclass
class PricingConfig:
    """Pricing configuration."""
    model: str = "black"                      # "black", "sabr", "local_vol", "heston", "mc"
    mc_paths: int = 10_000
    compute_greeks: bool = False
    measures: list[str] = field(default_factory=lambda: ["pv"])
    # measures: ["pv", "delta", "gamma", "vega", "dv01", "theta"]

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "mc_paths": self.mc_paths,
                "compute_greeks": self.compute_greeks,
                "measures": self.measures}

    @classmethod
    def from_dict(cls, d: dict) -> PricingConfig:
        return cls(model=d.get("model", "black"),
                   mc_paths=d.get("mc_paths", 10_000),
                   compute_greeks=d.get("compute_greeks", False),
                   measures=d.get("measures", ["pv"]))


# ---- Request ----

@dataclass
class PricingRequest:
    """Complete pricing request — everything needed to compute a price.

    Maps to protobuf: message PricingRequest { string request_id = 1; ... }
    """
    valuation_date: str                                  # ISO date
    trades: list[dict] = field(default_factory=list)      # TradeEnvelope.to_dict()
    market_data: dict = field(default_factory=dict)       # MarketDataEnvelope.to_dict()
    config: dict = field(default_factory=dict)            # PricingConfig.to_dict()
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, str] = field(default_factory=dict)
    # metadata: {"user": "desk_a", "priority": "high", "source": "bloomberg"}

    def to_dict(self) -> dict[str, Any]:
        return {"request_id": self.request_id,
                "valuation_date": self.valuation_date,
                "market_data": self.market_data,
                "trades": self.trades,
                "config": self.config,
                "metadata": self.metadata}

    @classmethod
    def from_dict(cls, d: dict) -> PricingRequest:
        return cls(request_id=d.get("request_id", str(uuid.uuid4())),
                   valuation_date=d["valuation_date"],
                   market_data=d.get("market_data", {}),
                   trades=d.get("trades", []),
                   config=d.get("config", {}),
                   metadata=d.get("metadata", {}))

    def get_market_data(self) -> MarketDataEnvelope:
        return MarketDataEnvelope.from_dict(self.market_data)

    def get_trades(self) -> list[TradeEnvelope]:
        return [TradeEnvelope.from_dict(t) for t in self.trades]

    def get_config(self) -> PricingConfig:
        return PricingConfig.from_dict(self.config) if self.config else PricingConfig()


# ---- Response ----

@dataclass
class TradeResult:
    """Result for a single trade."""
    trade_id: str
    status: str = "ok"                         # "ok" | "error"
    pv: float = 0.0
    currency: str = "USD"
    greeks: dict[str, float] = field(default_factory=dict)
    risk: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"trade_id": self.trade_id, "status": self.status,
                "pv": self.pv, "currency": self.currency}
        if self.greeks:
            d["greeks"] = self.greeks
        if self.risk:
            d["risk"] = self.risk
        if self.diagnostics:
            d["diagnostics"] = self.diagnostics
        if self.error_message:
            d["error_message"] = self.error_message
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TradeResult:
        return cls(trade_id=d["trade_id"], status=d.get("status", "ok"),
                   pv=d.get("pv", 0.0), currency=d.get("currency", "USD"),
                   greeks=d.get("greeks", {}), risk=d.get("risk", {}),
                   diagnostics=d.get("diagnostics", {}),
                   error_message=d.get("error_message", ""))


@dataclass
class PricingResponse:
    """Complete pricing response.

    Maps to protobuf: message PricingResponse { string request_id = 1; ... }
    """
    request_id: str
    status: str = "ok"                        # "ok" | "error" | "partial"
    results: list[dict] = field(default_factory=list)  # TradeResult.to_dict()
    compute_time_ms: float = 0.0
    server_version: str = ""
    errors: list[dict] = field(default_factory=list)
    # errors: [{"trade_id": "T1", "code": "BOOTSTRAP_FAILED", "message": "..."}]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"request_id": self.request_id, "status": self.status,
                "results": self.results, "compute_time_ms": self.compute_time_ms}
        if self.server_version:
            d["server_version"] = self.server_version
        if self.errors:
            d["errors"] = self.errors
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PricingResponse:
        return cls(request_id=d["request_id"], status=d.get("status", "ok"),
                   results=d.get("results", []),
                   compute_time_ms=d.get("compute_time_ms", 0.0),
                   server_version=d.get("server_version", ""),
                   errors=d.get("errors", []))

    def get_results(self) -> list[TradeResult]:
        return [TradeResult.from_dict(r) for r in self.results]


# ---- Convenience builders ----

def irs_trade(trade_id: str, currency: str, fixed_rate: float,
              maturity: str, notional: float = 1_000_000,
              direction: str = "payer") -> dict:
    """Build a TradeEnvelope dict for an IRS."""
    return TradeEnvelope(
        trade_id=trade_id,
        instrument_type="irs",
        params={"fixed_rate": fixed_rate, "currency": currency,
                "maturity": maturity, "notional": notional},
        direction=direction,
    ).to_dict()


def bond_trade(trade_id: str, coupon_rate: float, maturity: str,
               face_value: float = 100, currency: str = "USD") -> dict:
    """Build a TradeEnvelope dict for a bond."""
    return TradeEnvelope(
        trade_id=trade_id,
        instrument_type="bond",
        params={"coupon_rate": coupon_rate, "maturity": maturity,
                "face_value": face_value, "currency": currency},
    ).to_dict()


def quotes_market_data(quotes: list[dict], fixings: dict | None = None) -> dict:
    """Build a MarketDataEnvelope dict from raw quotes."""
    return MarketDataEnvelope(
        mode="quotes", quotes=quotes,
        fixings=fixings or {},
    ).to_dict()


def curves_market_data(curves: list[dict]) -> dict:
    """Build a MarketDataEnvelope dict from pre-built curves."""
    return MarketDataEnvelope(mode="curves", curves=curves).to_dict()
