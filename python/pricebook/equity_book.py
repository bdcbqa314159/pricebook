"""Equity position management: EquityBook, position aggregation, limits.

EquityBook extends the generic Book pattern with equity-specific
aggregation by underlying ticker, sector, and currency.

    from pricebook.equity_book import (
        EquityBook, EquityLimits, EquityPosition,
    )

    book = EquityBook("US_Vol")
    book.add(trade, ticker="AAPL", sector="tech", spot=180.0)
    positions = book.positions_by_ticker()
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pricebook.pricing_context import PricingContext
from pricebook.trade import Trade


# ---- Position dataclasses ----

@dataclass
class EquityPosition:
    """Net exposure to a single underlying."""
    ticker: str
    sector: str
    currency: str
    net_notional: float
    delta_exposure: float  # delta-adjusted exposure
    long_notional: float
    short_notional: float
    trade_count: int


@dataclass
class SectorExposure:
    """Aggregate exposure by sector."""
    sector: str
    net_notional: float
    long_notional: float
    short_notional: float
    n_names: int


@dataclass
class EquityLimitBreach:
    """Limit violation for equity book."""
    limit_type: str
    limit_name: str
    limit_value: float
    actual_value: float
    details: str = ""


@dataclass
class EquityLimits:
    """Position limits for an equity book.

    Attributes:
        max_notional_per_name: per-ticker notional cap.
        max_notional_per_sector: per-sector notional cap.
        max_net_exposure: maximum absolute net exposure (sum of signed notionals).
        max_gross_exposure: maximum gross exposure (sum of |notionals|).
        max_beta_exposure: beta-weighted exposure cap (requires beta in metadata).
    """
    max_notional_per_name: dict[str, float] = field(default_factory=dict)
    max_notional_per_sector: dict[str, float] = field(default_factory=dict)
    max_net_exposure: float | None = None
    max_gross_exposure: float | None = None
    max_beta_exposure: float | None = None


# ---- Equity trade entry ----

@dataclass
class EquityTradeEntry:
    """A trade with equity-specific metadata.

    The Trade itself wraps any instrument (option, future, swap, equity itself);
    the EquityBook needs ticker/sector/spot to aggregate properly.
    """
    trade: Trade
    ticker: str
    sector: str = "other"
    currency: str = "USD"
    spot: float = 0.0
    beta: float = 1.0
    delta_per_unit: float = 1.0  # delta sensitivity (1.0 for stock, ≤1 for options)


# ---- Equity Book ----

class EquityBook:
    """A named container of equity trades with per-name and per-sector aggregation.

    Args:
        name: book name.
        limits: optional position limits.
        currency: book reporting currency.
    """

    def __init__(
        self,
        name: str,
        limits: EquityLimits | None = None,
        currency: str = "USD",
    ):
        self.name = name
        self.limits = limits or EquityLimits()
        self.currency = currency
        self._entries: list[EquityTradeEntry] = []

    def add(
        self,
        trade: Trade,
        ticker: str,
        sector: str = "other",
        currency: str | None = None,
        spot: float = 0.0,
        beta: float = 1.0,
        delta_per_unit: float = 1.0,
    ) -> None:
        """Add an equity trade with metadata."""
        self._entries.append(EquityTradeEntry(
            trade=trade, ticker=ticker, sector=sector,
            currency=currency or self.currency,
            spot=spot, beta=beta, delta_per_unit=delta_per_unit,
        ))

    @property
    def entries(self) -> list[EquityTradeEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def n_entries(self) -> int:
        return len(self._entries)

    @property
    def n_names(self) -> int:
        return len({e.ticker for e in self._entries})

    @property
    def n_sectors(self) -> int:
        return len({e.sector for e in self._entries})

    # ---- Aggregations ----

    def positions_by_ticker(self) -> list[EquityPosition]:
        """Aggregate net positions per ticker."""
        agg: dict[str, dict] = {}
        for e in self._entries:
            t = e.trade
            signed_notional = t.direction * t.notional_scale * _instrument_notional(t.instrument)
            abs_notional = abs(signed_notional)

            if e.ticker not in agg:
                agg[e.ticker] = {
                    "ticker": e.ticker, "sector": e.sector, "currency": e.currency,
                    "net": 0.0, "delta": 0.0, "long": 0.0, "short": 0.0, "count": 0,
                }
            d = agg[e.ticker]
            d["net"] += signed_notional
            d["delta"] += signed_notional * e.delta_per_unit
            if signed_notional > 0:
                d["long"] += abs_notional
            else:
                d["short"] += abs_notional
            d["count"] += 1

        return [
            EquityPosition(
                ticker=d["ticker"], sector=d["sector"], currency=d["currency"],
                net_notional=d["net"], delta_exposure=d["delta"],
                long_notional=d["long"], short_notional=d["short"],
                trade_count=d["count"],
            )
            for d in sorted(agg.values(), key=lambda x: x["ticker"])
        ]

    def exposures_by_sector(self) -> list[SectorExposure]:
        """Aggregate exposure by sector."""
        sector_agg: dict[str, dict] = {}
        ticker_seen: dict[str, set] = {}
        for e in self._entries:
            signed_notional = (
                e.trade.direction * e.trade.notional_scale
                * _instrument_notional(e.trade.instrument)
            )
            abs_notional = abs(signed_notional)
            if e.sector not in sector_agg:
                sector_agg[e.sector] = {"net": 0.0, "long": 0.0, "short": 0.0}
                ticker_seen[e.sector] = set()
            sector_agg[e.sector]["net"] += signed_notional
            if signed_notional > 0:
                sector_agg[e.sector]["long"] += abs_notional
            else:
                sector_agg[e.sector]["short"] += abs_notional
            ticker_seen[e.sector].add(e.ticker)

        return [
            SectorExposure(
                sector=sec,
                net_notional=d["net"],
                long_notional=d["long"],
                short_notional=d["short"],
                n_names=len(ticker_seen[sec]),
            )
            for sec, d in sorted(sector_agg.items())
        ]

    def net_exposure(self) -> float:
        """Sum of signed notionals across all positions."""
        return sum(
            e.trade.direction * e.trade.notional_scale * _instrument_notional(e.trade.instrument)
            for e in self._entries
        )

    def gross_exposure(self) -> float:
        """Sum of absolute notionals."""
        return sum(
            abs(e.trade.direction * e.trade.notional_scale * _instrument_notional(e.trade.instrument))
            for e in self._entries
        )

    def beta_weighted_exposure(self) -> float:
        """Beta-weighted exposure: Σ (signed_notional × beta)."""
        return sum(
            e.trade.direction * e.trade.notional_scale
            * _instrument_notional(e.trade.instrument) * e.beta
            for e in self._entries
        )

    def total_delta(self) -> float:
        """Total delta-adjusted exposure."""
        return sum(p.delta_exposure for p in self.positions_by_ticker())

    # ---- PV ----

    def pv(self, ctx: PricingContext) -> float:
        """Aggregate PV across all trades that have a pv_ctx method."""
        total = 0.0
        for e in self._entries:
            inst = e.trade.instrument
            if hasattr(inst, "pv_ctx"):
                total += e.trade.pv(ctx)
        return total

    # ---- Limits ----

    def check_limits(self) -> list[EquityLimitBreach]:
        """Check all configured limits and return breaches."""
        breaches: list[EquityLimitBreach] = []

        # Per-name limits
        if self.limits.max_notional_per_name:
            ticker_notionals = {p.ticker: abs(p.net_notional) for p in self.positions_by_ticker()}
            for ticker, limit in self.limits.max_notional_per_name.items():
                actual = ticker_notionals.get(ticker, 0.0)
                if actual > limit:
                    breaches.append(EquityLimitBreach(
                        "per_name", ticker, limit, actual,
                        f"{ticker} notional {actual:,.0f} exceeds {limit:,.0f}",
                    ))

        # Per-sector limits
        if self.limits.max_notional_per_sector:
            sector_notionals = {s.sector: abs(s.net_notional) for s in self.exposures_by_sector()}
            for sector, limit in self.limits.max_notional_per_sector.items():
                actual = sector_notionals.get(sector, 0.0)
                if actual > limit:
                    breaches.append(EquityLimitBreach(
                        "per_sector", sector, limit, actual,
                        f"sector {sector} notional {actual:,.0f} exceeds {limit:,.0f}",
                    ))

        # Net exposure
        if self.limits.max_net_exposure is not None:
            actual = abs(self.net_exposure())
            if actual > self.limits.max_net_exposure:
                breaches.append(EquityLimitBreach(
                    "net_exposure", f"book:{self.name}",
                    self.limits.max_net_exposure, actual,
                    f"net exposure {actual:,.0f} exceeds {self.limits.max_net_exposure:,.0f}",
                ))

        # Gross exposure
        if self.limits.max_gross_exposure is not None:
            actual = self.gross_exposure()
            if actual > self.limits.max_gross_exposure:
                breaches.append(EquityLimitBreach(
                    "gross_exposure", f"book:{self.name}",
                    self.limits.max_gross_exposure, actual,
                    f"gross exposure {actual:,.0f} exceeds {self.limits.max_gross_exposure:,.0f}",
                ))

        # Beta-weighted exposure
        if self.limits.max_beta_exposure is not None:
            actual = abs(self.beta_weighted_exposure())
            if actual > self.limits.max_beta_exposure:
                breaches.append(EquityLimitBreach(
                    "beta_exposure", f"book:{self.name}",
                    self.limits.max_beta_exposure, actual,
                    f"beta exposure {actual:,.0f} exceeds {self.limits.max_beta_exposure:,.0f}",
                ))

        return breaches


# ---- Helpers ----

def _instrument_notional(instrument: object) -> float:
    """Extract notional from an instrument, defaulting to 0."""
    if hasattr(instrument, "notional"):
        return float(instrument.notional)
    return 0.0
