"""Commodity position management: CommodityBook, aggregation, limits.

Mirrors the equity desk pattern (:mod:`pricebook.equity_book`) for the
commodity world. Positions are organised by commodity (WTI, Brent, gold,
copper, natgas, wheat …), grouped into sectors (energy, metals,
agriculture), and tracked across tenor buckets so the desk can manage
term-structure exposure independently of outright direction.

    book = CommodityBook("Energy_Desk", valuation_date=date(2024, 1, 15))
    book.add(trade, commodity="WTI", sector="energy", unit="bbl",
             quantity=10_000, reference_price=72.0,
             delivery_date=date(2024, 6, 15))
    positions = book.positions_by_commodity()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pricebook.trade import Trade


# ---- Position dataclasses ----

@dataclass
class CommodityPosition:
    """Net position in a single commodity, aggregated across deliveries."""
    commodity: str
    sector: str
    unit: str
    net_quantity: float
    long_quantity: float
    short_quantity: float
    net_notional: float
    long_notional: float
    short_notional: float
    trade_count: int


@dataclass
class CommoditySectorExposure:
    """Aggregate notional exposure by sector."""
    sector: str
    net_notional: float
    long_notional: float
    short_notional: float
    n_commodities: int


@dataclass
class TermStructureBucket:
    """Net notional within a tenor bucket of a single commodity book."""
    bucket_label: str
    net_notional: float
    n_positions: int


@dataclass
class CommodityLimitBreach:
    """Limit violation for a commodity book."""
    limit_type: str
    limit_name: str
    limit_value: float
    actual_value: float
    details: str = ""


@dataclass
class CommodityLimits:
    """Position limits for a commodity book.

    Attributes:
        max_notional_per_commodity: per-commodity notional cap.
        max_notional_per_sector: per-sector notional cap.
        max_net_notional: maximum |signed sum| of all positions.
        max_gross_notional: maximum sum of |notionals|.
        max_notional_per_tenor: per-tenor-bucket notional cap.
    """
    max_notional_per_commodity: dict[str, float] = field(default_factory=dict)
    max_notional_per_sector: dict[str, float] = field(default_factory=dict)
    max_net_notional: float | None = None
    max_gross_notional: float | None = None
    max_notional_per_tenor: dict[str, float] = field(default_factory=dict)


# ---- Trade entry ----

@dataclass
class CommodityTradeEntry:
    """A trade with commodity-specific metadata.

    The trade wraps any instrument; the book needs commodity / sector /
    quantity / reference price / delivery date for proper aggregation.
    """
    trade: Trade
    commodity: str
    sector: str = "other"
    unit: str = "unit"
    quantity: float = 0.0          # always non-negative; sign comes from trade.direction
    reference_price: float = 0.0   # price per unit, in book ccy
    delivery_date: date | None = None
    currency: str = "USD"


# ---- Tenor bucketing ----

def commodity_tenor_bucket(
    valuation_date: date,
    delivery_date: date | None,
) -> str:
    """Bucket a delivery date into a coarse tenor label.

    Buckets are inclusive at the upper boundary:
    ``front`` (≤30d), ``≤6M``, ``≤1Y``, ``≤2Y``, ``>2Y``.
    """
    if delivery_date is None:
        return "unknown"
    days = (delivery_date - valuation_date).days
    if days <= 30:
        return "front"
    if days <= 180:
        return "≤6M"
    if days <= 365:
        return "≤1Y"
    if days <= 730:
        return "≤2Y"
    return ">2Y"


# ---- Commodity book ----

class CommodityBook:
    """A named container of commodity trades with per-commodity, per-sector
    and per-tenor aggregation, plus optional limit checking.

    Args:
        name: book name.
        valuation_date: as-of date for tenor bucketing.
        limits: optional :class:`CommodityLimits`.
        currency: book reporting currency.
    """

    def __init__(
        self,
        name: str,
        valuation_date: date,
        limits: CommodityLimits | None = None,
        currency: str = "USD",
    ):
        self.name = name
        self.valuation_date = valuation_date
        self.limits = limits or CommodityLimits()
        self.currency = currency
        self._entries: list[CommodityTradeEntry] = []

    def add(
        self,
        trade: Trade,
        commodity: str,
        sector: str = "other",
        unit: str = "unit",
        quantity: float = 0.0,
        reference_price: float = 0.0,
        delivery_date: date | None = None,
        currency: str | None = None,
    ) -> None:
        """Add a commodity trade with metadata."""
        self._entries.append(CommodityTradeEntry(
            trade=trade, commodity=commodity, sector=sector, unit=unit,
            quantity=quantity, reference_price=reference_price,
            delivery_date=delivery_date,
            currency=currency or self.currency,
        ))

    @property
    def entries(self) -> list[CommodityTradeEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def n_entries(self) -> int:
        return len(self._entries)

    @property
    def n_commodities(self) -> int:
        return len({e.commodity for e in self._entries})

    @property
    def n_sectors(self) -> int:
        return len({e.sector for e in self._entries})

    # ---- Aggregations ----

    def positions_by_commodity(self) -> list[CommodityPosition]:
        """Aggregate net positions per commodity (across all deliveries)."""
        agg: dict[str, dict] = {}
        for e in self._entries:
            sign = e.trade.direction * e.trade.notional_scale
            qty = sign * e.quantity
            notional = qty * e.reference_price

            if e.commodity not in agg:
                agg[e.commodity] = {
                    "commodity": e.commodity, "sector": e.sector, "unit": e.unit,
                    "net_qty": 0.0, "long_qty": 0.0, "short_qty": 0.0,
                    "net_n": 0.0, "long_n": 0.0, "short_n": 0.0,
                    "count": 0,
                }
            d = agg[e.commodity]
            d["net_qty"] += qty
            d["net_n"] += notional
            if qty > 0:
                d["long_qty"] += qty
                d["long_n"] += notional
            elif qty < 0:
                d["short_qty"] += -qty
                d["short_n"] += -notional
            d["count"] += 1

        return [
            CommodityPosition(
                commodity=d["commodity"], sector=d["sector"], unit=d["unit"],
                net_quantity=d["net_qty"],
                long_quantity=d["long_qty"], short_quantity=d["short_qty"],
                net_notional=d["net_n"],
                long_notional=d["long_n"], short_notional=d["short_n"],
                trade_count=d["count"],
            )
            for d in sorted(agg.values(), key=lambda x: x["commodity"])
        ]

    def exposures_by_sector(self) -> list[CommoditySectorExposure]:
        """Aggregate notional exposure by sector."""
        sector_agg: dict[str, dict] = {}
        commodities_seen: dict[str, set] = {}
        for e in self._entries:
            sign = e.trade.direction * e.trade.notional_scale
            notional = sign * e.quantity * e.reference_price

            if e.sector not in sector_agg:
                sector_agg[e.sector] = {"net": 0.0, "long": 0.0, "short": 0.0}
                commodities_seen[e.sector] = set()
            sector_agg[e.sector]["net"] += notional
            if notional > 0:
                sector_agg[e.sector]["long"] += notional
            elif notional < 0:
                sector_agg[e.sector]["short"] += -notional
            commodities_seen[e.sector].add(e.commodity)

        return [
            CommoditySectorExposure(
                sector=sec,
                net_notional=d["net"],
                long_notional=d["long"],
                short_notional=d["short"],
                n_commodities=len(commodities_seen[sec]),
            )
            for sec, d in sorted(sector_agg.items())
        ]

    def exposures_by_tenor(self) -> list[TermStructureBucket]:
        """Aggregate notional exposure by tenor bucket."""
        agg: dict[str, dict] = {}
        for e in self._entries:
            sign = e.trade.direction * e.trade.notional_scale
            notional = sign * e.quantity * e.reference_price
            bucket = commodity_tenor_bucket(self.valuation_date, e.delivery_date)
            if bucket not in agg:
                agg[bucket] = {"notional": 0.0, "count": 0}
            agg[bucket]["notional"] += notional
            agg[bucket]["count"] += 1

        return [
            TermStructureBucket(
                bucket_label=b, net_notional=d["notional"], n_positions=d["count"],
            )
            for b, d in sorted(agg.items())
        ]

    def net_notional(self) -> float:
        """Sum of signed notionals across all positions."""
        return sum(
            e.trade.direction * e.trade.notional_scale * e.quantity * e.reference_price
            for e in self._entries
        )

    def gross_notional(self) -> float:
        """Sum of absolute notionals across all positions."""
        return sum(
            abs(e.trade.direction * e.trade.notional_scale * e.quantity * e.reference_price)
            for e in self._entries
        )

    # ---- Limits ----

    def check_limits(self) -> list[CommodityLimitBreach]:
        """Check all configured limits and return any breaches."""
        breaches: list[CommodityLimitBreach] = []

        if self.limits.max_notional_per_commodity:
            commodity_n = {
                p.commodity: abs(p.net_notional)
                for p in self.positions_by_commodity()
            }
            for commodity, lim in self.limits.max_notional_per_commodity.items():
                actual = commodity_n.get(commodity, 0.0)
                if actual > lim:
                    breaches.append(CommodityLimitBreach(
                        "per_commodity", commodity, lim, actual,
                        f"{commodity} notional {actual:,.0f} exceeds {lim:,.0f}",
                    ))

        if self.limits.max_notional_per_sector:
            sector_n = {
                s.sector: abs(s.net_notional)
                for s in self.exposures_by_sector()
            }
            for sector, lim in self.limits.max_notional_per_sector.items():
                actual = sector_n.get(sector, 0.0)
                if actual > lim:
                    breaches.append(CommodityLimitBreach(
                        "per_sector", sector, lim, actual,
                        f"sector {sector} notional {actual:,.0f} exceeds {lim:,.0f}",
                    ))

        if self.limits.max_net_notional is not None:
            actual = abs(self.net_notional())
            if actual > self.limits.max_net_notional:
                breaches.append(CommodityLimitBreach(
                    "net_notional", f"book:{self.name}",
                    self.limits.max_net_notional, actual,
                    f"net notional {actual:,.0f} exceeds {self.limits.max_net_notional:,.0f}",
                ))

        if self.limits.max_gross_notional is not None:
            actual = self.gross_notional()
            if actual > self.limits.max_gross_notional:
                breaches.append(CommodityLimitBreach(
                    "gross_notional", f"book:{self.name}",
                    self.limits.max_gross_notional, actual,
                    f"gross notional {actual:,.0f} exceeds {self.limits.max_gross_notional:,.0f}",
                ))

        if self.limits.max_notional_per_tenor:
            tenor_n = {
                b.bucket_label: abs(b.net_notional)
                for b in self.exposures_by_tenor()
            }
            for tenor, lim in self.limits.max_notional_per_tenor.items():
                actual = tenor_n.get(tenor, 0.0)
                if actual > lim:
                    breaches.append(CommodityLimitBreach(
                        "per_tenor", tenor, lim, actual,
                        f"tenor {tenor} notional {actual:,.0f} exceeds {lim:,.0f}",
                    ))

        return breaches
