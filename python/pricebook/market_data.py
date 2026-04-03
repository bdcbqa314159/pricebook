"""Market data snapshots, curve building pipeline, and historical data."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.pricing_context import PricingContext
from pricebook.vol_surface import FlatVol


# ---------------------------------------------------------------------------
# Quote types
# ---------------------------------------------------------------------------


class QuoteType(Enum):
    DEPOSIT_RATE = "deposit_rate"
    SWAP_RATE = "swap_rate"
    CDS_SPREAD = "cds_spread"
    VOL_POINT = "vol_point"
    FX_SPOT = "fx_spot"


@dataclass
class Quote:
    """A single market observation."""

    quote_type: QuoteType
    tenor: str  # e.g. "3M", "5Y", "10Y"
    value: float
    currency: str = "USD"
    name: str = ""  # optional label, e.g. issuer for CDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "quote_type": self.quote_type.value,
            "tenor": self.tenor,
            "value": self.value,
            "currency": self.currency,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Quote:
        return cls(
            quote_type=QuoteType(d["quote_type"]),
            tenor=d["tenor"],
            value=d["value"],
            currency=d.get("currency", "USD"),
            name=d.get("name", ""),
        )


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


@dataclass
class MarketDataSnapshot:
    """A dated collection of market quotes."""

    snapshot_date: date
    quotes: list[Quote] = field(default_factory=list)

    def add(self, quote: Quote) -> None:
        self.quotes.append(quote)

    def get_quotes(self, quote_type: QuoteType, currency: str = "USD",
                   name: str = "") -> list[Quote]:
        """Filter quotes by type, currency, and optional name."""
        return [
            q for q in self.quotes
            if q.quote_type == quote_type
            and q.currency == currency
            and (not name or q.name == name)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_date": self.snapshot_date.isoformat(),
            "quotes": [q.to_dict() for q in self.quotes],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MarketDataSnapshot:
        return cls(
            snapshot_date=date.fromisoformat(d["snapshot_date"]),
            quotes=[Quote.from_dict(q) for q in d["quotes"]],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> MarketDataSnapshot:
        return cls.from_dict(json.loads(s))


# ---------------------------------------------------------------------------
# Tenor parsing
# ---------------------------------------------------------------------------

_TENOR_MAP = {
    "1D": 1 / 365, "1W": 7 / 365, "2W": 14 / 365, "1M": 1 / 12,
    "2M": 2 / 12, "3M": 3 / 12, "6M": 6 / 12, "9M": 9 / 12,
    "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "4Y": 4.0, "5Y": 5.0,
    "7Y": 7.0, "10Y": 10.0, "15Y": 15.0, "20Y": 20.0, "30Y": 30.0,
}


def tenor_to_years(tenor: str) -> float:
    """Convert a tenor string like '5Y' or '3M' to a year fraction."""
    if tenor in _TENOR_MAP:
        return _TENOR_MAP[tenor]
    # Parse NM or NY
    if tenor.endswith("Y"):
        return float(tenor[:-1])
    if tenor.endswith("M"):
        return float(tenor[:-1]) / 12.0
    if tenor.endswith("W"):
        return float(tenor[:-1]) * 7 / 365
    if tenor.endswith("D"):
        return float(tenor[:-1]) / 365
    raise ValueError(f"Cannot parse tenor: {tenor}")


def tenor_to_date(ref: date, tenor: str) -> date:
    """Convert a tenor to a target date from reference."""
    years = tenor_to_years(tenor)
    return date.fromordinal(ref.toordinal() + int(years * 365))


# ---------------------------------------------------------------------------
# Curve building pipeline
# ---------------------------------------------------------------------------

@dataclass
class CurveConfig:
    """Configuration for building a single curve from snapshot quotes."""

    curve_name: str
    quote_type: QuoteType
    currency: str = "USD"
    name: str = ""  # for CDS: issuer name


@dataclass
class PipelineConfig:
    """Configuration for building a full PricingContext from a snapshot."""

    discount_config: CurveConfig | None = None
    credit_configs: dict[str, CurveConfig] = field(default_factory=dict)
    vol_config: dict[str, str] = field(default_factory=dict)  # surface_name -> quote name filter
    fx_pairs: list[tuple[str, str]] = field(default_factory=list)


class MissingQuoteError(Exception):
    """Raised when required quotes are not found in snapshot."""


def _build_discount_curve(
    ref: date, quotes: list[Quote],
) -> DiscountCurve:
    """Build a discount curve from deposit/swap rate quotes."""
    if not quotes:
        raise MissingQuoteError("No quotes for discount curve")

    sorted_quotes = sorted(quotes, key=lambda q: tenor_to_years(q.tenor))
    dates = [tenor_to_date(ref, q.tenor) for q in sorted_quotes]
    # Simple approach: treat all as continuously compounded zero rates
    dfs = [math.exp(-q.value * tenor_to_years(q.tenor)) for q in sorted_quotes]
    return DiscountCurve(ref, dates, dfs)


def _build_survival_curve(
    ref: date, quotes: list[Quote], recovery: float = 0.4,
) -> SurvivalCurve:
    """Build a survival curve from CDS spread quotes."""
    if not quotes:
        raise MissingQuoteError("No CDS spread quotes")

    sorted_quotes = sorted(quotes, key=lambda q: tenor_to_years(q.tenor))
    dates = [tenor_to_date(ref, q.tenor) for q in sorted_quotes]
    # Approximate: hazard ≈ spread / (1 - R)
    survs = [
        math.exp(-q.value / (1 - recovery) * tenor_to_years(q.tenor))
        for q in sorted_quotes
    ]
    return SurvivalCurve(ref, dates, survs)


def build_context(
    snapshot: MarketDataSnapshot,
    config: PipelineConfig,
) -> PricingContext:
    """Build a PricingContext from a snapshot and pipeline configuration."""

    discount = None
    if config.discount_config:
        dc = config.discount_config
        quotes = snapshot.get_quotes(dc.quote_type, dc.currency, dc.name)
        if not quotes:
            raise MissingQuoteError(
                f"No quotes for discount curve ({dc.quote_type.value}, {dc.currency})"
            )
        discount = _build_discount_curve(snapshot.snapshot_date, quotes)

    credit_curves: dict[str, SurvivalCurve] = {}
    for label, cc in config.credit_configs.items():
        quotes = snapshot.get_quotes(cc.quote_type, cc.currency, cc.name)
        if not quotes:
            raise MissingQuoteError(f"No CDS quotes for '{label}'")
        credit_curves[label] = _build_survival_curve(snapshot.snapshot_date, quotes)

    vol_surfaces: dict[str, object] = {}
    for surf_name, name_filter in config.vol_config.items():
        vols = snapshot.get_quotes(QuoteType.VOL_POINT, name=name_filter)
        if vols:
            # Use average vol as flat vol (simplification for educational library)
            avg = sum(q.value for q in vols) / len(vols)
            vol_surfaces[surf_name] = FlatVol(avg)

    fx_spots: dict[tuple[str, str], float] = {}
    for base, quote in config.fx_pairs:
        fx_quotes = [
            q for q in snapshot.quotes
            if q.quote_type == QuoteType.FX_SPOT
            and q.name == f"{base}/{quote}"
        ]
        if fx_quotes:
            fx_spots[(base, quote)] = fx_quotes[0].value

    return PricingContext(
        valuation_date=snapshot.snapshot_date,
        discount_curve=discount,
        credit_curves=credit_curves,
        vol_surfaces=vol_surfaces,
        fx_spots=fx_spots,
    )


# ---------------------------------------------------------------------------
# Historical data
# ---------------------------------------------------------------------------


class HistoricalData:
    """Time series of market data snapshots."""

    def __init__(self, snapshots: list[MarketDataSnapshot] | None = None):
        self._snapshots: dict[date, MarketDataSnapshot] = {}
        if snapshots:
            for s in snapshots:
                self._snapshots[s.snapshot_date] = s

    def add(self, snapshot: MarketDataSnapshot) -> None:
        self._snapshots[snapshot.snapshot_date] = snapshot

    @property
    def dates(self) -> list[date]:
        return sorted(self._snapshots.keys())

    @property
    def size(self) -> int:
        return len(self._snapshots)

    def get(self, d: date) -> MarketDataSnapshot:
        if d not in self._snapshots:
            raise KeyError(f"No snapshot for {d}")
        return self._snapshots[d]

    def to_dict_list(self) -> list[dict[str, Any]]:
        return [self._snapshots[d].to_dict() for d in self.dates]

    @classmethod
    def from_dict_list(cls, data: list[dict[str, Any]]) -> HistoricalData:
        snapshots = [MarketDataSnapshot.from_dict(d) for d in data]
        return cls(snapshots)

    @classmethod
    def from_json(cls, s: str) -> HistoricalData:
        return cls.from_dict_list(json.loads(s))

    def to_json(self) -> str:
        return json.dumps(self.to_dict_list(), indent=2)
