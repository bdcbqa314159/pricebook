"""Underlying / index identity — the general concept and its siblings (L0).

The gate audit (F4/S10 + §3c): a new asset class adds **one** L0 object — its identity,
as a sibling under a shared concept. `RateIndex` (rate_index.py) was standing alone; this
gives it something to be a sibling of. An `Underlying` is anything with a `name` and an
`asset_class` — what `MarketKey` (A5, L1) will key on. The rate/credit/inflation/FX/
equity/commodity identities all satisfy it. The non-rates siblings are **defined now,
populated later** (thin identity records), so credit and the rest have a shape to slot
into — the commodity `delivery_location`/`grade` especially (Brent ≠ WTI).

Provenance:
  quarry: — (new: the general identity concept the quarry never had)
  source: gate audit F4/S10; CLAUDE.md §3c (asset-class onboarding)
  oracle: each sibling reports its asset_class; RateIndex satisfies Underlying
  slice:  index-rework (Topic 0 gate rework, F4/S10)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from pricebook_ng.foundation.money import Currency, CurrencyPair, Unit
from pricebook_ng.foundation.tenor import Tenor


class AssetClass(Enum):
    RATES = "rates"
    CREDIT = "credit"
    INFLATION = "inflation"
    FX = "fx"
    EQUITY = "equity"
    COMMODITY = "commodity"


@runtime_checkable
class Underlying(Protocol):
    """What every index/underlying identity exposes — the key `MarketKey` (A5) keys on."""

    @property
    def name(self) -> str: ...
    @property
    def asset_class(self) -> AssetClass: ...


class InflationInterp(Enum):
    DAILY_LINEAR = "daily_linear"  # UST TIPS / OATi
    MONTHLY_FLAT = "monthly_flat"  # UK RPI


# ── sibling identities: defined now, populated later ─────────────────────────────
@dataclass(frozen=True)
class ReferenceEntity:
    """Credit identity — survival curves are keyed by it, CDS reference it, credit risk
    bumps by it (§3c multi-layer). Seniority/restructuring are L2 contract data."""

    name: str

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.CREDIT


@dataclass(frozen=True)
class InflationIndex:
    """A published price level (CPI/RPI/HICP) — not a rate. Indexation lag, interpolation
    style, and the base index."""

    name: str
    indexation_lag: Tenor  # typically 3M
    interpolation: InflationInterp
    base_index: float

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.INFLATION


@dataclass(frozen=True)
class FxFixing:
    """An FX fixing identity — pair, source and the fixing time (time-of-day is index
    metadata, never on the `Calendar` — S9)."""

    name: str
    pair: CurrencyPair
    source: str  # e.g. "WMR"
    fixing_time: str  # e.g. "16:00 London"

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.FX


@dataclass(frozen=True)
class EquityUnderlying:
    name: str
    exchange: str
    currency: Currency

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.EQUITY


@dataclass(frozen=True)
class CommodityUnderlying:
    """A commodity underlying — the unit, and the identity-defining delivery location and
    grade (Brent ≠ WTI; a gas delivery point) that would otherwise be discovered late."""

    name: str
    unit: Unit
    delivery_location: str
    grade: str

    @property
    def asset_class(self) -> AssetClass:
        return AssetClass.COMMODITY
