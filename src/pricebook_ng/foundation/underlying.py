"""Underlying / index identity — the general concept (L0).

The gate audit (F4/S10 + §3c): a new asset class adds **one** L0 object — its identity, as a
sibling under a shared concept. `Underlying` is anything with a `name` and an `asset_class` —
what `MarketKey` (A5, L1) keys on; `RateIndex` (rate_index.py) is the first (rates) instance.

**The SHAPE is complete; the FIELDS are not populated speculatively (ruling A3).** The earlier
sibling identity records (`ReferenceEntity`, `InflationIndex`, `FxFixing`, `EquityUnderlying`,
`CommodityUnderlying`) were cut: they had zero consumers and guessed fields (`fixing_time: str`,
a required `grade: str`), and a guessed field frozen into a schema is a costlier retrofit than
the file-touch to reintroduce it correctly. Each returns — with *validated* fields — when its
asset class actually ships (credit adds `ReferenceEntity`, etc.), one identity at a time (§3c).

Provenance:
  quarry: — (new: the general identity concept the quarry never had)
  source: gate audit F4/S10; CLAUDE.md §3c (asset-class onboarding); audit closure A3
  oracle: RateIndex satisfies Underlying (test_rate_index)
  slice:  index-rework (Topic 0 gate rework, F4/S10); trimmed audit-closure Phase 4 (A3)
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable


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
