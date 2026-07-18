"""Market-data keys — the namespace for the snapshot's keyed registry (L1).

Amendment A5: market data lives in the snapshot's `curves`/`spots`/`vols` maps,
keyed by `MarketKey(asset, id)` — a typed, exhaustive `AssetClass` namespace plus
an open string id (currency code / ticker / issuer). Namespacing removes the
latent collision between an FX currency "EUR" and an equity ticker "EUR".

Provenance:
  quarry: n/a (new-tree market vocabulary)
  source: redesign/02_spine.md Amendment A5
  oracle: namespacing test (FX "EUR" != equity "EUR"); exercised by every asset class
  slice:  market-snapshot-keyed
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AssetClass(Enum):
    FX = "FX"
    EQUITY = "EQUITY"
    CREDIT = "CREDIT"
    CMDTY = "CMDTY"
    INFLATION = "INFLATION"


@dataclass(frozen=True)
class MarketKey:
    """A namespaced market-data identifier: asset class + open id (ccy/ticker/issuer)."""

    asset: AssetClass
    id: str
