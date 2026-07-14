"""CreditModel — the model a CDS is priced under (L3).

Amendment A1: a model carries its economy. A `CreditModel` carries the
`MarketSnapshot` (for discounting) plus the calibrated credit content — the
bootstrapped `SurvivalCurve` and the recovery rate. `model.market` satisfies the
`CalibratedModel` protocol; the CDS engine reaches discount curve, survival, and
recovery through it.

Provenance:
  quarry: python/pricebook/models/ (credit) + core/survival_curve.py
  source: redesign/02_spine.md Amendment A1; single-name CDS
  oracle: par CDS reprices to zero through the engine (CDS-as-product slice)
  slice:  cds-product
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook_ng.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class CreditModel:
    """Discounting + hazard economy (in the snapshot) + recovery, for CDS pricing.
    The engine looks up each CDS's survival curve by issuer at
    `MarketKey(CREDIT, issuer)` (A5, multi-issuer)."""

    market: MarketSnapshot
    recovery: float
