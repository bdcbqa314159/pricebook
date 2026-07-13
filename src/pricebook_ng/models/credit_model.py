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
from pricebook_ng.market.survival_curve import SurvivalCurve


@dataclass(frozen=True)
class CreditModel:
    """Discounting market + calibrated hazard curve + recovery, for CDS pricing."""

    market: MarketSnapshot
    survival: SurvivalCurve
    recovery: float
