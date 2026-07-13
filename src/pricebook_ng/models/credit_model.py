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

from pricebook_ng.market.snapshot import MarketSnapshot, SurvivalHandle


@dataclass(frozen=True)
class CreditModel:
    """Discounting + hazard economy (both in the snapshot) + recovery, for CDS
    pricing. The survival curve is market data reached through `market.survival_curve`
    (ruling §5.1), so a credit greek bumps the snapshot and rebuilds the model."""

    market: MarketSnapshot
    recovery: float

    @property
    def survival(self) -> SurvivalHandle:
        curve = self.market.survival_curve
        if curve is None:
            raise ValueError("CreditModel needs a survival curve on the snapshot")
        return curve
