"""CurveSet — the keyed store of curves a snapshot carries (L1).

Doc 19 §2-§3: closed shapes × open keys. The shape is `CurveSet`; the keys are open.
Typed accessors — `discount(currency, collateral=None)` and `projection(index)` — over
ONE backing store keyed by `CurveKey`, so a new asset adds keys, not fields. This slice
populates one EUR discount key and two projection keys (the OIS index projects off its
own discount curve — the degenerate config; the IBOR index off a distinct curve). The
`collateral` argument is the ratified signature (so it is not re-broken later) but its
machinery is deferred: a non-None collateral raises. `survival`/`inflation` accessors and
a `{index → curve}` map for multiple projections arrive with their asset class / 2nd curve.

Provenance:
  quarry: python/pricebook/curves/ncurve_solver.py (curves-keyed-by-name concept)
  source: redesign/19 §2-§3 (closed shapes × open keys; typed CurveSet accessors)
  oracle: OIS projection is the discount curve; a EURIBOR swap prices to zero dual-curve
  slice:  dual-curve-euribor-estr (T1 slice 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from pricebook_ng.foundation import Currency, RateIndex
from pricebook_ng.market.curve import CurveHandle


class CurveRole(Enum):
    DISCOUNT = "discount"
    PROJECTION = "projection"


@dataclass(frozen=True)
class CurveKey:
    """Identifies a curve in the set: its role and the asset dimension it is keyed by
    (a `Currency` for discount, a `RateIndex` for projection). Doc 19's fuller asset
    keying — collateral, entity, underlying, pair — arrives with those asset classes."""

    role: CurveRole
    id: Currency | RateIndex


@dataclass(frozen=True)
class CurveSet:
    """One backing store of curves, reached through typed accessors (doc 19 §3)."""

    curves: Mapping[CurveKey, CurveHandle]

    def discount(self, currency: Currency, collateral: Currency | None = None) -> CurveHandle:
        if collateral is not None:
            raise NotImplementedError(
                "collateral-keyed discounting is deferred to the CSA/xccy slice"
            )
        return self.curves[CurveKey(CurveRole.DISCOUNT, currency)]

    def projection(self, index: RateIndex) -> CurveHandle:
        """The forward-projection curve for `index`. Single-curve is the degenerate
        config: an OIS index's projection curve IS its discount curve (same object)."""
        return self.curves[CurveKey(CurveRole.PROJECTION, index)]
