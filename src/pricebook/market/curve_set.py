"""Curves, keyed (L1).

A set of curves reached through a **typed accessor**, not by position or by string. The key
carries the asset dimension — here a currency and a role — so a new asset class later adds
*keys*, not fields, and nothing already written has to change shape.

Provenance:
  stock:  _stock/src/pricebook/market/curve_set.py (purified: the DISCOUNT role only; projection
          and survival roles arrive with the products that need them)
  source: CLAUDE.md-era doc 19 §3 (closed shapes × open keys)
  oracle: a missing curve raises a KeyError the engine turns into a PricingFailure
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from pricebook.foundation.money import Currency
from pricebook.market.curve import CurveHandle


class CurveRole(Enum):
    """What a curve is *for*.

    `REAL` arrives in tranche 2 as a **new key, not a new snapshot field** — which is the whole
    point of keying: an inflation product needs a second discount curve (`Pr` alongside `Pn`), and
    adding it costs one enum member while `MarketSnapshot`, `CurveSet` and every existing pricer
    stay exactly as they were."""

    DISCOUNT = "discount"
    REAL = "real"


@dataclass(frozen=True)
class CurveKey:
    """Identifies a curve: its role and the asset dimension it is keyed by."""

    role: CurveRole
    currency: Currency


@dataclass(frozen=True)
class CurveSet:
    """One backing store, reached through accessors that state the narrowing once."""

    curves: Mapping[CurveKey, CurveHandle]

    def real(self, currency: Currency) -> CurveHandle:
        """The real (inflation-linked) discount curve `Pr` for `currency`.

        Its own accessor rather than `discount(currency, role=...)`: the two curves answer
        different questions, and a role parameter would let a caller pass the nominal curve where
        the real one belongs by changing one argument. Raises on a missing curve — defaulting to
        the nominal one would silently price every inflation swap to zero, which is a plausible
        wrong number."""
        key = CurveKey(CurveRole.REAL, currency)
        try:
            return self.curves[key]
        except KeyError:
            raise KeyError(f"no real curve for {currency} in the curve set") from None

    def discount(self, currency: Currency) -> CurveHandle:
        """The discount curve for `currency`. Raises on an unknown key rather than defaulting to
        a flat-zero curve: a silently risk-free curve prices everything to its undiscounted
        notional, which is a wrong number that looks plausible."""
        key = CurveKey(CurveRole.DISCOUNT, currency)
        try:
            return self.curves[key]
        except KeyError:
            raise KeyError(f"no discount curve for {currency} in the curve set") from None
