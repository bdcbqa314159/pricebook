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
    """What a curve is *for*. Only discounting exists yet; projection and survival arrive with
    their first products, and the role is what keeps them from being told apart by type."""

    DISCOUNT = "discount"


@dataclass(frozen=True)
class CurveKey:
    """Identifies a curve: its role and the asset dimension it is keyed by."""

    role: CurveRole
    currency: Currency


@dataclass(frozen=True)
class CurveSet:
    """One backing store, reached through accessors that state the narrowing once."""

    curves: Mapping[CurveKey, CurveHandle]

    def discount(self, currency: Currency) -> CurveHandle:
        """The discount curve for `currency`. Raises on an unknown key rather than defaulting to
        a flat-zero curve: a silently risk-free curve prices everything to its undiscounted
        notional, which is a wrong number that looks plausible."""
        key = CurveKey(CurveRole.DISCOUNT, currency)
        try:
            return self.curves[key]
        except KeyError:
            raise KeyError(f"no discount curve for {currency} in the curve set") from None
