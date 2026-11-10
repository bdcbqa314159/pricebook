"""The European equity option (L2) - frozen data, prices nothing.

Provenance:
  stock:  none - the archived predecessor's equity products were never purified into this tree
  source: BROCKHAUS-2016 (docs/references.md) ch 5 - the European equity option on a
          dividend-paying underlying
  oracle: tests/L4/test_equity_vanilla.py - the engine's reduction to verified Black and put-call
          parity; this module is pure data and is exercised through them
  crosscheck: none - a frozen product description, not a numerical convention a library implements
  slice:  tranche 30 - the equity underlying: the equity & hybrids arc opens
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from pricebook.foundation.money import Money
from pricebook.market.equity import EquityAsset


class EquityOptionKind(Enum):
    """`omega = +1` (call) or `omega = -1` (put).

    Its own enum rather than reusing the cap/floor one: a cap is not a call in anyone's vocabulary,
    and tranche 5's `CapFloorKind` and the swaption's `SwaptionKind` already set the precedent that
    each product names its own direction. What they *share* is the `omega` shape, which is why
    `black_price` depends on that shape and not on any one of them.
    """

    CALL = "call"
    PUT = "put"

    @property
    def omega(self) -> float:
        return 1.0 if self is EquityOptionKind.CALL else -1.0


@dataclass(frozen=True)
class EuropeanEquityOption:
    """A European option on one equity asset: strike, expiry, notional, direction."""

    asset: EquityAsset
    strike: float
    expiry: date
    notional: Money
    kind: EquityOptionKind

    def __post_init__(self) -> None:
        if self.strike <= 0.0:
            raise ValueError(f"the strike must be positive, got {self.strike}")
