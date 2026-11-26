"""The `Trade` — how much of a contract, and which way (L2). Frozen data; prices nothing.

**The structural track's first representation piece (T82 / R1.1).** A `Trade` is the smallest piece of
"a book": a product, how many units of it, and the direction. Ten literature arcs built the catalogue of
*contracts*; this is the first type that describes a **holding** of one.

**It is L2 and not L6, and the distinction is the §2 contract rather than a filing preference.** L6 is
the stateful tier — it *remembers*. A `Trade` remembers nothing: it is frozen data describing an
exposure, so it belongs beside the products it refers to, and `trade_value` (the pure function that
prices it) sits at L4 with the engines it dispatches to. The R1 plan originally placed the whole
representation layer at L6; that was corrected at spec time, because `verify.py layers` would have
rejected it and rightly.

**`product` is typed `object`, deliberately.** There is no `Priceable` protocol in this tree and that is
a standing decision (see `risk/__init__.py`): the registry dispatches on `type(product)` and types its
own argument `object`, so the products stay deliberately unrelated types with no base class to share
(§6 — no abstraction without two consumers sharing a *shape*). A `Trade` inherits that choice rather
than inventing a protocol for itself.

Provenance:
  stock:  none — architecture-first; the structural track anchors on the catalogue, not a source
  source: CLAUDE.md §0 (the representation contract is the durable public half) and §1 (the two-track
          rule and the purpose-defined 1.0, which names "represent, value and risk a position")
  oracle: tests/L4/test_trade.py — the reduction to `price(product, model)` bit-identical for a long
          unit trade, the long/short asymmetry on a deliberately non-symmetric payoff, and the
          notional's absolute level against an independently composed reference (T43)
  crosscheck: none — frozen representation data
  slice:  tranche 82 — R1.1, the Trade type
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from pricebook.foundation.money import Money


class TradeDirection(Enum):
    """`sign = +1` (long) or `−1` (short).

    Its own enum rather than a bare `bool` or a signed notional. A boolean would need every reader to
    remember which way `True` points, and a signed notional would make "how much" and "which way" one
    field — so a sign error and a size error would be indistinguishable, which is exactly what the
    asymmetry oracle exists to tell apart. The `sign` shape matches `EquityOptionKind`'s `omega` and
    the cap/floor and swaption kinds: each names its own direction, and what they share is the sign.
    """

    LONG = 1
    SHORT = -1

    @property
    def sign(self) -> float:
        return float(self.value)


@dataclass(frozen=True)
class Trade:
    """`direction · notional · price(product, model)` — a holding of one contract.

    **`notional` is a count of units, carried as `Money` so its currency can be checked.** The amount
    says how many units of `product` the trade holds; the product itself carries its own contract size
    (an `EuropeanEquityOption`'s `notional`, a `VarianceSwap`'s `variance_notional`), so this
    multiplies the contract rather than redefining it. The currency is the part that is *checkable* —
    `trade_value` refuses a notional denominated differently from what the product pays in — and that
    guard is why this is a `Money` and not a float. *(Raised at the T82 handoff: a count carried as
    `Money` makes `notional × price` dimensionally money², and whether the field should become a plain
    `units: float` plus a currency guard, or stay as specced, is Cowork's to rule. Built as specced.)*
    """

    product: object
    notional: Money
    direction: TradeDirection

    def __post_init__(self) -> None:
        amount = self.notional.amount
        if not math.isfinite(amount):
            raise ValueError(f"a trade's notional must be finite, got {amount}")
        if amount <= 0.0:
            # The SIZE is positive and the DIRECTION carries the sign. Allowing a negative notional
            # would give two spellings of a short, and then a sign error and a size error could not
            # be told apart -- which is what the long/short oracle is for.
            raise ValueError(
                f"a trade's notional must be positive — use the direction for a short, got {amount}"
            )
