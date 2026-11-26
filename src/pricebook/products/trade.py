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
    """`direction · units · price(product, model)` — a holding of one contract.

    **`units` is DIMENSIONLESS, and that is forced rather than chosen** (ruled 0446). Every engine
    returns a **money** PV, and `trade_value = multiplier · pv` must also be money — so the
    multiplier carries no units. The first draft typed this field `Money`, which makes the product
    money², and the symptom was having to describe the amount as "the real multiplier" while the
    currency rode along "as the part that is checkable": a field doing two jobs because neither fit.
    The **product** already carries its own contract size and currency (a `$10M` swap is the product;
    its PV is in dollars); the **trade** carries *how many of it*. The currency is read off the
    `PricingResult`, never duplicated here — so there is nothing to disagree with, and the
    currency-mismatch guard the earlier shape needed is gone rather than fixed.

    This is the first representation type and `Position`, `PricingRequest` and the mark-to-market
    procedure all inherit its shape, which is why the dimensional honesty is worth more here than
    anywhere downstream.
    """

    product: object
    units: float
    direction: TradeDirection

    def __post_init__(self) -> None:
        if not math.isfinite(self.units):
            raise ValueError(f"a trade's units must be finite, got {self.units}")
        if self.units <= 0.0:
            # The SIZE is positive and the DIRECTION carries the sign. Allowing negative units would
            # give two spellings of a short, and then a sign error and a size error could not be told
            # apart -- which is what the long/short oracle is for.
            raise ValueError(
                f"a trade's units must be positive — use the direction for a short, got {self.units}"
            )
