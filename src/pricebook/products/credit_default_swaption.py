"""The knockout default swaption (L2). `OKANE-2008` ch 9.3.

The option to enter a CDS on `[expiry, maturity]` at a strike spread. Frozen data.

**Knockout only, and there is no flag saying so.** The standard contract knocks out if the name
defaults before expiry, and the knockout is not a separate term — it is already carried by the
forward risky annuity, which weights by survival to expiry. The non-knockout variant is
`knockout + front-end protection` (O'Kane's own decomposition) and is a *different instrument*, out
of scope here. A `knockout: bool` field that may only ever be `True` would be a knob for a
hypothetical future (§6), and would read as though both were priceable; the type's name and this
docstring state the scope instead. Adding the variant means adding its product, not flipping a flag.

Provenance:
  stock:  none — the archived predecessor had no CDS swaption
  source: OKANE-2008 ch 9.3 (the default swaption, knockout)
  oracle: tests/L4/test_forward_cds_default_swaption.py — put-call parity against the forward CDS,
          the zero-vol intrinsic, and a pointwise transcription of the Black composition
  crosscheck: none — frozen data, no numerics
  slice:  tranche 40 — the forward CDS and the default swaption
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from pricebook.foundation.money import Money
from pricebook.market.survival_curve import ReferenceEntity
from pricebook.products.forward_cds import ForwardPeriod


class SwaptionDirection(Enum):
    """Payer buys protection on exercise; receiver sells it.

    Its own enum rather than `ProtectionDirection`, though both are a two-sided `omega`: the words
    describe different things. A protection *direction* is who holds the CDS; a payer/receiver
    swaption is which way the *option* points, and the pair that reads naturally beside a strike is
    payer/receiver. They share a shape and not a meaning, which §6 keeps apart.
    """

    PAYER = "payer"
    RECEIVER = "receiver"

    @property
    def omega(self) -> float:
        return 1.0 if self is SwaptionDirection.PAYER else -1.0


@dataclass(frozen=True)
class CreditDefaultSwaption:
    """An option expiring at `expiry` on a CDS running from `expiry` to the schedule's end."""

    period: ForwardPeriod
    strike: float
    notional: Money
    direction: SwaptionDirection
    reference_entity: ReferenceEntity

    def __post_init__(self) -> None:
        if self.strike <= 0.0:
            raise ValueError(
                f"the strike spread must be positive, got {self.strike}: Black takes log(F/K), so "
                f"a non-positive strike is undefined rather than merely extreme")
