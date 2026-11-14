"""The credit default swap (L2) - frozen data, prices nothing.

Provenance:
  stock:  none - the archived predecessor had no credit product in this tree
  source: OKANE-2008 (docs/references.md) ch 6 - the CDS contract: a premium leg of `S0` on a risky
          annuity against a protection leg paying `(1-R)` on default
  oracle: tests/L4/test_credit_default_swap.py - exercised through the engine; the buyer/seller
          sign symmetry and the matured-contract guard reach this directly
  crosscheck: none - a frozen product description, not a numerical convention
  slice:  tranche 38 - the credit default swap: project five opens
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pricebook.foundation.money import Money
from pricebook.foundation.schedule import CouponSchedule


class ProtectionDirection(Enum):
    """`omega = +1` (protection buyer: pays premium, receives on default) or `-1` (seller).

    Its own enum rather than reusing a call/put or payer/receiver direction. A protection buyer is
    not a call holder in anyone's vocabulary, and tranches 5, 30 and the swaption each set the
    precedent that a product names its own direction (§6: a shared *word* is not a shared shape).
    What they share is the `omega` sign, which is why the pricer multiplies by it rather than
    branching.
    """

    BUYER = "buyer"
    SELLER = "seller"

    @property
    def omega(self) -> float:
        return 1.0 if self is ProtectionDirection.BUYER else -1.0


@dataclass(frozen=True)
class CreditDefaultSwap:
    """The premium schedule, the contractual spread, the notional, and which side is held.

    **Recovery is deliberately NOT a field.** `R` is a market/model *assumption* - the standard 40%,
    not a term of the contract - so it belongs on the model (§3), where the snapshot it belongs with
    lives. A recovery on the product would let two CDS on the same name carry different recoveries,
    which is not a thing the market quotes.
    """

    schedule: CouponSchedule
    spread: float
    notional: Money
    direction: ProtectionDirection

    def __post_init__(self) -> None:
        if self.spread < 0.0:
            raise ValueError(f"a contractual spread may not be negative, got {self.spread}")
