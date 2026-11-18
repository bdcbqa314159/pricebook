"""The vanilla fixed-vs-float interest-rate swap (L2) — frozen data, prices nothing.

**This product did not exist, which is a scope finding worth stating in the module that fixes it.**
Tranche 55's spec named "an existing swap/leg product and the single-curve swap pricer" as reuse
anchors; the tree had neither. It priced inflation swaps, caps, swaptions, equity options and
credit, and had never carried a plain fixed-vs-float IRS — the swap *pieces* were there
(`nominal_bootstrap.swap_annuity`, `ParSwapQuote`) but a quote is market data, not a trade. So the
collateralised-discounting arc had to bring the product with it.

**The tenor is carried, and it is constitutive rather than decoration** (§6's test): a 3M-floating
and a 6M-floating swap on the same dates are *different trades*, priced off different projection
curves, and the gap between them is the tenor basis this arc exists to price. It is read by the
pricer, so it is not a carried-but-unread field either.

**Five fields, which is the limit and not a problem** (§7): schedule, fixed rate, notional, tenor,
direction. Nothing here bundles - each is a separate fact a desk would quote - and the limit is a
ceiling on un-bundled vocabulary rather than a target.

Provenance:
  stock:  none - the archived predecessor had no vanilla IRS product
  source: LICHTERS-2015 (docs/references.md) ch 5 - the collateralised swap whose legs discount at
          the collateral rate; the product description itself is vocabulary, not a transcription
  oracle: tests/L4/test_csa_discounting.py - exercised through the registered pricer: a swap struck
          at the par rate is worth zero, payer and receiver are exact negatives, the payer gains
          when rates exceed its fixed rate, and a matured swap is PV 0
  crosscheck: none - a frozen product description
  slice:  tranche 55 - CSA discounting
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pricebook.foundation.money import Money
from pricebook.foundation.schedule import CouponSchedule
from pricebook.foundation.tenor import Tenor


class SwapDirection(Enum):
    """`omega = +1` (payer: pays the fixed leg, receives the floating one) or `-1` (receiver).

    Its own enum rather than reusing `SwaptionKind`, whose members carry exactly these semantics:
    they share the *shape* (`omega = +-1` on the fixed leg) but a swap is not a swaption, and a
    product field reading `SwaptionKind` would be the shared-word coupling §6 warns against wearing
    the right type. The tree's established pattern is a kind per product family unified by the
    `OptionDirection` protocol where the arithmetic is shared (`models/black.py` says so), and this
    is the fourth member of that family."""

    PAYER = "payer"
    RECEIVER = "receiver"

    @property
    def omega(self) -> float:
        return 1.0 if self is SwapDirection.PAYER else -1.0


@dataclass(frozen=True)
class InterestRateSwap:
    """Pays the fixed leg and receives the tenor-floating leg (or the reverse), on `schedule`.

    The fixed and floating legs share the schedule, which is the convention tranche 54's bootstrap
    calibrates on - so a swap priced here reprices the quote it was stripped from."""

    schedule: CouponSchedule
    fixed_rate: float
    notional: Money
    tenor: Tenor
    direction: SwapDirection

    def __post_init__(self) -> None:
        if self.notional.amount <= 0.0:
            raise ValueError(f"a swap needs a positive notional, got {self.notional}")
        if len(self.schedule.dates) < 2:
            raise ValueError("a swap schedule needs at least one period")
