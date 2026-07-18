"""European swaption as pure data (L2).

A `Swaption` is the right, at `expiry`, to enter the underlying (forward-starting)
`VanillaSwap`. Payer vs receiver is the underlying swap's direction
(`swap.pay_fixed`). Pure data — no `pv` method (CLAUDE.md 2); the Jamshidian
valuation lives in the L4 SwaptionEngine.

Provenance:
  quarry: python/pricebook/fixed_income/ (swaption)
  source: European swaption on a vanilla IRS
  oracle: put-call parity + ATM symmetry + sigma->0 intrinsic (S08)
  slice:  S08
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.products.swap import VanillaSwap


@dataclass(frozen=True)
class Swaption:
    """A European option (at `expiry`) to enter `swap` — payer iff swap.pay_fixed."""

    expiry: date
    swap: VanillaSwap
