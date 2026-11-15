"""The exact-recursion tranche pricer (L4). `OKANE-2008` ch 18.

**A second pricer for the same product**, which is the whole point: a `SyntheticCDOTranche` priced
against tranche 43's `LHPModel` gets the infinite-pool approximation, and against tranche 44a's
`RecursionModel` gets the exact finite-pool distribution. Same contract, different mathematics —
the Black-vs-Heston shape the `(product, model)` registry key exists for, and the registry refuses
a duplicate pair so neither registration can silently replace the other.

**The arithmetic is not duplicated.** A tranche is a CDS on its own survival whichever model
produced that survival, so the legs, the par spread and the pricer body are tranche 43's, reached
through the `TrancheModel` protocol: both models answer `tranche_survival_curve`, which is a shared
*shape* and not a shared word (§6). This module is the registration plus the recursion's named entry
points — re-exported so a caller (and an oracle) can reach the exact engine's legs by name without
knowing they are shared.

Provenance:
  stock:  none
  source: OKANE-2008 ch 18; the legs and the pricer body are engine/synthetic_cdo_tranche.py
          (tranche 43), the exact distribution is models/loss_recursion.py (tranche 44a)
  oracle: tests/L4/test_loss_recursion.py — the par reprice through this engine, the exact price
          differing from the LHP's on one tranche, the notional scaling, and the protection leg
          against an independent discounted-loss sum
  crosscheck: quantlib RecursiveLossModel / IntegralCDOEngine — deferred, dev-only
  slice:  tranche 44a — the exact loss distribution by recursion
"""

from __future__ import annotations

from pricebook.engine.registry import register
from pricebook.engine.synthetic_cdo_tranche import (
    par_spread,
    price_synthetic_cdo_tranche,
    tranche_annuity,
    tranche_protection,
)
from pricebook.models.loss_recursion import RecursionModel
from pricebook.products.synthetic_cdo_tranche import SyntheticCDOTranche

__all__ = [
    "par_spread",
    "price_recursion_tranche",
    "tranche_annuity",
    "tranche_protection",
]

#: The same body as tranche 43's, registered for the exact model. Registered by name rather than
#: re-decorating the imported function, because the registry would otherwise see one function
#: object under two keys and a future reader could not tell which registration was intended.
price_recursion_tranche = register(SyntheticCDOTranche, RecursionModel)(
    price_synthetic_cdo_tranche)
