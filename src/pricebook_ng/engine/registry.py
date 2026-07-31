"""Engine registry — structural product→pricer dispatch, inside L4 (CLAUDE.md §1).

The registry replaces an `isinstance`-on-instrument ladder: each pricer registers for its
product type, and `dispatch` looks the pricer up by `type(instrument)` — one lookup, open
to new products, never a growing `if/elif`. It lives inside L4 (no leaky top-level
`registry.py`). An unregistered product is a value (`PricingFailure`), never a raise.

Provenance:
  quarry: python/pricebook/pricing/registry.py
  source: CLAUDE.md §1 (engine registry inside L4; no isinstance ladders)
  oracle: slice-1/2 par-swap-to-zero still green through the registry-dispatched engine
  slice:  cash-instruments (T1 slice 3, Step A)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pricebook_ng.foundation import PricingFailure, PricingResult
from pricebook_ng.models.discounting_model import DiscountingModel

# The instrument param is `Any`: each pricer takes its own concrete product type, and the
# registry keys by that type, so dispatch is type-safe at the key even though the callable
# signatures are heterogeneous.
Pricer = Callable[[Any, DiscountingModel], "PricingResult | PricingFailure"]
_PRICERS: dict[type, Pricer] = {}


def register(product_type: type) -> Callable[[Pricer], Pricer]:
    """Register a pricer for `product_type` (used as a decorator on the pricer)."""

    def _register(pricer: Pricer) -> Pricer:
        _PRICERS[product_type] = pricer
        return pricer

    return _register


def dispatch(instrument: object, model: DiscountingModel) -> PricingResult | PricingFailure:
    """Price `instrument` by dispatching on its type to the registered pricer."""
    pricer = _PRICERS.get(type(instrument))
    if pricer is None:
        return PricingFailure(f"no pricer registered for {type(instrument).__name__}")
    return pricer(instrument, model)
