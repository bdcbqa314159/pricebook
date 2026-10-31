"""Engine registry — structural product→pricer dispatch, inside L4 (CLAUDE.md §1).

The registry replaces an `isinstance`-on-instrument ladder: each pricer registers for its
product type, and `dispatch` looks the pricer up by `type(instrument)` — one lookup, open
to new products, never a growing `if/elif`. It lives inside L4 (no leaky top-level
`registry.py`). An unregistered product is a value (`PricingFailure`), never a raise.

Provenance:
  stock:  _stock/src/pricebook/engine/registry.py (purified as-is, except that the
          `CalibratedModel` protocol it typed the model against is NOT carried: that protocol
          exists to declare a model's capabilities, and a discounting model has none to declare
          beyond carrying its market. It returns with the first model that has dynamics — and a
          pricer needing a capability will state it then, which is the point of a protocol)
  source: CLAUDE.md-era §1 (engine registry inside L4; no isinstance ladders)
  oracle: an unregistered product returns a PricingFailure, not a raise
          (test_an_unregistered_product_type_is_a_FAILURE)
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations

import inspect

from collections.abc import Callable
from typing import Any

from pricebook.foundation import PricingFailure, PricingResult

# The instrument param is `Any`: each pricer takes its own concrete product type, and the
# registry keys by that type, so dispatch is type-safe at the key even though the callable
# signatures are heterogeneous. The model param is `object` for now because no capability
# protocol exists yet to narrow it to; a pricer that needs one (a vol surface, a sampler) will
# declare it structurally rather than naming a concrete model type.
Pricer = Callable[..., "PricingResult | PricingFailure"]
_PRICERS: dict[type, Pricer] = {}


def register(product_type: type) -> Callable[[Pricer], Pricer]:
    """Register a pricer for `product_type` (used as a decorator on the pricer)."""

    def _register(pricer: Pricer) -> Pricer:
        _PRICERS[product_type] = pricer
        return pricer

    return _register


def dispatch(
    instrument: object, model: object, numerics: object | None = None
) -> PricingResult | PricingFailure:
    """Price `instrument` by dispatching on its type to the registered pricer.

    `numerics` carries the explicit reproducibility knobs (invariant 5). It is **optional at this
    boundary and required by the pricers that read it**: a closed-form pricer has no knobs and must
    not be made to accept one (§6b), while a pricer that integrates returns a `PricingFailure` when
    it is absent — never a guessed grid. That is what invariant 5 forbids: a *hidden default*, not
    an optional parameter. A pricer declares it needs numerics by accepting a third argument."""
    pricer = _PRICERS.get(type(instrument))
    if pricer is None:
        return PricingFailure(f"no pricer registered for {type(instrument).__name__}")
    if _takes_numerics(pricer):
        return pricer(instrument, model, numerics)
    return pricer(instrument, model)


def _takes_numerics(pricer: Pricer) -> bool:
    """Whether the pricer declares a third (numerics) parameter — so closed-form pricers keep
    their two-argument signature and nothing is threaded where it has no meaning."""
    try:
        return len(inspect.signature(pricer).parameters) >= 3
    except (TypeError, ValueError):
        return False
