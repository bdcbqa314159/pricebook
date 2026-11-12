"""Engine registry — structural product→pricer dispatch, inside L4 (CLAUDE.md §1).

The registry replaces an `isinstance`-on-instrument ladder: each pricer registers for its
`(product type, model type)` pair, and `dispatch` looks the pricer up by that pair — one lookup,
open to new products, never a growing `if/elif`. It lives inside L4 (no leaky top-level
`registry.py`). An unregistered pair is a value (`PricingFailure`), never a raise.

**The key is a PAIR, and tranche 33 is why.** The original key was the product type alone, which
silently assumed one pricer per product. That assumption broke the first time a product legitimately
had two models: the equity vanilla has Black (tranche 30), the local-volatility MC (32) and Heston
(33), and a second `@register(EuropeanEquityOption)` **replaced** the first with no warning — proved
empirically at tranche 33, where registering Heston would have removed Black from the table and
repriced every equity vanilla without a single test failing, since the signatures are identical.

The model type was always part of what identifies a pricer — §3's contract takes it as the second
argument — so widening the key records something already true rather than adding a concept. And the
overwrite becomes **loud**: registering the same pair twice raises, because a pricer silently
replaced is debt incurred without a ledger entry (§8).

**The model key may be a CONCRETE type or a PROTOCOL, because both are real.** Widening the key to
concrete types alone looked like a mechanical pass and was not: two of the eight engines — the
inflation cap/floor and the year-on-year swap — are typed against the `IndexRatioModel`
**protocol** and are polymorphic over every model satisfying it (Jarrow-Yildirim, both market
models, and a test's own stub). That polymorphism is the point of the protocol, not an accident of
the old key, so the registry records whichever of the two a pricer actually means:

* a pricer written for one model registers under that **concrete type**;
* a pricer written against a capability registers **once** under the **protocol**.

**Dispatch resolves exact first, then the unique protocol match.** An exact `(product, type(model))`
hit wins, because a pricer naming a concrete model is making a more specific claim than one naming a
capability. Otherwise the registry looks for protocol entries the model satisfies — and **two
matches is an error, not a coin flip**: a silent arbitrary choice between two pricers is the same
defect this tranche exists to remove, one level along. A model satisfying a registered protocol
needs no registration of its own, which is why a stub defined in `tests/` dispatches without `src/`
knowing it exists.

Provenance:
  stock:  _stock/src/pricebook/engine/registry.py (purified as-is, except that the
          `CalibratedModel` protocol it typed the model against is NOT carried: that protocol
          exists to declare a model's capabilities, and a discounting model has none to declare
          beyond carrying its market. It returns with the first model that has dynamics — and a
          pricer needing a capability will state it then, which is the point of a protocol)
  source: CLAUDE.md-era §1 (engine registry inside L4; no isinstance ladders)
  oracle: an unregistered product returns a PricingFailure, not a raise
          (test_an_unregistered_product_type_is_a_FAILURE)
  crosscheck: none — product→pricer dispatch, no numerics
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations


from collections.abc import Callable
from typing import Any

from pricebook.foundation import PricingFailure, PricingResult

# The instrument param is `Any`: each pricer takes its own concrete product type, and the
# registry keys by that type, so dispatch is type-safe at the key even though the callable
# signatures are heterogeneous. The model param is `object` for now because no capability
# protocol exists yet to narrow it to; a pricer that needs one (a vol surface, a sampler) will
# declare it structurally rather than naming a concrete model type.
Pricer = Callable[..., "PricingResult | PricingFailure"]
_PRICERS: dict[tuple[type, type], Pricer] = {}


class AmbiguousRegistration(RuntimeError):
    """Raised when a model satisfies two registered protocols for one product.

    The registry could pick either, and picking arbitrarily is exactly the silent-wrong-pricer
    failure the `(product, model)` key was introduced to stop. Failing loudly keeps the ambiguity
    a design question for whoever registered the second protocol, rather than a number nobody
    questions.
    """


class DuplicateRegistration(RuntimeError):
    """Raised when a `(product, model)` pair is registered twice.

    Its own type rather than a bare `RuntimeError`, so a caller can distinguish *this* failure — a
    programming error at import time — from anything raised while pricing. Deliberately an
    exception and not a `PricingFailure`: §3.4's "failure is a value" governs *pricing* outcomes,
    and this is a module failing to load correctly, which must stop everything rather than be
    handed to someone as a price.
    """


def register(product_type: type, model_type: type) -> Callable[[Pricer], Pricer]:
    """Register a pricer for the `(product_type, model_type)` pair (used as a decorator).

    Both types are named explicitly rather than derived from the pricer's annotations: an
    annotation can be a string under `from __future__ import annotations`, a protocol, or absent,
    and a registry that guessed would fail in whichever of those cases nobody tested.
    """

    def _register(pricer: Pricer) -> Pricer:
        key = (product_type, model_type)
        existing = _PRICERS.get(key)
        if existing is not None:
            raise DuplicateRegistration(
                f"{product_type.__name__} with {model_type.__name__} is already registered to "
                f"{existing.__name__}; registering {pricer.__name__} would replace it silently"
            )
        _PRICERS[key] = pricer
        return pricer

    return _register


def registered_pricers() -> dict[tuple[type, type], Pricer]:
    """A copy of the registry — named surface, so a test can see what is registered.

    A copy rather than the live dict: handing out the mapping itself would let a caller mutate the
    registry through the back door, which is the exact silent replacement this tranche set out to
    make impossible.
    """
    return dict(_PRICERS)


def dispatch(instrument: object, model: object) -> PricingResult | PricingFailure:
    """Price `instrument` by dispatching on its type to the registered pricer.

    **No `numerics` parameter yet, deliberately.** The stock threads an optional numerical-config
    argument here and inspects each pricer's signature to decide whether to pass it. That branch
    was carried into the first draft of this tranche and **ran in no test**, because the only
    pricer is closed-form and takes two arguments — an unexercised dispatch path at the baseline,
    which is the same dead-code shape the debt gate caught one commit earlier in
    `DiscountCurve.zero_rate`. The rule it answers to is the config's own: the numerical-config
    value *lands with the first engine that reads it*. When a pricer integrates, samples or steps
    a grid, it arrives then — with a consumer and a test — rather than sitting here as plumbing
    for a pricer that does not exist."""
    pricer = _resolve(type(instrument), model)
    if pricer is None:
        return PricingFailure(
            f"no pricer registered for {type(instrument).__name__} with {type(model).__name__}"
        )
    return pricer(instrument, model)


def _resolve(product_type: type, model: object) -> Pricer | None:
    """The exact-beats-protocol lookup. Returns `None` when nothing is registered.

    Separated from `dispatch` so the resolution rule is one readable thing and so a test can reach
    it; `dispatch` stays the §3 contract boundary that turns a miss into a `PricingFailure`.
    """
    exact = _PRICERS.get((product_type, type(model)))
    if exact is not None:
        return exact

    matches = [
        (key, pricer)
        for key, pricer in _PRICERS.items()
        if key[0] is product_type and _is_protocol(key[1]) and isinstance(model, key[1])
    ]
    if len(matches) > 1:
        names = ", ".join(sorted(key[1].__name__ for key, _ in matches))
        raise AmbiguousRegistration(
            f"{type(model).__name__} matches {len(matches)} registered protocols for "
            f"{product_type.__name__} ({names}); register it concretely to disambiguate"
        )
    return matches[0][1] if matches else None


def _is_protocol(candidate: type) -> bool:
    """Whether a model key is a runtime-checkable protocol rather than a concrete type.

    `isinstance` against a non-runtime-checkable protocol raises, so this is a guard and not a
    tidiness check: without it, registering a plain `Protocol` would make every dispatch for that
    product explode instead of simply not matching.
    """
    return getattr(candidate, "_is_protocol", False) and getattr(
        candidate, "_is_runtime_protocol", False
    )
