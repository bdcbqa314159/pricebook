"""Curves, keyed (L1).

A set of curves reached through a **typed accessor**, not by position or by string. The key
carries the asset dimension — here a currency and a role — so a new asset class later adds
*keys*, not fields, and nothing already written has to change shape.

Provenance:
  stock:  _stock/src/pricebook/market/curve_set.py (purified: the DISCOUNT role only; projection
          and survival roles arrive with the products that need them)
  source: CLAUDE.md-era doc 19 §3 (closed shapes × open keys)
  oracle: a missing curve raises a KeyError the engine turns into a PricingFailure
  crosscheck: none — a keyed container of curves, no numerics
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from pricebook.foundation.money import Currency
from pricebook.foundation.tenor import Tenor
from pricebook.market.curve import CurveHandle
from pricebook.market.survival_curve import ReferenceEntity, SurvivalCurve


class CurveRole(Enum):
    """What a curve is *for*.

    `REAL` arrives in tranche 2 as a **new key, not a new snapshot field** — which is the whole
    point of keying: an inflation product needs a second discount curve (`Pr` alongside `Pn`), and
    adding it costs one enum member while `MarketSnapshot`, `CurveSet` and every existing pricer
    stay exactly as they were."""

    DISCOUNT = "discount"
    REAL = "real"


@dataclass(frozen=True)
class CurveKey:
    """Identifies a curve: its role and the asset dimension it is keyed by."""

    role: CurveRole
    currency: Currency


@dataclass(frozen=True)
class ProjectionKey:
    """Identifies a forward-projection curve: its currency and its **tenor**.

    Its own key rather than a widened `CurveKey`, and its own store on `CurveSet`, because a
    projection curve is keyed by a dimension `DISCOUNT`/`REAL` do not have. Adding `tenor` to
    `CurveKey` would hang a `None` on every existing key — a carried-but-unread field (§6) on the
    one value object every pricer in the tree already constructs. The precedent is `survivals`,
    which got its own store for exactly this reason: a different keying dimension, not another
    role. *(Ruled at tranche 54's handoff, built at tranche 55 where the first pricer reads it.)*
    """

    currency: Currency
    tenor: Tenor


@dataclass(frozen=True)
class CurveSet:
    """Curves, reached through accessors that state the narrowing once.

    Two backing stores, not one, because a survival curve is a different *kind* of curve,
    not another `CurveRole` key, and it is keyed by `ReferenceEntity` rather than by currency
    (tranche 41b). Tranche 1 predicted survival would arrive as a role; when
    the CDS actually pulled it at tranche 38 the type did not fit: `CurveHandle` answers
    `df_at`, while a hazard curve answers `survival` and `default_density`. Keying it as a
    role would have made a hazard curve answer a discounting question — a shared word, not
    a shared shape (charter §6) — so it gets its own store and its own typed accessor."""

    curves: Mapping[CurveKey, CurveHandle]
    survivals: Mapping[ReferenceEntity, SurvivalCurve] = field(default_factory=dict)
    projections: Mapping[ProjectionKey, CurveHandle] = field(default_factory=dict)

    def real(self, currency: Currency) -> CurveHandle:
        """The real (inflation-linked) discount curve `Pr` for `currency`.

        Its own accessor rather than `discount(currency, role=...)`: the two curves answer
        different questions, and a role parameter would let a caller pass the nominal curve where
        the real one belongs by changing one argument. Raises on a missing curve — defaulting to
        the nominal one would silently price every inflation swap to zero, which is a plausible
        wrong number."""
        key = CurveKey(CurveRole.REAL, currency)
        try:
            return self.curves[key]
        except KeyError:
            raise KeyError(f"no real curve for {currency} in the curve set") from None

    def discount(self, currency: Currency) -> CurveHandle:
        """The discount curve for `currency`. Raises on an unknown key rather than defaulting to
        a flat-zero curve: a silently risk-free curve prices everything to its undiscounted
        notional, which is a wrong number that looks plausible."""
        key = CurveKey(CurveRole.DISCOUNT, currency)
        try:
            return self.curves[key]
        except KeyError:
            raise KeyError(f"no discount curve for {currency} in the curve set") from None

    def projection(self, currency: Currency, tenor: Tenor) -> CurveHandle:
        """The forward-projection curve for `(currency, tenor)`.

        Raises on a missing curve rather than falling back to the discount curve: that fallback is
        precisely the pre-crisis pricing error, and it would price every collateralised trade as if
        there were no tenor basis — a wrong number wearing the right shape.

        **Its absence is how the era is chosen — and that choice now lives in
        `curves_for_era`, not in each caller.** This accessor still never guesses: it answers for a
        projection curve that must exist, and raises when it does not. A caller that wants the era
        decided asks `curves_for_era`. *(Through tranche 58 the era test was written out at each
        consumer, and the comment here said the choice was "the caller's business"; re-audit #6's
        F-6.6 found two copies and §6's rule-of-two fired.)*"""
        try:
            return self.projections[ProjectionKey(currency, tenor)]
        except KeyError:
            raise KeyError(
                f"no projection curve for {currency} {tenor.label} in the curve set"
            ) from None

    def curves_for_era(
        self, currency: Currency, tenor: Tenor
    ) -> tuple[CurveHandle, CurveHandle]:
        """`(projection, discount)` for the era this curve set describes — **the one place the
        mode is selected.**

        A projection curve for `(currency, tenor)` means the post-crisis era, and its absence means
        the pre-crisis one, where the discount curve projects its own forwards. The era is read off
        the **market**, never a flag, so there is no second code path to atrophy — and because both
        branches return the same pair shape, every consumer is written once.

        **Why this lives at L1, and why it is one method rather than two call sites.** Through
        tranche 58 the lookup was written out twice — in the swap pricer (L4) and in the
        collateralised capstone (L5) — and the swap's helper docstring claimed to be "the one place
        the mode is selected" while that was false. Re-audit #6 (F-6.6) found the pair: a shared
        **shape**, not a shared word, so §6's rule-of-two fires. It belongs here because
        `projections` is here, below both consumers, so dependencies still point down. The real
        argument is not tidiness: **the single-curve design invariant is realised through exactly
        this lookup**, so single-sourcing it gives the invariant one structural home instead of a
        per-consumer re-derivation.
        """
        discount = self.discount(currency)
        if ProjectionKey(currency, tenor) in self.projections:
            return self.projection(currency, tenor), discount
        return discount, discount

    def survival(self, entity: ReferenceEntity) -> SurvivalCurve:
        """The survival curve for `entity`. Raises on an unknown key rather than defaulting to a
        never-defaults curve: survival ≡ 1 zeroes the protection leg and prices a CDS at minus
        its premium, which is a wrong number that looks plausible. Equally, it never falls back to
        *another* name's curve, which would be a wrong number wearing the right shape."""
        try:
            return self.survivals[entity]
        except KeyError:
            raise KeyError(
                f"no survival curve for {entity} in the curve set") from None
