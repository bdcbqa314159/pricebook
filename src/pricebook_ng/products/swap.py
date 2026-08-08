"""Vanilla interest-rate swap — pure product data (L2).

A frozen description with no behaviour (pricing lives in L4). A `VanillaSwap` is a
fixed leg and a float leg on one notional/currency. The legs are bundled value objects
(≤5 fields each): a swap of 8 loose primitives is the same un-bundled smell PLR0913
can't see (CLAUDE.md §3b). The float leg's projection index arrives with multicurve —
its second (projection) curve is the consumer that earns it (rule of two).

Provenance:
  quarry: python/pricebook/fixed_income/interest_rate_swap.py
  source: redesign/18 (T1 linear products); CLAUDE.md §2 (products are pure data)
  oracle: EURIBOR par swap reprices to zero NPV dual-curve through the L4 engine
  slice:  dual-curve-euribor-estr (T1 slice 2); float leg gained its index here
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook_ng.foundation import Currency, DayCountConvention, RateIndex, Schedule


@dataclass(frozen=True)
class FixedLeg:
    """A fixed leg: its accrual `schedule`, `day_count`, and fixed `rate`."""

    schedule: Schedule
    day_count: DayCountConvention
    rate: float


@dataclass(frozen=True)
class FloatLeg:
    """A floating leg: its accrual `schedule`, `day_count`, and the `index` it fixes on.
    The `index` selects the projection curve (`CurveSet.projection(index)`); for an OIS
    index that curve is the discount curve itself (single-curve degenerate)."""

    schedule: Schedule
    day_count: DayCountConvention
    index: RateIndex


@dataclass(frozen=True)
class VanillaSwap:
    """A fixed-for-float swap on one `notional`/`currency`, discounted under a CSA in
    `collateral` currency (`None` = own-currency/OIS). PV is taken from the fixed-rate
    PAYER's side (pay fixed, receive float); the receiver is its negative. (Relocation
    trigger: `collateral`/CSA moves to the L6 trade layer when L6 lands.)"""

    notional: float
    currency: Currency
    fixed_leg: FixedLeg
    float_leg: FloatLeg
    collateral: Currency | None = None


@dataclass(frozen=True)
class XccyBasisSwap:
    """A constant-notional cross-currency basis swap under a `collateral` (domestic) CSA:
    pay a domestic OIS-flat leg, receive the `foreign_leg` (its currency = `foreign_leg.index`'s)
    plus a `basis` spread, with notional exchanged at spot at start and maturity.

    Minimal single-curve-per-currency form (doc 18 §1): the domestic leg is OIS-flat, so under
    its own collateral it prices to par and the notional exchange nets to zero — its PV, and with
    it the FX spot (`N_foreign = N_domestic/S`), CANCEL. The mark is therefore the foreign leg
    (discounted on the foreign-collateral curve, projected on the foreign OIS) plus the basis
    annuity plus the foreign notional exchange. Tenor-basis, MtM-notional resets, and NDFs are
    deferred (named triggers). (Relocation trigger: collateral/CSA → the L6 trade layer.)"""

    notional: float
    foreign_leg: FloatLeg
    collateral: Currency
    basis: float
