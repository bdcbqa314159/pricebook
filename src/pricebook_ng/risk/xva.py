"""XVA — valuation adjustments on top of the risk-free mark (L5 risk & capital).

The first XVA: unilateral **CVA**, the market value of counterparty default risk.
Given a trade's *expected positive exposure* profile `EE(t) = E[(V(t))^+]` on a
time grid, CVA discounts the expected loss over each interval's default probability:

    CVA = (1 - R) * sum_i  EE(t_i) * DF(t_i) * (Q(t_{i-1}) - Q(t_i))

This is exactly a CDS **protection leg** (`_protection_pv`/`cds_pv` at zero spread)
with the unit notional replaced by the exposure profile — CVA is "buying protection
on your own counterparty exposure." Unilateral form: exposure and default are taken
independent (no wrong-way risk), own default ignored (that is DVA).

Exposure generation is deliberately upstream and out of scope here: `EE(t)` is an
input. For a deterministic trade it is analytic; for an optional/path-dependent one
it comes from a Monte-Carlo exposure engine — a later slice. This module is the XVA
*integrator*, keyed to the counterparty's survival curve in the snapshot (A5), so a
credit bump (`bump_curve`/`credit01`) already gives CVA sensitivity for free.

Provenance:
  quarry: python/pricebook/risk/ (xva)
  source: Gregory, The xVA Challenge; Brigo & Mercurio (unilateral CVA)
  oracle: unit-exposure CVA == CDS protection leg (cds_pv @ 0 spread); linearity; Q≡1 -> 0
  slice:  cva (unilateral)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.market.keys import MarketKey
from pricebook_ng.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class ExposureProfile:
    """Expected exposure `E[(±V(t))^+]` on a time grid — positive (EPE) for CVA,
    negative (ENE) for DVA. `grid[0]` is the valuation date; the sum runs over the
    intervals `(grid[i-1], grid[i]]`. `grid` and `ee` are aligned and same length."""

    grid: tuple[date, ...]
    ee: tuple[float, ...]


@dataclass(frozen=True)
class ExposurePair:
    """Both sides of a trade's exposure from one simulation: expected *positive*
    exposure (EPE, feeds CVA) and expected *negative* exposure (ENE = E[(-V)^+],
    feeds DVA). On a common grid."""

    epe: ExposureProfile
    ene: ExposureProfile


@dataclass(frozen=True)
class CreditParty:
    """A defaultable party for a bilateral adjustment: its survival curve (keyed in
    the snapshot, A5) and recovery. Bundled so `bcva` stays under the arg ceiling."""

    key: MarketKey
    recovery: float


def cva(
    profile: ExposureProfile, snapshot: MarketSnapshot, key: MarketKey, recovery: float
) -> float:
    """Unilateral CVA against the counterparty whose survival curve is at `key`.

    The survival curve is reached through the `CurveHandle` capability (`df` = Q, the
    A5 survival-as-discount-factor alias), so no concrete curve type is assumed."""
    survival = snapshot.curves[key]
    discount = snapshot.discount_curve
    grid, ee = profile.grid, profile.ee
    return (1.0 - recovery) * sum(
        ee[i] * discount.df(grid[i]) * (survival.df(grid[i - 1]) - survival.df(grid[i]))
        for i in range(1, len(grid))
    )


def dva(
    profile: ExposureProfile, snapshot: MarketSnapshot, key: MarketKey, recovery: float
) -> float:
    """Debit Valuation Adjustment — the mirror of CVA: expected gain from OUR OWN
    default while out of the money. It is exactly the CVA integral on the *negative*
    exposure profile (ENE) against our own survival curve at `key`; passing an ENE
    profile is the caller's responsibility (the math is identical, the sign lives in
    the exposure)."""
    return cva(profile, snapshot, key, recovery)


def bcva(
    exposure: ExposurePair,
    snapshot: MarketSnapshot,
    counterparty: CreditParty,
    self_party: CreditParty,
) -> float:
    """Bilateral credit adjustment `BCVA = CVA - DVA` — the net credit charge (the
    value adjustment to the risk-free mark is `-BCVA`). CVA uses the counterparty's
    curve on the positive exposure; DVA uses our own curve on the negative exposure.

    Unilateral pair: exposure and default independent, no first-to-default survival
    weighting (a later refinement multiplies each term by the other party's Q(t))."""
    cva_term = cva(exposure.epe, snapshot, counterparty.key, counterparty.recovery)
    dva_term = dva(exposure.ene, snapshot, self_party.key, self_party.recovery)
    return cva_term - dva_term
