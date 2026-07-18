"""XVA — valuation adjustments on top of the risk-free mark (L5 risk & capital).

The XVA *integrators* — all the same discounted sum over an exposure profile
(`ExposureProfile`/`ExposurePair`, generated upstream by the MC exposure engine),
differing only in the weight:

    CVA = (1 - R_C) * sum_i  EPE_i * DF_i * (Q_C,i-1 - Q_C,i)   counterparty default
    DVA = (1 - R_B) * sum_i  ENE_i * DF_i * (Q_B,i-1 - Q_B,i)   own default (= CVA on ENE)
    BCVA = CVA - DVA                                            net credit charge
    FVA = s_F * sum_i (EPE_i - ENE_i) * DF_i * S_i * tau_i      funding of the position
    KVA = gamma_K * sum_i K_i * DF_i * S_i * tau_i              cost of regulatory capital
    MVA = s_F * sum_i IM_i * DF_i * S_i * tau_i                 funding of initial margin

CVA/DVA weight exposure by a *protection leg* — default increments `(Q_{i-1}-Q_i)`
and `(1-R)`, exactly a CDS `cds_pv` at zero spread. FVA, KVA and MVA weight a profile
by a *funding annuity* — a rate over each interval `S*tau`, exactly the CDS RPV01
(FVA: funding spread on net exposure; KVA: cost of capital on the capital profile;
MVA: funding spread on the IM profile). Keyed to the survival curves in the snapshot
(A5), so a credit bump (`bump_curve`/`credit01`) gives XVA sensitivity for free.

Unilateral/independence scope: exposure and default independent (no wrong-way risk),
no first-to-default survival weighting, symmetric funding spread. Exposure generation
(`risk/exposure.py`) and capital/EAD generation (the RWA slice) are upstream.

Provenance:
  quarry: python/pricebook/risk/ (xva)
  source: Gregory, The xVA Challenge; Brigo & Mercurio
  oracle: unit-exposure CVA == protection leg; FVA/KVA/MVA == rate·RPV01; BCVA = CVA - DVA
  slice:  cva (unilateral); bcva (DVA/bilateral); fva (funding); kva (capital); mva (margin)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.market.keys import MarketKey
from pricebook_ng.market.snapshot import MarketSnapshot

_FUNDING_DC = DayCountConvention.ACT_360  # money-market funding accrual


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


def fva(
    exposure: ExposurePair, snapshot: MarketSnapshot, key: MarketKey, funding_spread: float
) -> float:
    """Funding Valuation Adjustment — the cost of funding an uncollateralised position.
    `FVA = FCA - FBA = s_F * Sum_i (EPE_i - ENE_i) * DF(t_i) * S(t_i) * tau_i`: the
    funding spread carried over each interval on the *net* exposure, discounted and
    survival-weighted (funding stops on default — `key` is the funding-relevant
    survival curve). Where CVA weights exposure by default increments and `(1-R)`, FVA
    weights it by the survival annuity `S*tau` — the CDS RPV01 structure.

    Scope: symmetric funding spread (one `s_F` for borrow/lend), a single survival curve
    (own vs joint is a later refinement). FVA/DVA overlap is a known modelling debate,
    out of scope here — this is the discounting-approach FVA on the given exposure."""
    net = ExposureProfile(
        exposure.epe.grid,
        tuple(e - n for e, n in zip(exposure.epe.ee, exposure.ene.ee)),
    )
    return _annuity_adjustment(net, snapshot, key, funding_spread)


def kva(
    capital: ExposureProfile, snapshot: MarketSnapshot, key: MarketKey, cost_of_capital: float
) -> float:
    """Capital Valuation Adjustment — the cost of holding regulatory capital over the
    trade's life: `KVA = gamma_K * Sum_i K(t_i) * DF(t_i) * S(t_i) * tau_i`, the cost of
    capital `gamma_K` charged on the capital profile `K(t)`, discounted and survival-
    weighted (capital is only held while alive — `key` is the relevant survival curve).
    The same survival annuity as FVA, with capital in place of net exposure.

    `K(t)` is an input (a `CapitalProfile`-shaped `ExposureProfile`): generating it from a
    regulatory model (SA-CCR EAD -> RWA -> capital) is the upstream RWA slice, exactly as
    the exposure engine is upstream of CVA."""
    return _annuity_adjustment(capital, snapshot, key, cost_of_capital)


def mva(
    im: ExposureProfile, snapshot: MarketSnapshot, key: MarketKey, funding_spread: float
) -> float:
    """Margin Valuation Adjustment — the funding cost of posting initial margin over the
    trade's life: `MVA = s_F * Sum_i IM(t_i) * DF(t_i) * S(t_i) * tau_i`, the same survival-
    weighted funding annuity as FVA/KVA with the IM profile in place of net exposure/capital.

    `IM(t)` is an input: generating it — SIMM, or a dynamic MC-quantile IM over the margin
    period of risk — is upstream, a later slice (as the exposure engine is upstream of CVA)."""
    return _annuity_adjustment(im, snapshot, key, funding_spread)


def _annuity_adjustment(
    profile: ExposureProfile, snapshot: MarketSnapshot, key: MarketKey, rate: float
) -> float:
    """`rate * Sum_i profile_i * DF(t_i) * S(t_i) * tau_i` — the survival-weighted annuity
    shared by FVA (funding spread on net exposure), KVA (cost of capital on the capital
    profile), and MVA (funding spread on the IM profile). The CDS RPV01 structure, so a unit
    profile gives `rate * RPV01`."""
    survival = snapshot.curves[key]
    discount = snapshot.discount_curve
    grid, values = profile.grid, profile.ee
    return rate * sum(
        values[i]
        * discount.df(grid[i])
        * survival.df(grid[i])
        * year_fraction(grid[i - 1], grid[i], _FUNDING_DC)
        for i in range(1, len(grid))
    )
