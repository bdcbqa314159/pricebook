"""The LHP synthetic-CDO tranche pricer (L4). `OKANE-2008` ch 16.

    Premium PV    = S · RPV01(Q) · N
    Protection PV = ∫ Z(t)(−dQ(t)) · N
    V             = Protection − Premium        (long protection)

**A tranche is a CDS on its own survival**, which is what makes this engine small. `Q(T) = 1 −
EL(T)/(K₂−K₁)` is the fraction of the tranche still outstanding, and the premium leg paying its
spread on the amortising outstanding notional *is* `risky_pv01(Q)` — so the tranche-39 premium leg
and the tranche-42 `−dQ` protection leg are both reused unchanged, through the
`SurvivalFunction` protocol.

**The protection leg passes `recovery = 0`, and that is not an oversight.** The `(1−R)` factor is
already inside the expected tranche loss (the LHP's loss is `(1−R)·p(T|Z)`), so applying it again
would double-count it. What the leg integrates here is the tranche-notional loss itself, which is
why the recovery argument is the identity.

**Deterministic, so no standard error.** The LHP is closed form; a finite-pool Monte-Carlo measures
the LHP's own approximation error and is a cross-check oracle, not the engine (the tranche-41/42
pattern).

Provenance:
  stock:  none — the archived predecessor had no tranche pricer
  source: OKANE-2008 ch 16; the legs are market/credit_legs.py (tranches 39, 42), the expected
          tranche loss is models/lhp.py (tranche 43)
  oracle: tests/L4/test_lhp_tranche.py — the par reprice, linearity in the spread, the
          equity/mezzanine/senior ordering, and the premium leg matching `risky_pv01(Q)` exactly
  crosscheck: quantlib GaussianLHPLossModel — deferred, dev-only
  slice:  tranche 43 — the LHP synthetic-CDO tranche
"""

from __future__ import annotations

from typing import Protocol

from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.market.snapshot import MarketSnapshot
from pricebook.market.survival_curve import ReferenceEntity
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.market.credit_legs import protection_leg_from_survival, risky_pv01
from pricebook.models.lhp import LHPModel
from pricebook.models.tranche_loss import (
    TrancheSurvival,
    tranche_survival_curve,
)
from pricebook.products.synthetic_cdo_tranche import SyntheticCDOTranche

class TrancheModel(Protocol):
    """A model that can say what fraction of a tranche is still outstanding.

    Two present consumers share this shape, not merely a word: tranche 43's `LHPModel` (the
    infinite-pool closed form) and tranche 44a's `RecursionModel` (the exact finite-pool
    distribution) answer the *same* question by different mathematics. Everything below needs only
    this, which is why one pricer serves both and the second registration is a line rather than a
    module — the Black-vs-Heston shape, with the registry refusing a duplicate `(product, model)`
    pair so neither can silently replace the other.
    """

    @property
    def market(self) -> MarketSnapshot: ...

    @property
    def entity(self) -> ReferenceEntity: ...

    def expected_loss_capped(self, strike: float, horizon: float) -> float: ...

    @property
    def protection_steps(self) -> int:
        """How finely to integrate `∫Z(−dQ)` for *this* model.

        Numerical config living with the thing that makes it expensive (§3.5). A closed-form
        survival is cheap to sample, so tranche 43 commits 800 steps; the exact recursion costs a
        96-node sweep over an `N`-credit convolution *per step*, so tranche 44a commits 100 — where
        the measured error is 8.1e-07, four significant figures tighter than any tranche is quoted.
        The engine cannot know this, and hard-coding one number would either make the LHP needlessly
        coarse or the recursion unusably slow.
        """
        ...


def _live_periods(product: SyntheticCDOTranche, valuation_date):
    """The premium periods still to pay — invariant 6, through the one shared atom (C10)."""
    return [
        period
        for period in product.schedule.periods()
        if not is_historical(period[1], valuation_date)
    ]


def _survival(product: SyntheticCDOTranche, model: TrancheModel):
    """The tranche's survival curve, with the pool's entity checked against the model.

    A tranche naming a pool the model does not carry must fail as a value rather than quietly
    price the model's own pool — a plausible number for the wrong portfolio.
    """
    if product.reference_entity != model.entity:
        raise KeyError(
            f"this tranche is on {product.reference_entity.name} but the model carries "
            f"{model.entity.name}")
    model.market.curves.survival(product.reference_entity)
    return tranche_survival_curve(
        model, product.bounds.attachment, product.bounds.detachment)


def tranche_annuity(product: SyntheticCDOTranche, model: TrancheModel) -> float:
    """`RPV01(Q)` per unit of tranche notional — the amortising outstanding notional."""
    market = model.market
    discount = market.curves.discount(product.notional.currency)
    return risky_pv01(
        _live_periods(product, market.valuation_date), discount, _survival(product, model))


def tranche_protection(
    product: SyntheticCDOTranche, model: TrancheModel, steps: int | None = None
) -> float:
    """`∫Z(−dQ)` per unit of tranche notional."""
    market = model.market
    discount = market.curves.discount(product.notional.currency)
    periods = _live_periods(product, market.valuation_date)
    if not periods:
        return 0.0
    horizon = discount.time_measure.year_fraction(periods[-1][1])
    if steps is None:
        steps = model.protection_steps
    # `recovery = 0` deliberately: the `(1−R)` is already inside the expected tranche loss, so
    # applying it again would double-count the recovery.
    return protection_leg_from_survival(
        discount, _survival(product, model), 0.0, horizon, steps)


def par_spread(
    product: SyntheticCDOTranche, model: TrancheModel, steps: int | None = None
) -> float:
    """The spread making the tranche worth zero — the quoted tranche spread."""
    annuity = tranche_annuity(product, model)
    if annuity <= 0.0:
        raise ValueError(
            "a par spread needs a positive risky annuity; this tranche has no premium periods left")
    return tranche_protection(product, model, steps) / annuity


@register(SyntheticCDOTranche, LHPModel)
def price_synthetic_cdo_tranche(
    product: SyntheticCDOTranche, model: TrancheModel
) -> PricingResult | PricingFailure:
    """`Protection − S·RPV01`, long protection, on the tranche's own notional."""
    currency = product.notional.currency
    if not _live_periods(product, model.market.valuation_date):
        # Invariant 6: a matured tranche is the shell's business, PV 0 before any market lookup.
        return PricingResult(pv=Money(0.0, currency))
    try:
        protection = tranche_protection(product, model)
        premium = product.spread * tranche_annuity(product, model)
    except KeyError as exc:
        return PricingFailure(f"cannot price the synthetic CDO tranche: {exc}")
    return PricingResult(
        pv=Money((protection - premium) * product.notional.amount, currency))
