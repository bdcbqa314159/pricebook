"""The large-homogeneous-pool (LHP) model (L3). `OKANE-2008` ch 16.2, 16.7.1.

Vasicek's infinite-homogeneous-pool limit of the one-factor Gaussian copula. In the limit of
infinitely many identical credits — one default probability `p(T)`, one recovery `R`, one factor
loading `β` — the law of large numbers collapses the *conditional* portfolio loss to a point mass
at its mean, and the whole loss distribution becomes closed form:

    L(T|Z) = (1−R)·Φ((C − βZ)/√(1−β²)),        C = Φ⁻¹(p(T))
    A(K)   = (C − √(1−β²)·Φ⁻¹(K/(1−R))) / β,   Pr(L ≥ K) = Φ(A(K))
    E[min(L,K)] = (1−R)·Φ₂(C, −A(K); −β) + K·Φ(A(K))                            (ch 16.7.1)

**Eq 16.1's bivariate normal is our `Φ₂`, and the correlation argument is `−β`** — verified, not
assumed (the T17 same-formula caution): against an independently computed, kink-split market-factor
integral, `−β` agrees to **1e-14** while `+β` is **62% to 240%** wrong.

**The closed form is undefined at exactly the two points the capital-structure identity needs, so
the branches below are load-bearing rather than defensive.** `A(K)` takes `Φ⁻¹(K/(1−R))`, which
needs `0 < K < 1−R` strictly, and `inverse_norm_cdf` refuses the endpoints rather than returning
`±∞`. The partition `Σ EL([aₖ₋₁,aₖ]) = (1−R)p` runs from `K = 0` to `K = 1`. So: `K ≤ 0` is zero,
and `K ≥ 1−R` is `(1−R)p` because the cap cannot bind — **the maximum portfolio loss is `1−R`, not
1**. With the branches the identity is exact to 2e-16; without them it raises.

**The pool has no per-name structure, which answers where the pool lives.** The LHP is the infinite
homogeneous limit, so there are no individual credits to reference: the pool's identity *is* its one
default curve, which `CurveSet.survivals` already keys by `ReferenceEntity` (tranche 41b). An index
is a reference entity for keying purposes, so no new type and no entity set — the tranche-43 spec's
"index handle vs explicit entity set" question dissolves rather than needing a choice.

Provenance:
  stock:  none — the archived predecessor had no tranche model
  source: OKANE-2008 ch 16.2 (the LHP model), ch 16.7.1 (the expected tranche loss and eq 16.1);
          `Φ₂` is foundation/bivariate_normal.py (tranche 36b), the copula tranche 41
  oracle: tests/L4/test_lhp_tranche.py — the capital-structure partition identity (exact), the
          closed form against a kink-split market-factor integral (1e-11), and the β→0 point-mass
          reduction
  crosscheck: quantlib GaussianLHPLossModel — deferred, dev-only
  slice:  tranche 43 — the LHP synthetic-CDO tranche
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.foundation.bivariate_normal import bivariate_normal_cdf
from pricebook.foundation.distributions import inverse_norm_cdf, norm_cdf
from pricebook.market.snapshot import MarketSnapshot
from pricebook.market.survival_curve import ReferenceEntity


@dataclass(frozen=True)
class TrancheSurvival:
    """`Q(T) = 1 − EL(T)/(K₂−K₁)` — the tranche's outstanding fraction, survival-curve *shaped*.

    A synthetic CDO tranche is a CDS on *this*: the premium leg pays its spread on the outstanding
    notional, which amortises exactly as expected loss eats the tranche. So it satisfies
    `credit_legs.SurvivalFunction` and the tranche-39 premium leg consumes it unchanged.
    """

    model: LHPModel
    attachment: float
    detachment: float

    def survival(self, t: float) -> float:
        return self.model.tranche_survival(self.attachment, self.detachment, t)


@dataclass(frozen=True)
class LHPModel:
    """The homogeneous pool: its default curve (by entity), its factor loading and its recovery."""

    market: MarketSnapshot
    entity: ReferenceEntity
    beta: float
    recovery: float

    def __post_init__(self) -> None:
        if not 0.0 < self.beta < 1.0:
            raise ValueError(
                f"the factor loading must satisfy 0 < β < 1, got {self.beta}: `A(K)` divides by "
                f"β, so exactly zero has a closed form (the loss is a point mass at (1−R)p) but "
                f"no `A(K)`, and |β| ≥ 1 makes the idiosyncratic weight vanish. The β → 0 limit is "
                f"approached, not reached — correlation of exactly zero is refused rather than "
                f"returning an infinity")
        if not 0.0 <= self.recovery < 1.0:
            raise ValueError(
                f"the recovery must sit in [0, 1), got {self.recovery}: at R = 1 there is no loss "
                f"to tranche and `K/(1−R)` divides by zero")

    @property
    def loss_given_default(self) -> float:
        """`1−R` — and so the **maximum** portfolio loss, which is the branch point below."""
        return 1.0 - self.recovery

    def default_probability(self, horizon: float) -> float:
        """`p(T)` for the pool, read through the snapshot by entity (§3)."""
        return 1.0 - self.market.curves.survival(self.entity).survival(horizon)

    def threshold(self, horizon: float) -> float:
        """`C(T) = Φ⁻¹(p(T))`."""
        return inverse_norm_cdf(self.default_probability(horizon))

    def loss_exceedance(self, strike: float, horizon: float) -> float:
        """`A(K)`, where `Pr(L ≥ K) = Φ(A(K))`. Defined only for `0 < K < 1−R`."""
        if not 0.0 < strike < self.loss_given_default:
            raise ValueError(
                f"A(K) needs 0 < K < 1−R = {self.loss_given_default}, got {strike}: it takes "
                f"Φ⁻¹(K/(1−R)), which the endpoints send to ±∞. Outside that range the capped "
                f"loss is closed form (0 below, (1−R)p above) and `expected_loss_capped` returns "
                f"it directly")
        return ((self.threshold(horizon)
                 - math.sqrt(1.0 - self.beta * self.beta)
                 * inverse_norm_cdf(strike / self.loss_given_default)) / self.beta)

    def expected_loss_capped(self, strike: float, horizon: float) -> float:
        """`E[min(L(T), K)]` — ch 16.7.1, with the two branches its own domain requires."""
        if strike <= 0.0:
            return 0.0
        if horizon <= 0.0:
            return 0.0
        if strike >= self.loss_given_default:
            # The cap cannot bind: `L ≤ 1−R` always, so the capped loss is the pool's whole
            # expected loss. This branch is what makes the `K = 1` end of the capital-structure
            # partition expressible at all.
            return self.loss_given_default * self.default_probability(horizon)
        exceedance = self.loss_exceedance(strike, horizon)
        return (self.loss_given_default
                * bivariate_normal_cdf(self.threshold(horizon), -exceedance, -self.beta)
                + strike * norm_cdf(exceedance))

    def tranche_expected_loss(
        self, attachment: float, detachment: float, horizon: float
    ) -> float:
        """`EL = E[min(L,K₂)] − E[min(L,K₁)]`, in pool-notional units."""
        return (self.expected_loss_capped(detachment, horizon)
                - self.expected_loss_capped(attachment, horizon))

    def tranche_survival(
        self, attachment: float, detachment: float, horizon: float
    ) -> float:
        """`Q = 1 − EL/(K₂−K₁)` — the fraction of the tranche still outstanding."""
        width = detachment - attachment
        if width <= 0.0:
            raise ValueError(
                f"a tranche needs positive width, got [{attachment}, {detachment}]")
        return 1.0 - self.tranche_expected_loss(attachment, detachment, horizon) / width

    def tranche_survival_curve(
        self, attachment: float, detachment: float
    ) -> TrancheSurvival:
        """The tranche's survival as an object the tranche-39 premium leg can consume."""
        return TrancheSurvival(self, attachment, detachment)
