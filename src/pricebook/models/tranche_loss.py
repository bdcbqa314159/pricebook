"""The tranche composition, once (L3). `OKANE-2008` ch 16, 18.

A tranche's expected loss and its outstanding fraction are the **same arithmetic** whichever model
produced the portfolio's capped expected loss:

    EL([K₁,K₂]) = E[min(L,K₂)] − E[min(L,K₁)]
    Q(K₁,K₂)    = 1 − EL/(K₂−K₁)

**Why this module exists.** Both compositions were written out in `LHPModel` (tranche 43) and again
in `RecursionModel` (tranche 44a) — identical code over a differing `expected_loss_capped`. That
duplication was not theoretical: tranche 43's sweep found the **width division** unpinned and it was
fixed *there*, against the LHP model, so tranche 44a's sweep found the same hole again in the second
copy. The lesson is not "apply the shared oracle to both implementations" but **stop having two
implementations**: one composition, one oracle, both models covered.

**Free functions over a protocol, not a base class or a mixin.** A mixin is the template-method
polymorphism §6 keeps these models away from — they are frozen value objects, and inheritance would
make the tranche arithmetic a thing they *are* rather than a thing done *to* them. Functions over a
protocol is the library's own shape (`market/credit_legs.py`, the engine registry, tranche 42's
`SurvivalFunction`), and it leaves both models concrete and frozen. *(Ruled by Cowork at tranche
44a.)*

Provenance:
  stock:  none
  source: OKANE-2008 ch 16 (the tranche expected loss and outstanding notional); the capped loss is
          each model's own — the LHP's closed form (tranche 43) or the exact recursion (tranche 44a)
  oracle: tests/L4/test_lhp_tranche.py — the width normalisation, asserted once on the shared
          function and so covering both models; the capital-structure partition in each model's
          own suite reaches this composition too
  crosscheck: none — a composition over a model's own expected loss, no numerics of its own
  slice:  tranche 44a — extracted when the second model made the duplication a measured hole
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class TrancheLossModel(Protocol):
    """A model that can say the portfolio's expected loss capped at `K`.

    One member, deliberately: everything a tranche needs follows from it, so this is the narrowest
    shape that supports the composition — and a protocol with a real member discriminates, which a
    memberless one cannot (tranche 33b).
    """

    def expected_loss_capped(self, strike: float, horizon: float) -> float: ...


def tranche_expected_loss(
    model: TrancheLossModel, attachment: float, detachment: float, horizon: float
) -> float:
    """`EL([K₁,K₂]) = E[min(L,K₂)] − E[min(L,K₁)]`, in pool-notional units."""
    return (model.expected_loss_capped(detachment, horizon)
            - model.expected_loss_capped(attachment, horizon))


def tranche_survival(
    model: TrancheLossModel, attachment: float, detachment: float, horizon: float
) -> float:
    """`Q = 1 − EL/(K₂−K₁)` — the fraction of the tranche still outstanding.

    **The width division is the line that was duplicated**, and dropping it survives a surprising
    amount: `Q` stays inside `[0,1]` and still decreases, and a premium-leg oracle that feeds
    `risky_pv01` the same curve telescopes past it. It is pinned once, here.
    """
    width = detachment - attachment
    if width <= 0.0:
        raise ValueError(
            f"a tranche needs positive width, got [{attachment}, {detachment}]")
    return 1.0 - tranche_expected_loss(model, attachment, detachment, horizon) / width


@dataclass(frozen=True)
class TrancheSurvival:
    """`Q(t)` as a survival curve the tranche-39 premium leg consumes unchanged.

    A synthetic CDO tranche is a CDS on *this*: the premium leg pays its spread on the outstanding
    notional, which amortises exactly as expected loss eats the tranche. So it satisfies
    `credit_legs.SurvivalFunction` without adapting anything.
    """

    model: TrancheLossModel
    attachment: float
    detachment: float

    def survival(self, t: float) -> float:
        return tranche_survival(self.model, self.attachment, self.detachment, t)


def tranche_survival_curve(
    model: TrancheLossModel, attachment: float, detachment: float
) -> TrancheSurvival:
    """The tranche's survival, bound to a model."""
    return TrancheSurvival(model, attachment, detachment)
