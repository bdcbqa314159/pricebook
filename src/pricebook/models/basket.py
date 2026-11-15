"""The basket survival curve and the basket model (L3). `OKANE-2008` ch 15.1-15.2, §12.2.4.

    Q_B(t|Z) = Π_i Q_i(t|Z)          — conditional independence (tranche 41's copula)
    Q_B(t)   = ∫ Q_B(t|Z) φ(Z) dZ    — integrated over the market factor

`Q_B` is the probability that **no** name in the basket defaults by `t`, which is exactly what a
first-to-default contract is written on: the basket's first default is the basket curve's default,
so the product prices as a CDS on `Q_B`.

**Why this is L3 and not L1.** It is built from a `GaussianCopula`, which is L3 — an L1 module
importing it would be the upward edge `verify.py acyclic` forbids. So the basket curve is a model
*output* that happens to be survival-curve **shaped**: it answers `survival(t)` and nothing more,
which is all the premium leg needs. (The tranche-42 spec left the layer open, to be ruled by
whichever the import direction forced; it forces this one.)

**It is not a `SurvivalCurve`, and cannot be.** A `SurvivalCurve` is a piecewise-constant *hazard*;
`Q_B` is an integral whose implied hazard varies continuously, so it has no pillars and no
piecewise-constant form to fit. That is why the protection leg here is
`credit_legs.protection_leg_from_survival` (which differences `Q`) rather than
`credit_legs.protection_leg` (which reads a hazard per pillar) — the latter **plateaus 1.56e-03
from the truth** on `Q_B`, unimproved by refinement.

Provenance:
  stock:  none — the archived predecessor had no basket
  source: OKANE-2008 ch 15.1-15.2 (default baskets, the first-to-default basket), on the §12.2.4
          basket-survival construction; the conditional survivals are tranche 41's copula
  oracle: tests/L4/test_first_to_default_basket.py — the Fréchet sandwich
          `Π Q_i ≤ Q_B ≤ min Q_i`, the ρ=0 product and ρ=±1 widest-name limits, the exact
          single-name reduction, and the rate-asserted market-factor integral
  crosscheck: quantlib NthToDefault / basket machinery — deferred, dev-only
  slice:  tranche 42 — the first-to-default basket, the first product through the copula
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.foundation.distributions import norm_cdf
from pricebook.foundation.quadrature import standard_normal_expectation
from pricebook.market.snapshot import MarketSnapshot
from pricebook.market.survival_curve import ReferenceEntity, SurvivalCurve
from pricebook.models.gaussian_copula import GaussianCopula

#: Gauss-Hermite nodes for the market-factor integral. The integrand is a product of smooth
#: conditional survivals for `|β| < 1`, so it converges fast: measured below 1e-07 by 16 nodes on
#: the tranche-42 fixture. 64 is the copula's own committed count, kept so a basket and its names
#: are integrated on one rule.
_DEFAULT_FACTOR_NODES = 64


@dataclass(frozen=True)
class BasketSurvival:
    """`Q_B(t)` for the names a copula carries — survival-curve *shaped*, not a `SurvivalCurve`."""

    copula: GaussianCopula
    nodes: int = _DEFAULT_FACTOR_NODES

    def survival(self, t: float) -> float:
        """`Pr(no name has defaulted by t)`.

        **The all-unit-loading case is analytic, and that is not an optimisation.** At `|β| = 1`
        each conditional survival is an indicator in `Z`, so the product is a step — and
        Gauss-Hermite cannot integrate a step. Measured on the tranche-42 fixture, the quadrature
        came out **6.1% below** `min_i Q_i` at `t = 5` and, at `t = 3`, **above** it, which is an
        impossible probability (the Fréchet upper bound violated). The closed form is elementary:
        every name is driven by `Z` alone, so the product is `1{Z > max_i C_i}` and
        `Q_B = 1 − Φ(max_i C_i)`, which equals `min_i Q_i` exactly — for `β = −1` as well as `+1`,
        since the indicator's *orientation* flips with the sign but the probability does not.

        A basket **mixing** unit and sub-unit loadings is refused rather than integrated: the
        integrand is then a step times a smooth part, which the quadrature gets silently wrong, and
        nothing in the library needs it (§3.4 — a wrong number is worse than a refusal).
        """
        if t <= 0.0:
            return 1.0
        loadings = [abs(name.beta) for name in self.copula.names]
        at_unit = [loading >= 1.0 for loading in loadings]
        if all(at_unit):
            return 1.0 - norm_cdf(
                max(name.threshold(t) for name in self.copula.names))
        if any(at_unit):
            raise ValueError(
                f"this basket mixes unit and sub-unit factor loadings ({loadings}), and the "
                f"market-factor integrand is then a step times a smooth part — Gauss-Hermite gets "
                f"it silently wrong (6% on a comparable fixture, and it can breach the Fréchet "
                f"bound). All-unit loadings are taken analytically; sub-unit ones integrate "
                f"cleanly. Nothing needs the mixture, so it is refused rather than approximated.")

        names = self.copula.names
        return standard_normal_expectation(
            lambda z: math.prod(1.0 - name.conditional_default(t, z) for name in names),
            nodes=self.nodes)

    def default_probability(self, t: float) -> float:
        """`1 − Q_B(t)` — the probability that at least one name has defaulted by `t`."""
        return 1.0 - self.survival(t)


@dataclass(frozen=True)
class BasketModel:
    """A copula plus the homogeneous recovery a basket's loss is measured with.

    Composes rather than copies, as `CreditSwaptionModel` does: everything about the names is the
    copula's. **Recovery sits here and not on the product**, following tranche 38's ratified
    ruling — `R` is a market assumption, not a term of the contract, and a recovery on the product
    would let two baskets on the same names carry different ones. (The tranche-42 spec listed it as
    a product field; that ruling is the older and the more considered, so it wins.)
    """

    copula: GaussianCopula
    recovery: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.recovery <= 1.0:
            raise ValueError(f"the recovery must sit in [0, 1], got {self.recovery}")

    @property
    def market(self) -> MarketSnapshot:
        """§3 — the engine reaches market data only through this."""
        return self.copula.market

    def curve_for(self, entity: ReferenceEntity) -> SurvivalCurve:
        """The named entity's own marginal, for the single-name comparisons."""
        for name in self.copula.names:
            if name.entity == entity:
                return name.survival
        raise KeyError(f"no name {entity.name} in this basket's copula")

    def basket_survival(self, nodes: int = _DEFAULT_FACTOR_NODES) -> BasketSurvival:
        return BasketSurvival(self.copula, nodes)
