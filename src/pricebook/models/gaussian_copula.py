"""The one-factor Gaussian latent-variable model (L3). `OKANE-2008` ch 13 — the "copula".

    A_i = β_i·Z + √(1−β_i²)·ε_i ,    Z, ε_i ~ N(0,1) independent                     (13.3)
    C_i(T) = Φ⁻¹(1 − Q_i(T))                                                          (13.2)
    ρ_ij   = β_i·β_j                                                                  (13.2)
    p_i(T|Z) = Φ( (C_i(T) − β_i·Z) / √(1−β_i²) )                                      (13.4)
    p_ij(T)  = Φ₂( C_i(T), C_j(T); ρ_ij )

Name `i` defaults by `T` exactly when `A_i < C_i(T)`, and the threshold is set so that happens with
the CDS-implied probability `1 − Q_i(T)`. The names are coupled only through the single market
factor `Z`, so **conditional on `Z` they are independent** — the property every correlation product
exploits, and the reason the joint has two computations that must agree.

**No `price()` and no product here.** The copula is a model plus its probability functions, oracled
directly — the same shape as `SurvivalCurve` at tranche 25. Tranche 42's first-to-default basket is
the product that pulls it, so it is not speculative: two queued tranches consume it immediately.

**Where the per-name marginals live, and why that is a question for the next tranche.** The names
are carried on the model, as the tranche-41 spec rules. They cannot currently be carried in the
snapshot: `CurveSet.survivals` is keyed by **`Currency`**, so two names in EUR are not
representable. That is fine while nothing prices — §3's "the engine reaches market data only
through `model.market`" binds engines — but tranche 42 brings the first engine, and then a model
holding market data outside the snapshot is a real tension. The shape of the fix already exists in
the tree (`equities` is keyed by `EquityAsset(ticker)`); raised for Cowork rather than decided here.

Provenance:
  stock:  none — the archived predecessor had no copula
  source: OKANE-2008 ch 13 — §13.2 (the latent variable and threshold), §13.3 (the one-factor
          form and its ρ = 0, ±1 limits), §13.4 (conditional independence)
  oracle: tests/L3/test_gaussian_copula.py — the market-factor integral recovering each marginal
          (the anchor, a different computation from the threshold), the three exact limits with a
          non-vacuous ρ = −1 fixture, and Φ₂ against the conditional-independence integral
  crosscheck: quantlib OneFactorGaussianCopula / GaussianLHPLossModel — deferred, dev-only
  slice:  tranche 41 — the Gaussian copula opens the multi-name half
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.foundation.bivariate_normal import bivariate_normal_cdf
from pricebook.foundation.distributions import inverse_norm_cdf, norm_cdf
from pricebook.foundation.quadrature import standard_normal_expectation
from pricebook.market.snapshot import MarketSnapshot
from pricebook.market.survival_curve import SurvivalCurve

#: Gauss-Hermite nodes for the market-factor integral. Measured: the marginal recovery reads 1e-08
#: by 16 nodes on a moderate `β`, and the committed 64 clears 1e-09 even at `β = 0.9` with a
#: far-out threshold, where convergence is **not monotone** (2.1e-04 at 8 nodes, 1.6e-03 at 16,
#: 1.9e-06 at 32) because the integrand approaches a step in `Z` as `β → 1`. 64 is the margin that
#: makes the non-monotone region irrelevant rather than a tuned value.
_DEFAULT_FACTOR_NODES = 64


@dataclass(frozen=True)
class CreditName:
    """One name's contribution: its factor loading and its CDS-implied marginal.

    Bundled rather than held as two parallel tuples on the copula, and **not** because of the §7
    field limit — `(market, betas, survivals)` would be three fields. The reason is that parallel
    tuples make a mis-pairing expressible: name 0's beta against name 1's curve is a wrong number
    with no symptom, and it is exactly the index bug the suite's distinct-`β`/distinct-`Q` premises
    exist to expose. Pairing them here makes it unrepresentable instead of merely tested.
    """

    beta: float
    survival: SurvivalCurve

    def __post_init__(self) -> None:
        if not math.isfinite(self.beta) or abs(self.beta) > 1.0:
            raise ValueError(
                f"a factor loading must satisfy |β| ≤ 1, got {self.beta}: `A_i = βZ + √(1−β²)ε` "
                f"has unit variance only then, and |β| > 1 makes the idiosyncratic weight "
                f"imaginary")

    def default_probability(self, horizon: float) -> float:
        """`1 − Q_i(T)` — the marginal the threshold is calibrated to."""
        return 1.0 - self.survival.survival(horizon)

    def threshold(self, horizon: float) -> float:
        """`C_i(T) = Φ⁻¹(1 − Q_i(T))` — eq 13.2."""
        return inverse_norm_cdf(self.default_probability(horizon))

    def conditional_default(self, horizon: float, factor: float) -> float:
        """`p_i(T|Z) = Φ((C_i − β_iZ)/√(1−β_i²))` — eq 13.4.

        At `β = ±1` the idiosyncratic weight is **exactly zero** and this is an indicator, not a
        division: the name's fate is fully determined by `Z`. The `ρ = ±1` reductions need exactly
        that case, so it is handled rather than guarded against.
        """
        idiosyncratic = 1.0 - self.beta * self.beta
        shifted = self.threshold(horizon) - self.beta * factor
        if idiosyncratic <= 0.0:
            return 1.0 if shifted > 0.0 else 0.0
        return norm_cdf(shifted / math.sqrt(idiosyncratic))


@dataclass(frozen=True)
class GaussianCopula:
    """The portfolio: one market factor, one latent variable per name."""

    market: MarketSnapshot
    names: tuple[CreditName, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("a copula needs at least one name")
        for position, name in enumerate(self.names):
            if name.survival.time_measure.anchor != self.market.valuation_date:
                raise ValueError(
                    f"name {position}'s survival curve is on a different time measure than the "
                    f"snapshot: anchored at {name.survival.time_measure.anchor}, the snapshot is "
                    f"valued at {self.market.valuation_date}. A shared clock is what makes one "
                    f"horizon mean the same thing for every name.")

    def threshold(self, index: int, horizon: float) -> float:
        """`C_i(T)` — eq 13.2."""
        return self.names[index].threshold(horizon)

    def asset_correlation(self, first: int, second: int) -> float:
        """`ρ_ij = β_i·β_j` — eq 13.2. The **product**, not either loading alone."""
        return self.names[first].beta * self.names[second].beta

    def conditional_default(self, index: int, horizon: float, factor: float) -> float:
        """`p_i(T|Z)` — eq 13.4."""
        return self.names[index].conditional_default(horizon, factor)

    def marginal_from_factor(
        self, index: int, horizon: float, nodes: int = _DEFAULT_FACTOR_NODES
    ) -> float:
        """`E_Z[p_i(T|Z)]`, which must equal `1 − Q_i(T)`.

        The correctness anchor: a Gauss-Hermite integral against an inverse CDF is a genuinely
        different computation from the threshold that produced it, so a sign or a weight error in
        eq 13.4 cannot cancel the way it does in `Φ(C_i) = 1 − Q_i`. The factor model must
        reproduce the single-name defaults it was calibrated to — the credit analogue of tranche
        8's known-curve round trip.
        """
        name = self.names[index]
        if 1.0 - name.beta * name.beta <= 0.0:
            # At |β| = 1 the conditional probability is an indicator in `Z`, and Gauss-Hermite
            # **cannot integrate a step**: the nodes are fixed, so refining does not converge —
            # measured 13%, 18%, 19%, 5% wrong at 16, 64, 256, 370 nodes, non-monotone and never
            # right. The expectation is elementary instead: for β = +1 the event `C − Z > 0` is
            # `Z < C`, so `E = Φ(C)`; for β = −1 it is `Z > −C`, so `E = 1 − Φ(−C) = Φ(C)`. Either
            # way it is `Φ(C_i) = 1 − Q_i`, the marginal itself. Returning the quadrature's answer
            # here would be a silent 18% error in a configuration the ρ = ±1 limits make natural.
            return name.default_probability(horizon)
        return standard_normal_expectation(
            lambda z: name.conditional_default(horizon, z), nodes=nodes)

    def joint_default(self, first: int, second: int, horizon: float) -> float:
        """`p_ij(T) = Φ₂(C_i, C_j; ρ_ij)` — the pairwise joint, in closed form.

        Φ₂ is tranche 36b's, and it is **exact at all three limits** this model reduces to
        (measured: `ρ = 0` gives the product, `ρ = 1` the minimum, `ρ = −1` the Fréchet lower
        bound, each to the last bit), because it is derived from Plackett's identity rather than a
        quadrature in `ρ`. Its regime — `|ρ| ≤ 0.99`, `Φ₂ ≳ 1e-8` — carries over; outside it
        `joint_default_from_factor` is the better side.
        """
        return bivariate_normal_cdf(
            self.threshold(first, horizon), self.threshold(second, horizon),
            self.asset_correlation(first, second))

    def joint_default_from_factor(
        self, first: int, second: int, horizon: float, nodes: int = _DEFAULT_FACTOR_NODES
    ) -> float:
        """`E_Z[p_i(T|Z)·p_j(T|Z)]` — the same joint by conditional independence.

        The product *inside* the expectation is the whole content of §13.4: unconditionally the
        defaults are dependent, conditionally they are not. A second computation of `p_ij` by a
        different route, which is what makes the agreement with `joint_default` evidence rather
        than a restatement.
        """
        near, far = self.names[first], self.names[second]
        return standard_normal_expectation(
            lambda z: near.conditional_default(horizon, z) * far.conditional_default(horizon, z),
            nodes=nodes)
