"""The exact portfolio loss distribution by recursion (L3). `OKANE-2008` ch 18.

Conditional on the market factor the credits are independent, so the loss distribution is built by
adding one credit at a time and convolving its loss into the buckets:

    f_{k+1}(m | Z) = f_k(m | Z)·(1 − q) + f_k(m−1 | Z)·q,     q = p(T|Z)
    f(m)           = ∫ f(m | Z) φ(Z) dZ
    E[min(L,K)]    = Σ_m f(m)·min(m·u, K),                    u = (1−R)/N

**This is the exact thing the LHP approximates, and the relationship is the anchor.** Tranche 43's
closed form is this recursion's `N → ∞` limit, and the convergence is **exactly `O(1/N)`** —
measured error ratios of 1.81, 1.97, 2.00, 1.99 per doubling of the pool. So the arc's loop closes:
the approximation built first becomes the reference the exact model is checked against, by a *rate*
rather than a tolerance.

**The recursion is exact at every finite `N`** — only the LHP is an approximation. Hence the mean
identity `Σ f(m)·m·u = (1−R)p(T)` holds for `N = 3` as tightly as for `N = 40`, which is a sharper
check than any tranche price.

**Conditional on `Z` this is the binomial, in closed form**, which is where the recursion step is
pinned: `f(m|Z) = C(N,m)q^m(1−q)^{N−m}`. Checking it *before* integrating over `Z` matters, because
a dropped or swapped branch still yields a distribution summing to one afterwards.

**The tranche composition is `models/tranche_loss.py`'s**, shared with the LHP: this model
supplies `expected_loss_capped` and nothing about tranches. The fused two-cap sweep that lived here
is gone with it — measured at 1.74x the shared form, which the `slow` budget affords (projected 6.4s
on CI against a 10s gate) and which one composition is worth.

**Cost is why this was split from base correlation** (tranche 44b): one `E[min(L,K)]` at 96 nodes
runs 4 ms at `N = 40`, 44 ms at `N = 160` and 179 ms at `N = 320`, against 0.4 s for tranche 43's
whole test file. The recursion is the arc's one expensive quadrature.

Provenance:
  stock:  none — the archived predecessor had no loss distribution
  source: OKANE-2008 ch 18 (building the full loss distribution by recursion); the conditional
          default probability is the copula's (tranche 41), the limit is the LHP's (tranche 43)
  oracle: tests/L4/test_loss_recursion.py — the binomial conditional pointwise, the distribution
          identities (sum, mean, non-negativity, the capital-structure partition), and the
          rate-asserted `O(1/N)` convergence to tranche 43's closed form
  crosscheck: quantlib RecursiveLossModel / IntegralCDOEngine — deferred, dev-only
  slice:  tranche 44a — the exact loss distribution by recursion
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.foundation.distributions import inverse_norm_cdf, norm_cdf
import numpy as np

from pricebook.foundation.quadrature import (
    standard_normal_expectation,
    standard_normal_nodes,
)
from pricebook.market.snapshot import MarketSnapshot
from pricebook.market.survival_curve import ReferenceEntity

#: Gauss-Hermite nodes for the market-factor integral. The conditional distribution is a smooth
#: function of `Z` for `|β| < 1` (it is a binomial in a `Φ` of `Z`), so convergence is fast; 96 is
#: the count the tranche-44a convergence measurements were taken at.
_DEFAULT_FACTOR_NODES = 96


@dataclass(frozen=True)
class RecursionModel:
    """A finite homogeneous pool: its default curve, factor loading, recovery and credit count."""

    market: MarketSnapshot
    entity: ReferenceEntity
    beta: float
    recovery: float
    pool_size: int

    def __post_init__(self) -> None:
        if not 0.0 < self.beta < 1.0:
            raise ValueError(
                f"the factor loading must satisfy 0 < β < 1, got {self.beta}: at |β| = 1 the "
                f"conditional default probability is an indicator and the recursion's `Z` "
                f"integral is a step the quadrature cannot take (the tranche-41 lesson)")
        if not 0.0 <= self.recovery < 1.0:
            raise ValueError(
                f"the recovery must sit in [0, 1), got {self.recovery}: at R = 1 there is no loss "
                f"to distribute")
        if self.pool_size < 2:
            raise ValueError(
                f"a pool needs at least two credits, got {self.pool_size}: one credit makes the "
                f"recursion a single Bernoulli draw and the `N → ∞` limit meaningless")

    @property
    def protection_steps(self) -> int:
        """100, measured rather than chosen.

        Each step costs a 96-node market-factor sweep over an `N`-credit convolution, so this is
        the arc's one expensive quadrature. Against a 3200-step reference the protection leg
        converges cleanly at second order — 3.3e-06, 8.1e-07, 2.0e-07, 4.9e-08, 1.2e-08 at 50, 100,
        200, 400, 800 steps — so 100 resolves **8.1e-07**, four significant figures tighter than a
        tranche is ever quoted, for a third of a second rather than three.
        """
        return 100

    @property
    def loss_given_default(self) -> float:
        return 1.0 - self.recovery

    @property
    def loss_unit(self) -> float:
        """`u = (1−R)/N` — one credit's loss as a fraction of the pool's notional."""
        return self.loss_given_default / self.pool_size

    def default_probability(self, horizon: float) -> float:
        return 1.0 - self.market.curves.survival(self.entity).survival(horizon)

    def threshold(self, horizon: float) -> float:
        return inverse_norm_cdf(self.default_probability(horizon))

    def conditional_default(self, horizon: float, factor: float) -> float:
        """`q = p(T|Z) = Φ((C − βZ)/√(1−β²))` — one credit's conditional default probability."""
        return norm_cdf(
            (self.threshold(horizon) - self.beta * factor)
            / math.sqrt(1.0 - self.beta * self.beta))

    def conditional_distribution(self, horizon: float, factor: float) -> tuple[float, ...]:
        """`f(m | Z)` for `m = 0 … N` — the recursion, one credit at a time.

        Conditional independence is what makes this a convolution at all: each credit either
        defaults (shifting the bucket by one) or does not, with probabilities that do not depend on
        the others once `Z` is fixed. For a homogeneous pool the result is the binomial in closed
        form, which the suite asserts pointwise — the recursion is the general machinery, and the
        binomial is the case where it can be checked exactly.
        """
        probability = self.conditional_default(horizon, factor)
        buckets = [1.0]
        for _ in range(self.pool_size):
            nxt = [0.0] * (len(buckets) + 1)
            for defaults, weight in enumerate(buckets):
                nxt[defaults] += weight * (1.0 - probability)
                nxt[defaults + 1] += weight * probability
            buckets = nxt
        return tuple(buckets)

    def loss_distribution(
        self, horizon: float, nodes: int = _DEFAULT_FACTOR_NODES
    ) -> tuple[float, ...]:
        """`f(m)` — the conditional distribution integrated over the market factor."""
        return tuple(
            standard_normal_expectation(
                lambda z, defaults=defaults: self.conditional_distribution(horizon, z)[defaults],
                nodes=nodes)
            for defaults in range(self.pool_size + 1))

    def _capped_payoffs(
        self, payoff: np.ndarray, horizon: float, nodes: int
    ) -> float:
        """`E[Σ_m f(m|Z)·payoff(m)]`, with the recursion run for **all `Z` nodes at once**.

        The recursion's only `Z`-dependence is the scalar `q = p(T|Z)`, so the convolution is the
        same arithmetic at every node and vectorises across them: one array of shape
        `(nodes, N+1)` advanced `N` times, instead of `nodes × N` interpreter steps. That is what
        brings this inside the `slow` budget on CI — where the file ran 18.3s against a 10s gate
        while measuring 5.3s locally, which is §9's point that a timing reading is only valid where
        it is taken, and a reminder that my machine is not CI.

        The **outer** sum stays `math.fsum` over weight-times-value, exactly as
        `standard_normal_expectation` does it, so vectorising the inner convolution does not change
        the summation order of the integral itself.
        """
        points, weights = standard_normal_nodes(nodes)
        factors = np.asarray(points)
        # The library's own `Φ`, evaluated per node rather than a vectorised substitute.
        # `scipy.special.ndtr` would vectorise in one call and would also mean this model's `Φ`
        # is **not** the `Φ` the copula and the LHP use — a convention divergence of exactly the
        # kind the register exists to prevent, for no measurable gain: the 96 scalar calls are
        # negligible beside the `N`-step convolution that was the actual cost.
        scale = math.sqrt(1.0 - self.beta * self.beta)
        threshold = self.threshold(horizon)
        probability = np.asarray(
            [norm_cdf((threshold - self.beta * float(factor)) / scale) for factor in factors])

        buckets = np.zeros((factors.size, self.pool_size + 1))
        buckets[:, 0] = 1.0
        for credit in range(self.pool_size):
            survive = buckets[:, : credit + 1] * (1.0 - probability)[:, None]
            default = buckets[:, : credit + 1] * probability[:, None]
            buckets[:, : credit + 2] = 0.0
            buckets[:, : credit + 1] += survive
            buckets[:, 1 : credit + 2] += default
        per_node = buckets @ payoff
        return math.fsum(
            weight * value for weight, value in zip(weights, per_node.tolist()))

    def expected_loss_capped(
        self, strike: float, horizon: float, nodes: int = _DEFAULT_FACTOR_NODES
    ) -> float:
        """`E[min(L(T), K)] = Σ_m f(m)·min(m·u, K)`."""
        if strike <= 0.0 or horizon <= 0.0:
            return 0.0
        losses = np.arange(self.pool_size + 1) * self.loss_unit
        return self._capped_payoffs(np.minimum(losses, strike), horizon, nodes)
