"""The variance replication's numerical config, and the snapshot it reads (L3).

**There is no pricing model here, and that is the point.** The fair variance of a variance swap is
*model-independent*: it is a weighted strip of traded option prices, so nothing is fitted and no
dynamics are assumed. What the engine still needs is a `MarketSnapshot` (§3 - the model carries it,
so a model/market mismatch cannot be expressed) and the **strike grid** the continuum integral
becomes in practice.

The grid is numerical config, which §3.5 says arrives with its first consumer and is never defaulted
into place - a hidden grid would make the fair variance silently depend on a number nobody chose, and
the grid's width is exactly what controls the truncation error. So it is an explicit field, and
`StrikeGrid` lives here rather than in the engine because an L3 model may not import L4.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 3.7 / 5.8 - the replication is model-free, which is
          why this object carries config and a snapshot rather than parameters
  oracle: tests/L4/test_variance_swap.py - the grid's validity regime is asserted directly (a
          truncated grid underprices; widening converges upward to `sigma^2`; refining improves at
          the trapezoid rate)
  crosscheck: none - numerical config plus a snapshot handle
  slice:  tranche 31 - the variance swap by static replication
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from pricebook.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class StrikeGrid:
    """The discrete strike strip, as multiples of the forward: `[lower*F, upper*F]` in `count` nodes.

    Expressed **relative to the forward** rather than in absolute strikes so one grid is meaningful
    across underlyings and expiries - the integrand's shape is a function of moneyness, not of the
    level of the index.

    **Its width is the truncation error and its count is the discretisation error**, and they fail in
    opposite directions: too narrow systematically *underprices* (the tails are real value the strip
    never sees), too coarse adds trapezoid error of either sign. Both are asserted in the oracle,
    because a grid chosen by eye is a reference with an unstated accuracy (§4).
    """

    lower: float
    upper: float
    count: int

    def __post_init__(self) -> None:
        if self.lower <= 0.0:
            raise ValueError(f"the grid's lower bound must be positive, got {self.lower}")
        if self.upper <= self.lower:
            raise ValueError(
                f"the grid's upper bound must sit above its lower bound, got {self.lower}..{self.upper}"
            )
        if self.count < 2:
            raise ValueError(f"integrating the strip needs at least two strikes, got {self.count}")

    def strikes(self, forward: float) -> tuple[float, ...]:
        """The grid's absolute strikes, **with the forward itself as a node**.

        The forward is included deliberately, not incidentally. The strip switches from puts to
        calls at `K = F`, and while the integrand is *continuous* there (put-call parity makes the
        two prices equal), its **derivative is not** - so an interval straddling `F` contains a kink
        and the trapezoid rule loses its second-order rate on it. Measured on a uniform grid that
        ignores `F`, successive refinements gave error ratios of 0.85, 22.1, 0.61, 3.25 - noise
        around no rate at all, because the error was dominated by how close `F` happened to fall to
        a node at each count.

        Putting `F` on the grid makes each side smooth, which restores the rate and lets the oracle
        assert convergence instead of merely hoping for it. The two sides get node counts in
        proportion to their widths, so the step is near-uniform across the whole strip.
        """
        span = self.upper - self.lower
        put_fraction = (1.0 - self.lower) / span
        put_nodes = max(2, round(put_fraction * (self.count - 1)) + 1)
        call_nodes = max(2, self.count - put_nodes + 1)
        puts = [
            forward * (self.lower + (1.0 - self.lower) * k / (put_nodes - 1))
            for k in range(put_nodes)
        ]
        calls = [
            forward * (1.0 + (self.upper - 1.0) * k / (call_nodes - 1)) for k in range(1, call_nodes)
        ]
        return tuple(puts + calls)


#: One term of the strip, already weighted: `(price, strike, forward) -> contribution`.
#:
#: The *weighted* contribution rather than a bare weight, so each product's weight is a single
#: division and the vanilla path stays **bit-identical** to tranche 31's `price / (K * K)`. A bare
#: `weight(K, F)` multiplied in afterwards would be `price * (1/K²)` — the same quantity in exact
#: arithmetic, one extra rounding, and T31's oracles would have had to be rebanded for nothing.
StripTerm = Callable[[float, float, float], float]


def _vanilla_term(price: float, strike: float, forward: float) -> float:
    """`1/K²` — the vanilla variance swap and the corridor (KWOK-ZHENG eq 1.19, eq 1.26)."""
    return price / (strike * strike)


def _gamma_term(price: float, strike: float, forward: float) -> float:
    """`1/(K·F₀)` — eq (1.24)'s `1/K` weight with its `1/F₀` normalisation folded in.

    Folded in deliberately: written this way the three products share one prefactor `2/T`, so the
    strip walk has exactly one varying part. In the prefactor instead it would have had two.
    """
    return price / (strike * forward)


@dataclass(frozen=True)
class VarianceStrip:
    """How a generalized variance swap re-weights and restricts the OTM strip (KWOK-ZHENG §1.4).

    Vocabulary, not abstraction (§6): the grid, the weight and the range are exactly what a strip is,
    and bundling them is what keeps `generalized_fair_variance` inside §7's five arguments. Built
    through the three named constructors rather than assembled by callers, so a weight and a range
    cannot be paired in a combination no product has.

    **`lower`/`upper` are absolute strike levels and are NOT the grid's bounds.** The grid's are
    multiples of the forward and express the quadrature's *truncation* — a numerical choice whose
    error T31 measures. These are a **contract term**: the corridor the trade names. Conflating them
    would make a wider grid change the payoff.
    """

    grid: StrikeGrid
    term: StripTerm
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower < 0.0:
            raise ValueError(f"a strip's lower bound cannot be negative, got {self.lower}")
        if self.upper <= self.lower:
            raise ValueError(
                f"a strip's upper bound must sit above its lower bound, got "
                f"{self.lower}..{self.upper}"
            )

    @classmethod
    def vanilla(cls, grid: StrikeGrid) -> VarianceStrip:
        """The `1/K²` strip over the whole range — tranche 31's variance swap."""
        return cls(grid, _vanilla_term, 0.0, math.inf)

    @classmethod
    def gamma(cls, grid: StrikeGrid) -> VarianceStrip:
        """The `1/(K·F₀)` strip over the whole range — the gamma swap, eq (1.24)."""
        return cls(grid, _gamma_term, 0.0, math.inf)

    @classmethod
    def corridor(cls, grid: StrikeGrid, lower: float, upper: float) -> VarianceStrip:
        """The `1/K²` strip restricted to `(lower, upper]` — the corridor swap, eq (1.26)."""
        return cls(grid, _vanilla_term, lower, upper)

    def restricted(self, forward: float) -> tuple[float, ...]:
        """The grid's strikes that lie in `(lower, upper]`, refusing a corridor off the forward.

        The replication splits at the forward — puts below, calls above — so a corridor entirely on
        one side would integrate one empty half and return a plausible number for a contract the
        formula does not describe. §1.4 states the assumption ("it is reasonable to choose the
        corridor so that F0 lies within the corridor"); this enforces it.
        """
        if not self.lower < forward <= self.upper:
            raise ValueError(
                f"the corridor {self.lower}..{self.upper} must contain the forward {forward}: "
                f"eq (1.26) splits the strip at the forward, so a corridor on one side of it "
                f"integrates an empty half"
            )
        return tuple(k for k in self.grid.strikes(forward) if self.lower < k <= self.upper)


@dataclass(frozen=True)
class VarianceReplicationModel:
    """The snapshot the strip reads, and the grid it integrates on."""

    market: MarketSnapshot
    grid: StrikeGrid
