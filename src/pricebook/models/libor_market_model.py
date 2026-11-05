"""The LIBOR market model (BGM/J) and its drift (L3). JAECKEL-2002 §12.2, eq 12.8.

A set of spanning forward rates evolving lognormally,

    df_i / f_i = µ_i(f, t)dt + σ_i(t)dW_i                                              (13.1)

with the drift that makes each forward a martingale under the right measure — transcribed from
**both** renderings, which agree:

    µ_i = { −σ_i · Σ_{k=i+1}^{N−1} (f_k τ_k)/(1 + f_k τ_k)·σ_k·ρ_ik      for i <  N−1
          {  0                                                           for i =  N−1
          { +σ_i · Σ_{k=N}^{i}     (f_k τ_k)/(1 + f_k τ_k)·σ_k·ρ_ik      for i ≥  N     (12.8)

`N` is the index of the numéraire bond `P_N` — the measure we are in. **Three things make this
formula easy to get wrong**, and each has its own oracle: the sign **flips** between branches; the
summation bounds **differ** (`i+1 … N−1` going up, `N … i` going down); and the middle branch is
**exactly zero**.

**That middle branch is the anchor, and it is the finance rather than the formula.** `µ_{N−1} = 0`
says the forward spanning the numéraire's own accrual period is a martingale under that measure — a
requirement that survives a transcription error in either sum, which is precisely what charter §4
asks a requirement oracle to do. Its mandatory companion is that a non-terminal drift is non-zero
and state-dependent, since `return 0.0` would satisfy the anchor at every numéraire.

**State-dependent, which is what makes it a market model.** Jäckel: *"the drift terms given by
equation (12.8) are clearly state-dependent and thus indirectly stochastic"* — so the drift must
move with the forwards, and one frozen at `f(0)` would be a different model. Oracled both by a shift
and by two exact reductions: the drift vanishes at zero forwards (every term carries
`f_k τ_k/(1+f_k τ_k)`) and at zero volatility (`σ_i` is an overall factor).

**Concrete, with no base class shared with MM-I or MM-II.** All three are "market models" by name,
and that is the shared *word* §6 warns about: MM-I and MM-II model inflation forward indices under an
inflation measure, this models nominal forward rates under a bond's forward measure, and none of
their parameter sets or payoff maps line up. Three concrete types.

**The correlation is a full matrix, and tranche 17 is the unit that pulled it.** Tranche 16 shipped a
single scalar `ρ` and said a real structure would arrive "with the unit that needs one (a swaption fit
discriminates correlation shapes; a caplet strip does not)". The European swaption is that unit: its
whole reason to need the LMM rather than a caplet strip is the forward-forward correlation, and a
**flat** `ρ` makes the *indexing* of `ρ_ik` invisible — transpose it, or offset it, and a flat matrix
returns the same number. `LmmCorrelation.flat` keeps the flat case one call away, and its equivalence
to the old scalar branch is proved bit-identically rather than argued.

Provenance:
  stock:  parked/ng/.../swaption_mc.py and models/mc_processes.py — mined for SHAPE and edge cases
          only via docs/stock/mc_base_inventory.md; never imported (§5)
  source: JAECKEL-2002 (docs/references.md) §12.2 eq 12.8 (the drift) and §13.2 eq 13.1 (the SDE),
          both read in `-raw` and `-layout`, which agree. MCINERNEY-ZASTAWNIAK ch 5 is held as the
          LMM model background (its §5.3 drifts and change of numéraire).
  oracle: tests/L3/test_libor_market_model.py — `µ_{N−1} = 0` EXACTLY at every numéraire (the
          martingale anchor) with its non-vacuity companion; the pointwise transcription of 12.8
          swept over every `(i, N)` pair; the sign flip; the differing summation bounds; the `ρ_ii = 1`
          self term; state dependence; and the two exact reductions at zero forwards and zero vol
  crosscheck: quantlib LiborForwardModel / LmmDriftCalculator — deferred to the optional QuantLib tier
  slice:  tranche 16 — the LIBOR market model by Monte Carlo
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class LmmTenor:
    """The spanning forward-rate structure: accruals `τ_i`, forwards `f_i`, volatilities `σ_i`.

    One bundle rather than three loose sequences, because they are indexed together and a caller
    who passed them in the wrong order would get a plausible wrong number — the lengths are checked
    once, here, instead of at every use."""

    accruals: tuple[float, ...]
    forwards: tuple[float, ...]
    volatilities: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.accruals)
        if not (n == len(self.forwards) == len(self.volatilities)):
            raise ValueError(
                f"accruals, forwards and volatilities must have the same length, got "
                f"{n}, {len(self.forwards)}, {len(self.volatilities)}"
            )
        if n == 0:
            raise ValueError("a tenor structure needs at least one forward")
        for tau in self.accruals:
            if tau <= 0.0:
                raise ValueError(f"an accrual must be positive, got {tau}")
        for sigma in self.volatilities:
            if sigma < 0.0:
                raise ValueError(f"a volatility cannot be negative, got {sigma}")

    def __len__(self) -> int:
        return len(self.accruals)


@dataclass(frozen=True)
class LmmCorrelation:
    """The FRA/FRA correlation `ρ_ij` — Jäckel §12.4's object, whose eq 12.12 gives one parametric form.

    A frozen matrix rather than a loose nested tuple, because the three things that make it a
    *correlation* matrix — square, symmetric, unit diagonal — are checked once here instead of
    trusted at every read. `ρ_ii = 1` living in the matrix is what lets `drift_matrix` drop its
    `1.0 if k == i else ρ` branch: the diagonal is data, not a special case.

    Positive-semidefiniteness is deliberately **not** checked here. It is a property of the
    covariance the simulation decomposes, and `foundation.linalg.pseudo_square_root` already refuses
    a non-PSD matrix with a rounding floor — the right place, since that is where it bites. Checking
    it twice would mean two thresholds that can disagree."""

    matrix: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        n = len(self.matrix)
        if n == 0:
            raise ValueError("a correlation matrix needs at least one entry")
        for i, row in enumerate(self.matrix):
            if len(row) != n:
                raise ValueError(f"a correlation matrix must be square; row {i} has {len(row)} of {n}")
        for i in range(n):
            if self.matrix[i][i] != 1.0:
                raise ValueError(
                    f"a correlation matrix must have a unit diagonal; entry ({i},{i}) is "
                    f"{self.matrix[i][i]}"
                )
            for j in range(n):
                if not -1.0 <= self.matrix[i][j] <= 1.0:
                    raise ValueError(
                        f"a correlation must lie in [-1, 1]; entry ({i},{j}) is {self.matrix[i][j]}"
                    )
                if self.matrix[i][j] != self.matrix[j][i]:
                    raise ValueError(
                        f"a correlation matrix must be symmetric; ({i},{j}) is "
                        f"{self.matrix[i][j]} but ({j},{i}) is {self.matrix[j][i]}"
                    )

    @classmethod
    def flat(cls, size: int, rho: float) -> LmmCorrelation:
        """Every distinct pair at `rho`, the diagonal at exactly `1.0` — tranche 16's scalar case."""
        return cls(tuple(tuple(1.0 if i == j else rho for j in range(size)) for i in range(size)))

    def __len__(self) -> int:
        return len(self.matrix)


@dataclass(frozen=True)
class LiborMarketModel:
    """The BGM/J model under the `P_N`-forward measure, with a flat correlation.

    `numeraire_index` is `N` of eq 12.8 — the bond whose forward measure the drift is written in.
    It ranges over `1 … n`, so `N − 1` is a valid forward index and the zero-drift branch is always
    reachable."""

    tenor: LmmTenor
    correlation: LmmCorrelation
    numeraire_index: int

    def __post_init__(self) -> None:
        if len(self.correlation) != len(self.tenor):
            raise ValueError(
                f"the correlation must be {len(self.tenor)}x{len(self.tenor)} to match the tenor, "
                f"got {len(self.correlation)}"
            )
        if not 1 <= self.numeraire_index <= len(self.tenor):
            raise ValueError(
                f"the numeraire index must lie between 1 and {len(self.tenor)}, got "
                f"{self.numeraire_index}"
            )

    def drift(self, index: int) -> float:
        """`µ_i(f, t)` — eq 12.8 at this model's own forwards, under the `P_N`-forward measure."""
        return self.drift_at(self.tenor.forwards, index)

    def drift_at(self, forwards: Sequence[float], index: int) -> float:
        """`µ_i(f, t)` — eq 12.8 at an **arbitrary state** `forwards`.

        **The state-taking form is the primary one, and that is not an optimisation detail.** Eq
        12.8 is state-dependent, so a simulation needs the drift at each step's *evolved* forwards,
        not at the model's initial ones."""
        n = len(self.tenor)
        if not 0 <= index < n:
            raise IndexError(f"forward index must lie in [0, {n}), got {index}")
        if len(forwards) != n:
            raise ValueError(f"the state must hold {n} forwards, got {len(forwards)}")
        if index == self.numeraire_index - 1:
            # EXACT, by construction: the forward spanning the numeraire's own period is a
            # martingale under that measure. This is the finance, which is why it is the anchor.
            return 0.0
        row = self.drift_matrix()[index]
        ratios = self.state_ratios(forwards)
        return math.fsum(ratios[k] * row[k] for k in range(n))

    def state_ratios(self, forwards: Sequence[float]) -> tuple[float, ...]:
        """`(f_k τ_k)/(1 + f_k τ_k) · σ_k` for each `k` — eq 12.8's **state-dependent** factor.

        Split from the constant part so a vectorised simulation can compute this once per step as an
        array and still use the *same* definition of 12.8 that the scalar `drift_at` uses. Without
        the split the path engine would have had to re-derive the formula, which §4 forbids."""
        tau = self.tenor.accruals
        sigma = self.tenor.volatilities
        return tuple(
            (forwards[k] * tau[k]) / (1.0 + forwards[k] * tau[k]) * sigma[k]
            for k in range(len(self.tenor))
        )

    def drift_matrix(self) -> tuple[tuple[float, ...], ...]:
        """`M[i][k]` — eq 12.8's **constant** part, so that `µ_i = Σ_k ratio_k · M[i][k]`.

        `M[i][k] = ±σ_i·ρ_ik` for `k` inside branch `i`'s summation span and `0` outside it, with the
        sign from the branch and `ρ_ii = 1` carried by the matrix itself. Writing 12.8 this way is what lets the scalar and
        vectorised paths share one definition: the engine multiplies a `(paths × n)` ratio array by
        this matrix, and `drift_at` takes one row of it. The row for `i = N−1` is all zeros, which is
        the martingale branch falling out of the construction rather than being special-cased twice."""
        n = len(self.tenor)
        sigma = self.tenor.volatilities
        numeraire = self.numeraire_index
        rows: list[tuple[float, ...]] = []
        for i in range(n):
            row = [0.0] * n
            if i != numeraire - 1:
                if i < numeraire - 1:
                    span, sign = range(i + 1, numeraire), -1.0
                else:
                    span, sign = range(numeraire, i + 1), 1.0
                for k in span:
                    row[k] = sign * sigma[i] * self.correlation.matrix[i][k]
            rows.append(tuple(row))
        return tuple(rows)
