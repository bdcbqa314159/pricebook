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

**One scope choice, stated.** `correlation` is a single scalar `ρ` applied to every distinct pair —
the flat-correlation case. Jäckel's `ρ_ik` is a full matrix, and §12.4 discusses parametric forms for
it; a real correlation structure arrives with the unit that needs one (a swaption fit discriminates
correlation shapes; a caplet strip does not), rather than speculatively here.

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
class LiborMarketModel:
    """The BGM/J model under the `P_N`-forward measure, with a flat correlation.

    `numeraire_index` is `N` of eq 12.8 — the bond whose forward measure the drift is written in.
    It ranges over `1 … n`, so `N − 1` is a valid forward index and the zero-drift branch is always
    reachable."""

    tenor: LmmTenor
    correlation: float
    numeraire_index: int

    def __post_init__(self) -> None:
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError(
                f"the correlation must lie in [-1, 1], got {self.correlation}"
            )
        if not 1 <= self.numeraire_index <= len(self.tenor):
            raise ValueError(
                f"the numeraire index must lie between 1 and {len(self.tenor)}, got "
                f"{self.numeraire_index}"
            )

    def drift(self, index: int) -> float:
        """`µ_i(f, t)` — eq 12.8, under the `P_N`-forward measure."""
        n = len(self.tenor)
        if not 0 <= index < n:
            raise IndexError(f"forward index must lie in [0, {n}), got {index}")

        numeraire = self.numeraire_index
        if index == numeraire - 1:
            # EXACT, by construction: the forward spanning the numeraire's own period is a
            # martingale under that measure. This is the finance, which is why it is the anchor.
            return 0.0

        sigma = self.tenor.volatilities
        if index < numeraire - 1:
            span = range(index + 1, numeraire)
            sign = -1.0
        else:
            span = range(numeraire, index + 1)
            sign = 1.0
        return sign * sigma[index] * math.fsum(self._term(k, index) for k in span)

    def _term(self, k: int, index: int) -> float:
        """`(f_k τ_k)/(1 + f_k τ_k)·σ_k·ρ_ik` — one summand of eq 12.8.

        `ρ_ii = 1`, not the flat `ρ`: the `i ≥ N` branch includes `k = i`, and using `ρ` there is a
        plausible-looking error that only a case with `k = i` in range can catch."""
        f = self.tenor.forwards[k]
        tau = self.tenor.accruals[k]
        rho = 1.0 if k == index else self.correlation
        return (f * tau) / (1.0 + f * tau) * self.tenor.volatilities[k] * rho
