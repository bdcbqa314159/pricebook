"""The Monte-Carlo exposure engine: CVA of a simulated netting set (L5) — tranche 26.

T25 integrated an exposure that arrived in closed form. This produces one by **simulation**, which
changes what the oracle can be: T25's closed form is now our own code on the other side of the
check, so it validates the *estimator* and nothing else. The process needs an anchor that does not
consult it, and the measure provides one for free.

**Two legs, and what each can see.**

* The **reduction** to T25 on a Gaussian mark pins the positive part, the averaging and the standard
  error against a known closed form. It is self-consistency (§4) — both sides share the exposure
  convention — so a wrong *drift* sails through it: the `+` transform of a mis-drifted path set is
  still averaged correctly.
* The **martingale identity** `E[V(t)/B_N(t)] = V(0)/B_N(0)` is exact *by construction* under the
  numéraire measure — no model result, no T25 — so it sees exactly what the reduction cannot. This
  is tranche 19's free control reused as an exposure oracle.

**The estimator is instrument-agnostic, and that is not a speculative hook.** It reads a matrix of
simulated values, so it neither knows nor cares whether they came from a Brownian mark or a swap
under the LMM — which is precisely what lets one estimator carry both oracle legs. Two present
consumers, rule-of-two satisfied (§6).

**The CVA is estimated PATHWISE, not by combining per-date standard errors.** Each path contributes
its own `∫ DF·V(t)⁺·dPD` and the CVA is the mean of those scalars. Combining per-date errors in
quadrature would assume the dates independent, and they are emphatically not — one path visits every
date — so it would understate the error. The pathwise form also satisfies invariant 7's correctness
point directly: the standard error comes from the *same* paths as the price, so the band describes
that price rather than a different sample.

Provenance:
  stock:  none - the archived predecessor's `risk/xva.py` was a phantom (tranche 24 pruned its rows)
  source: CREPEY-2026 (docs/references.md) ch I - the exposure/CVA frame; the martingale anchor is
          the numeraire-measure definition (JAECKEL-2002 §10.3's control-variate identity, tranche
          19), and the swap value from a simulated forward curve follows eqs 5.31/5.32
  oracle: tests/L5/test_mc_exposure.py - Leg A, the simulated EPE against T25's closed form at the
          MC rate and the MC CVA against T25's analytic CVA (cross-product, different exposure code
          each side); Leg B, the numeraire-relative swap value reproducing `V(0)/B(0,T_end)` at every
          observation date, exact by construction and improving as root-N; invariant 7's SE falling
          as root-N; the historical boundary read through the `is_historical` atom
  crosscheck: quantlib (exposure simulation / CVA on a netting set) - deferred to the optional
          QuantLib tier
  slice:  tranche 26 - the Monte-Carlo exposure engine
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from pricebook.foundation.mc_estimator import MonteCarloEstimate, estimate
from pricebook.foundation.money import Currency, Money
from pricebook.foundation.results import PricingResult
from pricebook.foundation.time_measure import TimeMeasure, is_historical
from pricebook.market.curve import CurveHandle
from pricebook.market.survival_curve import SurvivalCurve
from pricebook.models.libor_market_model import LmmTenor
from pricebook.products.swaption import SwapSpan
from pricebook.risk.credit_valuation_adjustment import CreditParties, ExposureProfile


def live_exposure_times(
    dates: Sequence[date],
    valuation_date: date,
    time_measure: TimeMeasure,
) -> tuple[float, ...]:
    """The year fractions of the exposure dates that are **not** historical (invariant 6).

    The one consumer that pulls `is_historical` out of the engines: an exposure grid reads the
    valuation-date boundary for the same reason a cashflow does, and routing it through the atom is
    what makes register row C10 `guarded - structurally` rather than `per-module`.
    """
    return tuple(
        time_measure.year_fraction(d) for d in dates if not is_historical(d, valuation_date)
    )


@dataclass(frozen=True)
class ExposureEstimate:
    """`EPE(t)` and `ENE(t)` per date, each as the L0 statistical primitive (mean, SE, n).

    `MonteCarloEstimate` rather than parallel tuples of floats and errors: §3 is explicit that the
    mean and its error travel together, and three fields here beat five.
    """

    times: tuple[float, ...]
    epe: tuple[MonteCarloEstimate, ...]
    ene: tuple[MonteCarloEstimate, ...]

    def profile(self) -> ExposureProfile:
        """T25's shape, so the closed-form CVA functional consumes this unchanged."""
        return ExposureProfile(self.times, tuple(e.mean for e in self.epe))

    def negative_profile(self) -> ExposureProfile:
        """The ENE in the same shape - what the DVA leg weights (tranche 27).

        `ExposureProfile` is agnostic about which side it holds: it is a non-negative profile on a
        time grid, and `E[V-]` is as non-negative as `E[V+]`. Reusing it keeps the bilateral
        integral one function instead of two near-copies that would drift.
        """
        return ExposureProfile(self.times, tuple(e.mean for e in self.ene))


@dataclass(frozen=True)
class SimulatedExposure:
    """`V(t)` per path on a time grid - the raw material the estimator and the CVA both read.

    Kept as the paths rather than reduced to means on construction, because the CVA's standard
    error has to be taken across paths of the *integral* (see the module docstring), which a
    per-date reduction has already thrown away.
    """

    times: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.times:
            raise ValueError("an exposure needs at least one date")
        if not self.values:
            raise ValueError("an exposure needs at least one path")
        if any(len(row) != len(self.times) for row in self.values):
            raise ValueError("every path must carry the same number of values as there are times")
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("times must be strictly ascending")
        if self.times[0] <= 0.0:
            raise ValueError(f"an exposure time must be positive, got {self.times[0]}")

    def estimate(self) -> ExposureEstimate:
        """`EPE(t) = mean(V(t)⁺)` and `ENE(t) = mean(V(t)⁻)`, each with its standard error."""
        values = np.asarray(self.values, dtype=float)
        positive = np.maximum(values, 0.0)
        negative = np.maximum(-values, 0.0)
        return ExposureEstimate(
            self.times,
            tuple(estimate(positive[:, i].tolist()) for i in range(len(self.times))),
            tuple(estimate(negative[:, i].tolist()) for i in range(len(self.times))),
        )


def _trapezoid_weights(times: Sequence[float]) -> list[float]:
    """The node weights of the trapezoid rule, so the integral is one dot product per path.

    Pulling them out is what makes the pathwise estimator possible: the weights are deterministic,
    so each path's integral is a weighted sum of its own exposures and the CVA's error is the error
    of those sums.
    """
    n = len(times)
    if n < 2:
        raise ValueError("integrating an exposure profile needs at least two grid points")
    weights = [0.0] * n
    for i in range(n - 1):
        half = 0.5 * (times[i + 1] - times[i])
        weights[i] += half
        weights[i + 1] += half
    return weights


def mc_exposure_cva(
    exposure: SimulatedExposure,
    survival: SurvivalCurve,
    discount: CurveHandle,
    loss_given_default: float,
    currency: Currency,
) -> PricingResult:
    """`LGD · ∫ DF(t)·V(t)⁺·λ(t)S(t) dt`, estimated across paths and carrying its own error.

    The mean reproduces T25's `unilateral_cva` on this exposure's EPE profile exactly — the weights
    are the same and the deterministic factors commute with the average — so the two are one
    quantity computed two ways, and the oracle asserts that identity rather than assuming it.
    """
    if not 0.0 <= loss_given_default <= 1.0:
        raise ValueError(f"loss given default must lie in [0, 1], got {loss_given_default}")
    if survival.time_measure != discount.time_measure:
        # C12, as in the closed-form functional: one `t` must denote one date on both curves.
        raise ValueError(
            "the survival and discount curves must share one time measure: "
            f"{survival.time_measure} vs {discount.time_measure}"
        )

    weights = _trapezoid_weights(exposure.times)
    deterministic = np.asarray(
        [
            w * discount.df_at(t) * survival.default_density(t)
            for w, t in zip(weights, exposure.times)
        ],
        dtype=float,
    )
    positive = np.maximum(np.asarray(exposure.values, dtype=float), 0.0)
    per_path = (positive @ deterministic) * loss_given_default
    sampled = estimate(per_path.tolist())
    return PricingResult(
        pv=Money(sampled.mean, currency),
        standard_error=sampled.standard_error,
    )


@dataclass(frozen=True)
class SwapValuesOnPaths:
    """A forward-starting swap's simulated value, in both the forms the two oracle legs need.

    `value` is the time-`T_j` value `V(T_j)` the exposure integral consumes; `numeraire_relative` is
    `V(T_j)/B(T_j,T_end)`, the martingale. One object because they share the whole computation and
    returning only one would force a second pass to recover the other.
    """

    value: tuple[tuple[float, ...], ...]
    numeraire_relative: tuple[tuple[float, ...], ...]


def swap_values_on_paths(
    paths: Sequence[Sequence[Sequence[float]]],
    tenor: LmmTenor,
    span: SwapSpan,
    strike: float,
    observed: Sequence[int],
) -> SwapValuesOnPaths:
    """A payer swap's value on each simulated forward curve — eqs 5.31/5.32 read at `T_j`.

    `observed[d]` is the **tenor index** of observation `d`, so the bonds are built from `T_j`
    outwards: `B(T_j,T_m) = Π_{k=j}^{m-1} 1/(1+f_k τ_k)` on that path's own curve. The swap's value
    is then `(B(T_j,T_first) − B(T_j,T_end)) − K·A(T_j)`, which is `A·(S−K)` with 5.31 substituted —
    the same arithmetic the swaption payoff uses, minus the positive part.

    **Every observation must sit at or before the span's first reset.** Past it the swap has paid a
    coupon, so it is no longer the claim it was at inception and `E[V/B_N] = V(0)/B_N(0)` is simply
    false — the oracle would be asserting something untrue rather than detecting an error. Refusing
    the input keeps the martingale leg exact instead of approximately right.
    """
    if not observed:
        raise ValueError("at least one observation index is needed")
    if any(j > span.first for j in observed):
        raise ValueError(
            f"every observation must sit at or before the span's first reset ({span.first}): "
            f"got {tuple(observed)}"
        )
    if any(j < 0 for j in observed):
        raise ValueError("an observation index cannot be negative")

    curves = np.asarray(paths, dtype=float)
    accruals = np.asarray(tenor.accruals, dtype=float)
    end = span.first + span.count
    values: list[np.ndarray] = []
    relatives: list[np.ndarray] = []

    for position, j in enumerate(observed):
        forwards = curves[:, position, j:]
        taus = accruals[j:]
        # `bonds[:, k] = B(T_j, T_{j+1+k})` — the running product along the live tenor.
        bonds = np.cumprod(1.0 / (1.0 + forwards * taus), axis=1)
        annuity = sum(
            accruals[m] * bonds[:, m - j] for m in range(span.first, end)
        )
        bond_first = bonds[:, span.first - j - 1] if span.first > j else np.ones(len(curves))
        bond_end = bonds[:, end - j - 1]
        value = (bond_first - bond_end) - strike * annuity
        values.append(value)
        relatives.append(value / bond_end)

    def transpose(columns: list[np.ndarray]) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(row) for row in np.column_stack(columns))

    return SwapValuesOnPaths(transpose(values), transpose(relatives))


def mc_bilateral_adjustment(
    exposure: SimulatedExposure,
    parties: CreditParties,
    discount: CurveHandle,
    currency: Currency,
) -> PricingResult:
    """`BCVA = CVA - DVA` from a simulated exposure, estimated **pathwise** with its own error.

    **Pathwise is required here, not merely convenient.** CVA weights `V(t)+` and DVA weights
    `V(t)-` on the *same* paths, so the two legs are strongly (negatively) related and the error of
    their difference is **not** the quadrature sum of their separate errors. Reconstructing the SE
    from per-leg errors would assume an independence that does not hold; each path contributing its
    own `CVA - DVA` scalar gets the covariance for free, and satisfies invariant 7's correctness
    point - the error comes from the same paths as the number.

    Each leg carries the *other* party's survival, which is the first-to-default conditioning: we
    only lose on their default if we are still there to lose it.
    """
    for curve in (parties.own, parties.counterparty):
        if curve.time_measure != discount.time_measure:
            raise ValueError(
                "the survival and discount curves must share one time measure: "
                f"{curve.time_measure} vs {discount.time_measure}"
            )
    weights = _trapezoid_weights(exposure.times)
    cva_weights = np.asarray(
        [
            w * discount.df_at(t) * parties.counterparty.default_density(t) * parties.own.survival(t)
            for w, t in zip(weights, exposure.times)
        ],
        dtype=float,
    )
    dva_weights = np.asarray(
        [
            w * discount.df_at(t) * parties.own.default_density(t) * parties.counterparty.survival(t)
            for w, t in zip(weights, exposure.times)
        ],
        dtype=float,
    )
    values = np.asarray(exposure.values, dtype=float)
    cva_paths = np.maximum(values, 0.0) @ cva_weights * parties.counterparty_lgd
    dva_paths = np.maximum(-values, 0.0) @ dva_weights * parties.own_lgd
    sampled = estimate((cva_paths - dva_paths).tolist())
    return PricingResult(
        pv=Money(sampled.mean, currency),
        standard_error=sampled.standard_error,
    )
