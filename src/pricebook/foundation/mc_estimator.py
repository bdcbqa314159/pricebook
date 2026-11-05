"""The Monte Carlo estimator: sample mean and standard error (L0). T15.

`estimate(samples)` returns the sample mean and `σ̂/√N`; `antithetic_estimate` and
`control_variate_estimate` are tranche 19's variance reducers (`JAECKEL-2002` §10.1, §10.3). Finance-free statistics, and the piece that
lets a Monte Carlo oracle assert `|MC − closed form| < k·SE` with the standard error **measured**
rather than a tolerance invented to make a test pass — the "tolerance is a finding, not a knob" rule
(charter §4) applied to the one oracle shape that cannot be exact.

**The sample divisor is `N − 1`, deliberately.** The population divisor `N` under-states the spread
of a finite sample, and an MC oracle that under-states its own error bar is an oracle that passes
when it should not. The difference is 10% at `N = 5` and the oracle pins it.

**A non-finite sample is refused, not averaged.** One `nan` sample makes the mean and the SE `nan` —
a number-shaped non-answer that propagates. That is the stock's own migration-hardening lesson
(non-finite poisoning), folded in as a pre-registered oracle rather than inherited as code.

**The reducers are UNBIASED by construction, and that ordering is the whole discipline.** A variance
reduction that moves the expectation is worse than none: it trades an honest wide error bar for a
confident wrong number. Antithetic adds the mirror draw, whose distribution is identical, so the
expectation cannot move. The control variate adds `β(g* − g)`, which is mean-zero **only because
`g*` is the EXACT `E[g]`** — centring on an approximation shifts the estimate by `β(g* − E[g])`, a
bias no path count removes. So `control_variate_estimate` takes the exact mean as a required
argument rather than computing one: the caller must have a closed form, and if it only has an
approximation it does not have a control.

**Finance-free, and that is the layer split.** These functions do not know what `X` or `Y` are — the
reduction *math* is L0 statistics; the *choice* of control is a finance judgement and belongs to the
engine that knows which closed form it owns (charter §3's `MonteCarloEstimate`/`PricingResult`
ruling, same shape).

Provenance:
  stock:  _stock/quarry/pricebook/... `_mc.py`'s estimator — mined for the SHAPE and for its
          hardening cases (`n_paths` validation, non-finite guards); never imported (§5)
  source: standard sample mean and standard error; JAECKEL-2002 ch 2/ch 14 for the role of the
          standard error as the MC convergence measure ("a single number or a convergence
          diagram?") — the statistic itself is textbook, not Jaeckel's. The reducers: JAECKEL-2002
          §10.1 (antithetic sampling, eqs 10.1/10.2 and the pairwise-average standard error) and
          §10.3 eqs 10.8/10.9 (the control-variate estimator and the optimal `β*`), both read in
          `-raw` and `-layout`, which agree exactly. The variance factor `V[v](1 − ρ²)` is **not
          printed in the source** and is therefore DERIVED from 10.9 rather than transcribed — see
          the oracle note
  oracle: tests/L0/test_mc_estimator.py — mean and SE on a hand-computable sample, with the two
          wrong variants (population divisor, missing √N) asserted to differ; a constant sample's
          SE is exactly zero; the SE falls as 1/√N on a deterministic sample; a single sample and a
          non-finite sample are both refused. Tranche 19: antithetic exact (zero variance) on a
          function linear in the draw, with a convex companion that reduces but not to zero; the
          standard error taken from the PAIRWISE averages, asserted to differ from pooling `2N`
          dependent values; eqs 10.1/10.2's reduction condition asserted as an EQUIVALENCE, with a
          symmetric function as the case where it fails; `β*` pointwise against 10.9; the derived
          `(1 − ρ²)` factor as a theorem; and the centring bias shown to be many standard errors
          wide when `g*` is approximate rather than exact
  crosscheck: none — the textbook sample mean and standard error, no finance convention
  slice:  tranche 15 — the Monte Carlo spine
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class MonteCarloEstimate:
    """A Monte Carlo estimate and its own error bar.

    The two travel together because neither means anything alone: a mean without its standard error
    cannot be compared with a closed form, and an MC engine that returned only a price would make
    every downstream oracle invent a tolerance."""

    mean: float
    standard_error: float
    n_samples: int


@dataclass(frozen=True)
class ControlVariateEstimate:
    """A control-variate estimate: eq 10.8's adjusted mean, its reduced error bar, and the diagnostics.

    `beta` and `correlation` ride along because they are how a caller judges whether the control was
    worth using: the variance falls by `1 − ρ²`, so `correlation` *is* the reduction, and reporting
    it turns "the SE got smaller" into a number that can be checked against the theory."""

    mean: float
    standard_error: float
    n_samples: int
    beta: float
    correlation: float


def _checked(samples: Sequence[float], other: Sequence[float], what: str) -> None:
    if len(samples) != len(other):
        raise ValueError(
            f"{what} must have the same length, got {len(samples)} and {len(other)}"
        )
    for value in (*samples, *other):
        if not math.isfinite(value):
            raise ValueError(f"a sample is not finite ({value}); refusing to average it")


def antithetic_estimate(
    values: Sequence[float], mirrored: Sequence[float]
) -> MonteCarloEstimate:
    """§10.1 — the estimate over the PAIRWISE AVERAGES `½(v(z) + v(−z))`, not over `2N` values.

    *"If instead of viewing both `v_i = v(z_i)` and `ṽ_i = v(−z_i)` as individual samples, we only
    count the pairwise average `v̄_i = ½(v(z_i) + v(−z_i))` as an individual sample, all is well
    again, because the pairwise averages `v̄_i` are independent."* Pooling all `2N` values would
    report a smaller standard error from samples that are **not** independent — a confident wrong
    error bar, which is the one thing worse than a large one. So `n_samples` is `N`.

    Thin on purpose: the mirroring is the caller's (it owns the draws), and all this adds is the
    pairing and the insistence on where the error bar comes from."""
    _checked(values, mirrored, "values and their mirrors")
    return estimate(
        [0.5 * (value + mirror) for value, mirror in zip(values, mirrored, strict=True)]
    )


def control_variate_estimate(
    samples: Sequence[float],
    controls: Sequence[float],
    control_mean: float,
    beta: float | None = None,
) -> ControlVariateEstimate:
    """§10.3 eq 10.8 — `v̂_CV = (1/n) Σ [v(u_i) + β(g* − g(u_i))]`, with `β*` from eq 10.9.

    `control_mean` is `g*`, and it must be the **exact** `E[g]`: eq 10.8 is unbiased only because the
    added term has mean zero. An approximate `g*` biases the result by `β(g* − E[g])` for every path
    count, so it is a required argument and never inferred from the samples.

    `beta = None` uses the optimal `β* = Cov[v,g]/V[g]`, estimated from these same samples — which
    Jäckel notes *"can introduce a bias that is difficult to estimate"*, vanishing as `O(1/N)`; pass
    an explicit `beta` (1.0, or one fitted on a separate pilot run, which §10.3 offers as the remedy)
    when that matters."""
    _checked(samples, controls, "samples and controls")
    n = len(samples)
    if n < 2:
        raise ValueError(f"a control-variate estimate needs at least two samples, got {n}")

    mean_x = math.fsum(samples) / n
    mean_y = math.fsum(controls) / n
    variance_y = math.fsum((y - mean_y) ** 2 for y in controls) / (n - 1)
    if variance_y <= 0.0:
        raise ValueError(
            "the control has no variance, so the optimal beta is undefined; a constant control "
            "cannot explain any of the target's variation"
        )
    covariance = math.fsum(
        (x - mean_x) * (y - mean_y) for x, y in zip(samples, controls, strict=True)
    ) / (n - 1)
    variance_x = math.fsum((x - mean_x) ** 2 for x in samples) / (n - 1)

    chosen = covariance / variance_y if beta is None else beta
    adjusted = [
        x + chosen * (control_mean - y) for x, y in zip(samples, controls, strict=True)
    ]
    correlation = (
        covariance / math.sqrt(variance_x * variance_y) if variance_x > 0.0 else 0.0
    )
    reduced = estimate(adjusted)
    return ControlVariateEstimate(
        mean=reduced.mean,
        standard_error=reduced.standard_error,
        n_samples=n,
        beta=chosen,
        correlation=correlation,
    )


def estimate(samples: Sequence[float]) -> MonteCarloEstimate:
    """The sample mean and `σ̂/√N` of `samples`, with `σ̂` the `N−1` sample standard deviation."""
    n = len(samples)
    if n < 2:
        raise ValueError(
            f"a standard error needs at least two samples to estimate a spread, got {n}"
        )
    for value in samples:
        if not math.isfinite(value):
            raise ValueError(f"a sample is not finite ({value}); refusing to average it")

    mean = math.fsum(samples) / n
    # `fsum` on the squared deviations rather than the E[X^2] - E[X]^2 shortcut: the shortcut
    # cancels catastrophically when the mean dominates the spread, which is exactly the regime the
    # discounted samples of a simulation sit in.
    variance = math.fsum((x - mean) ** 2 for x in samples) / (n - 1)
    return MonteCarloEstimate(
        mean=mean, standard_error=math.sqrt(variance / n), n_samples=n
    )
