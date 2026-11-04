"""The Monte Carlo estimator: sample mean and standard error (L0). T15.

`estimate(samples)` returns the sample mean and `σ̂/√N`. Finance-free statistics, and the piece that
lets a Monte Carlo oracle assert `|MC − closed form| < k·SE` with the standard error **measured**
rather than a tolerance invented to make a test pass — the "tolerance is a finding, not a knob" rule
(charter §4) applied to the one oracle shape that cannot be exact.

**The sample divisor is `N − 1`, deliberately.** The population divisor `N` under-states the spread
of a finite sample, and an MC oracle that under-states its own error bar is an oracle that passes
when it should not. The difference is 10% at `N = 5` and the oracle pins it.

**A non-finite sample is refused, not averaged.** One `nan` sample makes the mean and the SE `nan` —
a number-shaped non-answer that propagates. That is the stock's own migration-hardening lesson
(non-finite poisoning), folded in as a pre-registered oracle rather than inherited as code.

Provenance:
  stock:  _stock/quarry/pricebook/... `_mc.py`'s estimator — mined for the SHAPE and for its
          hardening cases (`n_paths` validation, non-finite guards); never imported (§5)
  source: standard sample mean and standard error; JAECKEL-2002 ch 2/ch 14 for the role of the
          standard error as the MC convergence measure ("a single number or a convergence
          diagram?") — the statistic itself is textbook, not Jaeckel's
  oracle: tests/L0/test_mc_estimator.py — mean and SE on a hand-computable sample, with the two
          wrong variants (population divisor, missing √N) asserted to differ; a constant sample's
          SE is exactly zero; the SE falls as 1/√N on a deterministic sample; a single sample and a
          non-finite sample are both refused
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
