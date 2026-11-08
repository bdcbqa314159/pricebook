"""The survival curve: default probability from a hazard rate (L1).

Market data, not a model. A hazard rate `λ` is a *quote-side* description of credit — the same
status as a zero rate on the discount curve — and everything this module computes follows from it
by definition, with no model parameter fitted and no engine consulted. So it sits beside
`DiscountCurve` at L1, and §2's calibration split confirms it: a construction producing *market
data* is L1 even when it inverts quotes, which this does not yet do.

**Piecewise-constant hazard, because that is what the cumulation can get wrong.** `S(t) =
exp(−∫₀ᵗ λ(u)du)` degenerates to `e^{−λt}` under a flat hazard, so a flat-only curve could not
distinguish the integral from a point read of `λ(t)` — the fixture trap of §4. The piecewise form
is the simplest one whose cumulation is a real sum, and the oracle reads it beyond the first
segment where that sum has more than one term.

**CDS calibration is not here.** Inverting market CDS or defaultable-bond quotes *for* `λ` is a
curve bootstrap (L1, like `nominal_bootstrap`), and it arrives with the tranche whose product needs
it. A constructed hazard curve is what unilateral CVA pulls, and nothing speculative comes with it
(§6).

Provenance:
  stock:  none — the archived predecessor's `risk/xva.py` was a phantom (no survival curve existed
          to purify; tranche 24 pruned its register rows)
  source: CAPINSKI-ZASTAWNIAK (docs/references.md) ch 2 — the hazard function, the default time and
          its risk-neutral survival probability `S(t) = exp(−∫₀ᵗ λ(u)du)`, density `λ(t)S(t)`
  oracle: tests/L1/test_survival_curve.py — the cumulative hazard against an INDEPENDENTLY summed
          reference on a piecewise-constant fixture read beyond the first segment; the flat case
          exactly `exp(−λt)`; `d(1−S)/dt = λS` by central difference; monotonicity and the unit
          interval as requirement oracles
  crosscheck: quantlib DefaultProbabilityTermStructure / HazardRateCurve —
          deferred to the optional QuantLib tier
  slice:  tranche 25 — unilateral CVA: the XVA spine opens
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from datetime import date

from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure


@dataclass(frozen=True)
class SurvivalCurve:
    """Survival probabilities from a piecewise-constant hazard rate.

    `times` are year fractions on `time_measure` (ascending, starting at 0.0) and `hazards[i]` is
    the hazard on `[times[i], times[i+1])`, the last extending to every later horizon. Frozen, for
    the reason `DiscountCurve` is: a stressed copy is a new object, so a risk calculation cannot
    disturb the base market.
    """

    time_measure: TimeMeasure
    times: tuple[float, ...]
    hazards: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.times) != len(self.hazards):
            raise ValueError("times and hazards must have equal length")
        if not self.times:
            raise ValueError("a curve needs at least one pillar")
        if self.times[0] != 0.0:
            raise ValueError(f"the first pillar must be t=0.0, got {self.times[0]}")
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("times must be strictly ascending")
        if any(h < 0.0 for h in self.hazards):
            raise ValueError("a hazard rate may not be negative — survival would exceed 1")

    @classmethod
    def flat(cls, anchor: date, rate: float, day_count: DayCountConvention) -> SurvivalCurve:
        """A flat hazard: `S(t) = exp(−rate·t)` exactly, one pillar extrapolating forever."""
        return cls(TimeMeasure(anchor, day_count), (0.0,), (rate,))

    def hazard(self, t: float) -> float:
        """The hazard in force at `t` — piecewise constant, right-continuous at each pillar."""
        if t <= self.times[0]:
            return self.hazards[0]
        return self.hazards[bisect.bisect_right(self.times, t) - 1]

    def cumulative_hazard(self, t: float) -> float:
        """`∫₀ᵗ λ(u)du` — the sum of each segment's hazard times its overlap with `[0, t]`.

        Named surface rather than a step buried inside `survival`, so a test can reach the
        cumulation itself (§4: the wrong thing to oracle is only the exponential's output).
        """
        if t <= 0.0:
            return 0.0
        total = 0.0
        for i, hazard in enumerate(self.hazards):
            start = self.times[i]
            if t <= start:
                break
            end = self.times[i + 1] if i + 1 < len(self.times) else math.inf
            total += hazard * (min(t, end) - start)
        return total

    def survival(self, t: float) -> float:
        """`S(t) = exp(−∫₀ᵗ λ(u)du)` — the probability of no default by `t`."""
        return math.exp(-self.cumulative_hazard(t))

    def default_density(self, t: float) -> float:
        """`λ(t)S(t)` — the density of the default time, so `dPD(t) = λ(t)S(t)dt`.

        This is `d(1−S)/dt`, which the oracle checks by differencing `survival` rather than by
        restating this product (independent code on each side).
        """
        return self.hazard(t) * self.survival(t)
