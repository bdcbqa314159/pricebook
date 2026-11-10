"""Collateral / CSA (L5) - the collateralized exposure over the margin period of risk.

Under a **zero-threshold, zero-MTA variation-margin** CSA, collateral equals the netting set's mark
at the last margin call. So what remains exposed on a counterparty default is not `V(t)` but the
amount the mark has moved since collateral was last posted - the **gap over the margin period of
risk** `delta`:

    collateralized exposure(t) = (V(t) - V(t - delta))^+

Perfect margining (`delta = 0`) leaves nothing exposed; no CSA leaves `V(t)` itself. Between them,
`delta` is the whole economics of a CSA: collateral does not remove credit risk, it **shortens the
horizon over which the mark can move against you**.

**Collateral enters only through the exposure, and that is the design.** The gap is assembled into
the same `SimulatedExposure` the uncollateralized engine produces, so tranche 25/27's
`credit_adjustment` prices it **unchanged** - no collateralized CVA function, no second integral, no
branch in the credit code. A CSA changes what is exposed, not how an adjustment is computed, and the
module boundary says so.

**The positive part belongs here, not downstream.** A negative gap means the posted collateral
exceeds the current mark - we are over-collateralized and there is nothing to lose - so the floored
gap *is* the exposure, and an `ExposureProfile` is a non-negative object by construction. Leaving the
floor to the credit integral would let a negative "exposure" exist in between, which is not a
quantity this domain has.

**What this module does NOT model**, because the spine is fenced there (tranche 29's scope): a
non-zero threshold or minimum transfer amount (collateral posted only past a trigger), **initial
margin** and its MVA, ColVA (the funding cost of collateral), and any dependence between the gap and
the default itself (gap/wrong-way risk). Each is deferred, not approximated.

Provenance:
  stock:  none - the archived predecessor's `risk/xva.py` was a phantom (tranche 24 pruned its rows)
  source: CREPEY-2026 (docs/references.md) - the collateralized-exposure frame. The gap construction
          is standard CSA mechanics rather than a numbered equation, so there is nothing to
          transcribe and the closed-form check below is DERIVED (a driftless arithmetic mark's gap
          is `N(0, sigma^2 delta)`), which §4 rates above a transcription
  oracle: tests/L5/test_collateral.py - two exact reductions (`delta = 0` leaves nothing exposed;
          `agreement=None` returns the uncollateralized exposure bit-identically) each with the
          non-degenerate companion §4 demands; the simulated gap EPE against T25's
          `expected_positive_exposure` on the gap's own moments at the MC rate; the **flat-in-time**
          structural signature separating the gap from the mark; collateral never increasing the
          adjustment, and monotone in `delta`
  crosscheck: quantlib (collateralized exposure / CSA close-out) - deferred to the optional QuantLib
          tier
  slice:  tranche 29 - collateral/CSA, the last build of the XVA spine arc
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.risk.mc_exposure import SimulatedExposure

_UNIFORM_TOLERANCE = 1e-9


@dataclass(frozen=True)
class CollateralAgreement:
    """A zero-threshold, zero-MTA variation-margin CSA: its margin period and the mark it starts at.

    `initial_mark` is `V(0)`, the deterministic mark at inception. It is needed because the first
    exposure date's collateral was posted at `t - delta = 0`, which no simulated path contains - the
    value there is known rather than sampled, and substituting the first simulated value instead
    would silently compare a date with itself and report zero exposure.
    """

    margin_period: float
    initial_mark: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.margin_period) or not math.isfinite(self.initial_mark):
            raise ValueError(
                f"the margin period and initial mark must be finite, got "
                f"{self.margin_period}, {self.initial_mark}"
            )
        if self.margin_period < 0.0:
            raise ValueError(f"the margin period of risk may not be negative, got {self.margin_period}")


def collateralized_exposure(
    exposure: SimulatedExposure,
    agreement: CollateralAgreement | None,
) -> SimulatedExposure:
    """The exposure that survives a CSA: `(V(t) - V(t-delta))^+`, path by path.

    `agreement=None` returns the input **object**, so the uncollateralized case is the structural
    special case of this one and tranche 26/27's numbers provably cannot move - bit-identity by
    construction rather than by measurement. (`None` rather than a "no collateral" flag for the same
    reason tranche 27's `surviving=None` skips a multiply: the strongest proof is the one where
    nothing is computed at all.)

    `margin_period=0` is continuous margining, and gives exactly zero: `V(t) - V(t) = 0`. Exact by
    construction, not a numerical limit.

    The exposure grid must be the **margin-call schedule** - uniform, with spacing equal to
    `margin_period`, starting at `margin_period` - because `V(t-delta)` has to be a point the
    simulation actually visited. A caller handing in some other grid is refused rather than
    interpolated onto: an interpolated collateral balance is a different (and wrong) instrument.
    """
    if agreement is None:
        return exposure
    if agreement.margin_period == 0.0:
        return SimulatedExposure(
            exposure.times, tuple(tuple(0.0 for _ in row) for row in exposure.values)
        )

    times = exposure.times
    spacings = [b - a for a, b in zip((0.0,) + times, times)]
    if any(not math.isclose(s, spacings[0], rel_tol=_UNIFORM_TOLERANCE) for s in spacings):
        raise ValueError(
            f"a collateralized exposure needs a uniform margin-call grid, got spacings {spacings}"
        )
    if not math.isclose(spacings[0], agreement.margin_period, rel_tol=_UNIFORM_TOLERANCE):
        raise ValueError(
            f"the grid spacing must equal the margin period of risk: grid {spacings[0]} vs "
            f"margin period {agreement.margin_period}"
        )

    gaps = tuple(
        tuple(
            max(value - previous, 0.0)
            for value, previous in zip(row, (agreement.initial_mark,) + row[:-1])
        )
        for row in exposure.values
    )
    return SimulatedExposure(times, gaps)
