"""The SVJ / Bates model — Heston plus lognormal jumps (L3). `GATHERAL-2006` ch 5.

Heston's stochastic variance with a Merton-style lognormal jump process added to the **underlying
only**:

    dS = mu S dt + sqrt(v) S dZ1 + (e^{alpha + delta*eps} - 1) S dq     dq ~ Poisson(intensity)
    dv = -lambda (v - vbar) dt + eta sqrt(v) dZ2                        eps ~ N(0,1)

Ch 5's argument for it is the one T46 ran into from the other side: jumps generate the steep
short-dated skew stochastic volatility alone does not, and SVI's fit exposed exactly that gap.

**The jump process is independent of everything Heston carries, so the characteristic functions
multiply** — ch 5 states it outright ("the characteristic function for this process is just the
product of Heston and jump characteristic functions"), which is why tranche 33's inversion prices
this model with no new machinery.

**NOTATION, and it is a live trap.** Gatheral writes `lambda` for the Heston **mean-reversion
speed** and `lambda_J` for the **jump intensity**; they are different parameters in the same
formula. The reduction to Heston is at **zero intensity**, never at zero mean reversion (which would
destroy the variance process instead). `JumpParams.intensity` is `lambda_J`; mean reversion stays
`HestonParams.kappa`, so the two cannot be confused here even though the source's symbols nearly
invite it.

**The compensator, and the requirement that pins it.** `psi(u)` carries a `-iu*m*(e^{alpha+delta^2/2}
- 1)` drift term whose only job is to keep the jump factor drift-neutral, so that adding jumps does
not silently change the forward. The identity that catches a dropped or mis-signed compensator is
**`phi_jump(-i) = 1` exactly**: at `u = -i` the two bracketed terms become the same number and
cancel. (The spec asked for `phi_SVJ(-i) = 1`, which holds only where the carry is zero — this
tree's Heston factor ends in `+ iu*carry*T`, so the full function at `-i` is `F/S0` and it is the
*jump factor alone* that must be 1.)

Provenance:
  stock:  none - the archived predecessor had no SVJ model
  source: GATHERAL-2006 (docs/references.md) ch 5, "Stochastic Volatility Plus Jumps in the
          Underlying Only (SVJ)", p. 65 - the SDE and
          `phi_T(u) = e^{C vbar + D v} e^{psi(u) T}` with
          `psi(u) = -lambda_J iu (e^{alpha+delta^2/2} - 1) + lambda_J (e^{iu alpha - u^2 delta^2/2} - 1)`,
          transcribed in both `-raw` and `-layout` (§4) and reconciled
  oracle: tests/L4/test_bates_svj.py - the pointwise transcription of `phi_jump` in both
          renderings, the compensator identity `phi_jump(-i) = 1`, the factorisation
          `phi_SVJ = phi_Heston * phi_jump`, and the double reduction (to Heston at zero intensity,
          to the Merton series at zero vol-of-vol) with each reduction's blindness asserted
  crosscheck: quantlib BatesModel / BatesEngine - deferred to the optional QuantLib tier
  slice:  tranche 49 - the SVJ/Bates model
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

from pricebook.foundation.time_grid import checked_positive_horizon
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.heston import HestonParams, heston_characteristic_function


@dataclass(frozen=True)
class JumpParams:
    """The Merton jump block: Poisson `intensity` (jumps per year), and the log-jump's `mean` and
    `volatility`.

    Three fields that mean nothing apart — an intensity without a jump size describes no process —
    so they are §6's compelled bundling, kept concrete and frozen. Splitting the eight SVJ numbers
    into this plus `HestonParams` is what keeps `BatesParams` inside §7's limit without inventing a
    nine-field object or raising the limit."""

    intensity: float
    mean: float
    volatility: float

    def __post_init__(self) -> None:
        if self.intensity < 0.0:
            raise ValueError(f"the jump intensity must be non-negative, got {self.intensity}")
        if self.volatility < 0.0:
            raise ValueError(f"the jump volatility must be non-negative, got {self.volatility}")

    def compensator(self) -> float:
        """`E[e^J] - 1 = e^{alpha + delta^2/2} - 1` — the expected relative jump size.

        Named surface rather than an inline expression because the characteristic function and any
        simulation must use the *same* number; it is the one quantity in this model that appears in
        two places and could silently disagree."""
        return math.exp(self.mean + 0.5 * self.volatility * self.volatility) - 1.0


@dataclass(frozen=True)
class BatesParams:
    """The eight SVJ numbers, as two blocks that each mean something on their own."""

    heston: HestonParams
    jumps: JumpParams


@dataclass(frozen=True)
class BatesModel:
    """The SVJ parameters bound to the snapshot they were quoted against (§3)."""

    market: MarketSnapshot
    params: BatesParams


def jump_characteristic_function(jumps: JumpParams, u: complex, years: float) -> complex:
    """`phi_jump(u) = exp(T * psi(u))` with ch 5's

        psi(u) = m (e^{iu alpha - u^2 delta^2 / 2} - 1) - iu m (e^{alpha + delta^2/2} - 1)

    the second term being the compensator. `intensity = 0` returns exactly `1`, which is what makes
    the reduction to Heston exact rather than merely close: no arithmetic runs at all."""
    checked_positive_horizon(years, "the characteristic function")
    if jumps.intensity == 0.0:
        return 1.0 + 0j
    variance = jumps.volatility * jumps.volatility
    characteristic = cmath.exp(1j * u * jumps.mean - u * u * variance / 2.0) - 1.0
    psi = jumps.intensity * characteristic - 1j * u * jumps.intensity * jumps.compensator()
    return cmath.exp(psi * years)


def bates_characteristic_function(
    params: BatesParams, u: complex, years: float, carry: float
) -> complex:
    """`phi_SVJ(u) = phi_Heston(u) * phi_jump(u)` — ch 5's factorisation, as one multiplication.

    The carry stays entirely inside the Heston factor, where tranche 33 put it; the jump factor is
    drift-neutral by its own compensator. That split is what makes `phi_jump(-i) = 1` the testable
    statement rather than something entangled with the forward."""
    return heston_characteristic_function(
        params.heston, u, years, carry
    ) * jump_characteristic_function(params.jumps, u, years)
