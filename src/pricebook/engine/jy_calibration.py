"""Calibrating Jarrow-Yildirim's volatilities to a YoY floor strip (L4). BERHAN-2012 ch 5.

Two stages, and **only the second is fitted here**:

```
curve stage (tranche 2)   nominal curve + real curve from ZCIIS break-evens   model-free
vol   stage (this module) (sigma_r, sigma_I) fitted to a YoY floor strip      the fit
```

The curve stage stays pinned. Refitting curves against vol quotes launders a curve error into a
vol, so the snapshot arrives built and is read, never re-solved.

**Fixed, not fitted:** `kappa_n`, `kappa_r`, the correlations, and `sigma_n`. The thesis sources
`sigma_n` from *nominal* ATM caps -- an instrument this library has no engine for, and building
one to serve a calibration would be infrastructure ahead of its product. So `sigma_n` is a named
input with no default, and the `(sigma_r, sigma_I)` fit is **conditional on it**: a wrong `sigma_n`
biases the pair. That limit is faithful to the thesis's own staged calibration (Cowork ruled it
shed-and-named, thread 0035), not a gap this tranche leaves open by accident.

**Why a floor strip and not a ZCIIS strip -- the headline of tranche 6.** The stock's
`jy_calibrate` fits `(sigma_n, sigma_r, sigma_I)` to zero-coupon inflation swap quotes, which under
correct JY carry **no volatility information at all**: the fair rate is `Pr(0,T)/Pn(0,T) - 1`
(eqs 1.2.13/1.2.16), a pure curve relationship -- it is *because* the ZCIIS is model-free that
tranche 2 used it to build the real curve. The stock's objective is non-flat only through two
defects: it builds `Pn`, `Pr` from a Vasicek closed form with `theta = r0` (a model-implied flat
curve) where the market curves belong, and it applies a convexity adjustment to the ZCIIS when JY
puts convexity on the **YoY** swap (eq 2.5.11). Measured at a 5y quote, doubling `sigma_I` moves
its fair rate +10.3bp and doubling `sigma_r` +15.1bp -- enough to drive a least-squares fit against
basis-point quotes, and entirely artefact. It converges on its own bugs and reports success.
See `docs/tranches/06_jy_calibration.md` section 0.

**Layer note.** This is L4, with the pricers, not L3 with the models -- a calibrator that *inverts*
an engine must sit at or above that engine. The spine's "L3 MODELS + CALIBRATION" covers the
model-free kind (a curve bootstrap, tranche 2's break-even inversion); instrument-inverting vol
calibration is a consumer of `price()` and belongs beside it. No gate or charter change needed.

Provenance:
  stock:  _stock/quarry/pricebook/fixed_income/jarrow_yildirim.py::jy_calibrate -- rejected as a
          model, not purified: it fits vols to the wrong instrument (above). Mined for content
          only: the fixed-vs-free split, and that correlations come from historical estimation.
  source: BERHAN-2012 ch 5 (JY calibrated to YoY floor quotes, ATM only -- the thesis's own
          finding is that JY produces no smile, which this reproduces rather than papers over)
  oracle: tests/L4/test_jy_calibration.py -- synthetic parameter recovery (A1), with the two
          discriminators that stop A1 being vacuous: per-parameter sensitivity (A2a, "not flat")
          and multi-start uniqueness plus Jacobian conditioning (A2b, "not degenerate"); the
          objective vanishing at the truth and descending from a wrong start (A3); the cross-stage
          break-even round trip (A4); failure as a value (A5)
  slice:  tranche 6 -- calibrating Jarrow-Yildirim
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pricebook.engine.capfloor import price_inflation_capfloor
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.solvers import least_squares_fit
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.jarrow_yildirim import (
    HullWhiteParams,
    JarrowYildirim,
    JYCorrelations,
    JYParams,
)
from pricebook.products.inflation import InflationCapFloor

Quote = tuple[InflationCapFloor, float]

_FLOOR = 1e-10   # a volatility must be strictly positive to be fitted; this is the lower bound


@dataclass(frozen=True)
class JYFixedParams:
    """What the vol fit holds constant: the two mean reversions, the nominal vol, the correlations.

    A bundle rather than four loose arguments, because the calibrator otherwise takes seven and the
    rule's answer is to group the cohesive ones -- and these are cohesive in a stronger sense than
    convenience: they are exactly the parameters the thesis sources *elsewhere* (historical
    estimation for reversions and correlations, nominal ATM caps for `sigma_n`). The bundle is the
    staging of the calibration, made explicit."""

    kappa_nominal: float
    kappa_real: float
    sigma_nominal: float
    correlations: JYCorrelations


@dataclass(frozen=True)
class JYCalibration:
    """A successful vol fit. `params` is a complete `JYParams`, so a calibration's output is
    directly a model's input -- no unpacking step in which a caller can mix a fitted vol with the
    wrong fixed set."""

    params: JYParams
    rms_residual: float
    n_quotes: int
    iterations: int


def jy_params_from(fixed: JYFixedParams, sigma_real: float, sigma_index: float) -> JYParams:
    """Assemble the full parameter set from the fixed stage plus the two free volatilities."""
    return JYParams(
        nominal=HullWhiteParams(fixed.kappa_nominal, fixed.sigma_nominal),
        real=HullWhiteParams(fixed.kappa_real, sigma_real),
        index_vol=sigma_index,
        correlations=fixed.correlations,
    )


def _strip_pvs(
    quotes: Sequence[Quote], market: MarketSnapshot, fixed: JYFixedParams, x: Sequence[float]
) -> list[float] | PricingFailure:
    """Price every quoted floor at the parameter pair `x`, or say why it cannot be done."""
    # The floor pricer directly, not the registry's `dispatch`: this calibrator fits one known
    # instrument, so dispatching on a type it already knows is indirection with no consumer.
    model = JarrowYildirim(market, jy_params_from(fixed, x[0], x[1]))
    pvs: list[float] = []
    for product, _ in quotes:
        out = price_inflation_capfloor(product, model)
        if not isinstance(out, PricingResult):
            return PricingFailure(f"a quoted floor cannot be priced: {out.reason}")
        pvs.append(out.pv.amount)
    return pvs


def calibrate_jy_to_yoy_floors(
    quotes: Sequence[Quote],
    market: MarketSnapshot,
    fixed: JYFixedParams,
    start: tuple[float, float],
) -> JYCalibration | PricingFailure:
    """Fit `(sigma_r, sigma_I)` to `quotes`, a sequence of `(floor, quoted PV)` pairs.

    Overdetermined least squares: two free parameters against a strip of maturities, which is what
    separates them -- `sigma_I` enters `V^2` as `sigma_I^2 * delta`, flat within a period, while
    `sigma_r` enters through the `beta_r` terms and additionally scales all of the convexity `C`.
    Returns `PricingFailure` rather than raising, and rather than returning a plausible-looking
    parameter pair from a solve that did not converge."""
    if not quotes:
        return PricingFailure("cannot calibrate without at least one floor quote")
    if any(v <= 0.0 for v in start):
        return PricingFailure(f"the starting volatilities must be positive, got {start}")

    # Price the strip once at the start point. This validates the market (a snapshot missing the
    # real curve cannot price a floor, so it cannot calibrate to one) BEFORE entering the solver,
    # which is what keeps a `KeyError` from escaping mid-iteration as something other than a value.
    opening = _strip_pvs(quotes, market, fixed, start)
    if isinstance(opening, PricingFailure):
        return opening

    targets = [q for _, q in quotes]

    def residuals(x: Sequence[float]) -> list[float]:
        pvs = _strip_pvs(quotes, market, fixed, x)
        if isinstance(pvs, PricingFailure):
            # Unreachable in practice -- the opening price above proves the strip is priceable, and
            # the bounds keep every iterate in the valid region. Returning non-finite residuals
            # rather than raising means that if it ever IS reached, the solver reports
            # non-convergence and the caller gets a PricingFailure, which is the contract.
            return [math.inf] * len(targets)
        return [pv - target for pv, target in zip(pvs, targets)]

    outcome = least_squares_fit(
        residuals, start, bounds=((_FLOOR, _FLOOR), (math.inf, math.inf))
    )
    if not outcome.converged:
        return PricingFailure(
            f"the volatility fit did not converge from {start} on {len(quotes)} quotes"
        )

    rms = math.sqrt(2.0 * outcome.cost / len(quotes))   # cost is half the sum of squares
    return JYCalibration(
        params=jy_params_from(fixed, outcome.x[0], outcome.x[1]),
        rms_residual=rms,
        n_quotes=len(quotes),
        iterations=outcome.iterations,
    )
