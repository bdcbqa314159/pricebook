"""The full six-parameter JY calibration, and why it fails (L4). BERHAN-2012 §5.3.2.

§5.3.2 frees all six of `{κr, σr, σI, ρnr, ρrI, ρnI}` against the year-on-year floor **surface** and
reports the fit fails: *"we cannot fit to the whole surface"*, *"the presence of a strike skew makes
calibration unfeasible for non-ATM contracts"*.

**So this module does not claim a successful fit, and that is the point.** It is the honest
reproduction of a negative result: the machinery is correct, the documented failure is reproduced,
and — going further than the source — it is *explained*.

## There are TWO distinct failures, and the thesis names only one

**1. Underdetermination, and it is structural rather than numerical.** In `yoy_variance` the index
correlations enter **only as products with `σI`**:

    total += si*si*span
    total += 2*rho_ni * (sn*si/kn) * (span - beta_n)
    total -= 2*rho_ri * (sr*si/kr) * (span - beta_r)

so `V²` is a quadratic in `σI` whose linear coefficient is a **single pooled combination** of
`(ρnI, ρrI)`; and `ρnI` is **absent from `yoy_convexity` entirely**, so a floor price sees it *only*
through that pooled product. Three of the six — `(σI, ρrI, ρnI)` — are therefore not identifiable
from YoY floors **at any strike, by construction**.

Measured on *perfect* synthetic ATM data, three starts: `(κr, σr, ρnr)` recover exactly, while
`σI` spreads 27.5%, `ρrI` 21.5%, and `ρnI` **469%** — landing anywhere from +0.15 to −0.55 — with
every fit reporting an rms indistinguishable from zero. **A perfect residual at a wrong answer**, on
flawless data, before any skew enters. BERHAN does not report this half.

**2. Misspecification — the flat smile.** JY is Gaussian, so eq 2.6.5 is Black with a `V²` that
contains no strike: the implied YoY-cap vol is identical at every strike. That is what makes a
market skew unfittable, and it is the failure §5.3.2 names.

## No condition number is asserted anywhere

The smallest singular value of the six-parameter Jacobian sits **below the finite-difference noise
floor**: measured, the condition number swings three orders of magnitude with the step
(`2.5e7 → 2.1e10`) and differs again under central differences. A threshold on it would assert noise
and flake across the two CI operating systems. The discriminator is the **multi-start spread** —
behavioural, step-independent, identical on both OSes. `s₅/s₁ ≈ 3.4e-6` is the only stable number
and is documented here, never gated. (Charter §4; the spec's first draft proposed a condition-number
threshold and additionally claimed it *blows up* as non-ATM strikes are added — it does not, since a
Gram matrix cannot lose rank by gaining rows. Both were measured and withdrawn before building.)

**`(κn, σn)` arrive fixed from §5.3.1's fit (tranche 10b) and are never re-fitted** — refitting them
against YoY floors would launder the nominal-cap calibration into the inflation fit, which is the
same rule that keeps T6 from re-fitting the curve stage.

Provenance:
  stock:  _stock/quarry/pricebook/fixed_income/jarrow_yildirim.py — its `jy_calibrate` mined for
          the six-parameter objective's SHAPE only. Its calibration TARGET is the wrong instrument
          (tranche 6 §0: it fits vols to a ZCIIS that carries none), so nothing of its target is
          reused.
  source: BERHAN-2012 (docs/references.md) §5.3.2 — the six-parameter surface fit and its reported
          failure; §2.5/§2.6 for the forward model being inverted
  oracle: tests/L4/test_jy_full_calibration.py — O1 recovers the IDENTIFIABLE subset from every
          start (proving the solver; the forward it inverts was independently oracled in T4/T5);
          O2 is the discriminator, the multi-start SPREAD, with the pooling and `ρnI`'s absence from
          `C` asserted against the code; O3 the flat smile as a guard with its tautology noted;
          O4 the two failures asserted as DISTINCT, the underdetermination holding on ATM-only data
          before any skew; O5 failure is a value
  crosscheck: none — inverts the cap/floor forward map (tagged at engine/capfloor.py); QuantLib's
          inflation vol is surface-based, not a JY-to-floor calibrator
  slice:  tranche 11 — the full six-parameter fit, and reproducing why it fails
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from pricebook.engine.capfloor import price_inflation_capfloor
from pricebook.engine.jy_calibration import JYFixedParams
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.solvers import least_squares_fit
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.hull_white import HullWhiteParams
from pricebook.models.jarrow_yildirim import JarrowYildirim, JYCorrelations, JYParams
from pricebook.products.inflation import InflationCapFloor

Quote = tuple[InflationCapFloor, float]

_FLOOR = 1e-10      # a mean reversion and a volatility must be strictly positive to be fitted
_RHO_MAX = 0.999    # correlations live in (-1, 1); the open end keeps the Gaussian copula valid
_BP = 1.0e4         # residuals in basis points of notional, as in tranche 10(b)


@dataclass(frozen=True)
class JYFreeParams:
    """The six parameters §5.3.2 frees: the real Hull-White factor, the index vol, the correlations.

    Three fields, not six, because the model's own shape already groups them — `HullWhiteParams` and
    `JYCorrelations` are the bundles T4 and T5 established. A flat six-float signature would be two
    over the limit and would let a caller pass a correlation where a volatility belongs."""

    real: HullWhiteParams
    index_vol: float
    correlations: JYCorrelations

    def with_correlations(self, correlations: JYCorrelations) -> JYFreeParams:
        """A copy with different correlations — so an oracle can isolate one channel without
        rebuilding the whole set and risking a transposed field."""
        return replace(self, correlations=correlations)


@dataclass(frozen=True)
class JYSurfaceCalibration:
    """A completed six-parameter fit.

    **`free` is what the fit produced, not what it determined** — three of the six are not
    identifiable from this data (see the module docstring), so a caller reading `free.index_vol` or
    either index correlation is reading one point from a valley of equally good answers. The
    residual says nothing about that: every start reaches an rms near zero."""

    free: JYFreeParams
    rms_residual: float
    n_quotes: int
    iterations: int


def jy_params_from_free(fixed: JYFixedParams, free: JYFreeParams) -> JYParams:
    """Assemble the full parameter set from §5.3.1's fixed stage plus the six free parameters.

    Note the fixed set's own `correlations` are **ignored**: §5.3.2 frees all three, so they come
    from `free`. Taking the whole `JYFixedParams` anyway keeps one staging object across T6, T10 and
    T11 rather than a second near-identical bundle differing only by which fields are live."""
    return JYParams(
        nominal=HullWhiteParams(fixed.kappa_nominal, fixed.sigma_nominal),
        real=free.real,
        index_vol=free.index_vol,
        correlations=free.correlations,
    )


def _free_from_vector(x: Sequence[float]) -> JYFreeParams:
    return JYFreeParams(
        real=HullWhiteParams(x[0], x[1]),
        index_vol=x[2],
        correlations=JYCorrelations(x[3], x[4], x[5]),
    )


def _surface_pvs(
    quotes: Sequence[Quote], market: MarketSnapshot, fixed: JYFixedParams, x: Sequence[float]
) -> list[float] | PricingFailure:
    """Price every quoted floor at the parameter vector `x`, or say why it cannot be done."""
    model = JarrowYildirim(market, jy_params_from_free(fixed, _free_from_vector(x)))
    pvs: list[float] = []
    for product, _ in quotes:
        out = price_inflation_capfloor(product, model)
        if not isinstance(out, PricingResult):
            return PricingFailure(f"a quoted floor cannot be priced: {out.reason}")
        pvs.append(out.pv.amount)
    return pvs


def calibrate_jy_to_floor_surface(
    quotes: Sequence[Quote],
    market: MarketSnapshot,
    fixed: JYFixedParams,
    start: tuple[float, float, float, float, float, float],
) -> JYSurfaceCalibration | PricingFailure:
    """Fit all six free parameters to `quotes`, a sequence of `(floor, quoted PV)` pairs.

    **It converges, and converging is not the same as determining.** The returned `free` is one
    point on a valley for three of its six components; `JYSurfaceCalibration`'s docstring says which,
    and the module's says why. Returning the fit anyway is the honest thing — the thesis performs
    this fit too — but a caller must not read the undetermined three as measurements."""
    if not quotes:
        return PricingFailure("cannot calibrate without at least one floor quote")
    kappa_r, sigma_r, sigma_i = start[0], start[1], start[2]
    if kappa_r <= 0.0 or sigma_r <= 0.0 or sigma_i <= 0.0:
        return PricingFailure(
            f"the starting mean reversion and volatilities must be positive, got {start[:3]}"
        )
    if any(abs(rho) > _RHO_MAX for rho in start[3:]):
        return PricingFailure(
            f"the starting correlations must lie within +/-{_RHO_MAX}, got {start[3:]}"
        )

    # Price once at the start point, so a missing curve is a value before the solver is entered
    # rather than a KeyError escaping mid-iteration (tranche 6's precedent).
    opening = _surface_pvs(quotes, market, fixed, start)
    if isinstance(opening, PricingFailure):
        return opening

    targets = [q for _, q in quotes]

    def residuals(x: Sequence[float]) -> list[float]:
        pvs = _surface_pvs(quotes, market, fixed, x)
        if isinstance(pvs, PricingFailure):
            return [math.inf] * len(targets)
        return [(pv - target) * _BP for pv, target in zip(pvs, targets)]

    lower = (_FLOOR, _FLOOR, _FLOOR, -_RHO_MAX, -_RHO_MAX, -_RHO_MAX)
    upper = (math.inf, math.inf, math.inf, _RHO_MAX, _RHO_MAX, _RHO_MAX)
    outcome = least_squares_fit(residuals, start, bounds=(lower, upper))
    if not outcome.converged:
        return PricingFailure(
            f"the six-parameter fit did not converge from {start} on {len(quotes)} quotes"
        )

    rms = math.sqrt(2.0 * outcome.cost / len(quotes)) / _BP
    return JYSurfaceCalibration(
        free=_free_from_vector(outcome.x),
        rms_residual=rms,
        n_quotes=len(quotes),
        iterations=outcome.iterations,
    )
