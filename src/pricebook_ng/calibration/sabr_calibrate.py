"""SABR smile calibration — fit (α, ρ, ν) to a quoted smile (L3).

A per-family calibrator beside `vol_strip.py`: given a single-expiry `SmileQuote`, fit the three
free SABR parameters (β is FIXED, carried on the spec) by bounded least-squares over the per-strike
lognormal-vol residual, COMPOSING the SAME `sabr_vol` atom the model and engine read (§3d). Failure
is a value (invariant 4): a degenerate smile (non-positive forward/strike — lognormal SABR undefined)
or a non-converged solve returns `CalibrationFailure`.

§3d F-identity (the load-bearing constraint): the calibrator derives the forward from the model's OWN
projection curve, via the SAME canonical accrual `SABRModel.black_vol` uses — so the fitted params,
written to a `SabrSurface` and read back by `SABRModel`, reproduce the calibrated smile at exactly the
F priced at, by construction (calibrate-to-quote == price-to-smile).

Provenance:
  quarry: python/pricebook/options/sabr.py (sabr_calibrate / calibrate_sabr_smile)
  source: Hagan et al. (2002); CLAUDE.md §1 (unified front) · §3d (shared atom, F-identity) · §7bb
  oracle: round-trip seed recovery; reprice identity via SABRModel; degenerate smile → failure
  slice:  sabr-calibrate (T1 slice 21)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from pricebook_ng.calibration.calibrate import CalibrationFailure, CalibrationResult, SolveConfig
from pricebook_ng.foundation import Accrual, RateIndex, root_nd
from pricebook_ng.market.building_blocks import forward
from pricebook_ng.market.quotes import SmileQuote
from pricebook_ng.market.vol_surface import SabrParams
from pricebook_ng.models.black import vol_time_measure
from pricebook_ng.models.protocols import CalibratedModel
from pricebook_ng.models.sabr import sabr_vol

_PARAM_CAP = 5.0  # α (vol scale) or ν (vol-of-vol) above 500% is unphysical — the box ceiling
_RHO_CAP = 0.9999  # ρ ∈ (−1, 1) open; the box approximates it (a real box, never a penalty term)


@dataclass(frozen=True)
class SabrCalibrationSpec:
    """What to fit: the `model` carrying the projection curve (its F is what `SABRModel.black_vol`
    reads — §3d), the `index` the smile is quoted on, the single-expiry `smile`, the FIXED `beta`,
    and the `solve` config (tolerance / iteration cap, reused verbatim)."""

    model: CalibratedModel
    index: RateIndex
    smile: SmileQuote
    beta: float
    solve: SolveConfig = field(default_factory=SolveConfig)


def calibrate_sabr_smile(
    spec: SabrCalibrationSpec,
) -> tuple[SabrParams, CalibrationResult] | CalibrationFailure:
    """Fit (α, ρ, ν) to `spec.smile` by bounded least-squares (no penalty). Returns the fitted
    `SabrParams` beside a `CalibrationResult`, or `CalibrationFailure` on a degenerate smile /
    non-convergence."""
    market = spec.model.market
    index, smile, beta = spec.index, spec.smile, spec.beta
    # F via the SAME atom + canonical accrual SABRModel.black_vol uses (§3d — no forward drift)
    accrual = Accrual(smile.expiry, smile.expiry + index.id.tenor, index.accrual.day_count)
    try:
        fwd = forward(market.curves.projection(index), accrual)
    except (ValueError, KeyError, ZeroDivisionError) as exc:
        return CalibrationFailure(str(exc))
    if fwd <= 0.0 or any(k <= 0.0 for k in smile.strikes):  # lognormal SABR undefined (cf. caplet #15)
        return CalibrationFailure("forward or strike ≤ 0: lognormal SABR undefined")
    t = vol_time_measure(market.valuation_date).year_fraction(smile.expiry)

    def residual(x: Sequence[float]) -> list[float]:
        params = SabrParams(x[0], beta, x[1], x[2])
        return [sabr_vol(fwd, k, t, params) - mv for k, mv in zip(smile.strikes, smile.vols)]

    atm = min(range(len(smile.strikes)), key=lambda i: abs(smile.strikes[i] - fwd))
    x0 = [smile.vols[atm] * fwd ** (1.0 - beta), -0.1, 0.3]  # ATM seed (ADAPTED from the quarry)
    bounds = ([1e-8, -_RHO_CAP, 1e-8], [_PARAM_CAP, _RHO_CAP, _PARAM_CAP])
    x, _jac, converged = root_nd(residual, x0, spec.solve.tolerance, spec.solve.max_iterations, bounds)
    if not converged:
        return CalibrationFailure("SABR smile fit did not converge")
    return SabrParams(x[0], beta, x[1], x[2]), CalibrationResult(
        residuals=tuple(residual(x)), converged=True, jacobian=None
    )
