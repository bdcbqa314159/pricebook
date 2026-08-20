"""Caplet-vol stripping — the first SOLVED volatility surface (L3).

The vol analogue of the curve bootstrap, in a new family: given ascending flat cap vols, strip
each maturity's MARGINAL caplet vol so the caps reprice — a sequential 1-D Brent solve per pillar.
It COMPOSES the same atoms the L4 caplet engine composes — `black()` (undiscounted), `forward`,
and `df` — never a private caplet formula and never the L4 engine itself (L3 ⊥ L4, §3d). A
per-family entry (a `Surface` is a different target than a `CurveSet`; vols are downstream of a
finished curve), reusing `SolveConfig` verbatim. Failure is a value (invariant 4): an infeasible
marginal (target outside the vol-attainable range) returns `CalibrationFailure`.

Provenance:
  quarry: python/pricebook/options/vol_calibration.py; options/capfloor.py (strip_caplet_vols)
  source: CLAUDE.md §1 (unified front over per-family solvers) · §3d (shared atoms); doc 22 (A) fork
  oracle: flat cap strips to a flat caplet-vol curve exactly; the surface reprices each cap to quote
  slice:  caplet-vol-strip (C2 slice 3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pricebook_ng.calibration.calibrate import CalibrationFailure, CalibrationResult, SolveConfig
from pricebook_ng.foundation import Accrual, DayCountConvention, RateIndex, TimeMeasure, brent
from pricebook_ng.market.building_blocks import forward
from pricebook_ng.market.quotes import CapQuote
from pricebook_ng.market.vol_surface import Surface
from pricebook_ng.models.black import black
from pricebook_ng.models.protocols import CalibratedModel
from pricebook_ng.products.option import OptionType

_ACT365F = DayCountConvention.ACT_365_FIXED
_VOL_CAP = 5.0  # a caplet vol above 500% is unphysical — treat the pillar as infeasible


@dataclass(frozen=True)
class VolCalibrationSpec:
    """What to strip: the `model` carrying the calibrated curves (A1 — the strip reads its
    curves, the vols are downstream of a finished curve), the `index` the caplets fix on, the
    ascending flat-cap `quotes`, and the `solve` config (reused verbatim — SEQUENTIAL/Brent)."""

    model: CalibratedModel
    index: RateIndex
    quotes: tuple[CapQuote, ...]
    solve: SolveConfig = field(default_factory=SolveConfig)


@dataclass(frozen=True)
class _Leg:
    """One caplet's fixed inputs (everything but the vol) — the undiscounted Black annuity
    `pay_df·τ` and the forward/strike/expiry it is struck on."""

    annuity: float  # pay_df · τ
    forward: float
    strike: float
    expiry_time: float

    def pv(self, vol: float) -> float:
        return self.annuity * black(self.forward, self.strike, vol, self.expiry_time, OptionType.CALL)


def _solve_vol(leg: _Leg, target_pv: float) -> float:
    """The caplet vol reproducing `target_pv`. `pv` is monotonic increasing in vol from the
    intrinsic (`vol→0`) to the undiscounted forward (`vol→∞`), so a bracket exists iff the
    target lies in that open range; otherwise the pillar is infeasible (raised → failure)."""
    def residual(vol: float) -> float:
        return leg.pv(vol) - target_pv

    lo, hi = 1e-9, 0.05
    f_lo = residual(lo)
    f_hi = residual(hi)
    while f_lo * f_hi > 0.0 and hi < _VOL_CAP:
        hi *= 2.0
        f_hi = residual(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(f"caplet vol not bracketable below {_VOL_CAP:.0%} (infeasible marginal)")
    return brent(residual, lo, hi)


def strip_caplet_vols(
    spec: VolCalibrationSpec,
) -> tuple[Surface, CalibrationResult] | CalibrationFailure:
    """Strip caplet vols from the flat cap quotes. Sequential: cap(mₖ) at its flat vol prices the
    whole strip; the marginal caplet k is solved so the cumulative stripped PV matches. Returns the
    term-structure `Surface` (populating the surfaces shape) beside a `CalibrationResult`; an
    infeasible pillar returns `CalibrationFailure` (invariant 4)."""
    quotes = sorted(spec.quotes, key=lambda q: q.maturity)
    market = spec.model.market
    currency = spec.index.id.currency
    discount = market.curves.discount(currency)
    projection = market.curves.projection(spec.index)
    day_count = spec.index.accrual.day_count
    tenor = spec.index.id.tenor
    tm = TimeMeasure(market.valuation_date, _ACT365F)

    legs: list[_Leg] = []
    vols: list[float] = []
    expiries: list[date] = []
    stripped_pv = 0.0  # Σ_{j<k} caplet_j(σ_j)
    for q in quotes:
        accrual = Accrual(q.maturity, q.maturity + tenor, day_count)
        leg = _Leg(
            annuity=discount.df(accrual.end) * accrual.year_fraction(),
            forward=forward(projection, accrual),
            strike=q.strike,
            expiry_time=tm.year_fraction(q.maturity),
        )
        # the flat quote prices the WHOLE cap: Σ_{j≤k} caplet_j(σ̄ₖ); the marginal is what's left
        cap_pv_flat = sum(existing.pv(q.flat_vol) for existing in legs) + leg.pv(q.flat_vol)
        target_marginal = cap_pv_flat - stripped_pv
        try:
            vol = _solve_vol(leg, target_marginal)
        except (ValueError, ZeroDivisionError) as exc:
            return CalibrationFailure(f"cap {q.maturity.isoformat()}: {exc}")
        legs.append(leg)
        vols.append(vol)
        expiries.append(q.maturity)
        stripped_pv += leg.pv(vol)

    # residuals: each cap repriced with the stripped vols vs its flat-quote PV (≈0 by construction)
    residuals: list[float] = []
    for k, q in enumerate(quotes):
        cap_flat = sum(legs[j].pv(q.flat_vol) for j in range(k + 1))
        cap_stripped = sum(legs[j].pv(vols[j]) for j in range(k + 1))
        residuals.append(cap_stripped - cap_flat)

    surface = Surface(tuple(vols), tuple(expiries))
    converged = all(abs(r) < spec.solve.tolerance for r in residuals)
    return surface, CalibrationResult(tuple(residuals), converged=converged, jacobian=None)
