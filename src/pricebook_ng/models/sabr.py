"""SABR lognormal implied vol + the SABR model (L3).

`sabr_vol(forward, strike, t, params)` is the Hagan et al. (2002) lognormal (Black-76) implied
volatility of the SABR dynamics — the closed-form ANALYTIC BLOCK of a dynamics, so it lives at L3,
not L0 (the `models/black.py` precedent, CLAUDE.md §1). Discounting is OUT (df ∉ its args); the L4
engine multiplies `df·N·τ·black(F, K, σ_SABR, t)`, exactly as for a flat vol.

`SABRModel` is the SECOND `BlackVol` implementer (after `BlackModel`) and the FIRST that USES the
`strike` argument — so a caplet grows a smile with NO change to the `BlackVol` signature (Q1
rule-of-two on the capability). It is READ, not solved: `SabrParams` are carried on the snapshot
(`market.surfaces[SurfaceKey(index)]` as a `SabrSurface`); calibration is a later slice. §3d
F-identity: `black_vol` derives the forward from its OWN curves using the SAME `forward` atom and
the index's CANONICAL accrual the engine's caplet composes — verified byte-identical for the
standard caplet — so the smile is evaluated at exactly the F the engine prices at, no drift.

Provenance:
  quarry: python/pricebook/models/sabr.py
  source: Hagan, Kumar, Lesniewski, Woodward (2002) "Managing Smile Risk" eq 2.17a/2.18
  oracle: sabr_vol reprices to an independent inline Hagan; ATM branch = eq 2.18; caplet → Black
  slice:  sabr-caplet (T1 slice 20)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation import Accrual, RateIndex
from pricebook_ng.market.building_blocks import forward
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.market.vol_surface import SabrParams, SabrSurface, SurfaceKey
from pricebook_ng.models.black import vol_time_measure

_ATM_LOG_TOL = 1e-12  # |log(F/K)| below this ⇒ the removable z/χ(z) singularity; use the ATM branch


def sabr_vol(forward: float, strike: float, t: float, params: SabrParams) -> float:
    """Hagan 2002 lognormal implied vol σ_B(F, K). β is fixed (carried on `params`). At F=K the
    z/χ(z) factor has a removable singularity → the ATM branch (eq 2.18). `t` is the option
    year-fraction on the canonical vol clock (§3d)."""
    a, b, rho, nu = params.alpha, params.beta, params.rho, params.nu
    one_mb = 1.0 - b
    if abs(math.log(forward / strike)) < _ATM_LOG_TOL:  # ATM, eq 2.18 (and the F=K limit)
        f_pow = forward**one_mb
        correction = (
            one_mb * one_mb / 24.0 * a * a / forward ** (2.0 * one_mb)
            + 0.25 * rho * b * nu * a / f_pow
            + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu
        )
        return a / f_pow * (1.0 + correction * t)
    log_fk = math.log(forward / strike)
    fk_pow = (forward * strike) ** (one_mb / 2.0)
    z = nu / a * fk_pow * log_fk
    # z/χ(z) → 1 as z→0 (removable): ν=0 gives z=χ=0 exactly (0/0), so take the limit for tiny z
    z_over_chi = 1.0 if abs(z) < 1e-12 else z / math.log(
        (math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho)
    )
    denom = fk_pow * (
        1.0 + one_mb * one_mb / 24.0 * log_fk**2 + one_mb**4 / 1920.0 * log_fk**4
    )
    correction = (
        one_mb * one_mb / 24.0 * a * a / (forward * strike) ** one_mb
        + 0.25 * rho * b * nu * a / fk_pow
        + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu
    )
    return a / denom * z_over_chi * (1.0 + correction * t)


@dataclass(frozen=True)
class SABRModel:
    """Carries `market` (A1). Satisfies `CalibratedModel` (`.market`) and `BlackVol` (`.black_vol`) —
    the SAME capability `BlackModel` exposes, now smile-aware. READ, not solved: reads `SabrParams`
    from `market.surfaces[SurfaceKey(index)]` (a `SabrSurface`); a flat `Surface` there is a config
    error surfaced as a value by the engine's catch (KeyError-family)."""

    market: MarketSnapshot

    def black_vol(self, index: RateIndex, expiry: date, strike: float) -> float:
        surface = self.market.surfaces[SurfaceKey(index)]
        if not isinstance(surface, SabrSurface):  # a flat Surface under a SABRModel is a config error
            raise ValueError(f"SABRModel expects a SabrSurface, got {type(surface).__name__}")
        params = surface.at_expiry(expiry)
        # §3d: the SAME forward atom + canonical index accrual the engine's caplet composes → same F
        canonical = Accrual(expiry, expiry + index.id.tenor, index.accrual.day_count)
        fwd = forward(self.market.curves.projection(index), canonical)
        t = vol_time_measure(self.market.valuation_date).year_fraction(expiry)
        return sabr_vol(fwd, strike, t, params)
