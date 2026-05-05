"""Recovery analytics: recovery-parameterised curves, adjusted repricing, Greeks.

Breaks the recovery-hazard entanglement: CDS spreads are observable,
but the hazard rate h = spread / (1-R) depends on the assumed recovery.
This module provides tools to reprice instruments across different
recovery assumptions while keeping CDS spreads constant.

    from pricebook.recovery_analytics import (
        recovery_curve_family,
        reprice_at_recovery,
        recovery_greeks,
        recovery_pv_surface,
    )

References:
    O'Kane, D. (2008). Modelling Single-name and Multi-name Credit
    Derivatives, Ch. 4 (Recovery and hazard rate duality).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.cds_market import build_cds_curve


# ---------------------------------------------------------------------------
# Recovery-parameterised curve family
# ---------------------------------------------------------------------------

def recovery_curve_family(
    cds_spreads: dict[int, float],
    discount_curve: DiscountCurve,
    reference_date: date,
    recoveries: list[float] | None = None,
) -> dict[float, SurvivalCurve]:
    """Bootstrap a family of survival curves at different recoveries.

    For the same CDS par spreads, each recovery assumption produces
    a different survival curve: h(R) = spread / (1-R).

    Higher R → higher h → lower survival → steeper curve.

    Args:
        cds_spreads: {tenor_years: par_spread}, e.g. {1: 0.005, 5: 0.01}.
        discount_curve: risk-free discount curve.
        reference_date: valuation date.
        recoveries: list of recovery rates to calibrate. Default: [0.2, 0.3, 0.4, 0.5, 0.6].

    Returns:
        {recovery: SurvivalCurve} dict.
    """
    if recoveries is None:
        recoveries = [0.20, 0.30, 0.40, 0.50, 0.60]

    family = {}
    for R in recoveries:
        if R >= 1.0 or R <= 0.0:
            import warnings
            warnings.warn(f"Skipping invalid recovery R={R} (must be in (0, 1))", stacklevel=2)
            continue
        curve = build_cds_curve(reference_date, cds_spreads, discount_curve, recovery=R)
        family[R] = curve

    return family


# ---------------------------------------------------------------------------
# Recovery-adjusted repricing
# ---------------------------------------------------------------------------

@dataclass
class RecoveryAdjustedResult:
    """Result of repricing at a non-standard recovery."""
    recovery: float
    pv: float
    pv_convention: float       # PV at convention R (40%)
    difference: float          # pv - pv_convention
    difference_pct: float      # difference / notional
    hazard_at_5y: float        # implied 5Y hazard for this R

    def to_dict(self) -> dict:
        return {
            "recovery": self.recovery, "pv": self.pv,
            "pv_convention": self.pv_convention,
            "difference": self.difference, "diff_pct": self.difference_pct,
            "hazard_5y": self.hazard_at_5y,
        }


def reprice_at_recovery(
    instrument,
    discount_curve: DiscountCurve,
    cds_spreads: dict[int, float],
    reference_date: date,
    target_recovery: float,
    convention_recovery: float = 0.4,
) -> RecoveryAdjustedResult:
    """Reprice a credit instrument at a non-standard recovery.

    Keeps CDS spreads constant but changes both:
    1. The survival curve (re-bootstrapped at target R)
    2. The recovery payment on the instrument

    This is the correct way to see recovery sensitivity —
    not just bumping R on the instrument while keeping the same curve.

    Supports CreditLinkedNote, CDS, RiskyBond (anything with
    dirty_price(curve, survival) or pv(curve, survival)).
    """
    # Bootstrap at convention R
    surv_conv = build_cds_curve(reference_date, cds_spreads, discount_curve,
                                recovery=convention_recovery)

    # Bootstrap at target R (same spreads, different h)
    surv_target = build_cds_curve(reference_date, cds_spreads, discount_curve,
                                  recovery=target_recovery)

    # Price at convention
    pv_conv = _price_instrument(instrument, discount_curve, surv_conv, convention_recovery)

    # Price at target: use target survival curve AND target recovery
    pv_target = _price_instrument(instrument, discount_curve, surv_target, target_recovery)

    # Implied hazard at 5Y
    from pricebook.day_count import DayCountConvention, year_fraction
    from dateutil.relativedelta import relativedelta
    t5 = reference_date + relativedelta(years=5)
    T = year_fraction(reference_date, t5, DayCountConvention.ACT_365_FIXED)
    q5 = surv_target.survival(t5)
    # Note: floor at 1e-15 can produce very large hazards (~35/T) for near-default names.
    # This is intentional — represents the extreme case correctly.
    h5 = -math.log(max(q5, 1e-15)) / max(T, 1e-10)

    notional = getattr(instrument, 'notional', None)
    if notional is None:
        raise ValueError("Instrument must have 'notional' attribute for recovery repricing")
    diff = pv_target - pv_conv

    return RecoveryAdjustedResult(
        recovery=target_recovery, pv=pv_target,
        pv_convention=pv_conv, difference=diff,
        difference_pct=diff / notional if notional > 0 else 0.0,
        hazard_at_5y=h5,
    )


def _price_instrument(instrument, curve, survival, recovery):
    """Price instrument with given survival curve and recovery."""
    old_recovery = getattr(instrument, 'recovery', None)
    try:
        if old_recovery is not None:
            instrument.recovery = recovery
        if hasattr(instrument, 'dirty_price'):
            return instrument.dirty_price(curve, survival)
        elif hasattr(instrument, 'pv'):
            return instrument.pv(curve, survival)
        else:
            raise TypeError(f"Cannot price {type(instrument).__name__}")
    finally:
        if old_recovery is not None:
            instrument.recovery = old_recovery


# ---------------------------------------------------------------------------
# Recovery Greeks (total: direct + indirect through h(R))
# ---------------------------------------------------------------------------

@dataclass
class RecoveryGreeksResult:
    """Recovery-adjusted Greeks decomposition."""
    total_dPV_dR: float        # full derivative including h(R) effect
    direct_effect: float       # dPV/dR with fixed survival curve
    indirect_effect: float     # effect through re-bootstrapped h(R)
    convexity: float           # d²PV/dR²
    base_recovery: float
    base_pv: float

    def to_dict(self) -> dict:
        return {
            "total_dPV_dR": self.total_dPV_dR,
            "direct": self.direct_effect,
            "indirect": self.indirect_effect,
            "convexity": self.convexity,
            "base_R": self.base_recovery, "base_pv": self.base_pv,
        }


def recovery_greeks(
    instrument,
    discount_curve: DiscountCurve,
    cds_spreads: dict[int, float],
    reference_date: date,
    base_recovery: float = 0.4,
    bump: float = 0.01,
) -> RecoveryGreeksResult:
    """Recovery-adjusted Greeks with direct/indirect decomposition.

    Total dPV/dR = direct_effect + indirect_effect

    - Direct: change R on the instrument, keep survival curve fixed
    - Indirect: change R → h changes → survival curve changes → PV changes
    - Total: change both simultaneously (the correct recovery sensitivity)

    Convexity = d²PV/dR² (second-order, centred)
    """
    surv_base = build_cds_curve(reference_date, cds_spreads, discount_curve,
                                 recovery=base_recovery)
    pv_base = _price_instrument(instrument, discount_curve, surv_base, base_recovery)

    # Total: bump R → re-bootstrap survival → reprice with new R and new curve
    surv_up = build_cds_curve(reference_date, cds_spreads, discount_curve,
                               recovery=base_recovery + bump)
    surv_dn = build_cds_curve(reference_date, cds_spreads, discount_curve,
                               recovery=base_recovery - bump)
    pv_up = _price_instrument(instrument, discount_curve, surv_up, base_recovery + bump)
    pv_dn = _price_instrument(instrument, discount_curve, surv_dn, base_recovery - bump)
    total = (pv_up - pv_dn) / (2 * bump)

    # Direct: bump R on instrument only, keep survival curve fixed
    pv_direct_up = _price_instrument(instrument, discount_curve, surv_base, base_recovery + bump)
    pv_direct_dn = _price_instrument(instrument, discount_curve, surv_base, base_recovery - bump)
    direct = (pv_direct_up - pv_direct_dn) / (2 * bump)

    # Indirect: total - direct
    indirect = total - direct

    # Convexity: d²PV/dR²
    convexity = (pv_up - 2 * pv_base + pv_dn) / (bump ** 2)

    return RecoveryGreeksResult(
        total_dPV_dR=total, direct_effect=direct, indirect_effect=indirect,
        convexity=convexity, base_recovery=base_recovery, base_pv=pv_base,
    )


# ---------------------------------------------------------------------------
# Recovery PV surface
# ---------------------------------------------------------------------------

@dataclass
class RecoverySurfacePoint:
    """One point on the recovery PV surface."""
    recovery: float
    pv: float
    hazard: float

    def to_dict(self) -> dict:
        return {"recovery": self.recovery, "pv": self.pv, "hazard": self.hazard}


def recovery_pv_surface(
    instrument,
    discount_curve: DiscountCurve,
    cds_spreads: dict[int, float],
    reference_date: date,
    recoveries: list[float] | None = None,
) -> list[RecoverySurfacePoint]:
    """PV surface across recovery values.

    Shows how the instrument's value changes as recovery moves,
    accounting for the induced change in hazard rates.
    """
    if recoveries is None:
        recoveries = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    from pricebook.day_count import DayCountConvention, year_fraction
    from dateutil.relativedelta import relativedelta

    points = []
    for R in recoveries:
        if R >= 1.0 or R <= 0.0:
            continue
        surv = build_cds_curve(reference_date, cds_spreads, discount_curve, recovery=R)
        pv = _price_instrument(instrument, discount_curve, surv, R)

        t5 = reference_date + relativedelta(years=5)
        T = year_fraction(reference_date, t5, DayCountConvention.ACT_365_FIXED)
        q = surv.survival(t5)
        h = -math.log(max(q, 1e-15)) / max(T, 1e-10)

        points.append(RecoverySurfacePoint(R, pv, h))

    return points
