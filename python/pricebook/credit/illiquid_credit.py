"""Illiquid credit: distressed pricing, EM conventions, recovery sensitivity.

Tools for names without liquid CDS markets or trading in distressed regime.

    from pricebook.illiquid_credit import distressed_recovery_sensitivity

    sens = distressed_recovery_sensitivity(cds, disc, sc, recovery_range=(0.1, 0.6))

References:
    O'Kane (2008). Ch. 6 — Distressed CDS and Recovery.
    Schönbucher (2003). Credit Derivatives Pricing Models. Ch. 4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from pricebook.cds import CDS
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


# ---- Restructuring clauses ----

RESTRUCTURING_CLAUSES = {
    "CR": "Full Restructuring (Old R) — all obligations deliverable",
    "MR": "Modified Restructuring (Mod R) — maturity cap on deliverables",
    "MM": "Modified-Modified Restructuring (Mod-Mod R) — wider maturity cap",
    "XR": "No Restructuring (No R) — only bankruptcy/failure-to-pay",
}


# ---- EM sovereign CDS conventions ----

EM_CONVENTIONS = {
    "standard": {"coupon_bps": 100, "recovery": 0.25, "restructuring": "CR"},
    "hy_sovereign": {"coupon_bps": 500, "recovery": 0.25, "restructuring": "CR"},
    "ig_sovereign": {"coupon_bps": 25, "recovery": 0.40, "restructuring": "MR"},
}


# ---- Distressed analytics ----

@dataclass
class RecoverySensitivity:
    """Recovery sensitivity analysis for distressed names."""
    base_pv: float
    base_recovery: float
    recovery_dv01: float        # PV change per 1% recovery shift
    pv_at_recoveries: dict[float, float]  # {recovery: PV}

    def to_dict(self) -> dict:
        return {"base_pv": self.base_pv, "base_recovery": self.base_recovery,
                "recovery_dv01": self.recovery_dv01,
                "pv_at_recoveries": self.pv_at_recoveries}


def distressed_recovery_sensitivity(
    cds: CDS,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery_range: tuple[float, float] = (0.05, 0.80),
    n_points: int = 16,
) -> RecoverySensitivity:
    """Analyse recovery sensitivity for distressed CDS.

    For distressed names (spread > 500bp), PV is dominated by recovery
    assumption, not spread. This function maps PV across recovery values.

    Args:
        recovery_range: (min_recovery, max_recovery) to evaluate.
        n_points: number of recovery levels to test.
    """
    base_pv = cds.pv(discount_curve, survival_curve)

    # Recovery DV01: PV change per 1% recovery increase
    cds_up = CDS(cds.start, cds.end, cds.spread, cds.notional,
                  recovery=min(cds.recovery + 0.01, 0.99))
    pv_up = cds_up.pv(discount_curve, survival_curve)
    recovery_dv01 = pv_up - base_pv

    # PV across recovery range
    pv_map = {}
    step = (recovery_range[1] - recovery_range[0]) / max(n_points - 1, 1)
    for i in range(n_points):
        r = recovery_range[0] + i * step
        r = min(max(r, 0.01), 0.99)
        cds_r = CDS(cds.start, cds.end, cds.spread, cds.notional, recovery=r)
        pv_map[round(r, 4)] = cds_r.pv(discount_curve, survival_curve)

    return RecoverySensitivity(
        base_pv=base_pv, base_recovery=cds.recovery,
        recovery_dv01=recovery_dv01, pv_at_recoveries=pv_map,
    )


def is_distressed(spread_bps: float, threshold_bps: float = 500.0) -> bool:
    """Check if a name is in distressed territory."""
    return spread_bps >= threshold_bps


def implied_default_prob(
    par_spread: float,
    recovery: float = 0.4,
    maturity_years: float = 5.0,
) -> float:
    """Rough implied cumulative default probability from par spread.

    Approximate: P(default by T) ≈ 1 - exp(-spread/(1-R) × T)

    More accurate than spread/(1-R) for high spreads.
    """
    if recovery >= 1:
        return 0.0
    hazard = par_spread / (1 - recovery)
    return 1 - math.exp(-hazard * maturity_years)


def recovery_breakeven(
    cds: CDS,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> float:
    """Recovery rate at which CDS PV = 0 (breakeven for protection buyer).

    Solves: protection_leg(R_be) = premium_leg.
    """
    from pricebook.solvers import brentq
    from pricebook.cds import protection_leg_pv, premium_leg_pv

    prem = premium_leg_pv(cds.start, cds.end, cds.spread, discount_curve,
                           survival_curve, cds.notional)

    def objective(r: float) -> float:
        prot = protection_leg_pv(cds.start, cds.end, discount_curve,
                                  survival_curve, r, cds.notional)
        return prot - prem

    try:
        return brentq(objective, 0.01, 0.99)
    except Exception:
        return cds.recovery
