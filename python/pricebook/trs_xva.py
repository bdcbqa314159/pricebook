"""TRS XVA: MVA, analytic CVA/DVA, KVA from SA-CCR, all-in pricing.

Wires the generic XVA framework to TRS-specific economics.
Unlike repos (constant exposure), TRS exposure is path-dependent
(equity moves, credit spreads widen).

    from pricebook.trs_xva import (
        trs_mva, trs_kva_from_sa_ccr, trs_analytic_cva,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.trs import TotalReturnSwap
from pricebook.discount_curve import DiscountCurve
from pricebook.day_count import DayCountConvention, year_fraction


def trs_mva(
    trs: TotalReturnSwap,
    simm_im: float | None = None,
    funding_spread: float = 0.002,
) -> float:
    """MVA: cost of funding initial margin over the TRS term.

    MVA = IM × funding_spread × T.

    If simm_im not provided, estimates as notional × 5% (rough proxy).
    """
    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    if simm_im is None:
        simm_im = trs.notional * 0.05  # 5% proxy
    return simm_im * funding_spread * T


def trs_kva_from_sa_ccr(
    trs: TotalReturnSwap,
    curve: DiscountCurve,
    counterparty_rw: float = 1.0,
    hurdle_rate: float = 0.10,
) -> float:
    """KVA from SA-CCR EAD.

    KVA = EAD × RW × 8% × hurdle × T.

    Uses simplified SA-CCR: EAD ≈ max(MTM, 0) + notional × supervisory_factor.
    """
    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    result = trs.price(curve)
    mtm = max(result.value, 0)

    # Supervisory factor by type
    sf = {"equity": 0.32, "bond": 0.005, "loan": 0.005, "cln": 0.005}
    factor = sf.get(trs._underlying_type, 0.10)

    ead = 1.4 * (mtm + trs.notional * factor * math.sqrt(min(T, 1.0)))
    capital = ead * counterparty_rw * 0.08
    return capital * hurdle_rate * T


def trs_analytic_cva(
    trs: TotalReturnSwap,
    curve: DiscountCurve,
    hazard_rate: float = 0.02,
    recovery: float = 0.4,
) -> float:
    """Analytic CVA approximation for a TRS.

    CVA ≈ (1-R) × ∫ EPE(t) × h × exp(-h×t) × df(t) dt

    For TRS: EPE ≈ |delta| × σ × S × √t (Gaussian approximation).
    Simplified to: CVA ≈ (1-R) × h × EPE_avg × T × df_avg.
    """
    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    result = trs.price(curve)

    # EPE approximation
    if trs._underlying_type == "equity":
        sigma = trs.sigma if trs.sigma > 0 else 0.20
        spot = float(trs.underlying) if isinstance(trs.underlying, (int, float)) else 100.0
        epe = trs.notional * sigma * math.sqrt(T) * 0.4  # ~40% of 1σ move
    else:
        # Bond/loan: EPE ≈ notional × spread × duration
        epe = max(abs(result.value), trs.notional * 0.02)

    # Survival-weighted EPE
    lgd = 1 - recovery
    df_mid = curve.df(trs.start + (trs.end - trs.start) // 2) if trs.start else 1.0
    surv = math.exp(-hazard_rate * T)

    cva = lgd * hazard_rate * epe * T * df_mid * (1 + surv) / 2

    return cva


def trs_analytic_dva(
    trs: TotalReturnSwap,
    curve: DiscountCurve,
    own_hazard: float = 0.01,
    own_recovery: float = 0.4,
) -> float:
    """Analytic DVA: own-credit benefit.

    Same formula as CVA but using own default probability and ENE.
    DVA is a benefit (reduces cost) — typically reported as positive.
    """
    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    result = trs.price(curve)

    # ENE ≈ negative exposure
    ene = max(-result.value, trs.notional * 0.01)

    lgd = 1 - own_recovery
    df_mid = curve.df(trs.start + (trs.end - trs.start) // 2) if trs.start else 1.0

    dva = lgd * own_hazard * ene * T * df_mid

    return dva


# ---------------------------------------------------------------------------
# Independent Amount (IA) — initial margin at inception
# ---------------------------------------------------------------------------

def trs_independent_amount(
    trs: TotalReturnSwap,
    ia_method: str = "percentage",
    ia_pct: float = 0.10,
    simm_im: float | None = None,
) -> float:
    """Compute Independent Amount (IA) for a TRS.

    IA is the initial margin posted at trade inception, distinct from
    variation margin (which changes daily with MTM).

    Methods:
    - "percentage": IA = notional × ia_pct (simple, common for bilateral).
    - "simm": IA = SIMM IM (regulatory, for UMR-compliant).
    - "fixed": IA = simm_im (explicitly provided amount).
    """
    if ia_method == "simm" and simm_im is not None:
        return simm_im
    if ia_method == "fixed" and simm_im is not None:
        return simm_im
    return trs.notional * ia_pct
