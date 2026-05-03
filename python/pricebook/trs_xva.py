"""TRS XVA: MVA, analytic CVA/DVA, KVA from SA-CCR, MC XVA, SIMM IM.

Wires the generic XVA framework to TRS-specific economics.
Unlike repos (constant exposure), TRS exposure is path-dependent
(equity moves, credit spreads widen).

    from pricebook.trs_xva import (
        trs_mva, trs_kva_from_sa_ccr, trs_analytic_cva,
        trs_simm_im, trs_mc_xva,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.trs import TotalReturnSwap
from pricebook.discount_curve import DiscountCurve
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.survival_curve import SurvivalCurve
from pricebook.pricing_context import PricingContext


def trs_simm_im(
    trs: TotalReturnSwap,
    curve: DiscountCurve,
    projection_curve: DiscountCurve | None = None,
) -> float:
    """Compute SIMM Initial Margin for a TRS.

    Wires trs_simm_sensitivities() → SIMMCalculator for proper
    ISDA SIMM v2.6 margin instead of the flat 5% proxy.
    """
    from pricebook.simm import SIMMCalculator, SIMMSensitivity
    from pricebook.regulatory.trs_capital import trs_simm_sensitivities

    sens = trs_simm_sensitivities(trs, curve, projection_curve)

    simm_inputs = []
    for d in sens.delta_sensitivities:
        simm_inputs.append(SIMMSensitivity(
            risk_class=d["risk_class"], bucket=d["bucket"],
            tenor=d["tenor"], delta=d["delta"],
        ))
    for v in sens.vega_sensitivities:
        simm_inputs.append(SIMMSensitivity(
            risk_class=v["risk_class"], bucket=v["bucket"],
            tenor=v["tenor"], vega=v.get("vega", 0.0),
        ))

    if not simm_inputs:
        return trs.notional * 0.05  # fallback

    return SIMMCalculator().compute(simm_inputs).total_margin


def trs_mva(
    trs: TotalReturnSwap,
    curve: DiscountCurve | None = None,
    simm_im: float | None = None,
    funding_spread: float = 0.002,
) -> float:
    """MVA: cost of funding initial margin over the TRS term.

    MVA = IM × funding_spread × T.

    IM resolution order:
    1. Explicit simm_im if provided.
    2. SIMM calculation via trs_simm_im() if curve provided.
    3. Fallback: notional × 5% proxy.
    """
    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    if simm_im is None:
        if curve is not None:
            simm_im = trs_simm_im(trs, curve)
        else:
            simm_im = trs.notional * 0.05  # proxy fallback
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


# ---------------------------------------------------------------------------
# Monte Carlo XVA (wires xva.py engine to TRS)
# ---------------------------------------------------------------------------

def trs_mc_xva(
    trs: TotalReturnSwap,
    ctx: PricingContext,
    cpty_survival: SurvivalCurve,
    own_survival: SurvivalCurve,
    cpty_recovery: float = 0.4,
    own_recovery: float = 0.4,
    funding_spread: float = 0.005,
    n_paths: int = 1000,
    n_steps: int = 12,
    rate_vol: float = 0.01,
    seed: int = 42,
):
    """Monte Carlo XVA for a TRS — full decomposition.

    Wires the generic xva.py MC engine to TRS pricing:
    1. simulate_exposures() diffuses rates, reprices TRS at each time step
    2. Computes EPE/ENE profiles
    3. total_xva_decomposition() → CVA, DVA, CFA, DFA, FVA, MVA, KVA

    Returns a TotalXVAResult from xva.py.
    """
    from pricebook.xva import (
        simulate_exposures, expected_positive_exposure,
        expected_negative_exposure, total_xva_decomposition,
    )

    T = year_fraction(trs.start, trs.end, DayCountConvention.ACT_365_FIXED)
    time_grid = [(i + 1) * T / n_steps for i in range(n_steps)]

    # Pricer compatible with simulate_exposures
    pricer = lambda c: trs.pv_ctx(c)

    # Simulate exposures
    pvs = simulate_exposures(pricer, ctx, time_grid, n_paths, rate_vol, seed)
    epe = expected_positive_exposure(pvs)
    ene = expected_negative_exposure(pvs)

    # IM profile for MVA (SIMM at each time step — use flat approximation)
    if ctx.discount_curve is not None:
        im_val = trs_simm_im(trs, ctx.discount_curve)
    else:
        im_val = trs.notional * 0.05
    im_profile = np.full(n_steps, im_val)

    return total_xva_decomposition(
        epe=epe, ene=ene, time_grid=time_grid,
        discount_curve=ctx.discount_curve,
        cpty_survival=cpty_survival, own_survival=own_survival,
        cpty_recovery=cpty_recovery, own_recovery=own_recovery,
        funding_spread=funding_spread,
        im_profile=im_profile,
    )
