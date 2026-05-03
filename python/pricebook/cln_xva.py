"""CLN XVA: SIMM IM, MVA, KVA, analytic CVA, wrong-way cost, MC XVA.

Wires the generic XVA framework to CLN-specific credit economics.

    from pricebook.cln_xva import (
        cln_simm_im, cln_mva, cln_kva,
        cln_analytic_cva, cln_wrong_way_cost, cln_mc_xva,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.cln import CreditLinkedNote
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.pricing_context import PricingContext
from pricebook.day_count import DayCountConvention, year_fraction


# ---------------------------------------------------------------------------
# SIMM IM
# ---------------------------------------------------------------------------

def cln_simm_im(
    cln: CreditLinkedNote,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
) -> float:
    """SIMM Initial Margin for a CLN.

    Maps CLN sensitivities to SIMM risk classes:
    - CSR: CS01 (credit spread delta)
    - GIRR: DV01 (rate delta)
    """
    from pricebook.simm import SIMMCalculator, SIMMSensitivity

    greeks = cln.greeks(discount_curve, survival_curve)

    simm_inputs = [
        SIMMSensitivity(
            risk_class="CSR", bucket="IG_corporate",
            tenor="5Y", delta=greeks["cs01"],
        ),
        SIMMSensitivity(
            risk_class="GIRR", bucket="USD",
            tenor="5Y", delta=greeks["dv01"],
        ),
    ]

    return SIMMCalculator().compute(simm_inputs).total_margin


# ---------------------------------------------------------------------------
# MVA
# ---------------------------------------------------------------------------

def cln_mva(
    cln: CreditLinkedNote,
    discount_curve: DiscountCurve | None = None,
    survival_curve: SurvivalCurve | None = None,
    simm_im: float | None = None,
    funding_spread: float = 0.002,
) -> float:
    """MVA = IM × funding_spread × T."""
    T = year_fraction(cln.start, cln.end, DayCountConvention.ACT_365_FIXED)
    if simm_im is None:
        if discount_curve is not None and survival_curve is not None:
            simm_im = cln_simm_im(cln, discount_curve, survival_curve)
        else:
            simm_im = cln.notional * 0.05
    return simm_im * funding_spread * T


# ---------------------------------------------------------------------------
# KVA
# ---------------------------------------------------------------------------

def cln_kva(
    cln: CreditLinkedNote,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    counterparty_rw: float = 1.0,
    hurdle_rate: float = 0.10,
) -> float:
    """KVA from SA-CCR EAD. SF=0.005 for credit.

    KVA = EAD × RW × 8% × hurdle × T.
    """
    T = year_fraction(cln.start, cln.end, DayCountConvention.ACT_365_FIXED)
    pv = cln.dirty_price(discount_curve, survival_curve)
    mtm = max(pv, 0)

    sf = 0.005  # credit supervisory factor
    ead = 1.4 * (mtm + cln.notional * sf * math.sqrt(min(T, 1.0)))
    capital = ead * counterparty_rw * 0.08
    return capital * hurdle_rate * T


# ---------------------------------------------------------------------------
# Analytic CVA
# ---------------------------------------------------------------------------

def cln_analytic_cva(
    cln: CreditLinkedNote,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    cpty_hazard: float = 0.02,
    cpty_recovery: float = 0.4,
) -> float:
    """Analytic CVA for a CLN position.

    EPE ≈ |CS01| × expected_spread_vol × √T.
    CVA = (1-R_cpty) × h_cpty × EPE × T × df_avg.

    CLN exposure is driven by spread moves: when spreads widen,
    CLN loses value → negative MTM → counterparty owes us less.
    The EPE comes from spread tightening (CLN gains value).
    """
    T = year_fraction(cln.start, cln.end, DayCountConvention.ACT_365_FIXED)
    greeks = cln.greeks(discount_curve, survival_curve)

    # EPE from spread moves: |CS01| × spread_vol × √T
    # CS01 is per 1bp (0.0001), spread_vol is in absolute hazard units
    # Scale: EPE($) = |CS01($/bp)| × (spread_vol / 0.0001) × √T
    spread_vol = 0.30  # 30% of hazard as typical vol
    epe = abs(greeks["cs01"]) * spread_vol * math.sqrt(T) * 10_000

    lgd = 1 - cpty_recovery
    df_mid = discount_curve.df(cln.start + (cln.end - cln.start) // 2)
    surv = math.exp(-cpty_hazard * T)

    return lgd * cpty_hazard * epe * T * df_mid * (1 + surv) / 2


# ---------------------------------------------------------------------------
# Wrong-way cost (correlated recovery)
# ---------------------------------------------------------------------------

def cln_wrong_way_cost(
    cln: CreditLinkedNote,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery_spec=None,
    n_sims: int = 50_000,
    seed: int = 42,
) -> float:
    """Wrong-way premium from correlated recovery.

    Compares stochastic recovery price vs deterministic.
    Positive result = wrong-way cost (stochastic < deterministic).
    """
    base = cln.dirty_price(discount_curve, survival_curve)

    if recovery_spec is None:
        from pricebook.recovery_pricing import RecoverySpec
        recovery_spec = RecoverySpec(
            mean=cln.recovery, std=0.15,
            distribution="beta", correlation_to_default=-0.3,
        )

    stoch_result = cln.price_stochastic_recovery(
        discount_curve, survival_curve, recovery_spec,
        n_sims=n_sims, seed=seed,
    )
    stoch = stoch_result.price if hasattr(stoch_result, 'price') else float(stoch_result)

    return base - stoch  # positive if stochastic < deterministic


# ---------------------------------------------------------------------------
# MC XVA (wires xva.py)
# ---------------------------------------------------------------------------

def cln_mc_xva(
    cln: CreditLinkedNote,
    ctx: PricingContext,
    survival_curve: SurvivalCurve,
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
    """MC XVA for a CLN — wires xva.simulate_exposures.

    Returns TotalXVAResult from xva.py.
    """
    from pricebook.xva import (
        simulate_exposures, expected_positive_exposure,
        expected_negative_exposure, total_xva_decomposition,
    )

    T = year_fraction(cln.start, cln.end, DayCountConvention.ACT_365_FIXED)
    time_grid = [(i + 1) * T / n_steps for i in range(n_steps)]

    # Pricer: CLN needs survival curve, which pv_ctx resolves from context
    # For MC, we diffuse rates only (credit spread diffusion would need survival curve bumps)
    pricer = lambda c: cln.pv_ctx(c)

    pvs = simulate_exposures(pricer, ctx, time_grid, n_paths, rate_vol, seed)
    epe = expected_positive_exposure(pvs)
    ene = expected_negative_exposure(pvs)

    im_val = cln_simm_im(cln, ctx.discount_curve, survival_curve)
    im_profile = np.full(n_steps, im_val)

    return total_xva_decomposition(
        epe=epe, ene=ene, time_grid=time_grid,
        discount_curve=ctx.discount_curve,
        cpty_survival=cpty_survival, own_survival=own_survival,
        cpty_recovery=cpty_recovery, own_recovery=own_recovery,
        funding_spread=funding_spread,
        im_profile=im_profile,
    )
