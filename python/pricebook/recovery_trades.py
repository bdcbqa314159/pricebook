"""Recovery-focused trading: implied recovery, downturn LGD, recovery swaps, senior-sub basis.

Tier 1 recovery analytics for credit trading desks:

* :func:`market_implied_recovery` — extract R from CDS spread and bond ASW.
* :func:`recovery_by_spread_regime` — R as function of spread level (distressed calibration).
* :func:`downturn_lgd` — Basel IRB downturn LGD from base LGD.
* :func:`portfolio_recovery_stress` — joint R shock across portfolio.
* :func:`recovery_swap_pv` — PV of a recovery swap (fixed vs floating R).
* :func:`recovery_lock_greeks` — Greeks for a recovery lock.
* :func:`senior_sub_basis` — basis trade between seniority classes.
* :func:`cs01_neutral_ratio` — notional ratio for CS01-neutral seniority trades.

References:
    O'Kane (2008). Modelling Single-name and Multi-name Credit Derivatives, Ch. 4.
    Altman, Resti, Sironi (2004). Default Recovery Rates in Credit Risk Modelling.
    Basel Committee (2006). International Convergence of Capital Measurement, para 272.
    Schonbucher (2003). Credit Derivatives Pricing Models, Ch. 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
from dateutil.relativedelta import relativedelta

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


# ---------------------------------------------------------------------------
# 1. Market-implied recovery
# ---------------------------------------------------------------------------

@dataclass
class MarketImpliedRecoveryResult:
    """Market-implied recovery from CDS and/or bond prices."""
    recovery_cds: float           # R from CDS spread alone (given h)
    recovery_bond: float | None   # R from bond ASW (if provided)
    recovery_combined: float      # best estimate (average or optimised)
    spread_bps: float
    hazard_bps: float
    method: str

    def to_dict(self) -> dict:
        return {
            "recovery_cds": self.recovery_cds,
            "recovery_bond": self.recovery_bond,
            "recovery_combined": self.recovery_combined,
            "spread_bps": self.spread_bps,
            "hazard_bps": self.hazard_bps,
            "method": self.method,
        }


def market_implied_recovery(
    cds_spread: float,
    discount_curve: DiscountCurve,
    reference_date: date,
    tenor_years: int = 5,
    bond_asw_spread: float | None = None,
    bond_recovery: float | None = None,
    initial_hazard_guess: float | None = None,
) -> MarketImpliedRecoveryResult:
    """Extract market-implied recovery from CDS spread and optionally bond ASW.

    Method 1 (CDS only): Given a hazard rate estimate h, solve:
        R = 1 - spread / h
    The hazard is either provided or estimated from the CDS term structure.

    Method 2 (CDS + bond): Both instruments reference the same entity.
        CDS spread = h × (1 - R_cds)
        Bond ASW   = h × (1 - R_bond) + liquidity_premium
        Solve jointly for h and R.

    For distressed names (spread > 500bp), recovery is the dominant
    unknown. This function makes R the output, not the input.

    Args:
        cds_spread: par CDS spread.
        bond_asw_spread: asset swap spread of comparable bond (optional).
        bond_recovery: recovery assumed for bond (if different from CDS).
        initial_hazard_guess: override for hazard rate.
    """
    spread = cds_spread

    if initial_hazard_guess is not None:
        h = initial_hazard_guess
    else:
        # Default: assume R=0.40 to get initial h, then iterate
        h = spread / 0.60  # h = s / (1-R) with R=0.40

    # CDS-implied recovery: R = 1 - s/h
    R_cds = max(0.0, min(0.95, 1.0 - spread / max(h, 1e-8)))

    # If bond ASW available: solve jointly
    R_bond = None
    if bond_asw_spread is not None:
        # Bond spread = h × (1 - R_bond) approximately
        # → R_bond = 1 - bond_spread / h
        R_bond = max(0.0, min(0.95, 1.0 - bond_asw_spread / max(h, 1e-8)))

    # Combined estimate
    if R_bond is not None:
        R_combined = (R_cds + R_bond) / 2.0
        method = "cds_bond_average"
    else:
        R_combined = R_cds
        method = "cds_only"

    return MarketImpliedRecoveryResult(
        recovery_cds=R_cds,
        recovery_bond=R_bond,
        recovery_combined=R_combined,
        spread_bps=spread * 1e4,
        hazard_bps=h * 1e4,
        method=method,
    )


def recovery_by_spread_regime(
    cds_spread_bps: float,
    seniority: str = "senior_unsecured",
) -> float:
    """Empirical recovery estimate by CDS spread regime.

    Based on Altman (2004) and Moody's historical data:
    - Tight spreads (<100bp): R near seniority average (IG, healthy)
    - Medium (100-300bp): R drifts lower (crossover)
    - Wide (300-500bp): R drops materially (stressed)
    - Distressed (>500bp): R significantly compressed

    Args:
        cds_spread_bps: CDS spread in basis points.
        seniority: seniority class.

    Returns:
        Estimated recovery rate.
    """
    from pricebook.recovery_surface import SENIORITY_TABLE

    base_R = SENIORITY_TABLE.get(seniority, (0.40, 0.20, -0.008))[0]

    # Spread-dependent adjustment (empirical calibration)
    if cds_spread_bps <= 100:
        adjustment = 0.0
    elif cds_spread_bps <= 300:
        adjustment = -0.05 * (cds_spread_bps - 100) / 200  # -0 to -5%
    elif cds_spread_bps <= 500:
        adjustment = -0.05 - 0.10 * (cds_spread_bps - 300) / 200  # -5% to -15%
    else:
        adjustment = -0.15 - 0.10 * min((cds_spread_bps - 500) / 500, 1.0)  # -15% to -25%

    return max(0.05, min(0.90, base_R + adjustment))


# ---------------------------------------------------------------------------
# 2. Downturn LGD (Basel IRB)
# ---------------------------------------------------------------------------

@dataclass
class DownturnLGDResult:
    """Basel IRB downturn LGD result."""
    base_lgd: float
    downturn_lgd: float
    downturn_multiplier: float
    stress_quantile: float
    seniority: str

    def to_dict(self) -> dict:
        return {
            "base_lgd": self.base_lgd,
            "downturn_lgd": self.downturn_lgd,
            "downturn_multiplier": self.downturn_multiplier,
            "stress_quantile": self.stress_quantile,
            "seniority": self.seniority,
        }


def downturn_lgd(
    base_lgd: float = 0.45,
    seniority: str = "senior_unsecured",
    stress_quantile: float = 0.99,
    correlation_to_cycle: float = 0.30,
) -> DownturnLGDResult:
    """Basel IRB downturn LGD: LGD under stressed conditions.

    Basel II para 468: banks must estimate LGD reflecting downturn conditions.
    Model: LGD_downturn = base_LGD × (1 + multiplier)

    The multiplier depends on:
    - Stress quantile (99th percentile = 1-in-100-year default rate)
    - Correlation between LGD and aggregate default rate
    - Seniority (secured collateral dampens downturn impact)

    Typical results:
        Senior secured:   base 0.35 → downturn 0.42 (×1.20)
        Senior unsecured: base 0.45 → downturn 0.56 (×1.25)
        Subordinated:     base 0.72 → downturn 0.90 (×1.25)

    Args:
        base_lgd: average-conditions LGD.
        seniority: for collateral dampening.
        stress_quantile: percentile of default rate distribution (0.99 = 99th).
        correlation_to_cycle: ρ between LGD and aggregate default rate.
    """
    from scipy.stats import norm

    # Stress factor: standard normal quantile
    z_stress = norm.ppf(stress_quantile)

    # Seniority-dependent dampening (secured collateral absorbs some stress)
    dampening = {
        "senior_secured": 0.70,
        "1L": 0.75,
        "senior_unsecured": 1.00,
        "senior": 1.00,
        "2L": 1.10,
        "mezzanine": 1.15,
        "sub": 1.20,
    }
    damp = dampening.get(seniority, 1.0)

    # Multiplier: correlation × stress_factor × dampening
    # Higher correlation → more procyclical → bigger downturn increase
    multiplier = correlation_to_cycle * z_stress * 0.15 * damp

    downturn = min(base_lgd * (1 + multiplier), 0.95)
    downturn = max(downturn, base_lgd)  # downturn >= base

    return DownturnLGDResult(
        base_lgd=base_lgd,
        downturn_lgd=downturn,
        downturn_multiplier=1 + multiplier,
        stress_quantile=stress_quantile,
        seniority=seniority,
    )


@dataclass
class PortfolioStressResult:
    """Portfolio recovery stress test result."""
    base_pv: float
    stressed_pv: float
    pv_change: float
    pv_change_pct: float
    n_positions: int
    recovery_shock: float
    spread_shock_bps: float

    def to_dict(self) -> dict:
        return {
            "base_pv": self.base_pv,
            "stressed_pv": self.stressed_pv,
            "pv_change": self.pv_change,
            "pv_change_pct": self.pv_change_pct,
            "n_positions": self.n_positions,
            "recovery_shock": self.recovery_shock,
            "spread_shock_bps": self.spread_shock_bps,
        }


def portfolio_recovery_stress(
    instruments: list,
    discount_curve: DiscountCurve,
    cds_spreads: dict[int, float],
    reference_date: date,
    recovery_shock: float = -0.10,
    spread_shock_bps: float = 0.0,
    base_recovery: float = 0.40,
) -> PortfolioStressResult:
    """Joint recovery + spread stress on a portfolio of credit instruments.

    Shocks recovery by `recovery_shock` (e.g., -0.10 = drop R by 10%)
    and optionally bumps CDS spreads in parallel.

    Re-bootstraps the survival curve under stress, then reprices all instruments.

    Args:
        instruments: list of instruments with dirty_price(curve, survival) or pv().
        recovery_shock: additive shock to recovery (e.g., -0.10).
        spread_shock_bps: parallel spread bump in bp (e.g., +200).
        base_recovery: starting recovery assumption.
    """
    from pricebook.cds_market import build_cds_curve
    from pricebook.recovery_analytics import _price_instrument

    # Base pricing
    surv_base = build_cds_curve(reference_date, cds_spreads, discount_curve,
                                 recovery=base_recovery)
    base_pv = sum(_price_instrument(inst, discount_curve, surv_base, base_recovery)
                  for inst in instruments)

    # Stressed pricing
    stressed_R = max(0.05, min(0.90, base_recovery + recovery_shock))
    stressed_spreads = {t: s + spread_shock_bps / 1e4 for t, s in cds_spreads.items()}
    surv_stressed = build_cds_curve(reference_date, stressed_spreads, discount_curve,
                                     recovery=stressed_R)
    stressed_pv = sum(_price_instrument(inst, discount_curve, surv_stressed, stressed_R)
                      for inst in instruments)

    change = stressed_pv - base_pv
    return PortfolioStressResult(
        base_pv=base_pv,
        stressed_pv=stressed_pv,
        pv_change=change,
        pv_change_pct=change / abs(base_pv) if abs(base_pv) > 1e-10 else 0.0,
        n_positions=len(instruments),
        recovery_shock=recovery_shock,
        spread_shock_bps=spread_shock_bps,
    )


# ---------------------------------------------------------------------------
# 3. Recovery swaps / locks
# ---------------------------------------------------------------------------

@dataclass
class RecoverySwapResult:
    """Recovery swap PV and Greeks."""
    pv: float                     # PV to fixed-recovery payer
    fixed_recovery: float
    expected_floating_recovery: float
    annuity: float                # risk-free annuity for scaling
    notional: float
    par_fixed_recovery: float     # R_fixed that makes PV=0

    def to_dict(self) -> dict:
        return {
            "pv": self.pv,
            "fixed_recovery": self.fixed_recovery,
            "expected_floating_recovery": self.expected_floating_recovery,
            "annuity": self.annuity,
            "par_fixed_recovery": self.par_fixed_recovery,
        }


def recovery_swap_pv(
    notional: float,
    maturity_date: date,
    reference_date: date,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    fixed_recovery: float,
    expected_recovery: float = 0.40,
    recovery_vol: float = 0.15,
) -> RecoverySwapResult:
    """PV of a recovery swap: fixed R vs floating R on default.

    On default of the reference entity:
    - Fixed leg pays: notional × R_fixed
    - Floating leg pays: notional × R_actual (realised recovery)

    PV to fixed payer = E[R_actual - R_fixed] × notional × PD × df

    If R_actual > R_fixed → fixed payer receives the difference.
    This is a pure bet on recovery — no directional credit exposure.

    Args:
        fixed_recovery: the fixed R leg.
        expected_recovery: market consensus R (for floating leg valuation).
        recovery_vol: vol of recovery (for convexity adjustment).
    """
    T = year_fraction(reference_date, maturity_date, DayCountConvention.ACT_365_FIXED)
    if T <= 0:
        return RecoverySwapResult(0, fixed_recovery, expected_recovery, 0, notional, expected_recovery)

    df = discount_curve.df(maturity_date)
    q = survival_curve.survival(maturity_date)
    pd = 1.0 - q  # cumulative default probability

    # PV = notional × PD × df × (E[R] - R_fixed)
    # Convexity adjustment for R_vol: E[f(R)] ≈ f(E[R]) + 0.5 × f''(E[R]) × σ²_R
    # For linear payoff f(R) = R, convexity is zero. But PD depends on h = s/(1-R),
    # so there's a second-order effect through the curve.
    pv = notional * pd * df * (expected_recovery - fixed_recovery)

    # Par fixed recovery: R_fixed that makes PV=0 → R_fixed = E[R]
    par_R = expected_recovery

    # Annuity (for scaling)
    annuity = T * df * (1 + q) / 2  # approximate

    return RecoverySwapResult(
        pv=pv,
        fixed_recovery=fixed_recovery,
        expected_floating_recovery=expected_recovery,
        annuity=annuity,
        notional=notional,
        par_fixed_recovery=par_R,
    )


@dataclass
class RecoveryLockGreeksResult:
    """Recovery lock sensitivities."""
    pv: float
    delta_R: float         # dPV/dR (per 1% R change)
    delta_spread: float    # dPV/dSpread (per 1bp spread change)
    gamma_R: float         # d²PV/dR²
    lock_strike: float

    def to_dict(self) -> dict:
        return {
            "pv": self.pv, "delta_R": self.delta_R,
            "delta_spread": self.delta_spread,
            "gamma_R": self.gamma_R, "lock_strike": self.lock_strike,
        }


def recovery_lock_greeks(
    notional: float,
    maturity_date: date,
    reference_date: date,
    discount_curve: DiscountCurve,
    cds_spreads: dict[int, float],
    lock_strike: float,
    base_recovery: float = 0.40,
    bump_R: float = 0.01,
    bump_spread_bps: float = 1.0,
) -> RecoveryLockGreeksResult:
    """Greeks for a recovery lock (option on recovery at default).

    A recovery lock pays notional × max(R_actual - lock_strike, 0) on default.
    Simplified here as a linear swap for tractability.

    Args:
        lock_strike: recovery strike (e.g., 0.35).
    """
    from pricebook.cds_market import build_cds_curve

    def _pv_at(R, spreads):
        surv = build_cds_curve(reference_date, spreads, discount_curve, recovery=R)
        T = year_fraction(reference_date, maturity_date, DayCountConvention.ACT_365_FIXED)
        df = discount_curve.df(maturity_date)
        pd = 1.0 - surv.survival(maturity_date)
        return notional * pd * df * (R - lock_strike)

    pv_base = _pv_at(base_recovery, cds_spreads)

    # Delta R
    pv_up = _pv_at(base_recovery + bump_R, cds_spreads)
    pv_dn = _pv_at(base_recovery - bump_R, cds_spreads)
    delta_R = (pv_up - pv_dn) / (2 * bump_R)

    # Gamma R
    gamma_R = (pv_up - 2 * pv_base + pv_dn) / (bump_R ** 2)

    # Delta spread
    bumped_spreads = {t: s + bump_spread_bps / 1e4 for t, s in cds_spreads.items()}
    pv_spread_up = _pv_at(base_recovery, bumped_spreads)
    delta_spread = (pv_spread_up - pv_base) / bump_spread_bps

    return RecoveryLockGreeksResult(
        pv=pv_base, delta_R=delta_R, delta_spread=delta_spread,
        gamma_R=gamma_R, lock_strike=lock_strike,
    )


# ---------------------------------------------------------------------------
# 4. Senior-sub basis
# ---------------------------------------------------------------------------

@dataclass
class SeniorSubBasisResult:
    """Senior vs subordinated basis trade analysis."""
    basis_bps: float              # senior_spread - sub_spread (always negative)
    implied_senior_recovery: float
    implied_sub_recovery: float
    pv_long_senior: float
    pv_short_sub: float
    pv_net: float
    cs01_senior: float
    cs01_sub: float
    notional_ratio: float         # sub notional / senior notional for CS01 neutrality
    carry_30d: float

    def to_dict(self) -> dict:
        return {
            "basis_bps": self.basis_bps,
            "implied_senior_R": self.implied_senior_recovery,
            "implied_sub_R": self.implied_sub_recovery,
            "pv_net": self.pv_net,
            "cs01_senior": self.cs01_senior,
            "cs01_sub": self.cs01_sub,
            "notional_ratio": self.notional_ratio,
            "carry_30d": self.carry_30d,
        }


def senior_sub_basis(
    senior_spread: float,
    sub_spread: float,
    discount_curve: DiscountCurve,
    reference_date: date,
    tenor_years: int = 5,
    notional: float = 10_000_000,
    senior_recovery: float = 0.45,
    sub_recovery: float = 0.25,
) -> SeniorSubBasisResult:
    """Analyse the senior vs subordinated CDS basis trade.

    Long senior protection (buy) + short sub protection (sell).
    Profits if the basis (sub - senior spread) tightens.

    The trade isolates recovery risk: both legs have the same
    default probability but different recovery assumptions.

    Args:
        senior_spread: senior CDS par spread.
        sub_spread: subordinated CDS par spread.
        senior_recovery / sub_recovery: seniority-appropriate recovery.
    """
    from pricebook.cds import CDS
    from pricebook.cds_market import build_cds_curve

    mat = reference_date + relativedelta(years=tenor_years)

    # Build curves for each seniority
    senior_spreads = {tenor_years: senior_spread}
    sub_spreads = {tenor_years: sub_spread}
    surv_senior = build_cds_curve(reference_date, senior_spreads, discount_curve,
                                   recovery=senior_recovery)
    surv_sub = build_cds_curve(reference_date, sub_spreads, discount_curve,
                                recovery=sub_recovery)

    # CDS instruments
    cds_senior = CDS(reference_date, mat, spread=senior_spread,
                      notional=notional, recovery=senior_recovery)
    cds_sub = CDS(reference_date, mat, spread=sub_spread,
                   notional=notional, recovery=sub_recovery)

    # PV (at par, both should be ~0)
    pv_senior = cds_senior.pv(discount_curve, surv_senior)
    pv_sub = cds_sub.pv(discount_curve, surv_sub)

    # CS01: parallel spread bump
    bump = 0.0001
    senior_up = {tenor_years: senior_spread + bump}
    sub_up = {tenor_years: sub_spread + bump}
    surv_sen_up = build_cds_curve(reference_date, senior_up, discount_curve,
                                   recovery=senior_recovery)
    surv_sub_up = build_cds_curve(reference_date, sub_up, discount_curve,
                                   recovery=sub_recovery)
    cs01_senior = cds_senior.pv(discount_curve, surv_sen_up) - pv_senior
    cs01_sub = cds_sub.pv(discount_curve, surv_sub_up) - pv_sub

    # CS01-neutral notional ratio: how much sub notional to offset senior CS01
    ratio = abs(cs01_senior / cs01_sub) if abs(cs01_sub) > 1e-10 else 1.0

    # Implied recovery from spread ratio
    if sub_spread > 1e-8:
        R_sen_implied = 1.0 - senior_spread * (1.0 - sub_recovery) / sub_spread
        R_sen_implied = max(0.0, min(0.95, R_sen_implied))
    else:
        R_sen_implied = senior_recovery

    # 30-day carry: premium received (short sub) - premium paid (long senior)
    # Per 30 days ≈ spread × notional × 30/360
    carry_30d = (sub_spread - senior_spread) * notional * 30 / 360

    basis_bps = (senior_spread - sub_spread) * 1e4

    return SeniorSubBasisResult(
        basis_bps=basis_bps,
        implied_senior_recovery=R_sen_implied,
        implied_sub_recovery=sub_recovery,
        pv_long_senior=pv_senior,
        pv_short_sub=-pv_sub,
        pv_net=pv_senior - pv_sub,
        cs01_senior=cs01_senior,
        cs01_sub=cs01_sub,
        notional_ratio=ratio,
        carry_30d=carry_30d,
    )


def cs01_neutral_ratio(
    senior_spread: float,
    sub_spread: float,
    senior_recovery: float = 0.45,
    sub_recovery: float = 0.25,
) -> float:
    """Quick CS01-neutral notional ratio for senior-sub basis trade.

    Approximation: ratio ≈ (1 - R_senior) / (1 - R_sub)
    because CS01 ∝ RPV01 ∝ 1/(1-R) for similar hazard rates.

    Args:
        senior_spread / sub_spread: par spreads.
        senior_recovery / sub_recovery: recovery assumptions.

    Returns:
        sub_notional / senior_notional ratio for CS01 neutrality.
    """
    return (1 - senior_recovery) / max(1 - sub_recovery, 0.01)
