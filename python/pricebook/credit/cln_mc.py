"""Monte Carlo pricing functions for CreditLinkedNote.

Extracted from CreditLinkedNote class methods to standalone functions.
Each function takes the CLN instance as the first argument.

Functions:
    cln_stochastic_recovery — MC with correlated (default, recovery) draws.
    cln_stochastic_intensity — MC under stochastic hazard rate model.
    cln_stochastic_intensity_from_curve — auto-calibrate model, then MC.
    cln_bilateral_mc — bilateral default (ref entity + issuer) via copula.
    cln_rec_vol_01 — recovery vol sensitivity (bump-and-reprice).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


def cln_stochastic_recovery(
    cln,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery_spec=None,
    n_sims: int = 50_000,
    seed: int = 42,
):
    """Price CLN with stochastic recovery correlated to default.

    Uses MC: for each path, draw correlated (default, recovery) pair
    per period. The wrong-way premium = difference vs fixed-recovery price.

    Args:
        cln: CreditLinkedNote instance.
        discount_curve: risk-free discount curve.
        survival_curve: credit survival curve.
        recovery_spec: RecoverySpec (from recovery_pricing module).
            If None, uses fixed recovery at cln.recovery.
    """
    from pricebook.cln import CLNResult
    from pricebook.recovery_pricing import RecoverySpec

    if recovery_spec is None:
        recovery_spec = RecoverySpec.fixed(cln.recovery)

    rng = np.random.default_rng(seed)
    pv_paths = np.zeros(n_sims)

    for i in range(1, len(cln.schedule)):
        t_start = cln.schedule[i - 1]
        t_end = cln.schedule[i]
        yf = year_fraction(t_start, t_end, cln.day_count)
        df = discount_curve.df(t_end)
        surv = survival_curve.survival(t_end)
        surv_prev = survival_curve.survival(t_start)
        default_prob = surv_prev - surv

        # Coupon conditional on survival
        if cln.floating:
            fwd = discount_curve.forward_rate(t_start, t_end)
            coupon = cln.notional * (fwd + cln.coupon_rate) * yf
        else:
            coupon = cln.notional * cln.coupon_rate * yf
        pv_paths += coupon * df * surv

        # Default in this period: draw default indicators and recoveries
        Z_D = rng.standard_normal(n_sims)
        threshold = norm.ppf(max(default_prob / max(surv_prev, 1e-10), 1e-15))
        period_defaults = Z_D < threshold

        # Correlated recovery
        R_samples = recovery_spec.sample(n_sims, systematic_factor=Z_D, seed=seed + i)

        # Recovery on default (per path)
        recovery_pv = R_samples * cln.notional * period_defaults * df
        pv_paths += recovery_pv

        # Leveraged loss
        if cln.leverage > 1.0:
            extra = (cln.leverage - 1.0) * (1.0 - R_samples) * cln.notional * period_defaults * df
            pv_paths -= extra

    # Principal at maturity conditional on survival
    pv_paths += cln.notional * discount_curve.df(cln.end) * survival_curve.survival(cln.end)

    total_pv = float(pv_paths.mean())

    # Decomposition (approximate)
    fixed_price = cln.dirty_price(discount_curve, survival_curve)
    credit_spread_approx = (fixed_price - total_pv) / max(
        cln.notional * cln._risky_annuity(discount_curve, survival_curve), 1e-10
    )

    return CLNResult(
        price=total_pv, coupon_pv=0.0, principal_pv=0.0,
        recovery_pv=0.0, credit_spread=credit_spread_approx,
        par_coupon=0.0,
    )


def cln_stochastic_intensity(
    cln,
    discount_curve: DiscountCurve,
    intensity_model,
    x0: float | None = None,
    n_paths: int = 50_000,
    n_steps: int = 200,
    seed: int = 42,
):
    """Price CLN via Monte Carlo under stochastic intensity.

    For each path:
      1. Simulate intensity lambda(t) from the model (Euler or exact scheme).
      2. Compute path survival Q_path(t) = exp(-int_0^t lambda ds) via left-endpoint
         rectangular rule (Glasserman 2003, S6.3).
      3. Accumulate coupon x df x Q and recovery x df x DeltaQ per period.
      4. Average across paths.

    Supports CIRPlusPlus, HWHazardRate, or CIRIntensity models.
    """
    from pricebook.cln import CLNResult

    T = year_fraction(cln.start, cln.end, DayCountConvention.ACT_365_FIXED)
    if x0 is None:
        # Use theta or first hazard as starting point
        if hasattr(intensity_model, 'theta'):
            x0 = intensity_model.theta
        else:
            x0 = 0.02

    # Simulate intensity paths
    result = intensity_model.simulate(x0, T, n_steps, n_paths, seed)
    lambda_paths = result.lambda_paths  # (n_paths, n_steps+1)
    dt = T / n_steps

    # Cumulative survival per path: Q(t) = exp(-int_0^t lambda ds)
    cum_integral = np.cumsum(lambda_paths[:, :-1] * dt, axis=1)
    # Prepend zero for t=0
    cum_integral = np.concatenate([np.zeros((n_paths, 1)), cum_integral], axis=1)
    path_survival = np.exp(-cum_integral)  # (n_paths, n_steps+1)

    # Map coupon periods to simulation steps
    coupon_pv_total = 0.0
    recovery_pv_total = 0.0

    for i in range(1, len(cln.schedule)):
        t_start = cln.schedule[i - 1]
        t_end = cln.schedule[i]
        yf_coupon = year_fraction(t_start, t_end, cln.day_count)
        t_frac = year_fraction(cln.start, t_end, DayCountConvention.ACT_365_FIXED)
        t_frac_prev = year_fraction(cln.start, t_start, DayCountConvention.ACT_365_FIXED)

        step_end = min(int(t_frac / dt), n_steps)
        step_prev = min(int(t_frac_prev / dt), n_steps)

        df = discount_curve.df(t_end)
        surv_end = path_survival[:, step_end]        # (n_paths,)
        surv_prev = path_survival[:, step_prev]      # (n_paths,)
        default_prob = np.maximum(surv_prev - surv_end, 0.0)

        # Coupon
        if cln.floating:
            fwd = discount_curve.forward_rate(t_start, t_end)
            coupon = cln.notional * (fwd + cln.coupon_rate) * yf_coupon
        else:
            coupon = cln.notional * cln.coupon_rate * yf_coupon

        coupon_pv_total += float(np.mean(coupon * df * surv_end))

        # Recovery on default
        recovery_pv_total += float(np.mean(
            cln.recovery * cln.notional * default_prob * df
        ))

        # Leveraged loss
        if cln.leverage > 1.0:
            extra_loss = (cln.leverage - 1.0) * (1.0 - cln.recovery) * cln.notional
            recovery_pv_total -= float(np.mean(extra_loss * default_prob * df))

    # Principal at maturity
    final_surv = float(np.mean(path_survival[:, -1]))
    principal_pv = cln.notional * discount_curve.df(cln.end) * final_surv

    total_pv = coupon_pv_total + principal_pv + recovery_pv_total

    return CLNResult(
        price=total_pv,
        coupon_pv=coupon_pv_total,
        principal_pv=principal_pv,
        recovery_pv=recovery_pv_total,
        credit_spread=0.0,
        par_coupon=0.0,
    )


def cln_stochastic_intensity_from_curve(
    cln,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    model_type: str = "cir++",
    xi: float = 0.1,
    kappa: float = 1.0,
    n_paths: int = 50_000,
    n_steps: int = 200,
    seed: int = 42,
    **model_kwargs,
):
    """Auto-calibrate stochastic model from survival curve, then price.

    Combines Phase 2 (curve -> model) + Phase 4 (model -> MC price).
    """
    from pricebook.hazard_rate_models import CIRPlusPlus, HWHazardRate

    if model_type == "cir++":
        model = CIRPlusPlus.from_survival_curve(
            survival_curve, kappa=kappa, xi=xi, **model_kwargs)
        x0 = model.theta
    elif model_type == "hw":
        model = HWHazardRate.from_survival_curve(
            survival_curve, a=kappa, sigma=xi)
        hazards = survival_curve.pillar_hazards()
        x0 = hazards[0][1] if hazards else 0.02
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return cln_stochastic_intensity(
        cln, discount_curve, model, x0, n_paths, n_steps, seed)


def cln_bilateral_mc(
    cln,
    discount_curve: DiscountCurve,
    ref_survival: SurvivalCurve,
    issuer_survival: SurvivalCurve,
    issuer_recovery: float = 0.4,
    correlation: float = 0.3,
    n_paths: int = 50_000,
    seed: int = 42,
):
    """Price bilateral CLN with dual default risk (ref entity + issuer).

    On reference default -> investor receives recovery x notional.
    On issuer default (no ref default) -> investor receives issuer_recovery x notional.
    On both -> investor receives min(recovery, issuer_recovery) x notional.
    On neither -> full coupon + principal.

    Simulates two correlated uniform defaults via Gaussian copula.
    Default check: name defaults in period i when PD(t_i) > U, i.e. Q(t_i) < 1 - U.

    Reference: Li (2000), "On Default Correlation", Journal of Fixed Income.
    """
    from pricebook.cln import CLNResult

    if not -1.0 <= correlation <= 1.0:
        raise ValueError(f"correlation must be in [-1, 1], got {correlation}")
    if cln.notional <= 0:
        raise ValueError(f"notional must be positive, got {cln.notional}")

    rng = np.random.default_rng(seed)
    T = year_fraction(cln.start, cln.end, DayCountConvention.ACT_365_FIXED)

    # Correlated normals via Cholesky (Li 2000)
    Z = rng.standard_normal((n_paths, 2))
    Z[:, 1] = correlation * Z[:, 0] + math.sqrt(1 - correlation**2) * Z[:, 1]

    # Convert to uniform via Phi
    U_ref = norm.cdf(Z[:, 0])
    U_issuer = norm.cdf(Z[:, 1])

    total_pv = np.zeros(n_paths)

    for path in range(n_paths):
        # Default times from inverse survival
        ref_defaulted = False
        issuer_defaulted = False
        pv = 0.0

        for i in range(1, len(cln.schedule)):
            t_start = cln.schedule[i - 1]
            t_end = cln.schedule[i]
            yf = year_fraction(t_start, t_end, cln.day_count)
            df = discount_curve.df(t_end)
            surv_ref = ref_survival.survival(t_end)
            surv_iss = issuer_survival.survival(t_end)

            # Check defaults this period
            if not ref_defaulted and surv_ref < 1 - U_ref[path]:
                ref_defaulted = True
            if not issuer_defaulted and surv_iss < 1 - U_issuer[path]:
                issuer_defaulted = True

            if ref_defaulted and issuer_defaulted:
                # Both default: recovery on funded, leverage loss on credit exposure
                ref_payout = cln.recovery * cln.notional
                lev_loss = (cln.leverage - 1.0) * (1 - cln.recovery) * cln.notional
                pv += max(min(ref_payout - lev_loss, issuer_recovery * cln.notional), 0) * df
                break
            elif ref_defaulted:
                # Reference defaults: recovery minus leverage loss
                ref_payout = cln.recovery * cln.notional
                lev_loss = (cln.leverage - 1.0) * (1 - cln.recovery) * cln.notional
                pv += max(ref_payout - lev_loss, 0) * df
                break
            elif issuer_defaulted:
                pv += issuer_recovery * cln.notional * df
                break
            else:
                # Coupon (conditional on both surviving)
                coupon = cln.notional * cln.coupon_rate * yf
                pv += coupon * df

        if not ref_defaulted and not issuer_defaulted:
            pv += cln.notional * discount_curve.df(cln.end)

        total_pv[path] = pv

    mean_pv = float(np.mean(total_pv))

    return CLNResult(
        price=mean_pv,
        coupon_pv=0.0,
        principal_pv=0.0,
        recovery_pv=0.0,
        credit_spread=0.0,
        par_coupon=0.0,
    )


def cln_rec_vol_01(
    cln,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery_spec=None,
    n_sims: int = 50_000,
    seed: int = 42,
) -> float:
    """Recovery vol sensitivity: PV change for 1% increase in recovery std.

    Higher recovery vol -> more wrong-way risk -> lower CLN PV.
    """
    from pricebook.recovery_pricing import RecoverySpec

    if recovery_spec is None:
        recovery_spec = RecoverySpec(mean=cln.recovery, std=0.15,
                                     correlation_to_default=-0.3)

    base = cln_stochastic_recovery(
        cln, discount_curve, survival_curve, recovery_spec, n_sims, seed,
    ).price

    bumped_spec = RecoverySpec(
        mean=recovery_spec.mean,
        std=recovery_spec.std + 0.01,
        distribution=recovery_spec.distribution,
        correlation_to_default=recovery_spec.correlation_to_default,
    )
    bumped = cln_stochastic_recovery(
        cln, discount_curve, survival_curve, bumped_spec, n_sims, seed,
    ).price

    return bumped - base
