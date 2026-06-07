"""Bermudan CDS swaption with stochastic hazard rate (CIR intensity).

The deterministic version in ``credit.bermudan_cds_swaption`` uses a fixed
survival curve.  This module adds spread-volatility dynamics by driving the
default intensity with a Cox-Ingersoll-Ross (CIR) process:

    dλ = κ(θ - λ) dt + σ√λ dW

The Bermudan is priced via Longstaff-Schwartz Monte Carlo (LSM) with CDS
continuation values computed analytically from the current CIR state.

    from pricebook.credit.stochastic_bermudan_cds import (
        stochastic_bermudan_cds_swaption,
        StochasticBermudanCDSResult,
    )

References:
    Brigo, D. & Mercurio, F. (2006). Interest Rate Models — Theory and
        Practice, 2nd ed., Chapter 22.
    Schonbucher, P. J. (2003). Credit Derivatives Pricing Models.
        Wiley Finance.
    Cox, J., Ingersoll, J. & Ross, S. (1985). A Theory of the Term
        Structure of Interest Rates. Econometrica 53(2).
    Longstaff, F. A. & Schwartz, E. S. (2001). Valuing American Options
        by Simulation: A Simple Least-Squares Approach. Review of
        Financial Studies 14(1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from pricebook.core.discount_curve import DiscountCurve


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StochasticBermudanCDSResult:
    """Pricing result for a Bermudan CDS swaption under CIR intensity."""

    price: float
    """Bermudan swaption price (present value, same sign convention as notional)."""

    european_price: float
    """Price restricted to the first exercise date only."""

    early_exercise_premium: float
    """Bermudan price minus European price."""

    exercise_probabilities: list[float]
    """Fraction of paths that exercise at each date (sums to total exercise rate)."""

    mean_exercise_time: float
    """Probability-weighted average exercise time in years."""

    spread_vol_contribution: float
    """Incremental price from stochastic hazard vs deterministic (hazard_rate_0 flat)."""

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "european_price": self.european_price,
            "early_exercise_premium": self.early_exercise_premium,
            "exercise_probabilities": self.exercise_probabilities,
            "mean_exercise_time": self.mean_exercise_time,
            "spread_vol_contribution": self.spread_vol_contribution,
        }


# ---------------------------------------------------------------------------
# CIR hazard simulation (exact via non-central chi-squared)
# ---------------------------------------------------------------------------

def _cir_step_noncentralchi2(
    lam: np.ndarray,
    dt: float,
    kappa: float,
    theta: float,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact CIR step using the non-central chi-squared representation.

    The CIR process at time t+dt given λ_t follows:

        c · λ_{t+dt} ~ χ²(d, c · λ_t · e^{-κ dt})

    where c = 4κ / (σ²(1 - e^{-κ dt})) and d = 4κθ / σ².

    Parameters
    ----------
    lam : shape (n_paths,) current hazard rates.
    dt  : time step.
    kappa, theta, sigma : CIR parameters.
    rng : numpy random generator.

    Returns
    -------
    shape (n_paths,) new hazard rates, non-negative.
    """
    exp_kdt = math.exp(-kappa * dt)
    sigma2 = sigma * sigma
    d = 4.0 * kappa * theta / sigma2          # degrees of freedom
    c = sigma2 * (1.0 - exp_kdt) / (4.0 * kappa)  # scaling factor
    nc = lam * exp_kdt / c                    # non-centrality parameter
    chi2 = rng.noncentral_chisquare(d, nc)    # shape (n_paths,)
    return np.maximum(c * chi2, 0.0)


# ---------------------------------------------------------------------------
# Analytical CDS PV under CIR intensity  (Riccati ODE solution)
# ---------------------------------------------------------------------------

def cir_cds_pv(
    hazard_rate: float | np.ndarray,
    mean_reversion: float,
    long_run_hazard: float,
    hazard_vol: float,
    remaining_years: float,
    strike_spread: float,
    discount_rate: float,
    recovery: float,
    n_coupons: int,
) -> float | np.ndarray:
    """Analytical CDS PV under CIR intensity.

    Uses the affine bond-price formula for the CIR model to compute survival
    probabilities.  For a unit-notional payer CDS with quarterly coupons and
    flat discount rate r:

        PV = LGD · ∫₀ᵀ r(t) · P(0,t) · S(t) dt
             - s · Σᵢ P(0,tᵢ) · S(tᵢ) · Δtᵢ

    where S(t) = E[exp(-∫₀ᵗ λ_u du)] is the CIR survival probability,
    P(0,t) = exp(-r·t), LGD = 1 - recovery, and s = strike_spread.

    Survival probability under CIR:

        S(t) = A(t) · exp(-B(t) · λ)

        h   = √(κ² + 2σ²)
        A(t)= [2h·exp((κ+h)t/2) / (2h + (κ+h)(exp(ht)-1))]^{2κθ/σ²}
        B(t)= 2(exp(ht)-1) / (2h + (κ+h)(exp(ht)-1))

    Parameters
    ----------
    hazard_rate     : current instantaneous hazard rate λ_t.
    mean_reversion  : κ — speed of mean reversion.
    long_run_hazard : θ — long-run mean hazard rate.
    hazard_vol      : σ — volatility of hazard rate.
    remaining_years : time to CDS maturity from current exercise date.
    strike_spread   : CDS coupon spread (decimal, e.g. 0.01 = 100 bps).
    discount_rate   : flat continuously compounded risk-free rate.
    recovery        : recovery rate (e.g. 0.4).
    n_coupons       : number of coupon periods remaining.

    Returns
    -------
    CDS PV (positive = in-the-money for payer).
    """
    if remaining_years <= 0.0:
        return np.zeros_like(hazard_rate) if isinstance(hazard_rate, np.ndarray) else 0.0

    kappa = mean_reversion
    theta = long_run_hazard
    sigma = hazard_vol
    lgd = 1.0 - recovery
    r = discount_rate
    lam = hazard_rate

    h = math.sqrt(kappa * kappa + 2.0 * sigma * sigma)
    two_h = 2.0 * h
    kappa_plus_h = kappa + h

    def _survival(t: float) -> float | np.ndarray:
        """CIR survival probability S(t) = A(t) exp(-B(t) λ)."""
        if t <= 0.0:
            return np.ones_like(lam) if isinstance(lam, np.ndarray) else 1.0
        exp_ht = math.exp(h * t)
        denom = two_h + kappa_plus_h * (exp_ht - 1.0)
        # A(t)
        numer_A = two_h * math.exp(kappa_plus_h * t / 2.0)
        log_A = (2.0 * kappa * theta / (sigma * sigma)) * math.log(numer_A / denom)
        A = math.exp(log_A)
        # B(t)
        B = 2.0 * (exp_ht - 1.0) / denom
        return A * np.exp(-B * lam)

    # Coupon leg: sum over quarterly dates
    dt_coupon = remaining_years / n_coupons
    coupon_pv: float | np.ndarray = 0.0
    for i in range(1, n_coupons + 1):
        t_i = i * dt_coupon
        df_i = math.exp(-r * t_i)
        s_i = _survival(t_i)
        coupon_pv = coupon_pv + df_i * s_i * dt_coupon

    # Protection leg: numerical integration over same grid
    prot_pv: float | np.ndarray = 0.0
    n_quad = max(n_coupons * 4, 40)
    dt_q = remaining_years / n_quad
    s_prev = _survival(0.0)
    for j in range(1, n_quad + 1):
        t_j = j * dt_q
        s_curr = _survival(t_j)
        t_mid = (t_j - dt_q / 2.0)
        df_mid = math.exp(-r * t_mid)
        # expected default in interval: -(dS/dt)·dt ≈ S_prev - S_curr
        prot_pv = prot_pv + df_mid * (s_prev - s_curr)
        s_prev = s_curr

    return lgd * prot_pv - strike_spread * coupon_pv


# ---------------------------------------------------------------------------
# Main pricer: LSM Bermudan CDS swaption under CIR
# ---------------------------------------------------------------------------

def stochastic_bermudan_cds_swaption(
    reference_date: date,
    exercise_dates: list[date],
    cds_maturity_years: float,
    strike_spread: float,
    discount_curve: DiscountCurve,
    hazard_rate_0: float,
    mean_reversion: float,
    long_run_hazard: float,
    hazard_vol: float,
    recovery: float = 0.4,
    notional: float = 10_000_000.0,
    is_payer: bool = True,
    n_paths: int = 50_000,
    n_steps_per_year: int = 12,
    seed: int = 42,
) -> StochasticBermudanCDSResult:
    """Price a Bermudan CDS swaption under stochastic (CIR) hazard rate.

    The holder may enter a CDS at any exercise date, paying (payer) or
    receiving (receiver) the strike spread until ``cds_maturity_years``
    from ``reference_date``.  The CIR intensity drives spread volatility.

    Algorithm (Longstaff-Schwartz):
    1. Simulate N CIR paths with exact non-central chi-squared steps.
    2. Track cumulative survival probability along each path.
    3. At each exercise date compute CDS PV analytically from (λ, t_remaining).
    4. Backward induction: regress continuation value on (λ, cum_survival)
       with polynomial basis; exercise when exercise value > continuation.

    Parameters
    ----------
    reference_date      : pricing date.
    exercise_dates      : sorted list of Bermudan exercise dates.
    cds_maturity_years  : CDS tenor from reference_date (e.g. 5.0).
    strike_spread       : CDS fixed coupon in decimal (e.g. 0.01 = 100 bps).
    discount_curve      : risk-free discount curve.
    hazard_rate_0       : initial instantaneous hazard rate λ₀.
    mean_reversion      : CIR κ parameter.
    long_run_hazard     : CIR θ (long-run mean hazard).
    hazard_vol          : CIR σ (hazard volatility).
    recovery            : recovery rate on default, default 0.4.
    notional            : trade notional.
    is_payer            : True = payer swaption (buy protection).
    n_paths             : number of Monte Carlo paths.
    n_steps_per_year    : simulation steps per year between exercise dates.
    seed                : random seed for reproducibility.

    Returns
    -------
    StochasticBermudanCDSResult
    """
    rng = np.random.default_rng(seed)

    # Convert exercise dates to year fractions
    exercise_times = []
    for ex in sorted(exercise_dates):
        delta = (ex - reference_date).days / 365.25
        exercise_times.append(delta)

    maturity_yrs = cds_maturity_years
    n_ex = len(exercise_times)

    # Discount rate (flat approximation from curve at maturity)
    mat_date_approx = date(
        reference_date.year + int(maturity_yrs),
        reference_date.month,
        reference_date.day,
    )
    try:
        df_mat = discount_curve.df(mat_date_approx)
        r_flat = -math.log(df_mat) / maturity_yrs if df_mat > 0 else 0.02
    except Exception:
        r_flat = 0.02

    # Build fine simulation grid
    all_times = sorted(set([0.0] + exercise_times))
    # Insert intermediate steps between exercise points
    sim_times: list[float] = [0.0]
    for i in range(1, len(all_times)):
        t_start = all_times[i - 1]
        t_end = all_times[i]
        n_sub = max(1, round((t_end - t_start) * n_steps_per_year))
        step = (t_end - t_start) / n_sub
        for j in range(1, n_sub + 1):
            sim_times.append(t_start + j * step)
    sim_times = sorted(set(round(t, 10) for t in sim_times))

    # Map exercise times to sim_times indices
    ex_indices = []
    for et in exercise_times:
        idx = min(range(len(sim_times)), key=lambda i: abs(sim_times[i] - et))
        ex_indices.append(idx)

    # Simulate CIR hazard paths
    n_sim = len(sim_times)
    lam_paths = np.empty((n_paths, n_sim))
    lam_paths[:, 0] = hazard_rate_0

    for step_i in range(1, n_sim):
        dt = sim_times[step_i] - sim_times[step_i - 1]
        lam_paths[:, step_i] = _cir_step_noncentralchi2(
            lam_paths[:, step_i - 1], dt,
            mean_reversion, long_run_hazard, hazard_vol, rng,
        )

    # Cumulative survival: S(0,t) = exp(-∫₀ᵗ λ du) ≈ exp(-Σ λ_i Δt_i)
    cum_log_surv = np.zeros(n_paths)
    surv_at = {}
    surv_at[0] = np.ones(n_paths)
    for step_i in range(1, n_sim):
        dt = sim_times[step_i] - sim_times[step_i - 1]
        cum_log_surv -= lam_paths[:, step_i - 1] * dt
        if step_i in ex_indices:
            surv_at[step_i] = np.exp(cum_log_surv)

    # Compute exercise values at each exercise date
    n_coupons = max(4, int(round(maturity_yrs * 4)))  # quarterly
    sign = 1.0 if is_payer else -1.0

    ex_values: list[np.ndarray] = []
    for k, (et, idx) in enumerate(zip(exercise_times, ex_indices)):
        remaining = maturity_yrs - et
        if remaining <= 0.0:
            ex_values.append(np.zeros(n_paths))
            continue
        n_coup_rem = max(1, int(round(remaining * 4)))
        lam_k = lam_paths[:, idx]
        surv_k = surv_at[idx]
        raw_pv = cir_cds_pv(
            lam_k, mean_reversion, long_run_hazard, hazard_vol,
            remaining, strike_spread, r_flat, recovery, n_coup_rem,
        )
        # Condition on not having defaulted: multiply by survival probability
        # (paths that defaulted have near-zero survival weight)
        exercise_val = sign * raw_pv * surv_k * notional
        ex_values.append(np.maximum(exercise_val, 0.0))

    # --- LSM backward induction -------------------------------------------
    # cashflow[p] = discounted cash flow received by path p
    cashflow = np.zeros(n_paths)
    exercise_at = np.full(n_paths, -1, dtype=int)  # which ex date exercised

    for k in range(n_ex - 1, -1, -1):
        et = exercise_times[k]
        idx = ex_indices[k]
        ev = ex_values[k]
        lam_k = lam_paths[:, idx]
        surv_k = surv_at[idx]

        # Discount factor from exercise date back to t=0
        try:
            ex_date = exercise_dates[k] if k < len(exercise_dates) else exercise_dates[-1]
            df_ex = discount_curve.df(ex_date)
        except Exception:
            df_ex = math.exp(-r_flat * et)

        # Discount future cashflows to this exercise date
        # (cashflow is already in PV at t=0; bring to t=et for comparison)
        future_cf_at_ex = cashflow / df_ex if df_ex > 1e-10 else cashflow

        # Regression on in-the-money paths
        itm = ev > 0.0
        if itm.sum() > 50:
            x1 = lam_k[itm]
            x2 = surv_k[itm]
            # Polynomial basis: [1, x1, x2, x1², x2², x1·x2]
            X = np.column_stack([
                np.ones(itm.sum()),
                x1, x2, x1 * x1, x2 * x2, x1 * x2,
            ])
            y = future_cf_at_ex[itm]
            try:
                coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                X_all = np.column_stack([
                    np.ones(n_paths),
                    lam_k, surv_k, lam_k * lam_k,
                    surv_k * surv_k, lam_k * surv_k,
                ])
                continuation = X_all @ coef
            except np.linalg.LinAlgError:
                continuation = future_cf_at_ex
        else:
            continuation = future_cf_at_ex

        # Exercise decision
        exercise_now = itm & (ev > continuation * df_ex)
        cashflow[exercise_now] = ev[exercise_now] * df_ex
        exercise_at[exercise_now] = k

    # Price = mean discounted payoff
    price = float(np.mean(cashflow))

    # European price: only first exercise date
    ev_first = ex_values[0]
    try:
        df_first = discount_curve.df(exercise_dates[0])
    except Exception:
        df_first = math.exp(-r_flat * exercise_times[0])
    european_price = float(np.mean(ev_first) * df_first)

    early_exercise_premium = price - european_price

    # Exercise probabilities per date
    exercise_probs = []
    for k in range(n_ex):
        prob = float(np.mean(exercise_at == k))
        exercise_probs.append(round(prob, 6))

    # Mean exercise time (probability weighted)
    exercised = exercise_at >= 0
    if exercised.sum() > 0:
        times_arr = np.array(exercise_times)
        exercised_times = times_arr[exercise_at[exercised]]
        mean_ex_time = float(np.mean(exercised_times))
    else:
        mean_ex_time = 0.0

    # Spread vol contribution: compare to deterministic (zero vol) price
    det_result = stochastic_bermudan_cds_swaption(
        reference_date, exercise_dates, cds_maturity_years, strike_spread,
        discount_curve, hazard_rate_0, mean_reversion, long_run_hazard,
        hazard_vol=1e-6,  # near-zero vol = deterministic
        recovery=recovery, notional=notional, is_payer=is_payer,
        n_paths=n_paths, n_steps_per_year=n_steps_per_year, seed=seed,
    ) if hazard_vol > 1e-4 else None

    spread_vol_contribution = (
        price - det_result.price if det_result is not None else 0.0
    )

    return StochasticBermudanCDSResult(
        price=price,
        european_price=european_price,
        early_exercise_premium=early_exercise_premium,
        exercise_probabilities=exercise_probs,
        mean_exercise_time=mean_ex_time,
        spread_vol_contribution=spread_vol_contribution,
    )


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def stochastic_vs_deterministic(
    reference_date: date,
    exercise_dates: list[date],
    cds_maturity_years: float,
    strike_spread: float,
    discount_curve: DiscountCurve,
    hazard_rate_0: float,
    mean_reversion: float,
    long_run_hazard: float,
    hazard_vol: float,
    recovery: float = 0.4,
) -> dict:
    """Compare stochastic vs deterministic Bermudan CDS swaption prices.

    Runs the CIR stochastic model alongside a near-zero vol (deterministic)
    version and quantifies the spread-volatility value.

    Parameters
    ----------
    reference_date      : pricing date.
    exercise_dates      : Bermudan exercise dates.
    cds_maturity_years  : CDS tenor in years from reference_date.
    strike_spread       : CDS coupon spread (decimal).
    discount_curve      : risk-free discount curve.
    hazard_rate_0       : initial hazard rate.
    mean_reversion      : CIR κ.
    long_run_hazard     : CIR θ.
    hazard_vol          : CIR σ.
    recovery            : recovery rate.

    Returns
    -------
    dict with keys: stochastic_price, deterministic_price, difference,
    spread_vol_contribution, stochastic_result, deterministic_result.
    """
    stoch = stochastic_bermudan_cds_swaption(
        reference_date, exercise_dates, cds_maturity_years, strike_spread,
        discount_curve, hazard_rate_0, mean_reversion, long_run_hazard,
        hazard_vol, recovery=recovery,
    )
    det = stochastic_bermudan_cds_swaption(
        reference_date, exercise_dates, cds_maturity_years, strike_spread,
        discount_curve, hazard_rate_0, mean_reversion, long_run_hazard,
        hazard_vol=1e-6, recovery=recovery,
    )
    diff = stoch.price - det.price
    return {
        "stochastic_price": stoch.price,
        "deterministic_price": det.price,
        "difference": diff,
        "spread_vol_contribution": diff,
        "stochastic_result": stoch,
        "deterministic_result": det,
    }
